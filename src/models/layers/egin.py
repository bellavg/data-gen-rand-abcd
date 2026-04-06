import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter

# Adopted from: https://github.com/YxRicardo/EGIN/blob/main/models/graphegin.py 

try:
	from model_utils import get_norm_layer, get_pyg_pool
except ImportError:  # pragma: no cover - fallback for package-style imports
	from models.model_utils import get_norm_layer, get_pyg_pool


def _get_activation(name: str) -> nn.Module:
	name = name.lower()
	if name == "relu":
		return nn.ReLU()
	if name == "gelu":
		return nn.GELU()
	if name == "elu":
		return nn.ELU()
	if name == "silu":
		return nn.SiLU()
	raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
	"""Simple MLP used by EGIN blocks."""

	def __init__(
		self,
		num_layers: int,
		input_dim: int,
		hidden_dim: int,
		output_dim: int,
		activation: str = "relu",
		dropout: float = 0.0,
	):
		super().__init__()
		if num_layers < 1:
			raise ValueError("num_layers must be >= 1")

		self.num_layers = num_layers
		self.activation = _get_activation(activation)
		self.dropout = nn.Dropout(dropout)

		if num_layers == 1:
			self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim)])
		else:
			linears = [nn.Linear(input_dim, hidden_dim)]
			for _ in range(num_layers - 2):
				linears.append(nn.Linear(hidden_dim, hidden_dim))
			linears.append(nn.Linear(hidden_dim, output_dim))
			self.linears = nn.ModuleList(linears)

	def forward(self, x: Tensor) -> Tensor:
		if self.num_layers == 1:
			return self.linears[0](x)

		h = x
		for linear in self.linears[:-1]:
			h = self.activation(linear(h))
			h = self.dropout(h)
		return self.linears[-1](h)


class GraphEGIN(nn.Module):
	"""
	PyG-only implementation of EGIN-style graph model.

	Supports:
	- edge-aware updates in dot-product or concat modes
	- learnable epsilon updates
	- optional edge MLP in concat mode
	- configurable normalization and graph readout
	- optional positional encoding integration
	"""

	def __init__(
		self,
		num_layers: int,
		num_mlp_layers: int,
		num_edge_feat: int,
		input_dim: int,
		hidden_dim: int,
		output_dim: int,
		final_dropout: float = 0.0,
		learn_eps: bool = False,
		dot_update: bool = False,
		edge_mlp: bool = False,
		edge_hidden_dim: int | None = None,
		norm_type: str = "batch",
		readout: str = "add",
		activation: str = "relu",
		pos_enc_dim: int = 0,
		pos_enc_mode: str = "none",
	):
		super().__init__()
		if num_layers < 2:
			raise ValueError("num_layers must be >= 2 (input layer + at least one EGIN block)")
		if num_edge_feat < 1:
			raise ValueError("num_edge_feat must be >= 1 for EGIN")

		self.num_layers = num_layers
		self.num_edge_feat = num_edge_feat
		self.learn_eps = learn_eps
		self.dot_update = dot_update
		self.edge_mlp = edge_mlp
		self.final_dropout = final_dropout
		self.pos_enc_dim = pos_enc_dim
		self.pos_enc_mode = pos_enc_mode.lower()
		self.pool_fn = get_pyg_pool(readout)
		self.act = _get_activation(activation)

		if self.pos_enc_mode not in {"none", "concat", "add"}:
			raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")

		self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

		effective_input_dim = input_dim
		if self.pos_enc_mode == "concat" and pos_enc_dim > 0:
			effective_input_dim += pos_enc_dim

		self.pos_add_proj = (
			nn.Linear(pos_enc_dim, effective_input_dim, bias=False)
			if self.pos_enc_mode == "add" and pos_enc_dim > 0
			else None
		)

		self.edge_hidden_dim = edge_hidden_dim if edge_hidden_dim is not None else num_edge_feat

		self.mlps = nn.ModuleList()
		self.norms = nn.ModuleList()

		if self.edge_mlp and not self.dot_update:
			self.edge_mlps = nn.ModuleList(
				[nn.Linear(num_edge_feat, self.edge_hidden_dim) for _ in range(self.num_layers - 1)]
			)
		else:
			self.edge_mlps = None

		for layer in range(self.num_layers - 1):
			node_dim = effective_input_dim if layer == 0 else hidden_dim

			if self.dot_update:
				mlp_in_dim = node_dim * num_edge_feat
			elif self.edge_mlp:
				mlp_in_dim = node_dim + self.edge_hidden_dim
			else:
				mlp_in_dim = node_dim + num_edge_feat

			self.mlps.append(
				MLP(
					num_layers=num_mlp_layers,
					input_dim=mlp_in_dim,
					hidden_dim=hidden_dim,
					output_dim=hidden_dim,
					activation=activation,
					dropout=final_dropout,
				)
			)
			self.norms.append(get_norm_layer(norm_type, hidden_dim))

		self.linears_prediction = nn.ModuleList()
		for layer in range(num_layers):
			if layer == 0:
				self.linears_prediction.append(nn.Linear(effective_input_dim, output_dim))
			else:
				self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

	def _validate_positional_encoding(self, pos_enc: Tensor | None) -> None:
		if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == "none":
			return
		if pos_enc.size(-1) != self.pos_enc_dim:
			raise ValueError(
				f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
			)

	def _integrate_positional_encoding(self, x: Tensor, pos_enc: Tensor | None) -> Tensor:
		if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == "none":
			return x
		if self.pos_enc_mode == "concat":
			return torch.cat([x, pos_enc], dim=-1)
		if self.pos_add_proj is None:
			raise ValueError("pos_enc_mode='add' requires pos_enc_dim > 0")
		return x + self.pos_add_proj(pos_enc)

	@staticmethod
	def _to_2d_edge_attr(edge_attr: Tensor) -> Tensor:
		if edge_attr.dim() == 1:
			return edge_attr.view(-1, 1)
		return edge_attr

	def _dot_update_aggregate(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int) -> Tensor:
		src, dst = edge_index
		num_nodes = h.size(0)
		edge_feat_dim = edge_attr.size(-1)

		# Match original adjacency-based behavior: aggregate over outgoing neighbors.
		msg = h[dst].unsqueeze(-1) * edge_attr.unsqueeze(1)
		pooled = scatter(msg.reshape(msg.size(0), -1), src, dim=0, dim_size=num_nodes, reduce="sum")

		if self.learn_eps:
			pooled = pooled + self.eps[layer] * h.repeat(1, edge_feat_dim)
		return pooled

	def _concat_update_aggregate(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int) -> Tensor:
		src, dst = edge_index
		num_nodes = h.size(0)

		node_agg = scatter(h[dst], src, dim=0, dim_size=num_nodes, reduce="sum")
		edge_rep = scatter(edge_attr, src, dim=0, dim_size=num_nodes, reduce="sum")

		if self.edge_mlp and self.edge_mlps is not None:
			edge_part = self.edge_mlps[layer](edge_rep)
		else:
			edge_part = edge_rep

		pooled = torch.cat((node_agg, edge_part), dim=-1)

		if self.learn_eps:
			edge_unit = torch.ones_like(edge_rep)
			if self.edge_mlp and self.edge_mlps is not None:
				edge_unit_part = self.edge_mlps[layer](edge_unit)
			else:
				edge_unit_part = edge_unit
			pooled = pooled + self.eps[layer] * torch.cat((h, edge_unit_part), dim=-1)

		return pooled

	def _egin_next_layer(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, layer: int) -> Tensor:
		if self.dot_update:
			pooled = self._dot_update_aggregate(h, edge_index, edge_attr, layer)
		else:
			pooled = self._concat_update_aggregate(h, edge_index, edge_attr, layer)

		h = self.mlps[layer](pooled)
		h = self.norms[layer](h)
		h = self.act(h)
		return h

	def egin_forward(
		self,
		x: Tensor,
		edge_index: Tensor,
		batch: Tensor,
		edge_attr: Tensor,
		pos_enc: Tensor | None = None,
	) -> Tensor:
		if edge_attr is None:
			raise ValueError("EGIN requires edge_attr tensor.")

		edge_attr = self._to_2d_edge_attr(edge_attr)

		if edge_attr.size(-1) != self.num_edge_feat:
			raise ValueError(
				f"Expected edge_attr feature size {self.num_edge_feat}, got {edge_attr.size(-1)}"
			)

		self._validate_positional_encoding(pos_enc)
		x = self._integrate_positional_encoding(x, pos_enc)

		hidden_rep = [x]
		h = x

		for layer in range(self.num_layers - 1):
			h = self._egin_next_layer(h, edge_index, edge_attr, layer)
			hidden_rep.append(h)

		score_over_layer = 0.0
		for layer, h_layer in enumerate(hidden_rep):
			pooled_h = self.pool_fn(h_layer, batch)
			logits = self.linears_prediction[layer](pooled_h)
			score_over_layer = score_over_layer + F.dropout(
				logits,
				p=self.final_dropout,
				training=self.training,
			)

		return score_over_layer

	def forward(
		self,
		x: Tensor,
		edge_index: Tensor,
		batch: Tensor,
		edge_attr: Tensor = None,
		edge_type: Tensor | None = None,
		pos_enc: Tensor | None = None,
	) -> Tensor:
		# edge_type is accepted for interface compatibility but unused.
		return self.egin_forward(
			x=x,
			edge_index=edge_index,
			batch=batch,
			edge_attr=edge_attr,
			pos_enc=pos_enc,
		)

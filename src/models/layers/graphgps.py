from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.typing import Adj, OptTensor

# Adopted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py 

try:
	from torch_geometric.nn.attention import PerformerAttention
except Exception:  # pragma: no cover
	PerformerAttention = None

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


class RedrawProjection:
	"""Helper for periodically redrawing Performer projection matrices."""

	def __init__(self, model: nn.Module, redraw_interval: Optional[int] = None):
		self.model = model
		self.redraw_interval = redraw_interval
		self.num_last_redraw = 0

	def redraw_projections(self):
		if not self.model.training or self.redraw_interval is None:
			return

		if self.num_last_redraw >= self.redraw_interval:
			if PerformerAttention is not None:
				modules = [m for m in self.model.modules() if isinstance(m, PerformerAttention)]
			else:
				modules = [m for m in self.model.modules() if hasattr(m, "redraw_projection_matrix")]

			for module in modules:
				module.redraw_projection_matrix()

			self.num_last_redraw = 0
			return

		self.num_last_redraw += 1


class GraphGPSLayer(nn.Module):
	"""Single GraphGPS block with local GINE and global attention."""

	def __init__(
		self,
		hidden_dim: int,
		dropout: float = 0.0,
		norm_type: str = "batch",
		activation: str = "relu",
		residual: bool = True,
		ffn: bool = True,
		heads: int = 4,
		attn_type: str = "multihead",
		attn_kwargs: Optional[Dict[str, Any]] = None,
	):
		super().__init__()
		self.residual = residual
		self.ffn = ffn

		local_nn = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			_get_activation(activation),
			nn.Linear(hidden_dim, hidden_dim),
		)
		local_conv = GINEConv(local_nn, edge_dim=hidden_dim)
		self.conv = GPSConv(
			hidden_dim,
			local_conv,
			heads=heads,
			attn_type=attn_type,
			attn_kwargs={} if attn_kwargs is None else attn_kwargs,
		)

		self.norm = get_norm_layer(norm_type, hidden_dim)
		self.act = _get_activation(activation)
		self.drop = nn.Dropout(dropout)

		if self.ffn:
			self.ffn_norm1 = get_norm_layer(norm_type, hidden_dim)
			self.ffn_linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
			self.ffn_linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
			self.ffn_act = _get_activation(activation)
			self.ffn_drop1 = nn.Dropout(dropout)
			self.ffn_drop2 = nn.Dropout(dropout)
			self.ffn_norm2 = get_norm_layer(norm_type, hidden_dim)

	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.ffn_drop1(self.ffn_act(self.ffn_linear1(x)))
		return self.ffn_drop2(self.ffn_linear2(x))

	def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, edge_attr: OptTensor) -> Tensor:
		x_in = x
		x = self.conv(x, edge_index, batch=batch, edge_attr=edge_attr)
		x = self.norm(x)
		x = self.act(x)
		x = self.drop(x)

		if self.residual and x.shape == x_in.shape:
			x = x + x_in

		if self.ffn:
			x = self.ffn_norm1(x)
			x = x + self._ff_block(x)
			x = self.ffn_norm2(x)

		return x


class GraphGPSEncoder(nn.Module):
	"""
	PyG-only GraphGPS encoder for graph-level tasks.

	Supports node/edge features, optional positional encoding, configurable
	normalization/readout, and optional Performer projection redraw.
	"""

	def __init__(
		self,
		in_dim: int,
		edge_dim: int,
		hidden_dim: int,
		out_dim: int,
		num_layers: int,
		use_input_proj: bool = True,
		use_edge_proj: bool = True,
		use_output_proj: bool = True,
		node_vocab_size: Optional[int] = None,
		edge_vocab_size: Optional[int] = None,
		pos_enc_dim: int = 0,
		pos_enc_mode: str = "none",
		pe_out_dim: Optional[int] = None,
		dropout: float = 0.0,
		norm_type: str = "batch",
		readout: str = "add",
		activation: str = "relu",
		residual: bool = True,
		ffn: bool = True,
		heads: int = 4,
		attn_type: str = "multihead",
		attn_kwargs: Optional[Dict[str, Any]] = None,
		performer_redraw_interval: Optional[int] = None,
	):
		super().__init__()
		if num_layers < 1:
			raise ValueError("num_layers must be >= 1")

		self.use_input_proj = use_input_proj
		self.use_edge_proj = use_edge_proj
		self.use_output_proj = use_output_proj
		self.node_vocab_size = node_vocab_size
		self.edge_vocab_size = edge_vocab_size
		self.hidden_dim = hidden_dim
		self.pos_enc_dim = pos_enc_dim
		self.pos_enc_mode = pos_enc_mode.lower()

		if self.pos_enc_mode not in {"none", "concat", "add"}:
			raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")

		if self.pos_enc_mode == "concat" and pos_enc_dim > 0:
			if pe_out_dim is None:
				pe_out_dim = min(pos_enc_dim, max(1, hidden_dim // 4))
			if pe_out_dim >= hidden_dim:
				raise ValueError("pe_out_dim must be smaller than hidden_dim for concat mode")
			node_out_dim = hidden_dim - pe_out_dim
			self.pe_norm = get_norm_layer(norm_type, pos_enc_dim)
			self.pe_lin = nn.Linear(pos_enc_dim, pe_out_dim)
			self.pe_add_lin = None
		elif self.pos_enc_mode == "add" and pos_enc_dim > 0:
			node_out_dim = hidden_dim
			self.pe_norm = get_norm_layer(norm_type, pos_enc_dim)
			self.pe_lin = None
			self.pe_add_lin = nn.Linear(pos_enc_dim, hidden_dim)
		else:
			node_out_dim = hidden_dim
			self.pe_norm = nn.Identity()
			self.pe_lin = None
			self.pe_add_lin = None

		if node_vocab_size is not None:
			self.node_encoder = nn.Embedding(node_vocab_size, node_out_dim)
			self.node_is_embedding = True
			self.node_identity = False
		else:
			if self.use_input_proj:
				self.node_encoder = nn.Linear(in_dim, node_out_dim)
				node_identity = False
			else:
				if in_dim != node_out_dim:
					raise ValueError(
						f"use_input_proj=False requires in_dim ({in_dim}) == node_out_dim ({node_out_dim})"
					)
				self.node_encoder = nn.Identity()
				node_identity = True
			self.node_is_embedding = False
			self.node_identity = node_identity

		if edge_vocab_size is not None:
			self.edge_encoder = nn.Embedding(edge_vocab_size, hidden_dim)
			self.edge_is_embedding = True
			self.edge_identity = False
		else:
			if self.use_edge_proj:
				self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
				edge_identity = False
			else:
				if edge_dim != hidden_dim:
					raise ValueError(
						f"use_edge_proj=False requires edge_dim ({edge_dim}) == hidden_dim ({hidden_dim})"
					)
				self.edge_encoder = nn.Identity()
				edge_identity = True
			self.edge_is_embedding = False
			self.edge_identity = edge_identity

		self.layers = nn.ModuleList(
			[
				GraphGPSLayer(
					hidden_dim=hidden_dim,
					dropout=dropout,
					norm_type=norm_type,
					activation=activation,
					residual=residual,
					ffn=ffn,
					heads=heads,
					attn_type=attn_type,
					attn_kwargs=attn_kwargs,
				)
				for _ in range(num_layers)
			]
		)

		if self.use_output_proj:
			self.out_proj = nn.Linear(hidden_dim, out_dim)
		else:
			if hidden_dim != out_dim:
				raise ValueError(
					f"use_output_proj=False requires hidden_dim ({hidden_dim}) == out_dim ({out_dim})"
				)
			self.out_proj = nn.Identity()
		self.redraw_projection = RedrawProjection(
			self.layers,
			redraw_interval=performer_redraw_interval,
		)

	def _validate_positional_encoding(self, pos_enc: OptTensor) -> None:
		if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == "none":
			return
		if pos_enc.size(-1) != self.pos_enc_dim:
			raise ValueError(
				f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
			)

	def _encode_nodes(self, x: Tensor, pos_enc: OptTensor = None) -> Tensor:
		if self.node_is_embedding:
			if x.dim() > 1:
				x = x.squeeze(-1)
			x = self.node_encoder(x.long())
		else:
			if x.dim() == 1:
				x = x.unsqueeze(-1)
			x = self.node_encoder(x)

		if pos_enc is not None and self.pos_enc_mode != "none" and self.pos_enc_dim > 0:
			pos_enc = self.pe_norm(pos_enc)
			if self.pos_enc_mode == "concat":
				x = torch.cat((x, self.pe_lin(pos_enc)), dim=-1)
			elif self.pos_enc_mode == "add":
				x = x + self.pe_add_lin(pos_enc)

		return x

	def _encode_edges(self, edge_attr: OptTensor, edge_index: Adj, ref_x: Tensor) -> Tensor:
		if edge_attr is None:
			if isinstance(edge_index, Tensor):
				num_edges = edge_index.size(1)
			else:
				num_edges = edge_index.nnz()
			return ref_x.new_zeros((num_edges, self.hidden_dim))

		if self.edge_is_embedding:
			if edge_attr.dim() > 1:
				edge_attr = edge_attr.squeeze(-1)
			return self.edge_encoder(edge_attr.long())

		if edge_attr.dim() == 1:
			edge_attr = edge_attr.unsqueeze(-1)
		return self.edge_encoder(edge_attr)

	def forward(
		self,
		x: Tensor,
		edge_index: Adj,
		batch: Tensor,
		edge_attr: OptTensor = None,
		edge_type: OptTensor = None,
		pos_enc: OptTensor = None,
	) -> Tensor:
		# edge_type is accepted for interface consistency and is unused in GraphGPS.
		_ = edge_type
		self._validate_positional_encoding(pos_enc)

		x = self._encode_nodes(x, pos_enc=pos_enc)
		edge_attr = self._encode_edges(edge_attr, edge_index=edge_index, ref_x=x)

		for layer in self.layers:
			x = layer(x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)

		x = self.out_proj(x)
		return x

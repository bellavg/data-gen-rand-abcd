import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch_geometric.nn import GINEConv
from torch_geometric.typing import Adj, OptTensor
from src.layers.positional_encoding import validate_positional_encoding, integrate_positional_encoding

# Adopted from: https://github.com/LUOyk1999/GNNPlus/blob/main/GNNPlus/layer/gine_conv_layer.py 

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


class GINEConvESLapPE(pyg_nn.conv.MessagePassing):
	"""
	GINE-style message passing with optional EquivStable LapPE scaling.

	Message:
		m_ij = ReLU(x_j + e_ij) * r_ij
	where r_ij is produced from LapPE distances when provided.
	"""

	def __init__(self, nn_module: nn.Module, eps: float = 0.0, train_eps: bool = False, edge_dim: int | None = None, **kwargs):
		kwargs.setdefault("aggr", "add")
		super().__init__(**kwargs)
		self.nn = nn_module
		self.initial_eps = eps

		if train_eps:
			self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float32))
		else:
			self.register_buffer("eps", torch.tensor([eps], dtype=torch.float32))

		in_channels = self._infer_first_mlp_in_channels(self.nn)
		self.lin = pyg_nn.Linear(edge_dim, in_channels) if edge_dim is not None else None

		# Scalar gating MLP over LapPE pairwise distance.
		self.mlp_r_ij = nn.Sequential(
			nn.Linear(1, in_channels),
			nn.ReLU(),
			nn.Linear(in_channels, 1),
			nn.Sigmoid(),
		)
		self.reset_parameters()

	@staticmethod
	def _infer_first_mlp_in_channels(module: nn.Module) -> int:
		first = module[0] if isinstance(module, nn.Sequential) else module
		if hasattr(first, "in_features"):
			return first.in_features
		if hasattr(first, "in_channels"):
			return first.in_channels
		raise ValueError("Unable to infer in_channels from nn_module; first layer must expose in_features or in_channels.")

	def reset_parameters(self):
		pyg_nn.inits.reset(self.nn)
		self.eps.data.fill_(self.initial_eps)
		if self.lin is not None:
			self.lin.reset_parameters()
		pyg_nn.inits.reset(self.mlp_r_ij)

	def forward(
		self,
		x: Tensor,
		edge_index: Adj,
		edge_attr: OptTensor = None,
		pe_lap_pe: OptTensor = None,
		size=None,
	) -> Tensor:
		x_pair = (x, x)
		out = self.propagate(edge_index, x=x_pair, edge_attr=edge_attr, PE=pe_lap_pe, size=size)
		out = out + (1 + self.eps) * x_pair[1]
		return self.nn(out)

	def message(self, x_j: Tensor, edge_attr: OptTensor, PE_i: OptTensor = None, PE_j: OptTensor = None) -> Tensor:
		if edge_attr is not None:
			if self.lin is not None:
				edge_attr = self.lin(edge_attr)
			elif x_j.size(-1) != edge_attr.size(-1):
				raise ValueError(
					"Node and edge feature dimensionalities do not match. "
					"Set edge_dim in GINEConvESLapPE to enable edge projection."
				)
			msg = (x_j + edge_attr).relu()
		else:
			msg = x_j.relu()

		if PE_i is None or PE_j is None:
			r_ij = torch.ones(msg.size(0), 1, device=msg.device, dtype=msg.dtype)
		else:
			r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
			r_ij = self.mlp_r_ij(r_ij)

		return msg * r_ij

	def __repr__(self):
		return f"{self.__class__.__name__}(nn={self.nn})"


class GINEConvLayer(nn.Module):
	"""Node-update GINE layer block with residual and optional FFN (PyG-only)."""

	def __init__(
		self,
		dim_in: int,
		dim_out: int,
		edge_dim: int | None,
		dropout: float = 0.0,
		residual: bool = False,
		ffn: bool = False,
		activation: str = "relu",
		norm_type: str = "batch",
		use_es_lap_pe: bool = False,
		train_eps: bool = False,
	):
		super().__init__()
		self.dropout = dropout
		self.residual = residual
		self.ffn = ffn
		self.use_es_lap_pe = use_es_lap_pe

		mlp = nn.Sequential(nn.Linear(dim_in, dim_out), _get_activation(activation), nn.Linear(dim_out, dim_out))

		if use_es_lap_pe:
			self.model = GINEConvESLapPE(mlp, train_eps=train_eps, edge_dim=edge_dim)
		else:
			self.model = GINEConv(mlp, train_eps=train_eps, edge_dim=edge_dim)

		self.norm = get_norm_layer(norm_type, dim_out)
		self.act = _get_activation(activation)
		self.drop = nn.Dropout(dropout)

		if self.ffn:
			self.ffn_norm1 = get_norm_layer(norm_type, dim_out)
			self.ffn_linear1 = nn.Linear(dim_out, dim_out * 2)
			self.ffn_linear2 = nn.Linear(dim_out * 2, dim_out)
			self.ffn_act = _get_activation(activation)
			self.ffn_drop1 = nn.Dropout(dropout)
			self.ffn_drop2 = nn.Dropout(dropout)
			self.ffn_norm2 = get_norm_layer(norm_type, dim_out)

	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.ffn_drop1(self.ffn_act(self.ffn_linear1(x)))
		return self.ffn_drop2(self.ffn_linear2(x))

	def forward(self, batch):
		x_in = batch.x
		edge_attr = getattr(batch, "edge_attr", None)

		if self.use_es_lap_pe:
			pe_lap_pe = getattr(batch, "pe_lap_pe", None)
			batch.x = self.model(batch.x, batch.edge_index, edge_attr=edge_attr, pe_lap_pe=pe_lap_pe)
		else:
			batch.x = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)

		batch.x = self.norm(batch.x)
		batch.x = self.act(batch.x)
		batch.x = self.drop(batch.x)

		if self.residual and x_in.shape == batch.x.shape:
			batch.x = x_in + batch.x

		if self.ffn:
			batch.x = self.ffn_norm1(batch.x)
			batch.x = batch.x + self._ff_block(batch.x)
			batch.x = self.ffn_norm2(batch.x)

		return batch


class GINEEncoder(nn.Module):
	"""Graph-level GINE encoder with optional positional encoding and LapPE edge scaling."""

	def __init__(
		self,
		in_dim: int,
		hid_dim: int,
		out_dim: int,
		num_layers: int,
		edge_dim: int | None,
		pos_enc_dim: int = 0,
		pos_enc_mode: str = "concat",
		use_input_proj: bool = True,
		dropout: float = 0.0,
		norm_type: str = "batch",
		readout: str = "mean",
		residual: bool = False,
		jk: str = "last",
		activation: str = "relu",
		use_es_lap_pe: bool = False,
		train_eps: bool = False,
	):
		super().__init__()
		if num_layers < 1:
			raise ValueError("num_layers must be >= 1")

		self.num_layers = num_layers
		self.dropout = dropout
		self.residual = residual
		self.pos_enc_dim = pos_enc_dim
		self.pos_enc_mode = pos_enc_mode.lower()
		self.use_input_proj = use_input_proj
		self.jk = jk.lower()
		self.use_es_lap_pe = use_es_lap_pe

		if self.pos_enc_mode not in {"none", "concat", "add"}:
			raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")
		if self.jk not in {"last", "sum", "cat"}:
			raise ValueError(f"Unknown jk mode: {jk}")

		effective_in_dim = in_dim + pos_enc_dim if self.pos_enc_mode == "concat" and pos_enc_dim > 0 else in_dim
		if self.use_input_proj:
			self.input_proj = nn.Linear(effective_in_dim, hid_dim)
		else:
			if effective_in_dim != hid_dim:
				raise ValueError(
					f"use_input_proj=False requires effective_in_dim ({effective_in_dim}) == hid_dim ({hid_dim})"
				)
			self.input_proj = nn.Identity()
		self.pos_add_proj = (
			nn.Linear(pos_enc_dim, hid_dim, bias=False)
			if self.pos_enc_mode == "add" and pos_enc_dim > 0
			else None
		)

		self.convs = nn.ModuleList()
		self.norms = nn.ModuleList()

		for layer_idx in range(num_layers):
			layer_in = hid_dim
			layer_out = out_dim if layer_idx == num_layers - 1 else hid_dim
			mlp = nn.Sequential(
				nn.Linear(layer_in, layer_out),
				_get_activation(activation),
				nn.Linear(layer_out, layer_out),
			)

			if use_es_lap_pe:
				conv = GINEConvESLapPE(mlp, train_eps=train_eps, edge_dim=edge_dim)
			else:
				conv = GINEConv(mlp, train_eps=train_eps, edge_dim=edge_dim)

			self.convs.append(conv)
			self.norms.append(get_norm_layer(norm_type, layer_out))

		self.out_dim = out_dim * num_layers if self.jk == "cat" else out_dim


	def forward(
		self,
		x: Tensor,
		edge_index: Adj,
		batch: Tensor,
		edge_attr: OptTensor = None,
		edge_type: OptTensor = None,
		pos_enc: OptTensor = None,
		pe_lap_pe: OptTensor = None,
	) -> Tensor:
		# edge_type is accepted for interface consistency and is unused in GINE.
		_ = edge_type
		self._validate_positional_encoding(pos_enc)
		if self.use_es_lap_pe and pe_lap_pe is None:
			pe_lap_pe = pos_enc

		x = self._integrate_positional_encoding_input(x, pos_enc)
		x = self.input_proj(x)
		if pos_enc is not None and self.pos_add_proj is not None:
			x = x + self.pos_add_proj(pos_enc)

		h_list = []

		for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
			x_in = x
			if self.use_es_lap_pe:
				x = conv(x, edge_index, edge_attr=edge_attr, pe_lap_pe=pe_lap_pe)
			else:
				x = conv(x, edge_index, edge_attr=edge_attr)

			x = norm(x)
			if layer_idx < self.num_layers - 1:
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)

			if self.residual and x.shape == x_in.shape:
				x = x + x_in

			h_list.append(x)

		if self.jk == "cat":
			return torch.cat(h_list, dim=-1)
		if self.jk == "sum":
			return torch.stack(h_list, dim=0).sum(dim=0)
		return h_list[-1]

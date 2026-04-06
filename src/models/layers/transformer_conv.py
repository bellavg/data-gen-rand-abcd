import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj, OptTensor

try:
	from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
	from models.model_utils import get_norm_layer


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


class TransformerConvLayer(nn.Module):
	"""Basic TransformerConv block with norm, activation, residual, and optional FFN."""

	def __init__(
		self,
		dim_in: int,
		dim_out: int,
		edge_dim: int | None = None,
		heads: int = 4,
		concat: bool = False,
		attn_dropout: float = 0.0,
		dropout: float = 0.0,
		norm_type: str = "batch",
		activation: str = "relu",
		residual: bool = True,
		ffn: bool = False,
		beta: bool = False,
		root_weight: bool = True,
	):
		super().__init__()
		self.residual = residual
		self.ffn = ffn

		self.conv = TransformerConv(
			in_channels=dim_in,
			out_channels=dim_out,
			heads=heads,
			concat=concat,
			beta=beta,
			dropout=attn_dropout,
			edge_dim=edge_dim,
			root_weight=root_weight,
		)

		self.conv_out_dim = dim_out * heads if concat else dim_out
		self.norm = get_norm_layer(norm_type, self.conv_out_dim)
		self.act = _get_activation(activation)
		self.drop = nn.Dropout(dropout)

		if self.ffn:
			self.ffn_norm1 = get_norm_layer(norm_type, self.conv_out_dim)
			self.ffn_linear1 = nn.Linear(self.conv_out_dim, self.conv_out_dim * 2)
			self.ffn_linear2 = nn.Linear(self.conv_out_dim * 2, self.conv_out_dim)
			self.ffn_act = _get_activation(activation)
			self.ffn_drop1 = nn.Dropout(dropout)
			self.ffn_drop2 = nn.Dropout(dropout)
			self.ffn_norm2 = get_norm_layer(norm_type, self.conv_out_dim)

	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.ffn_drop1(self.ffn_act(self.ffn_linear1(x)))
		return self.ffn_drop2(self.ffn_linear2(x))

	def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
		x_in = x
		x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
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


class TransformerConvEncoder(nn.Module):
	"""Graph-level TransformerConv encoder with optional positional encoding."""

	def __init__(
		self,
		in_dim: int,
		hid_dim: int,
		out_dim: int,
		num_layers: int,
		edge_dim: int | None = None,
		pos_enc_dim: int = 0,
		pos_enc_mode: str = "none",
		use_input_proj: bool = True,
		heads: int = 4,
		concat: bool = False,
		attn_dropout: float = 0.0,
		dropout: float = 0.0,
		norm_type: str = "batch",
		readout: str = "mean",
		activation: str = "relu",
		residual: bool = True,
		ffn: bool = False,
		beta: bool = False,
		root_weight: bool = True,
	):
		super().__init__()
		if num_layers < 1:
			raise ValueError("num_layers must be >= 1")

		self.num_layers = num_layers
		self.pos_enc_dim = pos_enc_dim
		self.pos_enc_mode = pos_enc_mode.lower()
		self.use_input_proj = use_input_proj

		if self.pos_enc_mode not in {"none", "concat", "add"}:
			raise ValueError(f"Unknown pos_enc_mode: {pos_enc_mode}")

		effective_in_dim = in_dim
		if self.pos_enc_mode == "concat" and pos_enc_dim > 0:
			effective_in_dim += pos_enc_dim

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

		self.layers = nn.ModuleList()
		current_dim = hid_dim

		for layer_idx in range(num_layers):
			layer_out = out_dim if layer_idx == num_layers - 1 else hid_dim
			layer = TransformerConvLayer(
				dim_in=current_dim,
				dim_out=layer_out,
				edge_dim=edge_dim,
				heads=heads,
				concat=concat,
				attn_dropout=attn_dropout,
				dropout=dropout,
				norm_type=norm_type,
				activation=activation,
				residual=residual,
				ffn=ffn,
				beta=beta,
				root_weight=root_weight,
			)
			self.layers.append(layer)
			current_dim = layer.conv_out_dim

		self.out_dim = current_dim

	def _validate_positional_encoding(self, pos_enc: OptTensor) -> None:
		if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == "none":
			return
		if pos_enc.size(-1) != self.pos_enc_dim:
			raise ValueError(
				f"Expected pos_enc with feature size {self.pos_enc_dim}, got {pos_enc.size(-1)}"
			)

	def _integrate_positional_encoding(self, x: Tensor, pos_enc: OptTensor) -> Tensor:
		if pos_enc is None or self.pos_enc_dim == 0 or self.pos_enc_mode == "none":
			return x

		if self.pos_enc_mode == "concat":
			return torch.cat([x, pos_enc], dim=-1)
		return x

	def forward(
		self,
		x: Tensor,
		edge_index: Adj,
		batch: Tensor,
		edge_attr: OptTensor = None,
		edge_type: OptTensor = None,
		pos_enc: OptTensor = None,
	) -> Tensor:
		# edge_type is accepted for interface consistency and is unused in TransformerConv.
		_ = edge_type
		self._validate_positional_encoding(pos_enc)

		x = self._integrate_positional_encoding(x, pos_enc)
		x = self.input_proj(x)

		if pos_enc is not None and self.pos_add_proj is not None:
			x = x + self.pos_add_proj(pos_enc)

		for layer in self.layers:
			x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

		return x

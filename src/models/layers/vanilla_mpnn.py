import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn  # Standard alias for PyG layers
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

# Project Imports
from models.model_utils import apply_norm, get_norm_layer


class VanillaMPNNConv(MessagePassing):
    """
    A vanilla Message Passing Neural Network layer optimized with PyG components.
    Message: MLP(concat(source_node, target_node, edge_feature))
    Update: MLP(concat(node, aggregated_messages))
    """

    def __init__(self, dim_in: int, hid_dim: int, edge_dim: int, norm_type="batch"):
        # We use 'add' aggregation as it is standard for MPNNs, though
        # Mean is often more stable for regression tasks.
        super(VanillaMPNNConv, self).__init__(aggr="add")

        # 1. Using gnn.Linear for GNN-specific weight initialization.
        # This is better for handling the feature distributions in graph-level tasks.
        self.msg_lin1 = gnn.Linear(2 * dim_in + edge_dim, hid_dim)
        self.msg_norm = get_norm_layer(norm_type, hid_dim)
        self.msg_act = nn.LeakyReLU()
        self.msg_lin2 = gnn.Linear(hid_dim, hid_dim)
        self.msg_act2 = nn.LeakyReLU()

        # Node Update MLP
        self.update_lin1 = gnn.Linear(dim_in + hid_dim, hid_dim)
        self.update_norm = get_norm_layer(norm_type, hid_dim)
        self.update_act = nn.LeakyReLU()
        self.update_lin2 = gnn.Linear(hid_dim, hid_dim)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        The batch tensor is explicitly unsqueezed if 1D to prevent indexing errors
        during the message/update steps in some PyG versions.
        """
        if batch is not None and batch.dim() == 1:
            batch_in = batch.unsqueeze(-1)
        else:
            batch_in = batch

        # Passing 'batch' into propagate automatically makes it available as
        # 'batch_i' (destination nodes) and 'batch_j' (source nodes) in message().
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, batch=batch_in)

    def message(self, x_i, x_j, edge_attr, batch_i=None):
        """
        Constructs messages. By using batch_i, normalization is performed
        per-individual graph in the mini-batch.
        """
        if batch_i is not None and batch_i.dim() == 2:
            batch_i = batch_i.squeeze(-1)

        # Message: Linear -> Graph-Aware Norm -> Activation -> Linear -> Activation
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        h = self.msg_lin1(msg_input)

        # Using the project utility to apply normalization safely with the batch tensor
        h = apply_norm(self.msg_norm, h, batch_i)
        h = self.msg_act(h)
        h = self.msg_lin2(h)
        return self.msg_act2(h)

    def update(self, aggr_out, x, batch=None):
        """
        Updates node features using aggregated messages.
        """
        if batch is not None and batch.dim() == 2:
            batch = batch.squeeze(-1)

        update_input = torch.cat([x, aggr_out], dim=-1)
        h = self.update_lin1(update_input)

        # Final update normalization per graph boundary
        h = apply_norm(self.update_norm, h, batch)
        h = self.update_act(h)
        return self.update_lin2(h)


class MPNNEncoder(nn.Module):
    """
    Encoder for stacking VanillaMPNNConv layers with Jumping Knowledge integration.
    """

    def __init__(
        self,
        hid_dim: int,
        num_layers: int,
        node_input_dim: int,
        edge_attr_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        norm_type: str = "batch",
        **kwargs,
    ):
        super(MPNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.jk_mode = kwargs.get("jk_mode", "cat")
        self.dropout = dropout

        # Use gnn.Linear for the initial projection to hid_dim
        if node_input_dim != hid_dim:
            self.jk_proj = gnn.Linear(node_input_dim, hid_dim)
        else:
            self.jk_proj = nn.Identity()

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            dim_in = node_input_dim if i == 0 else hid_dim
            self.convs.append(
                VanillaMPNNConv(
                    dim_in=dim_in,
                    hid_dim=hid_dim,
                    edge_dim=edge_attr_dim,
                    norm_type=norm_type,
                )
            )

    def layer_forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        batch: Tensor | None,
        layer_idx: int,
    ) -> Tensor:
        current_x = self.convs[layer_idx](
            x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )

        if layer_idx != self.num_layers - 1:
            current_x = F.dropout(current_x, p=self.dropout, training=self.training)
        return current_x

    def forward(
        self, x: Tensor, edge_index: Adj, edge_attr: Tensor, batch: Tensor | None = None
    ):
        """
        Forward pass with support for various Jumping Knowledge (JK) modes.
        The batch tensor is passed to every layer to ensure correct per-graph normalization.
        """
        x_jk = self.jk_proj(x)

        if self.jk_mode == "cat":
            h_list = [x_jk]
            current_x = x
            for i in range(self.num_layers):
                current_x = self.layer_forward(
                    current_x, edge_index, edge_attr, batch, i
                )
                h_list.append(current_x)
            return torch.cat(h_list, dim=1)

        elif self.jk_mode == "max":
            current_x = x
            res = x_jk
            for i in range(self.num_layers):
                current_x = self.layer_forward(
                    current_x, edge_index, edge_attr, batch, i
                )
                res = torch.max(res, current_x)
            return res

        elif self.jk_mode == "sum":
            current_x = x
            res = x_jk
            for i in range(self.num_layers):
                current_x = self.layer_forward(
                    current_x, edge_index, edge_attr, batch, i
                )
                res = res + current_x
            return res

        elif self.jk_mode == "last":
            current_x = x
            for i in range(self.num_layers):
                current_x = self.layer_forward(
                    current_x, edge_index, edge_attr, batch, i
                )
            return current_x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

try:
    from model_utils import get_norm_layer
except ImportError:  # pragma: no cover - fallback for package-style imports
    try:
        from models.model_utils import get_norm_layer
    except ImportError:
        from src.models.model_utils import get_norm_layer


class VanillaMPNNConv(MessagePassing):
    """
    A vanilla Message Passing Neural Network layer.
    Message: MLP(concat(source_node, target_node, edge_feature))
    Update: MLP(concat(node, aggregated_messages))
    """

    def __init__(self, dim_in: int, hid_dim: int, edge_dim: int, norm_type="batch"):
        super(VanillaMPNNConv, self).__init__(aggr="add")

        # Message MLP (concatenates x_i, x_j, edge_attr)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * dim_in + edge_dim, hid_dim),
            get_norm_layer(norm_type, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )

        # Node Update MLP (concatenates original node features and aggregated messages)
        self.update_mlp = nn.Sequential(
            nn.Linear(dim_in + hid_dim, hid_dim),
            get_norm_layer(norm_type, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNNEncoder(nn.Module):
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
        self.jk_mode = kwargs.get("jk_mode", "cat")  # Default to 'cat' if not provided
        self.dropout = dropout

        # Project initial embedding purely for Jumping Knowledge uniformity
        if node_input_dim != hid_dim:
            self.jk_proj = nn.Linear(node_input_dim, hid_dim)
        else:
            self.jk_proj = nn.Identity()

        # First layer expects `node_input_dim`; subsequent layers expect `hid_dim`.

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

    def forward(self, x, edge_index, batch, edge_attr):

        # Keep track of uniform dimension sizes for h_list
        h_list = [self.jk_proj(x)]
        current_x = x

        for i in range(self.num_layers):
            current_x = self.convs[i](current_x, edge_index, edge_attr)

            # Inter-layer dropout
            if i != self.num_layers - 1:
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)

            h_list.append(current_x)

        # Hardcoded Jumping Knowledge = 'cat'
        if self.jk_mode == "last":
            node_emb = h_list[-1]
        elif self.jk_mode == "max":
            node_emb = torch.stack(h_list, dim=-1).max(dim=-1)[0]
        elif self.jk_mode == "sum":
            node_emb = torch.stack(h_list, dim=-1).sum(dim=-1)
        elif self.jk_mode == "cat":
            node_emb = torch.cat(h_list, dim=1)

        return node_emb

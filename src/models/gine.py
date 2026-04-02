import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import get_pyg_pool
from torch_geometric.nn import GINEConv


class GINEEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        edge_dim,
        dropout=0.5,
        readout="mean",
        jk="cat",
    ):
        """
        Improved GINE Encoder based on Jumping Knowledge and robust MLPs.

        Args:
            jk (str): Jumping Knowledge strategy. 'cat' (concatenate all layers),
                      'last' (only use final layer), or 'sum' (sum all layers).
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk.lower()
        self.pool_fn = get_pyg_pool(readout)

        # Initial Encoders: project raw features to hidden dimension immediately
        self.node_encoder = nn.Linear(in_dim, hid_dim)
        self.edge_encoder = nn.Linear(edge_dim, hid_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # The inner MLP for GIN must be highly expressive (Linear -> BN -> ReLU -> Linear -> ReLU)
            inner_mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
            )

            # GINEConv handles edge features. We set train_eps=True to learn the center node weighting
            self.convs.append(GINEConv(nn=inner_mlp, train_eps=True, edge_dim=hid_dim))
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        # Calculate output dimension based on Jumping Knowledge strategy
        if self.jk == "cat":
            # If concatenating all layers + the initial projection
            self.out_dim = hid_dim * (num_layers + 1)
        else:
            self.out_dim = hid_dim

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Initial projections
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # 2. Store layer outputs for Jumping Knowledge
        h_list = [x]

        # 3. Message Passing
        for i in range(self.num_layers):
            h = self.convs[i](h_list[i], edge_index, edge_attr)
            h = self.batch_norms[i](h)

            # Dropout before the residual/next layer (unless it's the absolute last layer)
            if i != self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

            h_list.append(h)

        # 4. Graph-level Readout / Pooling for EVERY layer
        pooled_h_list = [self.pool_fn(h, batch) for h in h_list]

        # 5. Jumping Knowledge combination
        if self.jk == "cat":
            # Concatenate pooled embeddings from all layers: Shape (Batch, hid_dim * (num_layers + 1))
            graph_emb = torch.cat(pooled_h_list, dim=1)
        elif self.jk == "sum":
            # Sum pooled embeddings from all layers
            graph_emb = sum(pooled_h_list)
        else:  # 'last'
            # Only take the pooled embedding from the final layer
            graph_emb = pooled_h_list[-1]

        return graph_emb

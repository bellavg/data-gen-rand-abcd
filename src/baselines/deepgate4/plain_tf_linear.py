"""Vendored from zyzheng17/DeepGate4-ICLR-25, src/models/plain_tf_linear.py.

Source: https://github.com/zyzheng17/DeepGate4-ICLR-25/blob/master/src/models/plain_tf_linear.py
Upstream commit: 85e20742a2d4702426f94e86666cdcdb408fba8a (2026-06-25).
License: upstream declares NONE -- see PROVENANCE.md in this directory.

`Sparse_Transformer` is DeepGate4 itself -- it is what `--tf_arch sparse`
selects in upstream's run/train_large.sh, and what paper Section 3.5
("GAT-based Sparse Transformer") describes: the Multi-Head Attention of a
standard transformer block is replaced by a `GATConv` over the graph
augmented with virtual edges, keeping the Add&Norm + FeedForward structure.

Unmodified apart from deleting `import torch.nn.functional as F`, which
upstream imports but never uses in this file (ruff F401 would fail `ruff
check src` otherwise). No behavioural change.

The `num_layers=12` default is the published depth (paper Section 4.1: "The
depth of Sparse Transformer is 12"), and upstream never overrides it --
dg4.py constructs this class as `Sparse_Transformer(args, hidden=self.hidden)`,
so `--TF_depth` does NOT reach it. `heads=4`, `concat=True` and `dropout=0.1`
are likewise upstream's constructor defaults; none of the three appears in
the paper.
"""

import torch
from torch_geometric.nn import GATConv
from torch.nn import Linear, LayerNorm

class GATTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, concat=True, dropout=0.1, ff_hidden_dim=128):
        super(GATTransformerEncoderLayer, self).__init__()

        # GAT multi-head attention
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)

        # Feed-forward network (FFN)
        self.ffn = torch.nn.Sequential(
            Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
            torch.nn.ReLU(),
            Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
        )

        # Layer normalization
        self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
        self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # GAT layer with residual connection
        x_residual = x.clone()
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm1(x)   # Layer normalization

        # Feed-forward network with residual connection
        x_residual = x.clone()
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm2(x)   # Layer normalization

        return x



class Sparse_Transformer(torch.nn.Module):
    def __init__(self, args, hidden, num_layers=12, heads=4, concat=True, dropout=0.1):
        super(Sparse_Transformer, self).__init__()

        in_channels = hidden * 2
        out_channels = in_channels // heads

        ff_hidden_dim = 4 * hidden

        self.num_layers = num_layers

        self.tf_layers = torch.nn.ModuleList([
            GATTransformerEncoderLayer(in_channels if i == 0 else out_channels*heads if concat else out_channels,
                                       out_channels, heads=heads, concat=concat, dropout=dropout, ff_hidden_dim=ff_hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, g, hf, hs, mk):

        virtual_edge = g.global_virtual_edge

        virtual_edge = virtual_edge.T
        virtual_edge = virtual_edge[mk[g.nodes[virtual_edge[:,1].cpu()]]==0]
        virtual_edge = virtual_edge.T

        if virtual_edge.shape[1] == 0:
            return hf, hs

        h = torch.cat([hf,hs],dim=-1)
        for i in range(self.num_layers):
            h = self.tf_layers[i](h,virtual_edge)

        hf, hs = torch.chunk(h,2,dim=-1)

        return hf, hs

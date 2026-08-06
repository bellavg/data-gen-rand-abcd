"""Gamora's `SAGE_MULT` trunk, vendored from Yu-Maryland/Gamora.

Source: `abc2pyg/gnn_multitask.py:39-105` at commit
`344d5e4530072cd95a07f0557640ad641fd6cfcb` (branch `master`, not `main`).
See PROVENANCE.md in this directory for the licence position, which is more
ambiguous than GitHub's "NOASSERTION" badge suggests.

WHAT WAS TAKEN AND WHAT WAS NOT
-------------------------------
Upstream's class carries four methods. Two are vendored here and two are not:

  - `__init__` / `reset_parameters` -- verbatim.
  - `forward_nosampler(x, adj_t, device)` (upstream :86-105) -- vendored, with
    one documented change (see below). This is the full-graph path: it loops
    `for conv in self.convs: x = conv(x, adj_t)` over the whole adjacency, with
    no sampled neighbourhoods anywhere.
  - `forward(x, adjs)` (upstream :68-84) -- NOT vendored. It consumes the
    `(edge_index, e_id, size)` triples a `NeighborSampler` yields and slices
    `x[:size[1]]` per layer, so it only means anything inside upstream's
    sampled training loop.
  - `inference(x_all, subgraph_loader, device)` (upstream :107-139) -- NOT
    vendored, same reason: it iterates a `NeighborSampler` directly.

Dropping those two is the point of this port, not an accident of it -- see
regressor.py's module docstring for why a sampling-free baseline is required
here, and PROVENANCE.md for the fact that upstream's *released trainer* does
sample even though this method does not.

`forward_nosampler` is inherited but never called by the regressor: the three
classification heads it reads (`self.linear[1:4]`) are deleted there, so
calling it on a `GamoraGraphRegressor` would raise `IndexError`. It is kept
because it is the evidence for the claim above -- the computation the
regressor performs is this method's, re-inlined so the pooling and regression
head can be spliced in after the trunk (the same structure `../hoga/regressor.py`
uses).

THE ONE DELIBERATE CHANGE
-------------------------
Upstream stores `self.dropout = dropout` in `__init__` and then never reads
it: both `forward` and `forward_nosampler` hardcode `F.dropout(x, p=0.5)`. The
constructor argument is dead, so `--dropout 0.2` on upstream's CLI silently
does nothing. This copy uses `p=self.dropout`. At the default (`dropout=0.5`,
upstream's own argparse default) the two are numerically identical; the change
only makes the knob real instead of a trap.

Also note the first two lines of upstream's `forward_nosampler`,
`x.to(device)` and `adj_t.to(device)`, discard their return values and are
therefore no-ops. They are kept verbatim rather than "fixed" -- device
placement is Lightning's job here, and deleting them would be an undocumented
divergence for no benefit. They are the reason the method takes a `device`
argument it does not use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import SAGEConv


class SAGE_MULT(torch.nn.Module):  # upstream's class name, kept verbatim.
    """4-layer GraphSAGE stack + shared linear/BN + three classification heads.

    Constructor arguments match upstream exactly. `num_layers` counts SAGEConv
    layers (1 input layer + `num_layers - 2` hidden + 1 more hidden), and
    `out_channels` sizes only the three classification heads, which
    `GamoraGraphRegressor` deletes.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_MULT, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # two linear layer for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))

        self.bn0 = BatchNorm1d(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.linear:
            lin.reset_parameters()

    def forward_nosampler(self, x, adj_t, device):
        # tensor placement
        x.to(device)
        adj_t.to(device)

        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        x1 = self.linear[1](x)  # for xor
        x2 = self.linear[2](x)  # for maj
        x3 = self.linear[3](x)  # for roots
        return x1, x2, x3

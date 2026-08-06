"""Vendored from BUPT-GAMMA/PolarGate, layers.py, with documented deletions.

Source: https://github.com/BUPT-GAMMA/PolarGate/blob/master/layers.py
Commit: 4bbb23e965ec5e9b17835878f5c50e108f4a7df7 (2024-10-08, the repository's
last push).
License: **upstream declares none** -- see PROVENANCE.md in this directory.
There is no LICENSE_UPSTREAM here for that reason, unlike ../hoga/ and
../openabc_synthnet/.

Everything from the import block down is byte-for-byte upstream apart from the
three deletions below, once line endings are normalised (upstream is CRLF, this
copy is LF) -- verified by a line diff. Nothing was reordered, renamed, or
retuned, including the import order and the trailing space inside `MLP`'s
`'''The basic structure is refered from '''` docstring. Outside that: this
header docstring is added, upstream's `# coding=utf-8` first line and its two
trailing blank lines are dropped.

DELETION 1 -- `create_spectral_features`. Upstream calls it only when the
caller passes `init_emb=None` (model.py:60). It runs
`TruncatedSVD(n_components=64, n_iter=128)` over the N x N signed adjacency,
which is not viable here: this project's graphs reach config.MAX_NUM_GATES =
366,040 nodes and there are ~707k train graphs (~788k with val), so a
128-iteration randomized SVD per graph per epoch is off by orders of magnitude
in both time and memory.
`regressor.py` therefore always supplies real node features and raises if it
is ever handed `None`, which is what keeps this function unreachable rather
than merely unused. Its removal also drops the `sklearn`, `scipy.sparse` and
`torch_sparse.coalesce` imports, none of which are installed in this project's
environment.

DELETION 2 -- `message_and_aggregate` on both conv classes. Upstream's version
is `matmul(adj_t.set_value(None), x[0], reduce=self.aggr)` from `torch_sparse`,
which is not a dependency of this project and is not installed. Behaviour-
neutral on this port's call path: PyG only dispatches to
`message_and_aggregate` when `propagate` is given a `SparseTensor` or a
`torch.sparse` adjacency, and `regressor.py` always passes a dense
`[2, E] LongTensor`, so upstream's own `message` + `aggregate` path is the one
that runs in both codebases. Keeping the method would make the module
unimportable for no behavioural gain.

DELETION 3 -- the now-unused imports (`SparseTensor`, `matmul`, `coalesce`,
`scipy.sparse`, `TruncatedSVD`). Every remaining import is referenced.

A NOTE ON `restPolarGateConv`'S INVERTED BRANCH, recorded because this project
runs it in a regime the paper never has to define.

`aggr='min'` applies to BOTH propagate calls, so the inverted branch computes
`-min_j(h_j)` over the inverted fanin set. Paper Equation (5), which governs
NOT nodes, is `sigma(W [0, OPNOT_{j in N_i} h_j, h_i])` -- OPNOT (elementwise
negation) written as an operator over the neighbour set, with NO aggregator
specified, because a NOT gate has exactly one fanin and none is needed. At
|N_i| = 1 the code and every reading of Equation (5) coincide, so there is no
code/paper discrepancy in upstream's own graphs.

It becomes a real question here only. This project's AIGs have no NOT nodes,
so an AND gate can carry two inverted fanins and the inverted set can hold
more than one element -- at which point `-min_j(h_j) = max_j(-h_j)` and
"negate first, then OPAND" `min_j(-h_j) = -max_j(h_j)` are different
functions, and the paper says nothing about which is meant. This port takes
the released code's order, unmodified.
"""

from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import PairTensor, Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F


_norm_layer_factory = {
    'batchnorm': nn.BatchNorm1d,
}

_act_layer_factory = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
}


class MLP(nn.Module):
    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5,
                 sigmoid=False, tanh=False):
        super(MLP, self).__init__()
        '''
        The basic structure is refered from 
        '''
        assert num_layer >= 2, 'The number of layers shoud be larger or equal to 2.'
        if norm_layer in _norm_layer_factory.keys():
            self.norm_layer = _norm_layer_factory[norm_layer]
        if act_layer in _act_layer_factory.keys():
            self.act_layer = _act_layer_factory[act_layer]
        if p_drop > 0:
            self.dropout = nn.Dropout

        fc = []
        # 1st layer
        fc.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
        for _ in range(num_layer - 2):
            fc.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                fc.append(self.norm_layer(dim_hidden))
            if act_layer:
                fc.append(self.act_layer(inplace=True))
            if p_drop > 0:
                fc.append(self.dropout(p_drop))
        # last layer
        fc.append(nn.Linear(dim_hidden, dim_pred))
        # sigmoid
        if sigmoid:
            fc.append(nn.Sigmoid())
        if tanh:
            fc.append(nn.Tanh())
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        out = self.fc(x)
        return out


class PolarGateConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        norm_emb: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        if first_aggr:
            self.lin_b = Linear(2 * in_dim, out_dim, bias)
            self.lin_u = Linear(2 * in_dim, out_dim, bias)
        else:
            self.lin_b = Linear(3 * in_dim, out_dim, bias)
            self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.first_aggr:
            out_b = self.propagate(pos_edge_index, x=x)
            out_b = self.lin_b(torch.cat([out_b, x[1]], dim=-1))

            out_u = self.propagate(neg_edge_index, x=x)
            out_u = self.lin_u(torch.cat([out_u, x[1]], dim=-1))
            out = torch.cat([out_b, out_u], dim=-1)
        else:
            F_in = self.in_dim
            out_b1 = self.propagate(pos_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_b2 = self.propagate(neg_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_b = torch.cat([out_b1, out_b2, x[1][..., :F_in]], dim=-1)
            out_b = self.lin_b(out_b)

            out_u1 = self.propagate(pos_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_u2 = self.propagate(neg_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_u = torch.cat([out_u1, out_u2, x[1][..., F_in:]], dim=-1)
            out_u = self.lin_u(out_u)

            out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


class restPolarGateConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool = False,
        bias: bool = True,
        norm_emb: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'min')
        # kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        self.lin_b = Linear(3 * in_dim, out_dim, bias)
        self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        F_in = self.in_dim

        # update positive embeddings
        out_b1 = self.propagate(pos_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_b2 = self.propagate(neg_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_b = torch.cat([out_b1, out_b2 * -1, x[1][..., :F_in]], dim=-1)
        out_b = self.lin_b(out_b)

        # update negative embeddings
        out_u1 = self.propagate(pos_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_u2 = self.propagate(neg_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_u = torch.cat([out_u1, out_u2 * -1, x[1][..., F_in:]], dim=-1)
        out_u = self.lin_u(out_u)

        out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')

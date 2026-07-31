"""Vendored from zyzheng17/DeepGate4-ICLR-25, src/models/tfmlp.py.

Source: https://github.com/zyzheng17/DeepGate4-ICLR-25/blob/master/src/models/tfmlp.py
Upstream commit: 85e20742a2d4702426f94e86666cdcdb408fba8a (2026-06-25).
License: upstream declares NONE -- see PROVENANCE.md in this directory.

`TFMlpAggr` is the attention-weighted message aggregator the DeepGate2
tokenizer (dg2.py) uses for its AND/NOT structural and functional updates.

Unmodified apart from deleting two names upstream imports but never references
here, `torch_geometric.typing.Adj` and `.mlp.MLP` (ruff F401). `OptTensor` from
the same import IS used and is kept. No behavioural change.

Worth knowing when reading `message()`: `softmax` is taken over a node's
incoming edges, so for any node of in-degree 1 -- every NOT gate, and every
synthetic PO -- it returns a constant 1 and `msg_q`/`msg_k`/`attn_lin` receive
exactly zero gradient. That is upstream's behaviour, and the paper notes the
same fact when motivating its CUDA kernel (Section 3.7: "when the in-degree is
1, we can skip the computation entirely, as the softmax result is
straightforward, i.e. alpha = 1"). It is not a symptom of the port.
"""

import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn

from typing import Optional
from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor

class TFMlpAggr(MessagePassing):
    '''
    The message propagation methods described in NeuroSAT (2 layers without dropout) and CircuitSAT (2 layers, dim = 50, dropout - 20%).
    Cite from NeuroSAT:
    `we sum the outgoing messages of each of a node’s neighbors to form the incoming message.`
    '''
    def __init__(self, in_channels, ouput_channels=64, reverse=False, mlp_post=None):
        super(TFMlpAggr, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.msg_post = None if mlp_post is None else mlp_post
        self.attn_lin = nn.Linear(ouput_channels + ouput_channels, 1)

        self.msg_q = nn.Linear(in_channels, ouput_channels)
        self.msg_k = nn.Linear(in_channels, ouput_channels)
        self.msg_v = nn.Linear(in_channels, ouput_channels)


    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr: Optional[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key
        h_attn_q_i = self.msg_q(x_i)
        h_attn = self.msg_k(x_j)
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        # x_j -> value
        t = self.msg_v(x_j) * a_j
        return t

    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out

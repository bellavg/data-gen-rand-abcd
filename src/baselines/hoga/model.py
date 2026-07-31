"""Vendored from cornell-zhang/HOGA, model.py, with one documented fix.

Source: https://github.com/cornell-zhang/HOGA/blob/master/model.py
License: BSD 3-Clause (see LICENSE_UPSTREAM in this directory).

THREE DELIBERATE MODIFICATIONS, in `MultiheadAttention.forward` (below). Upstream
reshapes the projections to `[b, l, head_dim, -1]` and computes
`einsum('bldh,bndh->blnh')`, giving scores indexed `[batch, query, key, head]`
-- then applies `Softmax(dim=-1)`, which normalizes over the **head** axis, not
the key axis. The attention weights therefore never form a distribution over
hop neighbours; they form one across heads, and the model cannot express
"attend to hop 3".

The authority for calling this wrong is the paper, Equation (5):
`S = softmax(QK^T)`, described as "the self-attention matrix widely used in
Transformer [16]" (citing Vaswani et al.) -- i.e. normalized row-wise over
keys. The sibling class `MultiheadAttentionMix` also softmaxes over the key
axis, and it is the class upstream's published `run.sh` actually instantiates
(`main_gamora.py` builds `HOGA(..., attn_type="mix")`), so the vanilla class
edited here was never exercised by the released experiment. Treat Mix as
corroboration on the softmax *axis* only, not as a general reference
implementation: its own reshape, `view(batch_size * num_heads, -1, head_dim)`,
splits the flattened (seq x feature) axis rather than the feature axis, so for
`heads > 1` its "heads" are chopped-up sequence rows and it is not multi-head
attention either. At `heads == 1` the two reshapes coincide, which is what
`test_single_head_matches_upstream_mix_implementation` pins.

This project selected `attn_type="vanilla"` (labelled "recommended for general
use cases"), which is how the defect reached a live path here. The forward pass
below is rewritten to normalize over keys, written out as an explicit
matmul -> scale -> softmax(dim=-1) -> matmul.

`F.scaled_dot_product_attention` would express the same thing and fuse it, and
was used here initially, but it is NOT safe at this model's shapes: the
attention batch axis is the NODE count (one independent hops x hops problem
per node), which reaches config.MAX_NUM_GATES = 366040 for a single large AIG.
Both fused backends put that axis on a CUDA grid y/z dimension capped at 65535
(FlashAttention-2 forward: `dim3 grid(num_m_block, params.b, params.h)`), and
PyTorch's dispatcher does not check batch against the grid limit -- so a fused
backend is selected and then fails at launch with `invalid configuration
argument`. Keep the lost saving in proportion: the score matmuls are only ~1%
of this module's arithmetic -- the four `Linear(256, 256)` projections dominate
-- so fusing would have won activation bytes and kernel count, not FLOPs, and
at 6 slots the score/prob tensors are ~1.0 GB against a ~45 GB peak.

The head split also changes layout, from upstream's interleaved
`[b, l, head_dim, heads]` to the standard contiguous `[b, heads, l, head_dim]`.
For training from scratch this is immaterial -- the two differ by a fixed
permutation of input channels, which the dense Q/K/V and output projections
absorb, and i.i.d. init makes the induced function distributions identical.
It does mean checkpoints predating this change decode to a different function,
but the softmax fix invalidates those regardless.

The Gamora-specific 3-head output (`self.linear[1:4]`,
returning `x1`/`x2`/`x3`/`layer_atten` for xor/maj/root node classification) is
left in place for fidelity to the original file -- this project's
`regressor.py` does not call `HOGA.forward()` directly; it reuses the module
list up through the fused node+neighbor-attention representation and attaches
its own graph-level pooling + regression head instead (see regressor.py for
why: this project's target is one scalar per whole graph, not a per-node
classification).
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm, Dropout, Softmax


'''
Slightly modified multihead attention for Gamora
'''
class MultiheadAttentionMix(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super(MultiheadAttentionMix, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query_projection = Linear(input_dim, input_dim)
        self.key_projection = Linear(input_dim, input_dim)
        self.value_projection = Linear(input_dim, input_dim)

        # Linear projection for the output of the attention heads
        self.output_projection = Linear(input_dim, input_dim)

        self.dropout = Dropout(dropout)
        self.softmax = Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections for queries, keys, and values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape the projected queries, keys, and values
        query = query.view(batch_size * self.num_heads, -1, self.head_dim)
        key = key.view(batch_size * self.num_heads, -1, self.head_dim)
        value = value.view(batch_size * self.num_heads, -1, self.head_dim)

        # Compute the scaled dot-product attention
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply the mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute the output of the attention heads
        attention_output = torch.bmm(attention_probs, value)

        # Reshape and project the output of the attention heads
        attention_output = attention_output.view(batch_size, -1, self.input_dim)
        attention_output = self.output_projection(attention_output)

        return attention_output, attention_probs

'''
Vanilla multihead attention (recommended for general use cases)
'''
class MultiheadAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query_projection = Linear(input_dim, input_dim)
        self.key_projection = Linear(input_dim, input_dim)
        self.value_projection = Linear(input_dim, input_dim)

        # Linear projection for the output of the attention heads
        self.output_projection = Linear(input_dim, input_dim)

        # Restored from upstream, but with dim=-1 now meaning the KEY axis
        # under the [batch, heads, seq, seq] score layout (upstream's layout
        # left heads last, which is the bug this class fixes).
        self.dropout = Dropout(dropout)
        self.softmax = Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """Standard scaled dot-product attention over the hop (sequence) axis.

        MODIFIED vs upstream -- see this module's docstring for the full
        rationale. Upstream reshaped to `[b, l, head_dim, -1]` and softmaxed
        `[b, l, n, h]` over `dim=-1`, i.e. over the *head* axis rather than
        the key axis, so the scores never normalized into a distribution over
        hop neighbours. This normalizes over keys, per Equation (5) of the
        paper ("S = softmax(QK^T) ... the self-attention matrix widely used in
        Transformer") and per `MultiheadAttentionMix`, the class upstream's own
        run.sh actually instantiates (`attn_type="mix"`).

        Returns `(output, attention_probs)`, matching upstream's contract. The
        probabilities are `[batch, heads, seq, seq]` here rather than upstream's
        `[b, l, n, h]`, since the head axis moved.
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections for queries, keys, and values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape to [batch, heads, seq, head_dim], the layout SDPA expects.
        def _split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        attn_mask = None
        if mask is not None:
            # This class accepted a `mask` argument and then silently ignored
            # it upstream -- only MultiheadAttentionMix ever applied one. Since
            # the rewrite has to decide what the parameter means, it follows
            # Mix's convention (`masked_fill(mask == 0, -inf)`, so nonzero =
            # keep), expressed as SDPA's boolean keep-mask. `[b, 1, l, n]`
            # broadcasts against the `[b, heads, l, n]` scores. No caller in
            # this repo passes a mask; this exists so the parameter is honest
            # rather than inert.
            attn_mask = mask.unsqueeze(1) != 0

        # Written out rather than delegated to F.scaled_dot_product_attention,
        # deliberately. In this model the attention batch axis is the NODE
        # count -- one independent (num_hop_slots x num_hop_slots) problem per
        # node -- so `batch_size` here reaches config.MAX_NUM_GATES (366040)
        # for a single large AIG. Both fused SDPA backends put that batch axis
        # on a CUDA grid y/z dimension (FlashAttention-2 forward launches
        # `dim3 grid(num_m_block, params.b, params.h)`; the mem-efficient
        # kernel `dim3(m_blocks, num_heads, num_batches)`), and those axes are
        # capped at 65535. PyTorch's SDPA dispatcher does not check batch
        # against the grid limit, so a fused backend would be selected and then
        # fail at launch with `invalid configuration argument` -- a sticky,
        # unrecoverable CUDA error minutes into a 72h job.
        #
        # The fusion was never the point of this rewrite; the softmax axis was
        # (see the class docstring). Writing the three steps explicitly keeps
        # that fix exactly, costs ~15% of the trunk speedup, and matches paper
        # Eq. 5 (`S = softmax(QK^T)`, then `SV`) more legibly besides. The
        # scores are [nodes, heads, slots, slots] -- at 6 slots and 32 heads
        # that is ~0.8 GB at the worst-case node count, against a ~45 GB peak.
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * (self.head_dim**-0.5)
        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(~attn_mask, float("-inf"))

        # dim=-1 is the KEY axis under the [batch, heads, seq, seq] layout
        # above. This is the corrected axis; upstream's einsum left heads last
        # and normalized over those instead.
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)

        # Reshape and project the output of the attention heads
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.input_dim
        )
        attention_output = self.output_projection(attention_output)

        return attention_output, attention_probs

class HOGA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_hops, heads, attn_dropout=0.0, attn_type="vanilla", use_bias=False):
        super(HOGA, self).__init__()
        self.num_layers = num_layers
        self.num_hops = num_hops

        self.lins = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.trans = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, bias=use_bias))
        self.lins.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.lins.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.gates.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        if attn_type == "vanilla":
            self.trans.append(MultiheadAttention(hidden_channels, heads, dropout=attn_dropout))
        else:
            self.trans.append(MultiheadAttentionMix(hidden_channels, heads, dropout=attn_dropout))
        self.lns.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
            self.gates.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
            if attn_type == "vanilla":
                self.trans.append(MultiheadAttention(hidden_channels, heads, dropout=attn_dropout))
            else:
                self.trans.append(MultiheadAttentionMix(hidden_channels, heads, dropout=attn_dropout))
            self.lns.append(LayerNorm(hidden_channels))

        # Linear layers for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.linear.append(Linear(hidden_channels, out_channels, bias=use_bias))
        self.linear.append(Linear(hidden_channels, out_channels, bias=use_bias))
        self.linear.append(Linear(hidden_channels, out_channels, bias=use_bias))

        self.bn = BatchNorm1d(hidden_channels)
        self.attn_layer = Linear(2 * hidden_channels, 1)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for gate in self.gates:
            gate.reset_parameters()
        for li in self.linear:
            li.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x):
        # Current implementation: use a shared linear layer for all hop-wise features
        # Note: apply separate layers for different hop-wise features may further improve accuracy
        x = self.lins[0](x)

        for i, tran in enumerate(self.trans):
            x = self.lns[i](self.gates[i](x)*(tran(x, x, x)[0]))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        target = x[:,0,:].unsqueeze(1).repeat(1,self.num_hops-1,1)
        split_tensor = torch.split(x, [1, self.num_hops-1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x = (node_tensor + neighbor_tensor).squeeze()
        x = self.linear[0](x)
        x = self.bn(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots

        return x1, x2, x3, layer_atten

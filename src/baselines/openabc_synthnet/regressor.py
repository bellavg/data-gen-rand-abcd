"""Graph-level optimizability regressor built on OpenABC-D's SynthNet GNN encoder.

Adapts NYU-MLDA/OpenABC's SynthNet (see model.py in this directory, vendored
near-verbatim from models/qor/SynthNetV3/model.py) to this project's task:
single-fixed-algorithm optimizability regression from a raw AIG, no synthesis
recipe conditioning.

Edge direction matches upstream by default (`upstream_edge_direction=True`).
OpenABC-D builds each edge as node -> fanin (`andAIG2Graphml.py:56` for AND
nodes, :71 for the PO buffer) and hands `list(G.edges)` straight to
`edge_index` with no reversal, so under PyG's default
`flow="source_to_target"` messages travel toward the primary inputs and every
node ends up summarising its *fanout* cone. This project's own graphs are
built the other way round (fanin -> node, `data/data_utils.py:150`), so
`encode()` reverses `edge_index` before the GCN sees it. Pass
`upstream_edge_direction=False` to run the trunk on this project's native
direction instead, where each node summarises its *fanin* cone -- arguably the
better inductive bias for optimizability, since whether a node can be
collapsed depends on the logic feeding it, not on what it drives. Both are
single-direction; neither adds reverse edges. Run both and report the pair.

Two further deviations from the upstream architecture, both intentional:
  - `SynthFlowEncoder` and the 4 parallel `SynthConv` branches are dropped.
    Those exist upstream to condition on a variable-length synthesis recipe
    (`synVec`) across OpenABC-D's many recipes; this project trains on a
    single fixed algorithm (config.VALID_ALGORITHMS == {"Orchestrate"}), so
    every sample's "recipe" is identical and that branch would only learn a
    constant offset.
  - A terminal Sigmoid is added on the final FC output to match this
    project's [0, 1] optimizability target convention (upstream SynthNet has
    no output bound, since it isn't reused elsewhere).

Everything else -- NodeEncoder, GCNConv, GNN, and the FC-stack depth/width --
matches the upstream paper's own train.py defaults unless overridden. These
aren't just repo defaults: they match the "Net3" configuration published in
Table 3 of the OpenABC-D paper (Chowdhury et al., arXiv:2110.11292) --
AIG-embedding input dim I=4 (3-dim categorical node-type embedding + 1-dim
inverted-predecessor count), GCN layer dims L1=L2=64, a 4-layer FC stack
(178-512-512-512-1 upstream, dr=0.2), batch size 64, initial LR 0.001, Adam,
80 epochs, and MSE loss -- Section 4.1's text confirms the last four for all
three configs. The FC input width differs from the paper's 178 because that
includes the recipe-conv output this project's regressor deliberately drops
(see above); the graph-embedding portion alone (128 = 64 * 2 pooled) is
unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.openabc_synthnet.model import GNN, NodeEncoder

# Upstream train.py defaults (models/qor/SynthNetV3/train.py):
#   nodeEmbeddingDim = 3
#   GNN(..., emb_dim=64)                      (GNN.__init__ default)
#   SynthNet(..., gnn_embed_dim=128, num_fc_layer=4, hidden_dim=512, drop_ratio=0.2)
DEFAULT_NODE_EMB_DIM = 3
DEFAULT_GNN_HIDDEN_DIM = 64
DEFAULT_NUM_FC_LAYER = 4
DEFAULT_FC_HIDDEN_DIM = 512
DEFAULT_DROP_RATIO = 0.2


def derive_node_type_index(x: torch.Tensor) -> torch.Tensor:
    """Map this project's one-hot node features to SynthNet's categorical node type.

    `x` is `[N, NODE_INPUT_DIM]`, one-hot over
    `[constant, primary_input, and_gate, primary_output]`
    (see data/data_utils.py:aig_to_pytorch_geometric). SynthNet's NodeEncoder
    expects a single categorical class index per node.
    """
    return x.argmax(dim=1)


def derive_num_inverted_predecessors(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Count incoming inverted edges per node.

    `edge_index[0]` is the fanin (source), `edge_index[1]` the node it feeds
    (target); `edge_attr[:, 1]` is 1.0 for an inverted signal, 0.0 otherwise
    (see data/data_utils.py:aig_to_pytorch_geometric, `e_type = [1 - inv, inv]`).
    SynthNet's GNN expects this scalar count as a raw (non-one-hot) node
    feature, concatenated with the node-type embedding.

    Returned as a Long tensor (it's a count) rather than float:
    `GNN.forward` builds `x = torch.cat([node_type, num_inverted_predecessors],
    dim=1)`, and `NodeEncoder.forward` then indexes `x[:, 0]` straight into an
    `nn.Embedding` lookup, which requires integer indices. If this were
    float, `torch.cat` would silently upcast the whole tensor (including the
    node-type column) to float and break that lookup.
    """
    target = edge_index[1]
    inverted = edge_attr[:, 1].long()
    counts = torch.zeros(num_nodes, dtype=torch.long, device=edge_attr.device)
    counts.scatter_add_(0, target, inverted)
    return counts


class SynthNetGraphRegressor(nn.Module):
    """SynthNet's GNN encoder + a graph-level regression head (recipe branch removed)."""

    def __init__(
        self,
        num_node_types: int = 4,
        node_emb_dim: int = DEFAULT_NODE_EMB_DIM,
        gnn_hidden_dim: int = DEFAULT_GNN_HIDDEN_DIM,
        num_fc_layer: int = DEFAULT_NUM_FC_LAYER,
        fc_hidden_dim: int = DEFAULT_FC_HIDDEN_DIM,
        drop_ratio: float = DEFAULT_DROP_RATIO,
        task_out_dim: int = 1,
        upstream_edge_direction: bool = True,
    ) -> None:
        super().__init__()
        if num_fc_layer < 2:
            raise ValueError(
                "num_fc_layer must be >= 2 (one input FC layer, one output FC layer)"
            )
        del num_node_types  # NodeEncoder reads the class count from model.allowable_features.

        self.drop_ratio = drop_ratio
        self.upstream_edge_direction = upstream_edge_direction

        node_encoder = NodeEncoder(emb_dim=node_emb_dim)
        # node_input_dim = node_emb_dim (categorical embedding) + 1 (num_inverted_predecessors scalar).
        self.gnn = GNN(node_encoder, node_emb_dim + 1, emb_dim=gnn_hidden_dim)
        gnn_embed_dim = gnn_hidden_dim * 2  # concat(global_max_pool, global_mean_pool)

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(gnn_embed_dim, fc_hidden_dim))
        for _ in range(num_fc_layer - 2):
            self.fcs.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
        self.fcs.append(nn.Linear(fc_hidden_dim, task_out_dim))

    def encode(self, batch) -> torch.Tensor:
        """Graph-level embedding, `(num_graphs, 2 * gnn_hidden_dim)`.

        Split out of `forward` so the trunk can be inspected on its own -- see
        `diagnose_synthnet_baseline.py`, which measures how much this varies
        between graphs.
        """
        num_nodes = batch.x.size(0)
        edge_index = batch.edge_index

        # Passed as a shim rather than assigned onto `batch`: the edge
        # direction below deliberately differs from the caller's, and
        # rewriting `batch.edge_index` in place would corrupt a Batch the
        # caller may still use.
        gnn_input = SimpleNamespace(
            batch=batch.batch,
            node_type=derive_node_type_index(batch.x),
            # Always counted on this project's own fanin -> node edges, never
            # on the (possibly reversed) edge_index handed to the GCN below:
            # upstream derives num_inverted_predecessors from the netlist at
            # graph-build time (andAIG2Graphml.py:47-57), not from edge_index,
            # so it means "inverted fanins of this node" under either
            # convention.
            num_inverted_predecessors=derive_num_inverted_predecessors(
                edge_index, batch.edge_attr, num_nodes
            ),
            edge_index=edge_index.flip(0)
            if self.upstream_edge_direction
            else edge_index,
        )
        return self.gnn(gnn_input)

    def forward(self, batch) -> torch.Tensor:
        """Args:
            batch: a `torch_geometric.data.Batch` with `.x`, `.edge_index`,
                `.edge_attr`, and `.batch` (graph-membership index).

        Returns:
            Tensor of shape `(num_graphs, task_out_dim)` in `[0, 1]`.
        """
        return self.head(self.encode(batch))

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Regression head over a graph embedding from `encode`.

        Separate from `forward` so callers that already hold an embedding do
        not have to re-run the trunk to get a prediction.
        """
        for fc in self.fcs[:-1]:
            h = F.relu(fc(h))
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
        return torch.sigmoid(self.fcs[-1](h))

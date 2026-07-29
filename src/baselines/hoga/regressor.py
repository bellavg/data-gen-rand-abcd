"""Graph-level optimizability regressor built on HOGA's hop-wise gated attention.

Adapts cornell-zhang/HOGA's `HOGA` module (see model.py in this directory,
vendored verbatim) from its released task -- per-node classification for
Gamora functional reasoning (xor/maj/root structural role) -- to this
project's task: a single scalar optimizability prediction per whole AIG.

IMPORTANT CAVEAT ON DEFAULTS: unlike SynthNet, the HOGA repo never released
training code or hyperparameters for its OpenABC-D QoR-prediction experiment
(the one actually analogous to this task) -- only the Gamora classification
task's `run.sh` was published (`--heads 32`), plus the model/hop-feature
building blocks with a note to "adjust them for your own tasks". So:
  - `heads=32` below is the one concrete value with a published source
    (run.sh), reused here.
  - `hidden_channels`, `num_layers`, `dropout`, and `num_hops` (propagation
    depth) have no published QoR-task default to inherit; they default to
    this project's own primary-model values (config.HIDDEN_DIM,
    config.NUM_LAYERS, config.DROPOUT) so the two baselines are at least
    comparable in scale to each other and to the primary model, not because
    the HOGA paper specifies them.

HOGA's trunk (hop-wise linear projection -> gated self-attention layers ->
node+neighbor fusion) is reused unmodified by subclassing `HOGA` itself and
overriding only `forward()`; the 3 Gamora classification heads
(`self.linear[1:3]`) are removed and replaced with global mean pooling +
a single `Linear -> ReLU -> Dropout -> Linear -> Sigmoid` regression head,
matching this project's existing head convention (models/base_model.py).

One shape-safety fix vs. upstream: `HOGA.forward()` ends the trunk with a
bare `.squeeze()`, which also collapses the node dimension when a batch
happens to contain exactly one total node (an edge case, but a real latent
bug). This version uses `.squeeze(1)` instead, which only removes the
intended singleton hop-dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from baselines.hoga.model import HOGA

# The one value with a published source: cornell-zhang/HOGA's run.sh
# (`python main_gamora.py --mapped 1 --heads 32 ...`), for the Gamora task
# HOGA does ship. No QoR-task config was ever released -- see module docstring.
DEFAULT_HEADS = 32


class HOGAGraphRegressor(HOGA):
    """HOGA's hop-wise gated-attention trunk + graph-level pooling + regression head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_hops: int,
        heads: int = DEFAULT_HEADS,
        attn_dropout: float = 0.0,
        attn_type: str = "vanilla",
        use_bias: bool = False,
        task_out_dim: int = 1,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,  # unused: the Gamora heads that consumed this are removed below.
            num_layers=num_layers,
            dropout=dropout,
            num_hops=num_hops,
            heads=heads,
            attn_dropout=attn_dropout,
            attn_type=attn_type,
            use_bias=use_bias,
        )
        # Drop the 3 Gamora node-classification heads (xor/maj/root); keep
        # self.linear[0] (hidden->hidden projection) and self.bn, which are
        # part of the shared trunk, not the Gamora-specific output.
        del self.linear[3]
        del self.linear[2]
        del self.linear[1]

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_channels // 2, task_out_dim),
        )

    def forward(self, batch) -> torch.Tensor:  # type: ignore[override]
        """Args:
            batch: a `torch_geometric.data.Batch` with `.hoga_x`
                (`[num_nodes_total, num_hop_slots, feat_dim]`, see
                hop_features.compute_hop_features) and `.batch`
                (graph-membership index).

        Returns:
            Tensor of shape `(num_graphs, task_out_dim)` in `[0, 1]`.
        """
        x = batch.hoga_x
        # --- HOGA's original trunk (model.py:HOGA.forward), unmodified except
        # for the `.squeeze(1)` shape-safety fix noted in the module docstring. ---
        x = self.lins[0](x)
        for i, tran in enumerate(self.trans):
            x = self.lns[i](self.gates[i](x) * (tran(x, x, x)[0]))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        target = x[:, 0, :].unsqueeze(1).repeat(1, self.num_hops - 1, 1)
        split_tensor = torch.split(x, [1, self.num_hops - 1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x = (node_tensor + neighbor_tensor).squeeze(1)
        x = self.linear[0](x)
        x = self.bn(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # --- end original trunk ---

        graph_embed = global_mean_pool(x, batch.batch)
        out = self.regression_head(graph_embed)
        return torch.sigmoid(out)

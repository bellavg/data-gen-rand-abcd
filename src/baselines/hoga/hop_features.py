"""Hop-wise feature precomputation for HOGA, adapted from cornell-zhang/HOGA's utils.py.

Source: https://github.com/cornell-zhang/HOGA/blob/master/utils.py (`graph2adj`,
`preprocess`, lines 1-63). License: BSD 3-Clause (see LICENSE_UPSTREAM).

Two deliberate deviations from the vendored algorithm, both documented here
rather than made silently:

1. Dependency substitution: upstream builds the normalized adjacency via
   `torch_sparse.SparseTensor`, which is not a dependency of this project (see
   pyproject.toml -- only `torch-scatter` is pinned as an optional PyG extra,
   and adding a second compiled PyG extension would mean re-pinning the HPC
   venv's wheel index). This module reimplements the exact same normalized
   directed adjacency (`graph2adj` below matches upstream's math verbatim)
   using `scipy.sparse` (already a project dependency) plus native
   `torch.sparse` COO tensors instead. The propagation itself -- Â^k X for
   k = 1..num_hops -- is unchanged.

2. Bug fix in the directed branch: upstream's `preprocess()` computes the
   "transposed" (reverse-direction) hop features as
   `high_order_features_tran = norm_adj @ high_order_features_tran`, i.e. it
   reuses the *forward* adjacency (`norm_adj`) instead of the reverse one
   (`norm_adj_tran`) for the reverse branch -- see the upstream source linked
   above. This looks like a copy-paste bug: it makes the reverse branch a
   duplicate of the forward branch rather than genuine reverse propagation.
   `compute_hop_features` below uses `norm_adj_tran` correctly for the
   reverse branch. Flagged explicitly since it changes the published
   algorithm's behavior, even though it's a correction rather than a
   simplification.

   NOTE the directed branch is no longer the default. `train_baseline.py`'s
   `--hoga_directed` defaults to False, matching the paper (Section 3.1
   defines a single symmetric-normalized adjacency, stacking K+1 slots) and
   upstream (whose `--directed` is `action='store_true'` and whose published
   run.sh never passes it). Directed mode remains available -- it was chosen
   here originally to capture fanin-ward and fanout-ward structure for AIGs,
   where causal cones matter -- but it doubles the slot width to 1 + 2K and
   costs ~1.85x throughout, so it is opt-in rather than standard. The
   `directed=True` defaults on the functions below are historical; every
   caller in this project passes the flag explicitly, and
   `train_baseline.py`'s argparse default is the operative one.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from torch_geometric.data import Data


def _graph2adj(adj: csr_matrix) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Matches upstream `graph2adj` exactly, operating on a scipy CSR matrix.

    Uses `.reshape(-1)` rather than upstream's `.squeeze()` to flatten the
    per-node degree vector: `.squeeze()` collapses a single-node graph's
    `(1, 1)` sum down to a 0-d array, which then crashes the boolean-mask
    assignment below (`TypeError: ... does not support item assignment`).
    `.reshape(-1)` always yields a 1-d array of length `num_nodes`.
    """
    degree_vec = np.asarray(adj.sum(axis=1)).reshape(-1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(degree_vec, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0
    degree_matrix = diags(d_inv_sqrt, 0)
    dad = degree_matrix @ (adj @ degree_matrix)
    ad = adj @ (degree_matrix @ degree_matrix)
    da = degree_matrix @ (degree_matrix @ adj)
    return dad.tocsr(), ad.tocsr(), da.tocsr()


def _scipy_csr_to_torch_sparse(mat: csr_matrix) -> torch.Tensor:
    coo = mat.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    # check_invariants=False is passed explicitly rather than left to the
    # global default: torch warns once per process otherwise ("Sparse invariant
    # checks are implicitly disabled"), which with persistent dataloader
    # workers means one such block per worker in every job log. The indices
    # come straight from scipy's own COO, so they are well-formed by
    # construction and the checks would only add per-graph overhead.
    return torch.sparse_coo_tensor(
        indices, values, coo.shape, check_invariants=False
    ).coalesce()


def compute_hop_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int,
    directed: bool = True,
) -> torch.Tensor:
    """Stack `[x, Â x, Â² x, ...]` (and the reverse direction if `directed`).

    Args:
        x: Raw node features, `[num_nodes, feat_dim]`.
        edge_index: `[2, num_edges]`, `edge_index[0]` = source, `[1]` = target
            (matches data/data_utils.py's AIG edge convention: fanin -> node).
        num_nodes: Number of nodes in this graph.
        num_hops: Number of propagation steps per direction.
        directed: If True (NOT the default -- see this module's docstring;
            the paper and upstream's released run are both undirected),
            propagate along both the
            forward (fanin -> node) and reverse (node -> fanin) directions
            separately, giving `1 + 2 * num_hops` stacked feature sets. If
            False, symmetrize the graph first and propagate once, giving
            `1 + num_hops` stacked feature sets.

    Returns:
        Tensor of shape `[num_nodes, 1 + num_hops * (2 if directed else 1), feat_dim]`.
    """
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    values = np.ones(row.shape[0], dtype=np.float32)

    if not directed:
        # Symmetrize: an undirected edge in both directions.
        sym_row = np.concatenate([row, col])
        sym_col = np.concatenate([col, row])
        sym_values = np.concatenate([values, values])
        adj = csr_matrix((sym_values, (sym_row, sym_col)), shape=(num_nodes, num_nodes))
        dad, _, _ = _graph2adj(adj)
        norm_adj = _scipy_csr_to_torch_sparse(dad)

        feats = [x]
        h = x.clone()
        for _ in range(num_hops):
            h = torch.sparse.mm(norm_adj, h)
            feats.append(h)
        return torch.stack(feats, dim=1)

    adj = csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))
    _, _, da = _graph2adj(adj)
    _, _, da_tran = _graph2adj(adj.transpose().tocsr())
    norm_adj = _scipy_csr_to_torch_sparse(da)
    norm_adj_tran = _scipy_csr_to_torch_sparse(da_tran)

    feats = [x]
    h_fwd = x.clone()
    h_rev = x.clone()
    for _ in range(num_hops):
        h_fwd = torch.sparse.mm(norm_adj, h_fwd)
        h_rev = torch.sparse.mm(norm_adj_tran, h_rev)  # corrected vs. upstream, see module docstring
        feats.append(h_fwd)
        feats.append(h_rev)
    return torch.stack(feats, dim=1)


def num_hop_slots(num_hops: int, directed: bool = True) -> int:
    """Total width of the stacked hop dimension, i.e. the `num_hops` argument HOGA's
    model constructor expects (confusingly named the same as this module's `num_hops`,
    which is the *propagation depth* rather than the stacked width)."""
    return 1 + num_hops * (2 if directed else 1)


class HopFeatureCache(torch.utils.data.Dataset):
    """Wraps a per-split AIG dataset, attaching HOGA hop-stacked features.

    Hop features depend only on graph structure (`edge_index`) and raw node
    features (`x`), so they *can* be computed once per sample and cached to
    disk (keyed by the sample's stable `graph_path`, plus `num_hops`/`directed`
    so different hop configs never collide) rather than recomputed every epoch.

    `cache_dir=None` disables the on-disk cache and computes hop features
    in-process on every access instead. **This is the right choice at this
    project's scale**, because the cache is not size-viable here: output is
    `[num_nodes, 1 + num_hops, 4]` float32, i.e. 96 B/node at num_hops=5
    undirected (the default; 176 B/node if `directed=True`), and this
    dataset's Orchestrate train+val splits total ~32.4e9
    nodes across ~788k graphs (~40k nodes/graph average) -- about 3.1 TB in
    ~788k files. Measured against a Snellius 8 TiB / 3M-inode scratch quota
    already ~90% full, that overruns both the space and the inode budget by
    roughly an order of magnitude. Recomputation is cheap by comparison
    (a handful of O(nnz) scipy/torch sparse ops per graph) and parallelises
    across dataloader workers, overlapped with GPU compute.

    Pass a `cache_dir` only when the split is small enough to fit -- e.g. a
    subset experiment -- in which case `precompute_all()` can populate it on
    CPU ahead of a GPU job.
    """

    def __init__(
        self,
        base_dataset,
        num_hops: int,
        cache_dir: str | Path | None = None,
        directed: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.num_hops = num_hops
        self.directed = directed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _cache_path(self, idx: int) -> Path:
        graph_path = self.base_dataset.samples[idx].graph_path
        # Mix in file size + mtime, matching data/dataset.py's own
        # _stable_graph_cache_name convention, so a graph regenerated at the
        # same path (e.g. a re-run of the data-creation pipeline) gets a new
        # cache key instead of silently reusing stale hop features. Falls
        # back to a path-only key if the file can't be stat'd (e.g. in unit
        # tests using synthetic paths that don't exist on disk).
        token = graph_path
        try:
            st = Path(graph_path).stat()
            token = f"{graph_path}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            pass
        key = hashlib.sha1(token.encode("utf-8")).hexdigest()
        suffix = "d" if self.directed else "u"
        return self.cache_dir / f"{key}_h{self.num_hops}_{suffix}.pt"

    def _compute_for(self, data: Data) -> torch.Tensor:
        return compute_hop_features(
            data.x, data.edge_index, int(data.num_nodes), self.num_hops, self.directed
        )

    def __getitem__(self, idx: int) -> Data:
        data = self.base_dataset[idx]

        if self.cache_dir is None:
            data.hoga_x = self._compute_for(data)
            return data

        cache_path = self._cache_path(idx)

        if cache_path.exists():
            data.hoga_x = torch.load(cache_path)
            return data

        hoga_x = self._compute_for(data)
        tmp_path = cache_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}")
        torch.save(hoga_x, tmp_path)
        os.replace(tmp_path, cache_path)
        data.hoga_x = hoga_x
        return data

    def precompute_all(self, log_every: int = 500) -> None:
        """Populate the on-disk cache for every sample in this split, on CPU.

        Only useful for subset experiments small enough to cache (see the class
        docstring); the full-scale training path runs with `cache_dir=None` and
        never calls this. Serial by design -- it exists for small splits, not
        as a throughput path.
        """
        if self.cache_dir is None:
            raise ValueError(
                "precompute_all() needs a cache_dir; with cache_dir=None hop "
                "features are computed on demand and there is nothing to "
                "populate. See this class's docstring for why None is the "
                "default at full dataset scale."
            )
        total = len(self)
        for idx in range(total):
            self[idx]
            if log_every and (idx + 1) % log_every == 0:
                print(f"[hoga hop cache] {idx + 1}/{total} done", flush=True)


def collate_hoga_batch(data_list: Sequence[Data]):
    """Collate a list of `Data` (each carrying `.hoga_x`) into a `Batch`."""
    from torch_geometric.data import Batch

    return Batch.from_data_list(list(data_list))

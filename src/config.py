import os

BATCH_SIZE = 32
WEIGHT_DECAY = 0.00396
ENCODER_NAME = "gcn"
NUM_LAYERS = 4
HIDDEN_DIM = 128
JK_MODE = "sum"
POOLING_TYPE = "mean"
PE_TYPE = "level"
HEADS = 4
LR = 0.0003
DROPOUT = 0.15
HEAD_DROPOUT = 0.3
POS_ENC_DIM = 32
NORM_TYPE = "layer"
DYNAMIC_BATCHING = True
MAX_TOTAL_NODES_PER_BATCH = 3_000_000
PIN_MEMORY = True
PERSISTENT_WORKERS = True
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
TORCH_COMPILE = True
MIN_LR = 1e-6
PATIENCE = 3

SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
WARMUP_STEPS = 10_000
WARMUP_START_LR = 1e-6
LOG_EVERY_N_STEPS = 1000
MAX_BATCH_COMPUTE_REPORTS = 4

# Static dataset/task constants
MAX_DEPTH = 24972
# Max depth should be rounded as there will be + 1 for primary outputs for depth encoded into AIG.

MAX_NUM_GATES = 366040

TASK_OUT_DIM = 1  # Regression task with single output value
NODE_INPUT_DIM = 4  # [constant, pi, and_gate, po]
# Exact-compression track only: data.exact_graph.fold_inversions_into_x
# appends an inverted-fanin count column and drops edge_attr entirely, so the
# exact model reads one more node feature and no edge features.
EXACT_NODE_INPUT_DIM = NODE_INPUT_DIM + 1
EDGE_ATTR_DIM = 2  # [normal edge, primary output edge]
NORMALIZE_EDGES = False
# Sparsification configuration
SPARSIFICATION_RANDOM_DROPOUT_RATE = 0.3  # target percentage of edges to drop
SPARSIFICATION_SPANNER_STRETCH = 3.0      # stretch factor for the spanner algorithm
SPARSIFICATION_PAGERANK_KEEP_RATIO = 0.8  # fraction of nodes to keep (0.0 to 1.0)
SPARSIFICATION_PAGERANK_ALPHA = 0.85      # damping factor for PageRank
SPARSIFICATION_SEED = 42

# Summarization / coarsening configuration.
# SUMMARIZATION_PARAMS is what the precompute driver actually passes to each
# registered method, so these are the settings a production run uses; the
# defaults in data/summarization.py exist only to keep the functions callable
# on their own.
SUMMARIZATION_SEED = 42
# Refinement/propagation depth is tied to the encoder depth (C1): information
# travels exactly NUM_LAYERS hops in the model, so merging nodes that agree to
# that depth is what the model cannot distinguish.
SUMMARIZATION_DEPTH = NUM_LAYERS
# Target fraction of nodes removed by ConvMatch, the one ratio-driven method
# left in the set.  Picked to
# match the sparsification sweep's mid-point so the Pareto fronts are
# comparable at a matched compression point (C8).
SUMMARIZATION_REDUCTION_RATIO = 0.5

SUMMARIZATION_PARAMS: dict[str, dict] = {
    # S1 — level-bounded cone coarsening (domain-specific).
    "cone": {"max_chain_length": 4, "level_band": 0},
    # S2 — graded WL / bisimulation.  count_cap=None is exact colour
    # refinement; set it to 1 for the bisimulation endpoint.
    #
    # pe_aware is set explicitly on both WL entries because it is the single
    # biggest lever on compression, and leaving it to the function default
    # would hide that.  Folding `level` into the initial colours keeps nodes
    # of different depth apart, which is what makes apply_merge_map's
    # min-pooled `level` exact — but on a deep datapath circuit it means
    # almost nothing merges.  Node retention measured end to end (50
    # unrandomized tier0 seed designs, provisional — see
    # summarization_notes.md):
    #
    #   design  levels  pe_aware=True  pe_aware=False
    #   sqrt      5059          99.0%            1.4%
    #   div       4373          95.2%            0.8%
    #   c6288      121          41.5%            2.0%
    #   jpeg        40           5.4%            1.7%
    #   aes         27           1.8%            1.4%
    #
    # So True is right for the lossy track (whose model consumes the PE) and
    # wrong for the exact track (whose model is pe_type="none").
    "wl": {
        "depth": SUMMARIZATION_DEPTH,
        "count_cap": None,
        "direction": "backward",
        "pe_aware": True,
    },
    # S2-exact — the same colour refinement, rewritten by
    # data.exact_graph.apply_exact_merge_map instead of apply_merge_map and
    # trained with models.base_model_exact.ExactGraphBaseModel.
    #
    # count_cap and direction here are EXACTNESS REQUIREMENTS, not tunables:
    #   count_cap=None     — capping compares neighbour sets, not multisets,
    #                        so a super-node no longer determines the number
    #                        of messages each member received.
    #   direction="backward" — the direction the GNN aggregates in.  Forward
    #                        or both merges nodes the model can still tell
    #                        apart, which is not lossless.
    # pe_aware=False is stated for intent, not effect: summarize_graph runs
    # fold_inversions_into_x *before* clustering on this path, and that drops
    # `level` entirely, so the flag is already inert here (measured: same 26
    # classes on adder either way).  It says what the exact track means —
    # its model has no positional encoding — and stops the value drifting to
    # True if the fold ever stops dropping `level`.  NOTE the wl/wl_exact
    # retention gap therefore comes from the fold, not from this flag.
    #
    # Only `depth` is free, and it is tied to the model's layer count by
    # summarize_graphs.assert_exact_depth_supports_model: exactness holds for
    # num_layers <= depth, no further.
    "wl_exact": {
        "depth": SUMMARIZATION_DEPTH,
        "count_cap": None,
        "direction": "backward",
        "pe_aware": False,
    },
    # S3 — A-ConvMatch (convolution matching), the general SOTA bar.
    # num_probes replaces the reference's exact kNN over the SGC embedding, so it
    # sets how good the candidate set is; 8 rather than the function default of 2
    # buys a measurable improvement in the paper's own objective at a real cost in
    # time and memory.  Numbers, and the reason not to go higher, live in
    # summarization_notes.md under "S3 measured against the paper's own
    # objective" — kept there rather than duplicated here.
    "convmatch": {
        "reduction_ratio": SUMMARIZATION_REDUCTION_RATIO,
        "sgc_depth": SUMMARIZATION_DEPTH,
        "num_probes": 8,
        "seed": SUMMARIZATION_SEED,
    },
}

# Dynamic partitioning heuristic: k = max(MIN_K, min(MAX_K, num_nodes // TARGET_NODES_PER_PART))
TARGET_NODES_PER_PART = 10_000  # target number of nodes per partition
MIN_K = 2  # minimum number of partitions
MAX_K = 32  # maximum number of partitions
PARTITION_SEED = 42

_user = os.environ.get("USER", "")
SPARSIFICATION_REPLACE_PATH = (
    f"/scratch-shared/{_user}/aig_train_run",
    f"/scratch-shared/{_user}/aig_mask_cache"
)

# Algorithms currently supported for training (train.py --algorithm validation).
# This project only trains on Orchestrate-optimized graphs.
VALID_ALGORITHMS = {"Orchestrate"}

# All algorithm names that appear in existing tier0/tier1/tier2 filenames on
# disk (the data-creation pipeline ran all four). Used by dataset_utils.py to
# parse filenames regardless of which algorithm is currently trained on.
KNOWN_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}

# Output dimension used by models (set equal to hidden dim by default)
OUTPUT_DIM = HIDDEN_DIM


ENCODER_KWARGS_DEFAULTS = {
    "node_input_dim": NODE_INPUT_DIM,
    "edge_attr_dim": EDGE_ATTR_DIM,
    "hid_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "jk_mode": JK_MODE,
    "norm_type": NORM_TYPE,
    "normalize_edges": NORMALIZE_EDGES,
}


def get_output_dim_for_encoder(encoder_name, encoder_kwargs):
    """
    Calculates the output dimension of the encoder based on its architecture
    and Jumping Knowledge (JK) strategy.
    """
    hid_dim = int(encoder_kwargs["hid_dim"])
    jk_mode = encoder_kwargs.get("jk_mode", "cat")  # Default to 'cat' if not specified

    if jk_mode == "cat":
        # Concatenation stacks all layer outputs + initial embedding
        num_layers = int(encoder_kwargs["num_layers"])
        return hid_dim * (num_layers + 1)
    else:
        # 'mean', 'max', 'sum', or 'last' all result in the same hid_dim
        return hid_dim


# GCN is the only GNN encoder model retained in the registry.
_ENCODER_MODULE_MAP = {
    "gcn": ("models.layers.gcn", "GCNEncoder"),
}
_ENCODER_CACHE: dict = {}


class _LazyEncoderRegistry(dict):
    """Dict-like registry that imports an encoder class on first access."""

    def __getitem__(self, key: str):
        if key not in _ENCODER_CACHE:
            if key not in _ENCODER_MODULE_MAP:
                raise KeyError(
                    f"Unknown encoder: {key!r}. Valid: {sorted(_ENCODER_MODULE_MAP)}"
                )
            module_path, class_name = _ENCODER_MODULE_MAP[key]
            import importlib

            mod = importlib.import_module(module_path)
            _ENCODER_CACHE[key] = getattr(mod, class_name)
        return _ENCODER_CACHE[key]

    def __contains__(self, key):
        return key in _ENCODER_MODULE_MAP

    def keys(self):
        return _ENCODER_MODULE_MAP.keys()

    def __iter__(self):
        return iter(_ENCODER_MODULE_MAP)

    def __len__(self):
        return len(_ENCODER_MODULE_MAP)


ENCODER_REGISTRY = _LazyEncoderRegistry()

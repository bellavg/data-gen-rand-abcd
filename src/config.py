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
# Target fraction of nodes removed by the two ratio-driven methods.  Picked to
# match the sparsification sweep's mid-point so the Pareto fronts are
# comparable at a matched compression point (C8).
SUMMARIZATION_REDUCTION_RATIO = 0.5

SUMMARIZATION_PARAMS: dict[str, dict] = {
    # S0 — control: no compression, exercises the pipeline end to end.
    "identity": {},
    # S1 — level-bounded cone coarsening (domain-specific).
    "cone": {"max_chain_length": 4, "level_band": 0},
    # S2 — graded WL / bisimulation.  count_cap=None is exact colour
    # refinement; set it to 1 for the bisimulation endpoint.
    "wl": {
        "depth": SUMMARIZATION_DEPTH,
        "count_cap": None,
        "direction": "backward",
    },
    # S3 — A-ConvMatch (convolution matching), the general SOTA bar.
    "convmatch": {
        "reduction_ratio": SUMMARIZATION_REDUCTION_RATIO,
        "sgc_depth": SUMMARIZATION_DEPTH,
        "seed": SUMMARIZATION_SEED,
    },
    # S4 — spectral / local-variation, the domain-blind control.  The node cap
    # bounds the eigensolver's cost; larger graphs fall back to heavy-edge.
    "spectral": {
        "reduction_ratio": SUMMARIZATION_REDUCTION_RATIO,
        "variant": "local_variation",
        "num_eigenvectors": 4,
        "max_spectral_nodes": 5_000,
    },
    # S5 — LSH / UGC hashing, the cheap linear-time tier.  Driven by
    # reduction_ratio rather than bin_width because the reference implementation
    # calibrates the bin width per dataset too rather than fixing it, and a fixed
    # bin width makes compression depend on graph size instead of on the knob.
    #
    # This does NOT put S5 on the same footing as S3/S4 for a matched-compression
    # comparison (C8).  S5's retention is capped by the number of distinct node
    # descriptors, which on an AIG is usually well below 0.5 — so at this setting
    # the calibration is inoperative and S5 returns its finest partition, more
    # compressed than requested.  The achieved ratio must be read off the
    # precompute stats, never assumed to be this value.  See lsh_coarsening.
    "lsh": {
        "reduction_ratio": SUMMARIZATION_REDUCTION_RATIO,
        "num_projections": 8,
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

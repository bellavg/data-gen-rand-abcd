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

# Eval-time batching (test.py). Deliberately SEPARATE from the two names above
# rather than a bumped value of them: those govern training, the trained
# checkpoints came from a 3M budget, and changing them would also change the
# training batch-plan cache key. Forward-only eval holds no gradients or
# optimizer state, so it has headroom for a larger budget.
# These are the single source of truth for eval batching — test.sh/test_cpu.sh
# deliberately do NOT pass them, so all 9 configs cannot drift apart.
EVAL_DYNAMIC_BATCHING = True
EVAL_MAX_TOTAL_NODES_PER_BATCH = 5_000_000
# Lower than PREFETCH_FACTOR: in-flight host memory is (num_workers x
# prefetch_factor) batches, and a node-budget eval batch is roughly an order of
# magnitude larger than a 32-graph training batch. Neither eval SLURM script
# requests an explicit --mem.
EVAL_PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
TORCH_COMPILE = True
MIN_LR = 1e-6
PATIENCE = 3

# WandB destination, shared by train.py and test.py so the two can't drift
# onto different projects/entities.
WANDB_PROJECT = "AIG-SUMMARIZE"
WANDB_ENTITY = "isabella-v-gardner-university-of-amsterdam"

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

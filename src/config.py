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
# Raised 5M -> 8M off a measured 5M eval run on an H100 80GB: peak GPU memory
# plateaued at ~40% (~34GB), i.e. ~6.8GB per 1M nodes. Activation memory scales
# ~linearly with nodes/edges per batch (GCN weights at hid=128/4 layers are
# negligible), so 8M projects to ~54GB / ~68% and leaves room for allocator
# fragmentation and for a peak between NVML samples. Do NOT expect a
# proportional speedup: that same run showed SM Active ~100% at 82% occupancy
# with FP32 pipeline ~5% and DRAM ~37% — the kernels are latency-bound on
# message-passing gather/scatter, so a bigger budget only amortizes per-batch
# collate/H2D/launch overhead. Raising this INVALIDATES cross-config hardware
# comparisons; every config must be re-run at the same value (see EVALUATION.md).
EVAL_MAX_TOTAL_NODES_PER_BATCH = 8_000_000
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

# Train/val/test grouping strategies (AIGGraphRegressionDataset split_by).
# These correspond to the standard train/test protocols used across the AIG-ML
# literature (OpenABC-D Task-1/Task-2, LOSTIN transductive/inductive,
# Lsoformer IP-Inductive/Recipe-Inductive, ABC-RL/INVICTUS/DeHNN design-disjoint):
# "design" holds out whole base circuits (unseen-IP / IP-Inductive, leak-safe
# default); "recipe" holds out whole ABC synthesis recipes -- every tier0 step
# plus its tier1/tier2 descendants for a given recipe ID -- across ALL designs,
# so circuits stay seen but a held-out recipe never appears anywhere in train
# (seen-IP/unseen-recipe, Recipe-Inductive); "random" applies no grouping at
# all (per-row split, the leakiest baseline, useful only as an ablation
# control to quantify how much leaky evaluation inflates reported performance).
VALID_SPLIT_BY = {"design", "recipe", "random"}
SPLIT_BY = "design"

# The reduction axis. "none" is the full-graph baseline; the other three name a
# family in src/data. Also the argparse choices for --reduction_type in test.py
# and benchmark.py, and the set run_label_for/parse_run_label agree on.
VALID_REDUCTION_TYPES = ("none", "sparsification", "partition", "summarization")


def run_label_for(
    algorithm: str,
    reduction_type: str,
    reduction_method: str | None,
    split_by: str = SPLIT_BY,
) -> str:
    """The directory name a configuration's checkpoints and logs live under.

    Single source of truth for train.py, test.py and benchmark.py. It used to be
    reimplemented in all three, they drifted, and the drift cost a result:
    train.py labelled by method alone, so ``--partition random`` and
    ``--split_by random`` both resolved to ``Orchestrate_random`` and wrote
    ``save_top_k=3`` checkpoints each into one directory. test.py's "best"
    selection then picked the partitioning run's epoch 3 (val_loss 0.0031) over
    the splitting run's epoch 16 (0.0048), and RQ1a's random-split row was
    measured on the partitioning model.

    The format is ``<algorithm>_<reduction_type>_<method>``. Carrying the
    *type* is what fixes the collision: method names are only unique within a
    family by luck. A non-default split reuses the same three slots as
    ``<algorithm>_none_<split_by>``, which is unambiguous because a reduction
    run never has type ``none``.

    The all-defaults configuration stays bare ``Orchestrate`` so the headline
    run keeps the directory it already has.

    Only the baseline is ever trained on a non-default split (see
    ``src/shell/test.sh``'s array mapping: there is nothing for
    ``sparsification:pagerank:recipe`` to evaluate), so three slots are enough.
    Asking for both a reduction and a non-default split has no representation
    here and raises, rather than returning a name that would quietly mean two
    things --- which is the failure this function exists to prevent.

    >>> run_label_for("Orchestrate", "none", None)
    'Orchestrate'
    >>> run_label_for("Orchestrate", "partition", "random")
    'Orchestrate_partition_random'
    >>> run_label_for("Orchestrate", "none", None, split_by="random")
    'Orchestrate_none_random'
    """
    if reduction_type != "none":
        if split_by != SPLIT_BY:
            raise ValueError(
                f"No run label exists for {reduction_type}/{reduction_method} on "
                f"the {split_by!r} split: the label format has three slots and a "
                f"reduction already uses them. Only the baseline is trained on a "
                f"non-default split. Extend the format before training this."
            )
        return f"{algorithm}_{reduction_type}_{reduction_method}"
    if split_by != SPLIT_BY:
        return f"{algorithm}_none_{split_by}"
    return algorithm


def parse_run_label(label: str) -> tuple[str, str, str | None, str]:
    """Inverse of :func:`run_label_for`.

    The label is not only a directory name: test.py names its result and
    prediction CSVs after it, and the analysis package has to recover the
    configuration from those filenames. That reader used to split positionally
    on underscores and take everything before the eval mode as the method,
    which silently produced ``partition_metis`` as a method name --- and since
    the figure code fails closed on an unregistered method, every real result
    would have been drawn as fabricated data. Parsing with the same definition
    that wrote the name is what stops the two from drifting.

    Returns ``(algorithm, reduction_type, reduction_method, split_by)``.

    >>> parse_run_label("Orchestrate")
    ('Orchestrate', 'none', None, 'design')
    >>> parse_run_label("Orchestrate_partition_random")
    ('Orchestrate', 'partition', 'random', 'design')
    >>> parse_run_label("Orchestrate_none_random")
    ('Orchestrate', 'none', None, 'random')
    """
    for reduction_type in VALID_REDUCTION_TYPES:
        if reduction_type == "none":
            continue
        head, sep, tail = label.partition(f"_{reduction_type}_")
        if sep:
            return head, reduction_type, tail, SPLIT_BY
    # A `none` type never names a method, so whatever follows it is the split.
    # Checking the value against VALID_SPLIT_BY keeps a design named
    # "some_none_thing" from being read as a splitting protocol.
    head, sep, tail = label.partition("_none_")
    if sep and tail in VALID_SPLIT_BY:
        return head, "none", None, tail
    return label, "none", None, SPLIT_BY

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

BATCH_SIZE = 32
WEIGHT_DECAY = 0.004
ENCODER_NAME = "transformer_conv"
NUM_LAYERS = 8
HIDDEN_DIM = 128
JK_MODE = "last"
POOLING_TYPE = "max"
PE_TYPE = "level"
HEADS = 4
LR = 0.0005
DROPOUT = 0.28
POS_ENC_DIM = 32
NORM_TYPE = "layer"
DYNAMIC_BATCHING = True
DYNAMIC_BUCKET_RULES = "250000:1,150000:2,100000:4,75000:8,50000:16"
PIN_MEMORY = True
PERSISTENT_WORKERS = True
MIN_LR = 1e-6
LOG_EVERY_N_STEPS = 1000

# Default Huber delta for training
HUBER_DELTA = 0.95

# Static dataset/task constants
MAX_DEPTH = 24972
# Max depth should be rounded as there will be + 1 for primary outputs for depth encoded into AIG.

MAX_NUM_GATES = 366040

TASK_OUT_DIM = 1  # Regression task with single output value
NODE_INPUT_DIM = 4  # [constant, pi, and_gate, po]
EDGE_ATTR_DIM = 2  # [normal edge, primary output edge]

VALID_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}

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
    # EGIN-specific
    "egin_kwargs": {
        "num_mlp_layers": None,
        "dot_update": None,
        "edge_mlp": None,
        "edge_hidden_dim": HIDDEN_DIM,
    },
    # Attention/Transformer/GPS specific
    "heads": HEADS,
    "concat": None,
    "performer_redraw_interval": None,
    "beta": None,
    "root_weight": None,
}

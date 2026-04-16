from models.layers.egin import GraphEGIN
from models.layers.gcn import GCNEncoder
from models.layers.gine import GINEEncoder
from models.layers.graphgps import GraphGPSEncoder
from models.layers.transformer_conv import TransformerConvEncoder
from models.layers.vanilla_mpnn import MPNNEncoder

MAX_DEPTH = 24972
# Max depth should be rounded as there will be + 1 for primary outputs for depth encoded into AIG.

MAX_NUM_GATES = 366040

TASK_OUT_DIM = 1  # Regression task with single output value
NODE_INPUT_DIM = 4  # [constant, pi, and_gate, po]
EDGE_ATTR_DIM = 2  # [normal edge, primary output edge]


ENCODER_REGISTRY = {
    "gcn": GCNEncoder,
    "gine": GINEEncoder,
    "graphgps": GraphGPSEncoder,
    "transformer_conv": TransformerConvEncoder,
    "vanilla_mpnn": MPNNEncoder,
    "egin": GraphEGIN,
}


VALID_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}

ENCODER_KWARGS_DEFAULTS = {
    # Unified encoder kwargs (set to None so the same dict can be passed
    # to any encoder and left unspecified fields will be handled by the
    # encoder's own defaults).
    # Use base-model naming: `node_input_dim` / `edge_attr_dim` instead of
    # `in_dim` / `edge_dim` so a single dict can be passed into UnifiedGraphBaseModel.
    "node_input_dim": NODE_INPUT_DIM,
    "edge_attr_dim": EDGE_ATTR_DIM,
    "hid_dim": None,
    "num_layers": None,
    "dropout": None,
    "jk_mode": None,
    "norm_type": None,
    # EGIN-specific
    "egin_kwargs": {
        "num_mlp_layers": None,
        "dot_update": None,
        "edge_mlp": None,
        "edge_hidden_dim": None,
    },
    # Attention/Transformer/GPS specific
    "heads": None,
    "concat": None,
    "performer_redraw_interval": None,
    "beta": None,
    "root_weight": None,
}


def get_output_dim_for_encoder(encoder_name, encoder_kwargs):
    """
    Calculates the output dimension of the encoder based on its architecture
    and Jumping Knowledge (JK) strategy.
    """
    if encoder_name == "egin":
        return TASK_OUT_DIM

    hid_dim = int(encoder_kwargs["hid_dim"])
    jk_mode = encoder_kwargs.get("jk_mode", "cat")  # Default to 'cat' if not specified

    if jk_mode == "cat":
        # Concatenation stacks all layer outputs + initial embedding
        num_layers = int(encoder_kwargs["num_layers"])
        return hid_dim * (num_layers + 1)
    else:
        # 'mean', 'max', 'sum', or 'last' all result in the same hid_dim
        return hid_dim

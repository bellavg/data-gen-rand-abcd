MAX_DEPTH = 24972
# Max depth should be rounded as there will be + 1 for primary outputs for depth encoded into AIG.

MAX_NUM_GATES = 366040

TASK_OUT_DIM = 1  # Regression task with single output value
NODE_INPUT_DIM = 4  # [constant, pi, and_gate, po]
EDGE_ATTR_DIM = 2  # [normal edge, primary output edge]


# Encoder classes are imported lazily on first access to avoid loading all six
# heavy modules (including PerformerAttention in graphgps.py) at startup.
# Each trial only uses one encoder, so the other five should not be paid for.
_ENCODER_MODULE_MAP = {
    "gcn": ("models.layers.gcn", "GCNEncoder"),
    "gine": ("models.layers.gine", "GINEEncoder"),
    "graphgps": ("models.layers.graphgps", "GraphGPSEncoder"),
    "transformer_conv": ("models.layers.transformer_conv", "TransformerConvEncoder"),
    "vanilla_mpnn": ("models.layers.vanilla_mpnn", "MPNNEncoder"),
    "egin": ("models.layers.egin", "GraphEGIN"),
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


ENCODER_REGISTRY = _LazyEncoderRegistry()


VALID_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}


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

from models.layers.egin import GraphEGIN
from models.layers.gcn import GCNEncoder
from models.layers.gine import GINEEncoder
from models.layers.graphgps import GraphGPSEncoder
from models.layers.rgcn import RGCNEncoder
from models.layers.transformer_conv import TransformerConvEncoder
from models.layers.vanilla_mpnn import MPNNEncoder

MAX_DEPTH = 24972
# Max depth should be rounded as there will be + 1 for primary outputs for depth encoded into AIG. 

MAX_NUM_GATES = 366040


ENCODER_REGISTRY = {
    "gcn": GCNEncoder,
    "gine": GINEEncoder,
    "graphgps": GraphGPSEncoder,
    "transformer_conv": TransformerConvEncoder,
    "vanilla_mpnn": MPNNEncoder,
    "rgcn": RGCNEncoder,
    "egin": GraphEGIN,
}


VALID_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}



MAX_DEPTH = None

MAX_NUM_NODES = None


ENCODER_REGISTRY = {
	"gcn": GCNEncoder,
	"gine": GINEEncoder,
	"graphgps": GraphGPSEncoder,
	"transformer_conv": TransformerConvEncoder,
	"vanilla_mpnn": MPNNEncoder,
	"rgcn": RGCNEncoder,
	"egin": GraphEGIN,
}

import torch.nn as nn
from src.baselines.dagnn import DAGNN

class DAGNNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, num_rels=2, dropout=0.0, readout='mean'):
        super().__init__()
        # DAGNN handles its own normalization and pooling internally via out_pool
        self.encoder = DAGNN(
            emb_dim=hid_dim, 
            hidden_dim=hid_dim, 
            num_rels=num_rels, 
            w_edge_attr=True, 
            num_layers=num_layers, 
            out_pool=readout, # passes mean, max, add directly
            in_feat=in_dim,
            dropout=dropout,
            num_class=hid_dim # Used as output embedding dim
        )
        
    def forward(self, data):
        return self.encoder(data)
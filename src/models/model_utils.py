import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


def identity_pool(x, batch=None):
    """No-op pool used when node embeddings should be preserved."""
    return x

def get_norm_layer(norm_type, dim):
    """Returns a normalization layer based on the string provided."""
    if norm_type is None or norm_type.lower() == 'none':
        return nn.Identity()
    elif norm_type.lower() == 'batch':
        return nn.BatchNorm1d(dim)
    elif norm_type.lower() == 'layer':
        return nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
        
        # GCNNorm, graph norm


def get_pyg_pool(pool_type):
    """Returns a PyG global pooling function."""
    if pool_type.lower() == 'mean':
        return global_mean_pool
    elif pool_type.lower() == 'max':
        return global_max_pool
    elif pool_type.lower() in ['add', 'sum']:
        return global_add_pool
    elif pool_type.lower() in ['identity', 'none']:
        return identity_pool
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")
        # todo add SAGPooling, ASAPooling? 
        
def dense_pool(x, pool_type):
    """Applies pooling over dense graphs of shape (Batch, Nodes, Features)"""
    if pool_type.lower() == 'mean':
        return x.mean(dim=1) if len(x.shape) == 3 else x.mean(dim=0, keepdim=True)
    elif pool_type.lower() == 'max':
        return x.max(dim=1)[0] if len(x.shape) == 3 else x.max(dim=0, keepdim=True)[0]
    elif pool_type.lower() in ['add', 'sum']:
        return x.sum(dim=1) if len(x.shape) == 3 else x.sum(dim=0, keepdim=True)
    elif pool_type.lower() in ['identity', 'none']:
        return x
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


def get_batch_positional_encoding(batch: object) -> torch.Tensor | None:
    """Return the collated `pos_enc` tensor from a PyG Batch (or Data) object.

    - If `batch` has no `pos_enc` attribute, returns `None`.
    - If `pos_enc` is 1D, it will be unsqueezed to shape `(N, 1)` and cast to float.
    - Otherwise returns `pos_enc.float()`.
    """
    pe = getattr(batch, "pos_enc", None)
    if pe is None:
        return None
    if pe.dim() == 1:
        pe = pe.unsqueeze(-1)
    return pe.float()

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from numpy import linalg as la
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

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

def get_pyg_pool(pool_type):
    """Returns a PyG global pooling function."""
    if pool_type.lower() == 'mean':
        return global_mean_pool
    elif pool_type.lower() == 'max':
        return global_max_pool
    elif pool_type.lower() in ['add', 'sum']:
        return global_add_pool
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")
        
def dense_pool(x, pool_type):
    """Applies pooling over dense graphs of shape (Batch, Nodes, Features)"""
    if pool_type.lower() == 'mean':
        return x.mean(dim=1) if len(x.shape) == 3 else x.mean(dim=0, keepdim=True)
    elif pool_type.lower() == 'max':
        return x.max(dim=1)[0] if len(x.shape) == 3 else x.max(dim=0, keepdim=True)[0]
    elif pool_type.lower() in ['add', 'sum']:
        return x.sum(dim=1) if len(x.shape) == 3 else x.sum(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

def compute_Dq_dynamic(dag: nx.DiGraph, target_node: int) -> np.ndarray:
    """
    Compute the diagonal frequency response matrix path_exists for a target node.
    """
    N = dag.number_of_nodes()
    path_exists = np.zeros(N)
    for i in range(N):
        path_exists[i] = nx.has_path(dag, i, target_node)
    return np.diag(path_exists)

def compute_gso_from_adj(adj_matrix: np.ndarray) -> torch.Tensor:
    """
    Computes the K Graph Shift Operators (GSOs) for a given adjacency matrix.
    Returns a tensor of shape (N, N, N) where K = N.
    """
    N = adj_matrix.shape[0]
    dag = nx.from_numpy_array(adj_matrix.T, create_using=nx.DiGraph())
    W = adj_matrix 
    
    try:
        W_inv = la.inv(W)
    except la.LinAlgError:
        W_inv = la.pinv(W)
        
    GSOs = np.array([W @ compute_Dq_dynamic(dag, i) @ W_inv for i in range(N)])
    return torch.tensor(GSOs, dtype=torch.float32)
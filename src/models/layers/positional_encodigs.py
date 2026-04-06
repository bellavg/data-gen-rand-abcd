import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path


# ==============================================================================
# Highly Scalable Encodings for Massive DAGs / AIGs (O(V + E) complexity)
# ==============================================================================
from torch_geometric.utils import k_hop_subgraph

class LocalRelativeEncoding:
    """
    Computes Directed Shortest Path distances for nodes within a local neighborhood.
    Stores results as expanded edge_index and edge_attr.
    """
    def __init__(self, max_hops: int = 3, attr_name: str = 'edge_rel_dist'):
        self.max_hops = max_hops
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        new_edges = []
        new_dists = []

        # For every node, find its local k-hop directed neighborhood
        for i in range(num_nodes):
            # Get nodes reachable from 'i' within max_hops
            subset, local_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=i, 
                num_hops=self.max_hops, 
                edge_index=edge_index, 
                relabel_nodes=True, 
                directed=True
            )
            
            # Simple BFS within this tiny subgraph to get distances from 'i'
            # (mapping[0] is the index of 'i' in the local subgraph)
            start_node = mapping[0].item()
            dists = self._local_bfs(start_node, local_edge_index, subset.size(0))
            
            for local_idx, d in enumerate(dists):
                if 0 < d <= self.max_hops:
                    global_idx = subset[local_idx].item()
                    new_edges.append([i, global_idx])
                    new_dists.append(d)

        # Convert to tensors
        data.edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        # Store distances as a feature (can be one-hot encoded later)
        setattr(data, self.attr_name, torch.tensor(new_dists, dtype=torch.float).unsqueeze(1))
        
        return data

    def _local_bfs(self, start_node, edge_index, num_nodes):
        dists = [-1] * num_nodes
        dists[start_node] = 0
        queue = [start_node]
        while queue:
            u = queue.pop(0)
            if dists[u] >= self.max_hops:
                continue
            # Find neighbors of u
            neighbors = edge_index[1][edge_index[0] == u]
            for v in neighbors:
                v = v.item()
                if dists[v] == -1:
                    dists[v] = dists[u] + 1
                    queue.append(v)
        return dists


class FastAIGDepthPE:
    # to do add depth by yoursel fin data rpocessing 
    """
    Computes Logic Depth highly efficiently for massive graphs using SciPy's C-backend.
    Primary Inputs have depth 0. Gates have depth = max(predecessor depths) + 1.
    
    If 'discrete=True': Outputs integer depths. Use this with `LearnedDepthEmbedding`.
    If 'discrete=False': Outputs continuous Sinusoidal vectors directly.
    """
    def __init__(self, dim: int = 16, attr_name: str = 'pos_enc', discrete: bool = False):
        self.dim = dim
        self.attr_name = attr_name
        self.discrete = discrete

    def __call__(self, data: Data) -> Data:
        # 1. Convert edge_index to a scipy sparse matrix
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        
        # 2. Identify Primary Inputs (nodes with in-degree == 0)
        in_degrees = np.array(adj.sum(axis=0)).flatten()
        primary_inputs = np.where(in_degrees == 0)[0]
        
        if len(primary_inputs) == 0:
            # Fallback if no primary inputs exist (cyclic graph)
            logic_depths = torch.zeros(data.num_nodes, dtype=torch.long)
        else:
            # 3. Use Bellman-Ford to find the longest path from ANY primary input.
            # We invert the adjacency matrix weights to trick shortest_path into finding longest paths.
            adj_neg = adj.copy()
            adj_neg.data = -np.ones_like(adj_neg.data)
            
            # Calculate shortest paths from all primary inputs (directed=True ensures forward logic flow)
            distances = shortest_path(adj_neg, directed=True, indices=primary_inputs, method='bellman-ford')
            
            # The logic depth is the maximum negative distance (inverted back to positive)
            valid_distances = np.where(np.isinf(distances), 0, distances)
            logic_depths = torch.tensor(-np.min(valid_distances, axis=0), dtype=torch.long)
            
        if self.discrete:
            setattr(data, self.attr_name, logic_depths)
        else:
            position = logic_depths.unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.dim, 2).float() * -(np.log(10000.0) / self.dim))
            
            pe = torch.zeros(data.num_nodes, self.dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            setattr(data, self.attr_name, pe)
            
        return data


class DAGPathCountPE:
    """
    Computes a scalable structural encoding for DAGs/AIGs.
    Uses Dynamic Programming to count the number of paths from Primary Inputs 
    to every node in linear time O(V + E).
    
    Provides 'structural neighborhood' context similar to Random Walks, but scales instantly.
    """
    def __init__(self, attr_name: str = 'path_pe', normalize: bool = True):
        self.attr_name = attr_name
        self.normalize = normalize

    def __call__(self, data: Data) -> Data:
        G = nx.DiGraph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.t().tolist())
        
        path_counts = torch.zeros(data.num_nodes, dtype=torch.float32)
        
        try:
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                if not preds:
                    path_counts[node] = 1.0  # Primary input
                else:
                    path_counts[node] = sum(path_counts[p] for p in preds)
        except nx.NetworkXUnfeasible:
            pass # Handle cyclic graphs gracefully
            
        if self.normalize:
            # log1p prevents massive path counts from destabilizing neural network gradients
            path_counts = torch.log1p(path_counts)
            
        # Reshape to (N, 1) so it acts as a valid feature dimension
        setattr(data, self.attr_name, path_counts.unsqueeze(1))
        
        return data


class AddSinusoidalPE:
    """
    Normal sinusoidal positional encoding based on an ordered node index.
    Useful mostly if nodes have a strict sequential or temporal ordering.
    """
    def __init__(self, dim: int = 16, attr_name: str = 'pos_enc'):
        self.dim = dim
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        position = torch.arange(data.num_nodes).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * -(np.log(10000.0) / self.dim))
        
        pe = torch.zeros(data.num_nodes, self.dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        setattr(data, self.attr_name, pe)
        return data


# ==============================================================================
# PyTorch Module for Learned Positional Encodings
# ==============================================================================

class LearnedDepthEmbedding(nn.Module):
    """
    If using FastAIGDepthPE(discrete=True), place this inside your UnifiedGraphBaseModel
    to map the integer depths to learned continuous vectors during the forward pass.
    """
    def __init__(self, max_depth: int, embed_dim: int):
        super().__init__()
        self.max_depth = max_depth
        self.embed = nn.Embedding(max_depth, embed_dim)

    def forward(self, depth_indices: torch.Tensor) -> torch.Tensor:
        # Clamp bounds to ensure that if a massive circuit with unseen depths 
        # is fed during inference, the program does not crash.
        clamped_indices = depth_indices.clamp(min=0, max=self.max_depth - 1)
        return self.embed(clamped_indices)


# ==============================================================================
# Factory Functions for Dynamic Configuration
# ==============================================================================

def get_pe_transform(pe_type: str, **kwargs):
    """
    Returns an instantiated Positional Encoding transform based on a text string.
    Use this in your Dataset/DataLoader pipeline configuration.
    """
    if pe_type is None or pe_type.lower() == 'none':
        # Return a dummy transform that passes data through untouched
        return lambda data: data
        
    pe_type = pe_type.lower()
    
    if pe_type in ['dag', 'dag_depth', 'fast_aig_depth']:
        return FastAIGDepthPE(**kwargs)
        
    elif pe_type in ['path', 'dag_path', 'path_count']:
        return DAGPathCountPE(**kwargs)
        
    elif pe_type in ['sinusoidal', 'sine']:
        return AddSinusoidalPE(**kwargs)
        
    else:
        raise ValueError(f"Unknown positional encoding transform: {pe_type}")


def get_pos_enc_layer(pe_type: str | None, pos_enc_dim: int = 16, max_depth: int = 1000) -> nn.Module:
    """
    Returns a PyTorch module for processing batch positional encodings.
    Initialize this inside your Base Model using a text string.
    """
    if pe_type is None or pe_type.lower() == 'none':
        # If no processing is needed (e.g., already continuous vectors), do nothing.
        return nn.Identity()
    elif pe_type.lower() in ['learned', 'learned_depth']:
        # Map discrete integer depths to continuous learned embeddings
        return LearnedDepthEmbedding(max_depth=max_depth, embed_dim=pos_enc_dim)
    else:
        raise ValueError(f"Unknown pos_enc layer type: {pe_type}")




def get_batch_positional_encoding(batch, keys=('pos_enc', 'pe', 'pe_lap_pe')):
    """Fetches positional encoding tensor from a batch using prioritized keys."""
    for key in keys:
        value = getattr(batch, key, None)
        if value is not None:
            return value
    return None

def validate_positional_encoding(pos_enc: torch.Tensor, pos_enc_dim: int, pos_enc_mode: str):
    """Common helper to validate PE dimensions across all encoders."""
    if pos_enc is None or pos_enc_dim == 0 or pos_enc_mode.lower() == 'none':
        return
    if pos_enc.size(-1) != pos_enc_dim:
        raise ValueError(
            f"Expected pos_enc with feature size {pos_enc_dim}, got {pos_enc.size(-1)}"
        )

def integrate_positional_encoding(x: torch.Tensor, pos_enc: torch.Tensor, pos_enc_dim: int, pos_enc_mode: str):
    """Common helper to concatenate PE to node features if mode is 'concat'."""
    if pos_enc is None or pos_enc_dim == 0 or pos_enc_mode.lower() != 'concat':
        return x
    return torch.cat([x, pos_enc], dim=-1)
import torch
from torch_geometric.data import Data, Batch

def and_gate_only_sparsification_vec(batch: Batch) -> Batch:
    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    device = x.device

    is_pi = x[:, 1] == 1.0
    is_po = x[:, 3] == 1.0
    removed_mask = is_pi | is_po
    kept_mask = ~removed_mask
    
    n = x.size(0)
    kept = kept_mask.nonzero(as_tuple=True)[0]
    
    old_to_new = torch.full((n,), -1, dtype=torch.long, device=device)
    old_to_new[kept] = torch.arange(len(kept), dtype=torch.long, device=device)
    
    u = edge_index[0]
    v = edge_index[1]
    
    u_is_pi = is_pi[u]
    v_is_po = is_po[v]
    u_kept = kept_mask[u]
    v_kept = kept_mask[v]
    
    mask_pi_to_kept = u_is_pi & v_kept
    mask_kept_to_po = u_kept & v_is_po
    mask_kept_to_kept = u_kept & v_kept
    
    sl_mask = mask_pi_to_kept | mask_kept_to_po
    if sl_mask.any():
        sl_nodes = torch.where(mask_pi_to_kept, v, u)[sl_mask]
        sl_attr = edge_attr[sl_mask]
        
        sl_cat = torch.cat([sl_nodes.unsqueeze(1).to(sl_attr.dtype), sl_attr], dim=1)
        sl_cat = torch.unique(sl_cat, dim=0)
        
        sl_nodes_unique = sl_cat[:, 0].long()
        sl_attr_unique = sl_cat[:, 1:]
        
        sl_nodes_new = old_to_new[sl_nodes_unique]
        
        new_src_sl = sl_nodes_new
        new_dst_sl = sl_nodes_new
        new_attr_sl = sl_attr_unique
    else:
        new_src_sl = torch.empty((0,), dtype=torch.long, device=device)
        new_dst_sl = torch.empty((0,), dtype=torch.long, device=device)
        new_attr_sl = torch.empty((0, edge_attr.size(1)), dtype=edge_attr.dtype, device=device)

    if mask_kept_to_kept.any():
        new_src_kk = old_to_new[u[mask_kept_to_kept]]
        new_dst_kk = old_to_new[v[mask_kept_to_kept]]
        new_attr_kk = edge_attr[mask_kept_to_kept]
    else:
        new_src_kk = torch.empty((0,), dtype=torch.long, device=device)
        new_dst_kk = torch.empty((0,), dtype=torch.long, device=device)
        new_attr_kk = torch.empty((0, edge_attr.size(1)), dtype=edge_attr.dtype, device=device)
        
    new_src = torch.cat([new_src_sl, new_src_kk])
    new_dst = torch.cat([new_dst_sl, new_dst_kk])
    new_attr = torch.cat([new_attr_sl, new_attr_kk])
    
    new_x = x[kept]
    
    if new_src.numel() > 0:
        new_edge_index = torch.stack([new_src, new_dst], dim=0)
        new_edge_attr = new_attr
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        new_edge_attr = torch.empty((0, edge_attr.size(1)), dtype=edge_attr.dtype, device=device)

    # We must properly rebuild the batch structures!
    # PyG Batch relies on batch.batch to map nodes to graphs
    if hasattr(batch, 'batch') and batch.batch is not None:
        new_batch_idx = batch.batch[kept]
    else:
        new_batch_idx = None
        
    out = Batch(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, batch=new_batch_idx)
    
    # We also need to recalculate ptr if it exists
    if hasattr(batch, 'ptr') and batch.ptr is not None:
        # ptr is the cumulative sum of nodes per graph
        # We can recompute it from new_batch_idx using torch.bincount
        if new_batch_idx is not None:
            counts = torch.bincount(new_batch_idx, minlength=batch.num_graphs)
            out.ptr = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)])
    
    # Copy other attributes correctly
    n_orig = x.size(0)
    for key in batch.keys():
        if key in ("x", "edge_index", "edge_attr", "num_nodes", "num_edges", "batch", "ptr"):
            continue
        val = batch[key]
        if isinstance(val, torch.Tensor) and val.dim() > 0 and val.size(0) == n_orig:
            setattr(out, key, val[kept])
        else:
            setattr(out, key, val)

    return out

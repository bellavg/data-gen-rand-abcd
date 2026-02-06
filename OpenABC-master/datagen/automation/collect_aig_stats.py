#!/usr/bin/env python3
"""
Collect statistics from AIG bench files and write CSV manifest.

Usage:
  python collect_aig_stats.py --root /path/to/OPENABC_DATASET/bench --out stats.csv

The script parses AIG bench files (the same format used in this repo) and
computes per-file features useful for ML: PI/PO counts, internal nodes, edge
types, longest path (depth), fanout stats, file size, etc.

This is intentionally self-contained (no intermediate graphml files).
"""
import argparse
import os
import re
import csv
import sys
from pathlib import Path
import networkx as nx
import networkx.algorithms.dag as nxdag


def parse_aig_bench(path):
    """Parse a bench AIG file into a directed graph with attributes.

    Returns: (G, meta)
      G: networkx.DiGraph with node attrs: node_type (PI/PO/Internal)
      meta: dict with counts collected during parse
    """
    node_id_map = {}
    AIG = nx.DiGraph()
    po_list = []
    single_input_map = {}
    idx = 0
    and_count = 0
    not_edges = 0
    buff_edges = 0

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if 'INPUT' in line:
                m = re.search(r'INPUT\((.*?)\)', line)
                if m:
                    name = m.group(1)
                    AIG.add_node(idx, node_id=name, node_type='PI')
                    node_id_map[name] = idx
                    idx += 1
            elif 'OUTPUT' in line:
                m = re.search(r'OUTPUT\((.*?)\)', line)
                if m:
                    po = m.group(1)
                    po_list.append(po)
            elif 'AND' in line:
                # out = AND(in1,in2)
                m1 = re.search(r'^(.*?)=', line)
                if not m1:
                    continue
                out = m1.group(1).strip()
                m = re.search(r'AND\((.*?),(.*?)\)', line)
                if not m:
                    continue
                in1 = m.group(1).strip()
                in2 = m.group(2).strip()
                # ensure inputs exist in mapping
                inputs = []
                for inp in (in1, in2):
                    if inp in node_id_map:
                        inputs.append(node_id_map[inp])
                    elif inp in single_input_map:
                        inputs.append(node_id_map[single_input_map[inp]])
                    else:
                        # placeholder PI if unseen
                        AIG.add_node(idx, node_id=inp, node_type='PI')
                        node_id_map[inp] = idx
                        inputs.append(idx)
                        idx += 1
                # add internal node
                AIG.add_node(idx, node_id=out, node_type='Internal')
                node_id_map[out] = idx
                # add edges from internal node to its inputs (direction: node -> predecessor)
                for src in inputs:
                    # detect inversion - in bench format inversion is represented by using inverted node name in mapping; heuristics omitted
                    AIG.add_edge(idx, src)
                    buff_edges += 1
                and_count += 1
                # if output is also a PO, add a PO node
                if out in po_list:
                    AIG.add_node(idx+1, node_id=out+"_po", node_type='PO')
                    AIG.add_edge(idx+1, idx)
                    node_id_map[out+"_po"] = idx+1
                    idx += 1
                idx += 1
            elif 'NOT' in line:
                m1 = re.search(r'^(.*?)=', line)
                if not m1:
                    continue
                out = m1.group(1).strip()
                m = re.search(r'NOT\((.*?)\)', line)
                if not m:
                    continue
                inp = m.group(1).strip()
                single_input_map[out] = inp
            elif 'BUFF' in line:
                m1 = re.search(r'^(.*?)=', line)
                if not m1:
                    continue
                out = m1.group(1).strip()
                m = re.search(r'BUFF\((.*?)\)', line)
                if not m:
                    continue
                inp = m.group(1).strip()
                # try to resolve input
                if inp in node_id_map:
                    src = node_id_map[inp]
                elif inp in single_input_map and single_input_map[inp] in node_id_map:
                    src = node_id_map[single_input_map[inp]]
                    not_edges += 1
                else:
                    AIG.add_node(idx, node_id=inp, node_type='PI')
                    src = idx
                    node_id_map[inp] = idx
                    idx += 1
                AIG.add_node(idx, node_id=out+"_po", node_type='PO')
                AIG.add_edge(idx, src)
                node_id_map[out+"_po"] = idx
                idx += 1
                buff_edges += 1
            else:
                # ignore other lines
                pass

    meta = {
        'and_count': and_count,
        'not_edges': not_edges,
        'buff_edges': buff_edges
    }
    return AIG, meta


def compute_stats(G, meta):
    node_types = {'PI':0,'PO':0,'Internal':0}
    for _, d in G.nodes(data=True):
        nt = d.get('node_type','Internal')
        node_types[nt] = node_types.get(nt,0) + 1
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    try:
        longest = nxdag.dag_longest_path_length(G)
    except Exception:
        longest = -1
    out_degrees = [d for _,d in G.out_degree()]
    avg_fanout = sum(out_degrees)/len(out_degrees) if out_degrees else 0
    max_fanout = max(out_degrees) if out_degrees else 0
    stats = {
        'PI': node_types.get('PI',0),
        'PO': node_types.get('PO',0),
        'Internal': node_types.get('Internal',0),
        'AND': meta.get('and_count',0),
        'NOT_edges': meta.get('not_edges',0),
        'BUFF_edges': meta.get('buff_edges',0),
        'nodes': node_count,
        'edges': edge_count,
        'longest_path': longest,
        'avg_fanout': avg_fanout,
        'max_fanout': max_fanout
    }
    return stats


def walk_and_collect(root, out_csv):
    root = Path(root)
    rows = []
    for p in root.rglob('*.aig'):
        try:
            G, meta = parse_aig_bench(str(p))
            stats = compute_stats(G, meta)
            stat_row = {
                'path': str(p),
                'size_bytes': p.stat().st_size,
            }
            stat_row.update(stats)
            rows.append(stat_row)
        except Exception as e:
            print(f"Error parsing {p}: {e}", file=sys.stderr)

    # write CSV
    if rows:
        keys = ['path','size_bytes','nodes','edges','PI','PO','Internal','AND','NOT_edges','BUFF_edges','longest_path','avg_fanout','max_fanout']
        with open(out_csv, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k,'') for k in keys})


def parse_args():
    p = argparse.ArgumentParser(description='Collect AIG bench statistics')
    p.add_argument('--root', required=True, help='Root folder to search for .aig files')
    p.add_argument('--out', required=True, help='Output CSV file')
    return p.parse_args()


def main():
    args = parse_args()
    walk_and_collect(args.root, args.out)


if __name__ == '__main__':
    main()

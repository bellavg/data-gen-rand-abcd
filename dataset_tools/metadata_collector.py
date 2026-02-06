#!/usr/bin/env python3
"""
Simple metadata collector for synthesis results.
Captures current ABC circuit statistics and appends directly to CSV.
"""

import re
import csv
import os
import subprocess

# Canonical CSV header for metadata (logical statistics only)
CSV_HEADER = "file_path,design,recipe_id,step_id,tier_id,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout"

def get_stats_from_file(stats_file):
    """Read ABC statistics from a temporary stats file."""
    try:
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Clean up temp file
            os.remove(stats_file)
            return content
    except:
        pass
    return ""

def parse_abc_stats(stats_output):
    """Parse ABC print_stats output and return statistics dictionary."""
    stats = {
        'nodes': 0, 'edges': 0, 'num_PI': 0, 'num_PO': 0, 'depth': 0,
        'avg_fanout': 0.0, 'max_fanout': 0
    }
    
    if not stats_output:
        return stats
    
    # Parse I/O
    io_match = re.search(r'i/o\s*=\s*(\d+)/\s*(\d+)', stats_output)
    if io_match:
        stats['num_PI'] = int(io_match.group(1))
        stats['num_PO'] = int(io_match.group(2))
    
    # Parse AND gates (nodes)
    and_match = re.search(r'and\s*=\s*(\d+)', stats_output)
    if and_match:
        stats['nodes'] = int(and_match.group(1))
    
    # Parse levels (depth)
    lev_match = re.search(r'lev\s*=\s*(\d+)', stats_output)
    if lev_match:
        stats['depth'] = int(lev_match.group(1))
    
    # Note: Area and delay are not captured as they require technology mapping
    # and are only meaningful for final mapped results, not intermediate logic steps
    
    # Calculate derived statistics for AIG
    if stats['nodes'] > 0:
        stats['edges'] = stats['nodes'] * 2 + stats['num_PI']
        stats['avg_fanout'] = 2.0
        stats['max_fanout'] = 2
    
    return stats

def append_to_csv(csv_file, file_path, design, recipe_id, step_id, stats):
    """Append a single row to the CSV file."""
    row = [
        file_path, design, recipe_id, step_id, '',  # tier_id empty for base AIGs
        stats['nodes'], stats['edges'], stats['num_PI'], stats['num_PO'], stats['depth'],
        stats['avg_fanout'], stats['max_fanout']
    ]
    
    # Append to CSV
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def collect_current_metadata(design, recipe_id, step_id, graph_dump_folder, stats_file):
    """
    Collect current circuit metadata and append to CSV.
    Called from ABC synthesis script at each step.
    """
    csv_file = os.path.join(graph_dump_folder, "metadata", f"{design}.csv")
    
    # Ensure CSV exists with header
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            f.write(f"{CSV_HEADER}\n")
    
    # Get circuit statistics from temporary stats file
    stats_output = get_stats_from_file(stats_file)
    stats = parse_abc_stats(stats_output)
    
    # Create file path
    file_path = f"base_aigs/{design}/{design}_syn{recipe_id}_step{step_id}.aig"
    
    # Append to CSV
    append_to_csv(csv_file, file_path, design, recipe_id, step_id, stats)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 6:
        print("Usage: python metadata_collector.py <design> <recipe_id> <step_id> <graph_dump_folder> <stats_file>")
        sys.exit(1)
    
    design = sys.argv[1]
    recipe_id = int(sys.argv[2])
    step_id = int(sys.argv[3])
    graph_dump_folder = sys.argv[4]
    stats_file = sys.argv[5]
    
    collect_current_metadata(design, recipe_id, step_id, graph_dump_folder, stats_file)
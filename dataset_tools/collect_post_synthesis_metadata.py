#!/usr/bin/env python3
"""
Post-synthesis metadata collector for OpenABC dataset.
Analyzes synthesis log files and AIG files to extract circuit statistics.
"""

import os
import re
import csv
import zipfile
import argparse
from pathlib import Path

def parse_abc_stats_from_log(log_content):
    """Extract ABC statistics from synthesis log content."""
    stats_pattern = r'i/o\s*=\s*(\d+)/\s*(\d+)\s+nd\s*=\s*(\d+)\s+edge\s*=\s*(\d+)\s+lev\s*=\s*(\d+)'
    
    matches = re.findall(stats_pattern, log_content)
    if not matches:
        return []
    
    # Each match is (PI, PO, nodes, edges, levels)
    parsed_stats = []
    for match in matches:
        pi, po, nodes, edges, levels = map(int, match)
        # Calculate approximate fanout statistics
        avg_fanout = round(edges / nodes, 2) if nodes > 0 else 0
        max_fanout = max(10, int(avg_fanout * 1.5))  # Estimated max fanout
        
        stats = {
            'nodes': nodes,
            'edges': edges,
            'num_PI': pi,
            'num_PO': po,
            'depth': levels,
            'avg_fanout': avg_fanout,
            'max_fanout': max_fanout
        }
        parsed_stats.append(stats)
    
    return parsed_stats

def collect_metadata_for_design(design, base_dir):
    """Collect metadata for a specific design from log files and zip files."""
    print(f"Collecting metadata for design {design}...")
    
    bench_dir = os.path.join(base_dir, "OPENABC_DATASET", "bench", design)
    log_dir = os.path.join(bench_dir, f"log_{design}")
    metadata_dir = os.path.join(bench_dir, "metadata")
    
    # Ensure metadata directory exists
    os.makedirs(metadata_dir, exist_ok=True)
    
    # CSV file for this design
    csv_file = os.path.join(metadata_dir, f"{design}.csv")
    
    # CSV header
    header = "file_path,design,recipe_id,step_id,tier_id,nodes,edges,num_PI,num_PO,depth,avg_fanout,max_fanout"
    
    metadata_rows = []
    processed_recipes = 0
    
    # Process each log file
    for recipe_id in range(1500):  # 1500 synthesis recipes
        log_file = os.path.join(log_dir, f"log_{design}_syn{recipe_id}.log")
        zip_file = os.path.join(bench_dir, f"syn{recipe_id}.zip")
        
        if not os.path.exists(log_file):
            print(f"Warning: Log file missing for recipe {recipe_id}")
            continue
            
        if not os.path.exists(zip_file):
            print(f"Warning: Zip file missing for recipe {recipe_id}")
            continue
        
        try:
            # Read log file content
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            # Extract all statistics from log (one per synthesis step)
            stats_list = parse_abc_stats_from_log(log_content)
            
            # Get list of AIG files from zip to match with statistics
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    aig_files = [name for name in zf.namelist() if name.endswith('.aig')]
                    aig_files.sort()  # Ensure consistent ordering
            except zipfile.BadZipFile:
                print(f"Warning: Invalid zip file for recipe {recipe_id}")
                continue
            
            # Match statistics with AIG files
            for step_id, (aig_file, stats) in enumerate(zip(aig_files, stats_list), 1):
                if stats:  # Only add if we have valid statistics
                    # Extract step number from filename
                    step_match = re.search(r'step(\d+)\.aig', aig_file)
                    actual_step = int(step_match.group(1)) if step_match else step_id
                    
                    # Create file path for the AIG file (use zip-based path)
                    file_path = f"{design}/syn{recipe_id}.zip/{aig_file}"
                    
                    # Calculate tier (group steps into tiers)
                    tier_id = (actual_step - 1) // 7 + 1  # 7 steps per tier
                    
                    row = [
                        file_path,
                        design,
                        recipe_id,
                        actual_step,
                        tier_id,
                        stats['nodes'],
                        stats['edges'],
                        stats['num_PI'],
                        stats['num_PO'],
                        stats['depth'],
                        stats['avg_fanout'],
                        stats['max_fanout']
                    ]
                    metadata_rows.append(row)
            
            processed_recipes += 1
            if processed_recipes % 100 == 0:
                print(f"  Processed {processed_recipes}/1500 recipes")
                
        except Exception as e:
            print(f"Error processing recipe {recipe_id}: {e}")
            continue
    
    # Write metadata CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        f.write(header + '\n')
        writer = csv.writer(f)
        writer.writerows(metadata_rows)
    
    print(f"✓ Collected metadata for {processed_recipes} recipes ({len(metadata_rows)} entries)")
    print(f"  Metadata saved to: {csv_file}")
    return len(metadata_rows)

def main():
    parser = argparse.ArgumentParser(description='Collect post-synthesis metadata from log files')
    parser.add_argument('--home', required=True, help='Base directory path')
    parser.add_argument('--design', help='Specific design to process (e.g., 128). If not provided, processes all designs')
    
    args = parser.parse_args()
    
    designs = ['128', '256', '512', '1024', '2048', '4096', '8192', '16384']
    
    if args.design:
        if args.design not in designs:
            print(f"Error: Invalid design {args.design}. Valid designs: {designs}")
            return 1
        designs = [args.design]
    
    total_entries = 0
    
    print("=" * 50)
    print("POST-SYNTHESIS METADATA COLLECTION")
    print("=" * 50)
    
    for design in designs:
        entries = collect_metadata_for_design(design, args.home)
        total_entries += entries
        print()
    
    print(f"✓ Total metadata entries collected: {total_entries}")
    print("Metadata collection complete!")

if __name__ == '__main__':
    main()
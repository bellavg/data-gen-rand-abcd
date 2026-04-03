import os
import re
import csv
import zipfile
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Regex updated to capture 'and=' or 'nd=' and skip missing edges
STATS_REGEX = re.compile(r"i/o\s*=\s*(\d+)/\s*(\d+).*?(?:and|nd)\s*=\s*(\d+).*?lev\s*=\s*(\d+)")
# Updated to handle 'synX' safely!
FILENAME_REGEX = re.compile(r"syn([0-9X]+)_step(\d+)")
ALGORITHMS = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]

def parse_single_log(args):
    zip_path, log_name, design_name, algorithm, tier_id = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            content = zf.read(log_name).decode('utf-8')
    except: return None
        
    stats = STATS_REGEX.findall(content)
    if len(stats) < 2: return None

    # Safely handle the recipe and step IDs, including 'synX'
    match = FILENAME_REGEX.search(log_name)
    if match:
        recipe_str = match.group(1)
        recipe_id = 0 if recipe_str == 'X' else int(recipe_str)
        step_id = int(match.group(2))
    else:
        recipe_id = 0
        step_id = 0

    t0_nodes, t0_depth = int(stats[0][2]), int(stats[0][3])
    t1_pi, t1_po, t1_nodes, t1_depth = map(int, stats[-1])
    
    opt_nodes = (t0_nodes - t1_nodes) / t0_nodes if t0_nodes > 0 else 0.0
    opt_depth = (t0_depth - t1_depth) / t0_depth if t0_depth > 0 else 0.0

    # --- DYNAMIC DESIGN NAME FOR TIER 2 ---
    # Extracts the Tier-1 algorithm that was used first
    csv_design_name = design_name
    if tier_id == 2:
        # log_name format: {design_name}_{tier1_algo}_{tier2_algo}_tier2_{suffix}.log
        if log_name.startswith(f"{design_name}_"):
            remainder = log_name[len(design_name)+1:]
            for a in ALGORITHMS:
                if remainder.startswith(f"{a}_"):
                    csv_design_name = f"{design_name}_{a}"
                    break

    return {
        "file_path": f"base_aigs/{design_name}/tier{tier_id}/{algorithm}/{log_name.replace('.log', '.aig')}",
        "design": csv_design_name, 
        "recipe_id": recipe_id, "step_id": step_id,
        "tier_id": tier_id, "algorithm": algorithm, "nodes": t1_nodes, "edges": 0,
        "num_PI": t1_pi, "num_PO": t1_po, "depth": t1_depth,
        "optimizability": round(opt_nodes, 4), "depth_optimizability": round(opt_depth, 4)
    }

def process_logs(design_dir, design_name, num_workers):
    logs_base = Path(design_dir) / "design_metadata" / "raw_logs" / "optimization_logs"
    csv_path = Path(design_dir) / "design_metadata" / f"{design_name}.csv"
    
    # --- INCREMENTAL CHECK ---
    existing_paths = set()
    existing_rows = []
    if csv_path.exists():
        print(f">>> Found existing CSV. Loading processed paths...")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_paths.add(row['file_path'])
                existing_rows.append(row)
        print(f"    Loaded {len(existing_paths)} existing entries.")

    tasks = []

    for tier in [1, 2]:
        tier_dir = logs_base / f"tier{tier}"
        if not tier_dir.exists(): continue
        for algo in ALGORITHMS:
            algo_dir = tier_dir / algo
            if not algo_dir.exists(): continue
            for zip_path in algo_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for log_name in zf.namelist():
                        if not log_name.endswith(".log"): continue
                        # Predict the file_path this log would create (keeping base design_name for accurate folder paths)
                        pred_path = f"base_aigs/{design_name}/tier{tier}/{algo}/{log_name.replace('.log', '.aig')}"
                        if pred_path not in existing_paths:
                            tasks.append((zip_path, log_name, design_name, algo, tier))

    if not tasks:
        print("✓ All logs already present in CSV. Nothing to do!")
        return

    print(f">>> Found {len(tasks)} NEW logs to parse. Starting {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        new_results = list(executor.map(parse_single_log, tasks))

    final_rows = existing_rows + [r for r in new_results if r is not None]
    
    # Save the combined data
    headers = ["file_path", "design", "recipe_id", "step_id", "tier_id", "algorithm", 
               "nodes", "edges", "num_PI", "num_PO", "depth", "optimizability", "depth_optimizability"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"✓ Updated Dataset: {csv_path} now has {len(final_rows)} total rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--design-name", required=True)
    parser.add_argument("--cpus", type=int, default=24)
    args = parser.parse_args()
    process_logs(args.design_dir, args.design_name, args.cpus)
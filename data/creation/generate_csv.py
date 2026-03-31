import os
import re
import csv
import zipfile
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Regex to match ABC's print_stats output
STATS_REGEX = re.compile(r"i/o\s*=\s*(\d+)/\s*(\d+).*?nd\s*=\s*(\d+).*?edge\s*=\s*(\d+).*?lev\s*=\s*(\d+)")
FILENAME_REGEX = re.compile(r"syn(\d+)_step(\d+)")

def parse_single_log(args):
    """Function to be run by each of the 24 cores."""
    zip_path, log_name, design_name, algorithm = args
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        content = zf.read(log_name).decode('utf-8')
        
    stats = STATS_REGEX.findall(content)
    if len(stats) < 2:
        return None

    match = FILENAME_REGEX.search(log_name)
    recipe_id = int(match.group(1)) if match else 0
    step_id = int(match.group(2)) if match else 0

    t0_nodes = int(stats[0][2])
    t1_pi, t1_po, t1_nodes, t1_edges, t1_depth = map(int, stats[-1])
    opt = (t0_nodes - t1_nodes) / t0_nodes if t0_nodes > 0 else 0.0

    return {
        "file_path": f"base_aigs/{design_name}/tier1/{algorithm}/{log_name.replace('.log', '.aig')}",
        "design": design_name,
        "recipe_id": recipe_id,
        "step_id": step_id,
        "tier_id": 1,
        "algorithm": algorithm,
        "nodes": t1_nodes,
        "edges": t1_edges,
        "num_PI": t1_pi,
        "num_PO": t1_po,
        "depth": t1_depth,
        "optimizability": round(opt, 4)
    }

def process_logs(design_dir, design_name, num_workers):
    logs_root = Path(design_dir) / "design_metadata" / "raw_logs" / "optimization_logs" / "tier1"
    csv_path = Path(design_dir) / "design_metadata" / f"{design_name}.csv"
    
    algorithms = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]
    tasks = []

    # 1. Gather all tasks first
    for algo in algorithms:
        zip_path = logs_root / algo / f"optimize_{algo}_{design_name}.zip"
        if not zip_path.exists(): continue
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for log_name in zf.namelist():
                if log_name.endswith(".log"):
                    tasks.append((zip_path, log_name, design_name, algo))

    # 2. Use all 24 cores to parse the logs in parallel
    print(f">>> Processing {len(tasks)} logs using {num_workers} cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(parse_single_log, tasks))

    # 3. Filter out Nones and write to CSV
    final_rows = [r for r in results if r is not None]
    
    headers = ["file_path", "design", "recipe_id", "step_id", "tier_id", "algorithm", 
               "nodes", "edges", "num_PI", "num_PO", "depth", "optimizability"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"✓ CSV Created: {csv_path} with {len(final_rows)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--design-name", required=True)
    parser.add_argument("--cpus", type=int, default=24)
    args = parser.parse_args()
    process_logs(args.design_dir, args.design_name, args.cpus)
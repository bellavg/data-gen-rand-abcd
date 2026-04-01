import os
import re
import csv
import zipfile
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Regex updated to capture 'and=' or 'nd=' and skip missing edges
STATS_REGEX = re.compile(r"i/o\s*=\s*(\d+)/\s*(\d+).*?(?:and|nd)\s*=\s*(\d+).*?lev\s*=\s*(\d+)")
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

    # Grab initial stats for BOTH nodes and depth (levels)
    t0_nodes = int(stats[0][2])
    t0_depth = int(stats[0][3])
    
    # Unpack the final metrics and default edges to 0
    t1_pi, t1_po, t1_nodes, t1_depth = map(int, stats[-1])
    t1_edges = 0 
    
    # Calculate optimizability for both metrics
    opt_nodes = (t0_nodes - t1_nodes) / t0_nodes if t0_nodes > 0 else 0.0
    opt_depth = (t0_depth - t1_depth) / t0_depth if t0_depth > 0 else 0.0

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
        "optimizability": round(opt_nodes, 4),
        "depth_optimizability": round(opt_depth, 4)
    }

def analyze_dataset(rows, design_name):
    """Prints a terminal health report of the dataset distribution."""
    if not rows:
        return

    total = len(rows)
    improved_nodes = sum(1 for r in rows if r["optimizability"] > 0)
    improved_depth = sum(1 for r in rows if r["depth_optimizability"] > 0)
    
    max_opt_nodes = max((r["optimizability"] for r in rows), default=0)
    max_opt_depth = max((r["depth_optimizability"] for r in rows), default=0)

    print(f"\n{'='*60}")
    print(f" 📊 DATASET HEALTH REPORT: {design_name}")
    print(f"{'='*60}")
    print(f" Total AIGs Analyzed  : {total}")
    print(f" Node Reductions      : {improved_nodes} AIGs ({improved_nodes/total:.1%}) | Max Shrink: {max_opt_nodes:.1%}")
    print(f" Depth Reductions     : {improved_depth} AIGs ({improved_depth/total:.1%}) | Max Shrink: {max_opt_depth:.1%}")
    print(f" Stuck at Local Min   : {total - improved_nodes} AIGs (0.0% node reduction)")
    
    print(f"\n --- Step-by-Step Breakdown (When does it hit the wall?) ---")
    
    # Group data by step_id
    steps_data = {}
    for r in rows:
        sid = r["step_id"]
        if sid not in steps_data:
            steps_data[sid] = {"node_opts": [], "depth_opts": []}
        steps_data[sid]["node_opts"].append(r["optimizability"])
        steps_data[sid]["depth_opts"].append(r["depth_optimizability"])
        
    # Print stats for each step
    for sid in sorted(steps_data.keys()):
        n_opts = steps_data[sid]["node_opts"]
        d_opts = steps_data[sid]["depth_opts"]
        
        avg_n = sum(n_opts) / len(n_opts) if n_opts else 0
        avg_d = sum(d_opts) / len(d_opts) if d_opts else 0
        non_zero_n = sum(1 for x in n_opts if x > 0)
        pct_non_zero = (non_zero_n / len(n_opts)) * 100
        
        print(f" Step {sid:2d} | Avg Node Opt: {avg_n:.4f} | Avg Depth Opt: {avg_d:.4f} | Non-Zero Runs: {pct_non_zero:>5.1f}%")
        
    print(f"{'='*60}\n")

    
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

    # 2. Use all cores to parse the logs in parallel
    print(f">>> Processing {len(tasks)} logs using {num_workers} cores...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(parse_single_log, tasks))

    # 3. Filter out Nones
    final_rows = [r for r in results if r is not None]
    
    # 4. Print the analysis report!
    analyze_dataset(final_rows, design_name)
    
    # 5. Write to CSV (added depth_optimizability to headers)
    headers = ["file_path", "design", "recipe_id", "step_id", "tier_id", "algorithm", 
               "nodes", "edges", "num_PI", "num_PO", "depth", "optimizability", "depth_optimizability"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"✓ COMPLETED: CSV saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--design-name", required=True)
    parser.add_argument("--cpus", type=int, default=24)
    args = parser.parse_args()
    process_logs(args.design_dir, args.design_name, args.cpus)
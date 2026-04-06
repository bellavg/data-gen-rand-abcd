#!/bin/bash
#SBATCH --job-name=8_full_csv
#SBATCH --time=00:59:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --output=logs/8_full_csv_%j.out

set -euo pipefail

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
EPS="${EPS:-1e-12}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-24}}"
DESIGNS_DIR="$BASE_DIR/data/designs"
MASTER_DIR="$DESIGNS_DIR/design_metadata"
MASTER_CSV="$MASTER_DIR/full_master.csv"
PER_ALGO_DIR="$MASTER_DIR"
REPORT_DIR="$BASE_DIR/logs/8_checks"
REPORT_PATH="$REPORT_DIR/full_master_report_${SLURM_JOB_ID:-manual}.txt"

DESIGNS=(
    "128" "256" "512" "1024" "2048" "4096" "8192" "16384"
    "ac97_ctrl" "aes" "aes_secworks" "aes_xcrypt" "apex1" "bc0" "bp_be"
    "c1355" "c5315" "c6288" "c7552" "dalu" "des3_area" "dft" "div"
    "dynamic_node" "ethernet" "fir" "fpu" "hyp" "i10" "i2c" "idft"
    "iir" "jpeg" "k2" "log2" "mainpla" "max" "mem_ctrl" "multiplier"
    "pci" "picosoc" "sasc" "sha256" "simple_spi" "sin" "spi" "sqrt"
    "square" "ss_pcm" "tinyRocket" "tv80" "usb_phy" "vga_lcd" "wb_conmax"
    "wb_dma"
)

SCRATCH_PARENT="${TMPDIR:-/tmp}"
JOB_SCRATCH="$(mktemp -d "$SCRATCH_PARENT/final_job8_${SLURM_JOB_ID:-manual}_XXXXXX")"
SCRATCH_CSV_ROOT="$JOB_SCRATCH/per_design_csv"
SCRATCH_ZIP_ROOT="$JOB_SCRATCH/tier1_zip"
SCRATCH_UNZIP_ROOT="$JOB_SCRATCH/unzipped_tier1"
SCRATCH_PER_ALGO_DIR="$JOB_SCRATCH/per_algorithm_csv"
BOOTSTRAP_MASTER="$JOB_SCRATCH/full_master.bootstrap.csv"
MASTER_TMP="$JOB_SCRATCH/full_master.updated.csv"
REPORT_TMP="$JOB_SCRATCH/full_master_report.tmp.txt"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

mkdir -p "$MASTER_DIR" "$PER_ALGO_DIR" "$REPORT_DIR" "$SCRATCH_CSV_ROOT" "$SCRATCH_ZIP_ROOT" "$SCRATCH_UNZIP_ROOT" "$SCRATCH_PER_ALGO_DIR"

echo "=================================================="
echo " JOB 8: Build full master CSV + add Tier-0 + full stats"
echo " Time: $(date)"
echo " Base dir: $BASE_DIR"
echo " Master output: $MASTER_CSV"
echo " Per-algorithm CSV dir: $PER_ALGO_DIR"
echo " Report output: $REPORT_PATH"
echo " Scratch dir: $JOB_SCRATCH"
echo " EPS: $EPS"
echo " Workers: $WORKERS"
echo "=================================================="

echo ">>> Staging per-design CSV files to scratch..."
first_file=1
included_count=0
missing_count=0

for design in "${DESIGNS[@]}"; do
    src_csv="$DESIGNS_DIR/$design/design_metadata/${design}.csv"
    dst_csv_dir="$SCRATCH_CSV_ROOT/$design"
    dst_csv="$dst_csv_dir/${design}.csv"

    if [[ ! -f "$src_csv" ]]; then
        echo "WARNING: Missing per-design CSV: $src_csv"
        missing_count=$((missing_count + 1))
        continue
    fi

    mkdir -p "$dst_csv_dir"
    cp -f "$src_csv" "$dst_csv"

    if [[ $first_file -eq 1 ]]; then
        cat "$dst_csv" > "$BOOTSTRAP_MASTER"
        first_file=0
    else
        tail -n +2 "$dst_csv" >> "$BOOTSTRAP_MASTER"
    fi

    included_count=$((included_count + 1))
    echo "Included CSV: $design"
done

if [[ $first_file -eq 1 ]]; then
    echo "ERROR: No per-design CSV files were found."
    exit 1
fi

if [[ $missing_count -gt 0 ]]; then
    echo "WARNING: $missing_count design CSV file(s) were missing; master CSV will still be produced from available designs."
fi

echo ">>> Staging tier1 zip logs to scratch..."
staged_zip_count=0
staged_design_sources=0
for design in "${DESIGNS[@]}"; do
    selected_algo=""
    selected_zip_count=0

    # Tier-0 stats are identical across tier1 algorithms for the same input AIG.
    # Prefer one source algorithm per design to avoid 4x redundant parsing.
    for algo in Orchestrate Syn4 C2RS Deepsyn; do
        src_dir="$DESIGNS_DIR/$design/design_metadata/raw_logs/optimization_logs/tier1/$algo"
        if [[ ! -d "$src_dir" ]]; then
            continue
        fi

        shopt -s nullglob
        zips=("$src_dir"/*.zip)
        shopt -u nullglob

        if [[ ${#zips[@]} -eq 0 ]]; then
            continue
        fi

        dst_dir="$SCRATCH_ZIP_ROOT/$design/$algo"
        mkdir -p "$dst_dir"
        for z in "${zips[@]}"; do
            cp -f "$z" "$dst_dir/"
            staged_zip_count=$((staged_zip_count + 1))
            selected_zip_count=$((selected_zip_count + 1))
        done
        selected_algo="$algo"
        break
    done

    if [[ -n "$selected_algo" ]]; then
        staged_design_sources=$((staged_design_sources + 1))
        echo "Tier0 source for $design: $selected_algo ($selected_zip_count zip[s])"
    else
        echo "WARNING: No tier1 zip source found for $design; no new tier0 rows will be derived for this design."
    fi
done

echo ">>> Staged $staged_zip_count tier1 zip archive(s) to scratch."
echo ">>> Tier0 source selected for $staged_design_sources design(s)."

python3 - "$BOOTSTRAP_MASTER" "$MASTER_TMP" "$SCRATCH_ZIP_ROOT" "$EPS" "$WORKERS" "$SCRATCH_UNZIP_ROOT" "$SCRATCH_PER_ALGO_DIR" "${#DESIGNS[@]}" "${DESIGNS[@]}" <<'PY' | tee "$REPORT_TMP"
import csv
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

master_in = Path(sys.argv[1])
master_out = Path(sys.argv[2])
zip_root = Path(sys.argv[3])
eps = float(sys.argv[4])
workers = max(1, int(sys.argv[5]))
unzip_root = Path(sys.argv[6])
algo_out_dir = Path(sys.argv[7])
expected_designs = int(sys.argv[8])
designs = sys.argv[9:]

algorithms = ["Orchestrate", "Deepsyn", "Syn4", "C2RS"]
tier0_source_order = ["Orchestrate", "Syn4", "C2RS", "Deepsyn"]
stats_regex = re.compile(r"i/o\s*=\s*(\d+)/\s*(\d+).*?(?:and|nd)\s*=\s*(\d+).*?lev\s*=\s*(\d+)")
filename_regex = re.compile(r"syn([0-9X]+)_step(\d+)")

required_columns = [
    "file_path",
    "design",
    "recipe_id",
    "step_id",
    "tier_id",
    "algorithm",
    "nodes",
    "edges",
    "num_PI",
    "num_PO",
    "depth",
    "optimizability",
    "depth_optimizability",
]
valid_algorithms = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}
null_algorithms = {"", "NA", "N/A", "None", "none", "null", "NULL"}
max_row_issues = 20

added_tier0 = 0
tier0_parse_failures = 0
unzipped_zip_count = 0
matched_tier1_log_count = 0
tier0_source_design_count = 0


def build_tier0_row_from_log(design, algo, log_name, content):
    prefix = f"{design}_{algo}_tier1_"
    if not log_name.startswith(prefix) or not log_name.endswith(".log"):
        return None

    stats = stats_regex.findall(content)
    if not stats:
        return None

    t0_pi, t0_po, t0_nodes, t0_depth = map(int, stats[0])
    suffix = log_name[len(prefix):-4]

    match = filename_regex.search(log_name)
    if match:
        recipe_str = match.group(1)
        recipe_id = 0 if recipe_str == "X" else int(recipe_str)
        step_id = int(match.group(2))
    else:
        recipe_id = 0
        step_id = 0

    return {
        "file_path": f"base_aigs/{design}/tier0/{suffix}.aig",
        "design": design,
        "recipe_id": recipe_id,
        "step_id": step_id,
        "tier_id": 0,
        "algorithm": "",
        "nodes": t0_nodes,
        "edges": 0,
        "num_PI": t0_pi,
        "num_PO": t0_po,
        "depth": t0_depth,
        "optimizability": 0.0,
        "depth_optimizability": 0.0,
    }


def process_zip_task(task):
    zip_path_str, extract_dir_str, design, algo = task
    zip_path = Path(zip_path_str)
    extract_dir = Path(extract_dir_str)

    local_unzipped_zip_count = 0
    local_matched_tier1_log_count = 0
    local_parse_failures = 0
    local_rows = []

    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        local_unzipped_zip_count = 1
    except zipfile.BadZipFile:
        return {
            "unzipped_zip_count": 0,
            "matched_tier1_log_count": 0,
            "tier0_parse_failures": 1,
            "tier0_rows": [],
        }
    except Exception:
        return {
            "unzipped_zip_count": 0,
            "matched_tier1_log_count": 0,
            "tier0_parse_failures": 1,
            "tier0_rows": [],
        }

    for log_path in extract_dir.rglob("*.log"):
        log_name = log_path.name
        prefix = f"{design}_{algo}_tier1_"
        if not log_name.startswith(prefix):
            continue

        local_matched_tier1_log_count += 1

        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            row = build_tier0_row_from_log(design, algo, log_name, content)
            if row is None:
                local_parse_failures += 1
                continue
            local_rows.append(row)
        except Exception:
            local_parse_failures += 1

    return {
        "unzipped_zip_count": local_unzipped_zip_count,
        "matched_tier1_log_count": local_matched_tier1_log_count,
        "tier0_parse_failures": local_parse_failures,
        "tier0_rows": local_rows,
    }


with open(master_in, "r", newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        print("ERROR: Bootstrap CSV has no header row.")
        sys.exit(1)

    missing = sorted(set(required_columns).difference(reader.fieldnames or []))
    if missing:
        print("ERROR: Missing required columns: " + ", ".join(missing))
        sys.exit(1)

    fieldnames = reader.fieldnames
    rows = list(reader)

existing_paths = {
    (row.get("file_path") or "").strip()
    for row in rows
    if (row.get("file_path") or "").strip()
}

zip_tasks = []
for design in designs:
    chosen_algo = None
    for algo in tier0_source_order:
        algo_dir = zip_root / design / algo
        if not algo_dir.exists():
            continue
        if any(algo_dir.glob("*.zip")):
            chosen_algo = algo
            break

    if chosen_algo is None:
        continue

    tier0_source_design_count += 1
    algo_dir = zip_root / design / chosen_algo
    for idx, zip_path in enumerate(sorted(algo_dir.glob("*.zip"))):
        extract_dir = unzip_root / design / chosen_algo / f"{zip_path.stem}_{idx}"
        zip_tasks.append((str(zip_path), str(extract_dir), design, chosen_algo))

if zip_tasks:
    active_workers = min(workers, len(zip_tasks))
    with ProcessPoolExecutor(max_workers=active_workers) as executor:
        futures = [executor.submit(process_zip_task, task) for task in zip_tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                tier0_parse_failures += 1
                continue

            unzipped_zip_count += result["unzipped_zip_count"]
            matched_tier1_log_count += result["matched_tier1_log_count"]
            tier0_parse_failures += result["tier0_parse_failures"]

            for row in result["tier0_rows"]:
                tier0_path = row["file_path"]
                if tier0_path in existing_paths:
                    continue
                rows.append(row)
                existing_paths.add(tier0_path)
                added_tier0 += 1

with open(master_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

algo_out_dir.mkdir(parents=True, exist_ok=True)
tier0_rows_shared = []
rows_by_algo_specific = {algo: [] for algo in algorithms}
for row in rows:
    tier_raw = (row.get("tier_id") or "").strip()
    if tier_raw == "":
        tier_value = 0
    else:
        try:
            tier_value = int(float(tier_raw))
        except ValueError:
            tier_value = -1

    if tier_value == 0:
        tier0_rows_shared.append(row)

    algo_value = (row.get("algorithm") or "").strip()
    if algo_value in rows_by_algo_specific:
        rows_by_algo_specific[algo_value].append(row)

rows_by_algo = {
    algo: list(tier0_rows_shared) + rows_by_algo_specific[algo]
    for algo in algorithms
}

algo_csv_paths = {}
for algo in algorithms:
    algo_path = algo_out_dir / f"full_master_{algo}.csv"
    with open(algo_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_algo[algo])
    algo_csv_paths[algo] = algo_path

non_empty_counts = {c: 0 for c in fieldnames}
tier_counts = Counter()
algo_counts = Counter()
tier_algo_counts = {1: Counter(), 2: Counter()}
per_design_counts = defaultdict(lambda: Counter())
row_issues = []

total_rows = 0
valid_rows = 0
match_rows = 0
invalid_rows = 0
node_opt_zero_total = 0
max_nodes = None
max_nodes_row = None
max_depth = None
max_depth_row = None
algo_range_count = Counter()
algo_opt_min = {}
algo_opt_max = {}
algo_depth_opt_min = {}
algo_depth_opt_max = {}

tier_numeric_counts = Counter()
tier_node_sum = Counter()
tier_depth_sum = Counter()

tier0_by_key = {}
tier1_by_key_algo = {}
tier2_records = []


def parse_tier2_source_algo(file_path, root_design):
    """Extract tier1 source algorithm from tier2 file name.

    Expected name pattern:
    {design}_{tier1_algo}_{tier2_algo}_tier2_syn{recipe}_step{step}.aig
    """
    name = Path(file_path).name
    prefix = f"{root_design}_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix):]
    for algo_name in algorithms:
        if remainder.startswith(f"{algo_name}_"):
            return algo_name
    return None

for line_no, row in enumerate(rows, start=2):
    total_rows += 1

    for col in fieldnames:
        value = (row.get(col) or "").strip()
        if value != "":
            non_empty_counts[col] += 1

    file_path = (row.get("file_path") or "").strip()
    root_design = ""
    if file_path.startswith("base_aigs/"):
        parts = file_path.split("/")
        if len(parts) >= 2 and parts[1]:
            root_design = parts[1]
    if not root_design and len(row_issues) < max_row_issues:
        row_issues.append(
            f"Row {line_no}: cannot derive root design from file_path='{file_path}'"
        )

    tier_raw = (row.get("tier_id") or "").strip()
    if tier_raw == "":
        tier_id = 0
    else:
        try:
            tier_id = int(float(tier_raw))
        except ValueError:
            tier_id = -1
            if len(row_issues) < max_row_issues:
                row_issues.append(f"Row {line_no}: invalid tier_id='{tier_raw}'")
    tier_counts[tier_id] += 1

    algorithm = (row.get("algorithm") or "").strip()
    if algorithm not in null_algorithms:
        algo_counts[algorithm] += 1

    if tier_id == 0:
        if algorithm not in null_algorithms and len(row_issues) < max_row_issues:
            row_issues.append(
                f"Row {line_no}: tier0 row should not have algorithm (found '{algorithm}')"
            )
    elif tier_id in (1, 2):
        if algorithm in null_algorithms and len(row_issues) < max_row_issues:
            row_issues.append(f"Row {line_no}: tier{tier_id} row is missing algorithm")
        elif algorithm not in valid_algorithms and len(row_issues) < max_row_issues:
            row_issues.append(
                f"Row {line_no}: tier{tier_id} has unknown algorithm '{algorithm}'"
            )
        else:
            tier_algo_counts[tier_id][algorithm] += 1
    elif len(row_issues) < max_row_issues:
        row_issues.append(f"Row {line_no}: unsupported tier_id={tier_id}")

    if root_design:
        per_design_counts[root_design][tier_id] += 1

    try:
        node_opt = float((row.get("optimizability") or "").strip())
        depth_opt = float((row.get("depth_optimizability") or "").strip())
        nodes = float((row.get("nodes") or "").strip())
        depth = float((row.get("depth") or "").strip())
    except ValueError:
        invalid_rows += 1
        continue

    if not (
        math.isfinite(node_opt)
        and math.isfinite(depth_opt)
        and math.isfinite(nodes)
        and math.isfinite(depth)
    ):
        invalid_rows += 1
        continue

    valid_rows += 1
    tier_numeric_counts[tier_id] += 1
    tier_node_sum[tier_id] += nodes
    tier_depth_sum[tier_id] += depth

    if abs(node_opt) <= eps:
        node_opt_zero_total += 1
    if abs(node_opt) <= eps and depth_opt > 0.0:
        match_rows += 1

    recipe_raw = (row.get("recipe_id") or "").strip()
    step_raw = (row.get("step_id") or "").strip()
    try:
        recipe_id_num = int(float(recipe_raw))
        step_id_num = int(float(step_raw))
    except ValueError:
        recipe_id_num = None
        step_id_num = None

    if root_design and recipe_id_num is not None and step_id_num is not None:
        if tier_id == 0:
            key = (root_design, recipe_id_num, step_id_num)
            if key not in tier0_by_key:
                tier0_by_key[key] = (nodes, depth, line_no, file_path)
        elif tier_id == 1 and algorithm not in null_algorithms:
            key = (root_design, algorithm, recipe_id_num, step_id_num)
            tier1_by_key_algo[key] = (nodes, depth, line_no, file_path)
        elif tier_id == 2:
            source_algo = parse_tier2_source_algo(file_path, root_design)
            tier2_records.append(
                (
                    root_design,
                    source_algo,
                    recipe_id_num,
                    step_id_num,
                    nodes,
                    depth,
                    line_no,
                    file_path,
                    algorithm,
                )
            )

    if algorithm not in null_algorithms:
        algo_range_count[algorithm] += 1
        if algorithm not in algo_opt_min:
            algo_opt_min[algorithm] = node_opt
            algo_opt_max[algorithm] = node_opt
            algo_depth_opt_min[algorithm] = depth_opt
            algo_depth_opt_max[algorithm] = depth_opt
        else:
            if node_opt < algo_opt_min[algorithm]:
                algo_opt_min[algorithm] = node_opt
            if node_opt > algo_opt_max[algorithm]:
                algo_opt_max[algorithm] = node_opt
            if depth_opt < algo_depth_opt_min[algorithm]:
                algo_depth_opt_min[algorithm] = depth_opt
            if depth_opt > algo_depth_opt_max[algorithm]:
                algo_depth_opt_max[algorithm] = depth_opt

    if max_nodes is None or nodes > max_nodes:
        max_nodes = nodes
        max_nodes_row = {
            "line": line_no,
            "design": (row.get("design") or "").strip(),
            "tier_id": (row.get("tier_id") or "").strip(),
            "algorithm": (row.get("algorithm") or "").strip(),
            "recipe_id": (row.get("recipe_id") or "").strip(),
            "step_id": (row.get("step_id") or "").strip(),
            "file_path": (row.get("file_path") or "").strip(),
        }

    if max_depth is None or depth > max_depth:
        max_depth = depth
        max_depth_row = {
            "line": line_no,
            "design": (row.get("design") or "").strip(),
            "tier_id": (row.get("tier_id") or "").strip(),
            "algorithm": (row.get("algorithm") or "").strip(),
            "recipe_id": (row.get("recipe_id") or "").strip(),
            "step_id": (row.get("step_id") or "").strip(),
            "file_path": (row.get("file_path") or "").strip(),
        }

rate_valid = (100.0 * match_rows / valid_rows) if valid_rows else 0.0
rate_total = (100.0 * match_rows / total_rows) if total_rows else 0.0

t01_checked = 0
t01_missing_parent = 0
t01_node_violations = 0
t01_depth_violations = 0
t01_samples = []

for (design_name, algo_name, recipe_id_num, step_id_num), (
    t1_nodes,
    t1_depth,
    t1_line,
    t1_path,
) in tier1_by_key_algo.items():
    parent = tier0_by_key.get((design_name, recipe_id_num, step_id_num))
    if parent is None:
        t01_missing_parent += 1
        continue

    t01_checked += 1
    t0_nodes, t0_depth, t0_line, t0_path = parent

    node_bad = t1_nodes > t0_nodes
    depth_bad = t1_depth > t0_depth
    if node_bad:
        t01_node_violations += 1
    if depth_bad:
        t01_depth_violations += 1

    if (node_bad or depth_bad) and len(t01_samples) < 10:
        t01_samples.append(
            {
                "design": design_name,
                "algo": algo_name,
                "recipe": recipe_id_num,
                "step": step_id_num,
                "t0_nodes": t0_nodes,
                "t1_nodes": t1_nodes,
                "t0_depth": t0_depth,
                "t1_depth": t1_depth,
                "t0_line": t0_line,
                "t1_line": t1_line,
                "t0_path": t0_path,
                "t1_path": t1_path,
            }
        )

t12_checked = 0
t12_missing_parent = 0
t12_missing_source = 0
t12_node_violations = 0
t12_depth_violations = 0
t12_samples = []

for (
    design_name,
    source_algo,
    recipe_id_num,
    step_id_num,
    t2_nodes,
    t2_depth,
    t2_line,
    t2_path,
    target_algo,
) in tier2_records:
    if source_algo is None:
        t12_missing_source += 1
        continue

    parent = tier1_by_key_algo.get((design_name, source_algo, recipe_id_num, step_id_num))
    if parent is None:
        t12_missing_parent += 1
        continue

    t12_checked += 1
    t1_nodes, t1_depth, t1_line, t1_path = parent

    node_bad = t2_nodes > t1_nodes
    depth_bad = t2_depth > t1_depth
    if node_bad:
        t12_node_violations += 1
    if depth_bad:
        t12_depth_violations += 1

    if (node_bad or depth_bad) and len(t12_samples) < 10:
        t12_samples.append(
            {
                "design": design_name,
                "source_algo": source_algo,
                "target_algo": target_algo,
                "recipe": recipe_id_num,
                "step": step_id_num,
                "t1_nodes": t1_nodes,
                "t2_nodes": t2_nodes,
                "t1_depth": t1_depth,
                "t2_depth": t2_depth,
                "t1_line": t1_line,
                "t2_line": t2_line,
                "t1_path": t1_path,
                "t2_path": t2_path,
            }
        )

errors = []
tier0 = tier_counts.get(0, 0)
tier1 = tier_counts.get(1, 0)
tier2 = tier_counts.get(2, 0)
other_tiers = sum(v for k, v in tier_counts.items() if k not in (0, 1, 2))

if total_rows == 0:
    errors.append("Master CSV has 0 data rows.")

if len(per_design_counts) != expected_designs:
    errors.append(
        f"Expected {expected_designs} root designs from file_path, found {len(per_design_counts)}"
    )

if other_tiers > 0:
    errors.append(f"Found {other_tiers} rows with unsupported tier_id values")

if tier0 > 0:
    if tier1 != 4 * tier0:
        errors.append(f"Tier1 count mismatch: expected {4 * tier0}, found {tier1}")
    if tier2 != 12 * tier0:
        errors.append(f"Tier2 count mismatch: expected {12 * tier0}, found {tier2}")
else:
    if tier1 > 0 and tier2 != 3 * tier1:
        errors.append(f"Tier2 count mismatch without tier0 rows: expected {3 * tier1}, found {tier2}")

if tier1 > 0:
    expected_t1_per_algo = tier1 // 4 if tier1 % 4 == 0 else None
    if expected_t1_per_algo is None:
        errors.append(f"Tier1 total ({tier1}) is not divisible by 4 algorithms")
    else:
        for algo in sorted(valid_algorithms):
            found = tier_algo_counts[1].get(algo, 0)
            if found != expected_t1_per_algo:
                errors.append(
                    f"Tier1 algorithm count mismatch for {algo}: expected {expected_t1_per_algo}, found {found}"
                )

if tier2 > 0:
    expected_t2_per_algo = tier2 // 4 if tier2 % 4 == 0 else None
    if expected_t2_per_algo is None:
        errors.append(f"Tier2 total ({tier2}) is not divisible by 4 algorithms")
    else:
        for algo in sorted(valid_algorithms):
            found = tier_algo_counts[2].get(algo, 0)
            if found != expected_t2_per_algo:
                errors.append(
                    f"Tier2 algorithm count mismatch for {algo}: expected {expected_t2_per_algo}, found {found}"
                )

per_design_errors = []
for design in sorted(per_design_counts.keys()):
    d0 = per_design_counts[design].get(0, 0)
    d1 = per_design_counts[design].get(1, 0)
    d2 = per_design_counts[design].get(2, 0)

    if d0 > 0:
        if d1 != 4 * d0:
            per_design_errors.append(f"{design}: tier1 expected {4 * d0}, found {d1}")
        if d2 != 12 * d0:
            per_design_errors.append(f"{design}: tier2 expected {12 * d0}, found {d2}")
    else:
        if d1 > 0 and d2 != 3 * d1:
            per_design_errors.append(
                f"{design}: tier2 expected {3 * d1} (no tier0 rows), found {d2}"
            )

if per_design_errors:
    errors.extend(per_design_errors[:20])
    if len(per_design_errors) > 20:
        errors.append(f"... and {len(per_design_errors) - 20} more per-design count mismatch(es)")

print("=== Tier-0 Augmentation Summary ===")
print(f"Master input CSV: {master_in}")
print(f"Master output temp CSV: {master_out}")
print(f"Scratch zip root: {zip_root}")
print(f"Scratch extraction root: {unzip_root}")
print(f"Per-algorithm output temp dir: {algo_out_dir}")
print("Per-algorithm membership rule: include all tier0 rows + rows where algorithm column equals target algorithm")
print(f"Parallel workers used: {workers if not zip_tasks else min(workers, len(zip_tasks))}")
print(f"Designs with a selected tier1 source for tier0: {tier0_source_design_count}")
print(f"Tier1 zip archives extracted in scratch: {unzipped_zip_count}")
print(f"Tier1 logs matched for tier0 derivation: {matched_tier1_log_count}")
print(f"Tier0 rows added this run: {added_tier0}")
print(f"Tier0 parse failures/skips: {tier0_parse_failures}")

print("Column summary (non-empty / total rows):")
for col in fieldnames:
    print(f"  {col}: {non_empty_counts[col]} / {total_rows}")

print("Tier summary:")
print(f"  tier0: {tier0}")
print(f"  tier1: {tier1}")
print(f"  tier2: {tier2}")
print(f"  other: {other_tiers}")

print("Tier means (valid numeric rows):")
for tier_value in sorted(tier_numeric_counts.keys()):
    count = tier_numeric_counts[tier_value]
    mean_nodes = tier_node_sum[tier_value] / count if count else 0.0
    mean_depth = tier_depth_sum[tier_value] / count if count else 0.0
    print(
        f"  tier{tier_value}: mean_nodes={mean_nodes:.6f}, "
        f"mean_depth={mean_depth:.6f}, rows={count}"
    )

print("Algorithm summary (non-empty values):")
if algo_counts:
    for algo in sorted(algo_counts.keys()):
        print(f"  {algo}: {algo_counts[algo]}")
else:
    print("  none")

print("Optimizability ranges by algorithm (valid numeric rows):")
if algo_range_count:
    for algo in sorted(algo_range_count.keys()):
        print(
            f"  {algo}: "
            f"node_opt[{algo_opt_min[algo]:.6f}, {algo_opt_max[algo]:.6f}], "
            f"depth_opt[{algo_depth_opt_min[algo]:.6f}, {algo_depth_opt_max[algo]:.6f}], "
            f"rows={algo_range_count[algo]}"
        )
else:
    print("  none")

print("=== Per-Algorithm CSV Stats ===")
per_algo_lengths = {}
for algo in algorithms:
    algo_rows = rows_by_algo.get(algo, [])
    per_algo_lengths[algo] = len(algo_rows)
    tier_breakdown = Counter()
    algo_valid = 0
    algo_invalid = 0
    algo_zero_node = 0
    algo_zero_node_pos_depth = 0
    node_opt_min = None
    node_opt_max = None
    depth_opt_min = None
    depth_opt_max = None

    for algo_row in algo_rows:
        tier_raw = (algo_row.get("tier_id") or "").strip()
        if tier_raw == "":
            tier_value = 0
        else:
            try:
                tier_value = int(float(tier_raw))
            except ValueError:
                tier_value = -1
        tier_breakdown[tier_value] += 1

        try:
            node_opt = float((algo_row.get("optimizability") or "").strip())
            depth_opt = float((algo_row.get("depth_optimizability") or "").strip())
        except ValueError:
            algo_invalid += 1
            continue

        if not (math.isfinite(node_opt) and math.isfinite(depth_opt)):
            algo_invalid += 1
            continue

        algo_valid += 1
        if abs(node_opt) <= eps:
            algo_zero_node += 1
        if abs(node_opt) <= eps and depth_opt > 0.0:
            algo_zero_node_pos_depth += 1

        if node_opt_min is None or node_opt < node_opt_min:
            node_opt_min = node_opt
        if node_opt_max is None or node_opt > node_opt_max:
            node_opt_max = node_opt
        if depth_opt_min is None or depth_opt < depth_opt_min:
            depth_opt_min = depth_opt
        if depth_opt_max is None or depth_opt > depth_opt_max:
            depth_opt_max = depth_opt

    tier_text = ", ".join(
        f"tier{tier_id}={count}" for tier_id, count in sorted(tier_breakdown.items())
    )
    if not tier_text:
        tier_text = "none"

    print(
        f"  {algo}: path={algo_csv_paths[algo]}, rows={len(algo_rows)}, "
        f"valid_numeric={algo_valid}, invalid_numeric={algo_invalid}"
    )
    print(f"    tier_breakdown: {tier_text}")
    print(
        f"    node_opt_zero={algo_zero_node}, "
        f"node_opt_zero_and_depth_pos={algo_zero_node_pos_depth}"
    )
    if node_opt_min is None:
        print("    ranges: N/A")
    else:
        print(
            f"    ranges: node_opt[{node_opt_min:.6f}, {node_opt_max:.6f}], "
            f"depth_opt[{depth_opt_min:.6f}, {depth_opt_max:.6f}]"
        )

if per_algo_lengths:
    min_algo = min(per_algo_lengths, key=per_algo_lengths.get)
    max_algo = max(per_algo_lengths, key=per_algo_lengths.get)
    min_len = per_algo_lengths[min_algo]
    max_len = per_algo_lengths[max_algo]
    delta = max_len - min_len
    denom = max(1, max_len)
    delta_pct = 100.0 * delta / denom
    print("Per-algorithm length consistency check:")
    print(
        f"  min={min_len} ({min_algo}), max={max_len} ({max_algo}), "
        f"delta={delta}, delta_pct={delta_pct:.4f}%"
    )
    if delta == 0:
        print("  status: equal")
    elif delta_pct <= 1.0:
        print("  status: essentially equal (<=1.0% spread)")
    else:
        print("  status: not essentially equal (>1.0% spread)")

print("=== Zero Node + Positive Depth Check ===")
print(f"Total rows: {total_rows}")
print(f"Valid numeric rows: {valid_rows}")
print(f"Invalid rows skipped: {invalid_rows}")
print(f"Node optimizability == 0 count (total): {node_opt_zero_total}")
print(f"Node optimizability == 0 percent (valid rows): {(100.0 * node_opt_zero_total / valid_rows) if valid_rows else 0.0:.6f}%")
print("Condition: abs(optimizability) <= eps AND depth_optimizability > 0")
print(f"Matches: {match_rows}")
print(f"Percent of valid rows: {rate_valid:.6f}%")
print(f"Percent of total rows: {rate_total:.6f}%")

print("=== Monotonic Sanity Checks ===")
print("Check A: tier0 should be >= tier1 (same design/recipe/step)")
print(f"  compared rows: {t01_checked}")
print(f"  missing tier0 parents: {t01_missing_parent}")
print(f"  node violations (tier1 > tier0): {t01_node_violations}")
print(f"  depth violations (tier1 > tier0): {t01_depth_violations}")
if t01_samples:
    print("  sample violations:")
    for s in t01_samples:
        print(
            "    - {design} {algo} syn{recipe} step{step}: "
            "nodes t0={t0_nodes} t1={t1_nodes}, depth t0={t0_depth} t1={t1_depth} "
            "(lines t0={t0_line}, t1={t1_line})".format(**s)
        )

print("Check B: tier1 should be >= tier2 (same design/source-algo/recipe/step)")
print(f"  compared rows: {t12_checked}")
print(f"  missing tier1 parents: {t12_missing_parent}")
print(f"  missing tier2 source algo parse: {t12_missing_source}")
print(f"  node violations (tier2 > tier1): {t12_node_violations}")
print(f"  depth violations (tier2 > tier1): {t12_depth_violations}")
if t12_samples:
    print("  sample violations:")
    for s in t12_samples:
        print(
            "    - {design} {source_algo}->{target_algo} syn{recipe} step{step}: "
            "nodes t1={t1_nodes} t2={t2_nodes}, depth t1={t1_depth} t2={t2_depth} "
            "(lines t1={t1_line}, t2={t2_line})".format(**s)
        )

if max_nodes is None or max_depth is None:
    print("Max nodes: N/A")
    print("Max depth: N/A")
else:
    print(f"Max nodes: {max_nodes:g}")
    print(
        "  at line={line}, design={design}, tier_id={tier_id}, algorithm={algorithm}, "
        "recipe_id={recipe_id}, step_id={step_id}".format(**max_nodes_row)
    )
    print(f"  file_path={max_nodes_row['file_path']}")

    print(f"Max depth: {max_depth:g}")
    print(
        "  at line={line}, design={design}, tier_id={tier_id}, algorithm={algorithm}, "
        "recipe_id={recipe_id}, step_id={step_id}".format(**max_depth_row)
    )
    print(f"  file_path={max_depth_row['file_path']}")

if row_issues:
    print("Validation row issues:")
    for issue in row_issues:
        print("  - " + issue)

if errors:
    print("Validation errors:")
    for error in errors:
        print("  - " + error)
    print("Validation note: errors reported above but master CSV was still generated as requested.")
else:
    print("Validation passed: master CSV structure and counts look complete.")
PY

mv -f "$MASTER_TMP" "$MASTER_CSV"
mv -f "$REPORT_TMP" "$REPORT_PATH"

algo_csv_moved=0
for algo in Orchestrate Deepsyn Syn4 C2RS; do
    src_csv="$SCRATCH_PER_ALGO_DIR/full_master_${algo}.csv"
    dst_csv="$PER_ALGO_DIR/full_master_${algo}.csv"
    if [[ -f "$src_csv" ]]; then
        mv -f "$src_csv" "$dst_csv"
        algo_csv_moved=$((algo_csv_moved + 1))
    else
        echo "WARNING: Missing per-algorithm CSV in scratch: $src_csv"
    fi
done

total_lines=$(wc -l < "$MASTER_CSV")
total_rows=$((total_lines - 1))

echo "=================================================="
echo " Master CSV completed"
echo " Designs included from per-design CSVs: $included_count"
echo " Missing per-design CSVs: $missing_count"
echo " Total rows written: $total_rows"
echo " Per-algorithm CSVs moved: $algo_csv_moved / 4"
for algo in Orchestrate Deepsyn Syn4 C2RS; do
    algo_csv="$PER_ALGO_DIR/full_master_${algo}.csv"
    if [[ -f "$algo_csv" ]]; then
        algo_lines=$(wc -l < "$algo_csv")
        algo_rows=$((algo_lines - 1))
        echo "   $algo rows: $algo_rows"
    fi
done
echo " Saved to: $MASTER_CSV"
echo " Per-algorithm CSV dir: $PER_ALGO_DIR"
echo " Report saved to: $REPORT_PATH"
echo "=================================================="

#!/bin/bash
#SBATCH --job-name=8_full_csv
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa
#SBATCH --output=logs/8_full_csv_%j.out

set -euo pipefail

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
DESIGNS_DIR="$BASE_DIR/data/designs"
MASTER_DIR="$DESIGNS_DIR/design_metadata"
MASTER_CSV="$MASTER_DIR/full_master.csv"

# --- Design List (0-54) ---
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

mkdir -p "$MASTER_DIR"

echo "=================================================="
echo " JOB 8: Building Master CSV from all design CSVs"
echo " Time: $(date)"
echo " Output: $MASTER_CSV"
echo "=================================================="

TMP_CSV="$(mktemp "$MASTER_DIR/full_master.tmp.XXXXXX.csv")"
trap 'rm -f "$TMP_CSV"' EXIT

first_file=1
included_count=0
missing_count=0

for design in "${DESIGNS[@]}"; do
    csv_path="$DESIGNS_DIR/$design/design_metadata/${design}.csv"

    if [[ ! -f "$csv_path" ]]; then
        echo "WARNING: Missing CSV for $design -> $csv_path"
        missing_count=$((missing_count + 1))
        continue
    fi

    if [[ $first_file -eq 1 ]]; then
        cat "$csv_path" > "$TMP_CSV"
        first_file=0
    else
        tail -n +2 "$csv_path" >> "$TMP_CSV"
    fi

    included_count=$((included_count + 1))
    echo "Included: $design"
done

if [[ $first_file -eq 1 ]]; then
    echo "ERROR: No per-design CSV files were found."
    exit 1
fi

if [[ $missing_count -gt 0 ]]; then
    echo "ERROR: $missing_count design CSV file(s) are missing."
    echo "ERROR: Refusing to write partial master CSV."
    exit 1
fi

echo ">>> Running integrity checks and column summary..."
python3 - "$TMP_CSV" "${#DESIGNS[@]}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

csv_path = sys.argv[1]
expected_designs = int(sys.argv[2])

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

with open(csv_path, newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        print("ERROR: Master CSV has no header row.")
        sys.exit(1)

    columns = reader.fieldnames
    missing_columns = [c for c in required_columns if c not in columns]
    if missing_columns:
        print("ERROR: Missing required columns: " + ", ".join(missing_columns))
        sys.exit(1)

    non_empty_counts = {c: 0 for c in columns}
    tier_counts = Counter()
    algo_counts = Counter()
    tier_algo_counts = {1: Counter(), 2: Counter()}
    per_design_counts = defaultdict(lambda: Counter())
    row_issues = []
    total_rows = 0

    for row_idx, row in enumerate(reader, start=2):
        total_rows += 1

        for col in columns:
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
                f"Row {row_idx}: cannot derive root design from file_path='{file_path}'"
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
                    row_issues.append(f"Row {row_idx}: invalid tier_id='{tier_raw}'")
        tier_counts[tier_id] += 1

        algorithm = (row.get("algorithm") or "").strip()
        if algorithm not in null_algorithms:
            algo_counts[algorithm] += 1

        if tier_id == 0:
            if algorithm not in null_algorithms and len(row_issues) < max_row_issues:
                row_issues.append(
                    f"Row {row_idx}: tier0 row should not have algorithm (found '{algorithm}')"
                )
        elif tier_id in (1, 2):
            if algorithm in null_algorithms and len(row_issues) < max_row_issues:
                row_issues.append(f"Row {row_idx}: tier{tier_id} row is missing algorithm")
            elif algorithm not in valid_algorithms and len(row_issues) < max_row_issues:
                row_issues.append(
                    f"Row {row_idx}: tier{tier_id} has unknown algorithm '{algorithm}'"
                )
            else:
                tier_algo_counts[tier_id][algorithm] += 1
        elif len(row_issues) < max_row_issues:
            row_issues.append(f"Row {row_idx}: unsupported tier_id={tier_id}")

        if root_design:
            per_design_counts[root_design][tier_id] += 1

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
            per_design_errors.append(
                f"{design}: tier1 expected {4 * d0}, found {d1}"
            )
        if d2 != 12 * d0:
            per_design_errors.append(
                f"{design}: tier2 expected {12 * d0}, found {d2}"
            )
    else:
        if d1 > 0 and d2 != 3 * d1:
            per_design_errors.append(
                f"{design}: tier2 expected {3 * d1} (no tier0 rows), found {d2}"
            )

if per_design_errors:
    errors.extend(per_design_errors[:20])
    if len(per_design_errors) > 20:
        errors.append(f"... and {len(per_design_errors) - 20} more per-design count mismatch(es)")

print("Column summary (non-empty / total rows):")
for col in columns:
    print(f"  {col}: {non_empty_counts[col]} / {total_rows}")

print("Tier summary:")
print(f"  tier0: {tier0}")
print(f"  tier1: {tier1}")
print(f"  tier2: {tier2}")
print(f"  other: {other_tiers}")

print("Algorithm summary (non-empty values):")
if algo_counts:
    for algo in sorted(algo_counts.keys()):
        print(f"  {algo}: {algo_counts[algo]}")
else:
    print("  none")

if row_issues:
    print("Validation row issues:")
    for issue in row_issues:
        print("  - " + issue)

if errors:
    print("Validation errors:")
    for error in errors:
        print("  - " + error)
    sys.exit(1)

print("Validation passed: master CSV structure and counts look complete.")
PY

mv "$TMP_CSV" "$MASTER_CSV"
trap - EXIT

total_lines=$(wc -l < "$MASTER_CSV")
total_rows=$((total_lines - 1))

echo "=================================================="
echo " Master CSV complete"
echo " Designs included: $included_count"
echo " Total rows: $total_rows"
echo " Saved to: $MASTER_CSV"
echo "=================================================="

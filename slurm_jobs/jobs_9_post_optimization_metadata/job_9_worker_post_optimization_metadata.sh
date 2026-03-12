#!/bin/bash

set -euo pipefail

if [ -z "${ALGORITHM:-}" ]; then
	echo "ERROR: ALGORITHM is required (Orchestrate|Deepsyn|Syn4|C2RS)"
	exit 1
fi

case "$ALGORITHM" in
	Orchestrate|Deepsyn|Syn4|C2RS)
		;;
	*)
		echo "ERROR: Unsupported ALGORITHM: $ALGORITHM"
		exit 1
		;;
esac

echo "=========================================="
echo "JOB 9: Post-Optimization Metadata Pipeline"
echo "=========================================="
echo "Algorithm: ${ALGORITHM}"
echo "Job ID: ${SLURM_JOB_ID:-manual_run}"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0

export PATH="$HOME/abc:$PATH"

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
DATASET_TOOLS_DIR="${BASE_DIR}/dataset_tools"
UPDATE_SCRIPT="${DATASET_TOOLS_DIR}/update_optimization_metadata.py"

DESIGN_GROUP="${DESIGN_GROUP:-all}"          # all | random | openabc
DESIGNS="${DESIGNS:-}"                       # comma/space-separated explicit list
METADATA_WORKERS="${METADATA_WORKERS:-${SLURM_CPUS_PER_TASK:-24}}"

ARCHIVE_FULL_DATASET="${ARCHIVE_FULL_DATASET:-true}"   # true | false
BACKUP_DIR="${BACKUP_DIR:-/scratch-shared/$USER/dataset_backups}"

if [[ "$DESIGN_GROUP" != "all" && "$DESIGN_GROUP" != "random" && "$DESIGN_GROUP" != "openabc" ]]; then
	echo "ERROR: DESIGN_GROUP must be one of: all, random, openabc"
	exit 1
fi

if [[ "$ARCHIVE_FULL_DATASET" != "true" && "$ARCHIVE_FULL_DATASET" != "false" ]]; then
	echo "ERROR: ARCHIVE_FULL_DATASET must be true or false"
	exit 1
fi

if ! [[ "$METADATA_WORKERS" =~ ^[0-9]+$ ]] || [ "$METADATA_WORKERS" -lt 1 ]; then
	echo "ERROR: METADATA_WORKERS must be a positive integer"
	exit 1
fi

if [ ! -d "$FULL_DATASET" ]; then
	echo "ERROR: FULL_DATASET not found: $FULL_DATASET"
	exit 1
fi

if [ ! -f "$UPDATE_SCRIPT" ]; then
	echo "ERROR: update script not found: $UPDATE_SCRIPT"
	exit 1
fi

if ! command -v abc >/dev/null 2>&1; then
	echo "ERROR: abc not found in PATH"
	exit 1
fi

echo "Loaded modules: 2025, foss/2025a, Python/3.13.1"
echo "Using abc: $(which abc)"
echo "Base directory: $BASE_DIR"
echo "FULL_DATASET: $FULL_DATASET"
echo "Design group: $DESIGN_GROUP"
echo "Explicit designs: ${DESIGNS:-<none>}"
echo "Metadata workers: $METADATA_WORKERS"
echo "Archive enabled: $ARCHIVE_FULL_DATASET"
echo ""

designs_file="$(mktemp "${TMPDIR:-/tmp}/job9_designs_${SLURM_JOB_ID:-local}_XXXXXX")"
trap 'rm -f "$designs_file"' EXIT

python3 - "$FULL_DATASET" "$DESIGN_GROUP" "$DESIGNS" > "$designs_file" <<'PY'
import sys
from pathlib import Path

full_dataset = Path(sys.argv[1])
design_group = sys.argv[2]
designs_raw = sys.argv[3]

random_designs = {"128", "256", "512", "1024", "2048", "4096", "8192", "16384"}
openabc_designs = {
    "i2c", "spi", "des3_area", "ss_pcm", "usb_phy", "sasc", "wb_dma", "simple_spi",
    "dynamic_node", "aes", "pci", "ac97_ctrl", "mem_ctrl", "tv80", "fpu",
    "wb_conmax", "tinyRocket", "aes_xcrypt", "aes_secworks", "jpeg", "bp_be",
    "ethernet", "vga_lcd", "picosoc", "dft", "idft", "fir", "iir", "sha256",
}

base_aigs = full_dataset / "base_aigs"
if not base_aigs.is_dir():
    raise SystemExit(f"ERROR: missing base_aigs directory: {base_aigs}")

available = sorted([p.name for p in base_aigs.iterdir() if p.is_dir()])
if not available:
    raise SystemExit("ERROR: no design folders found under base_aigs")

if design_group == "random":
    selected = [d for d in available if d in random_designs]
elif design_group == "openabc":
    selected = [d for d in available if d in openabc_designs]
else:
    selected = list(available)

explicit = [x.strip() for chunk in designs_raw.split() for x in chunk.split(",") if x.strip()]
if explicit:
    explicit_set = set(explicit)
    missing = sorted(explicit_set.difference(available))
    if missing:
        raise SystemExit("ERROR: explicit designs not found in base_aigs: " + ", ".join(missing))
    selected = [d for d in selected if d in explicit_set]

if not selected:
    raise SystemExit("ERROR: no designs selected after applying filters")

for design in selected:
    print(design)
PY

selected_designs_count=$(wc -l < "$designs_file" | tr -d ' ')
echo "Selected designs: $selected_designs_count"
echo ""

echo "=========================================="
echo "STEP 1/7: Pre-check Tier-1 graph completeness"
echo "=========================================="

python3 - "$FULL_DATASET" "$ALGORITHM" "$designs_file" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

full_dataset = Path(sys.argv[1])
algorithm = sys.argv[2]
designs_file = Path(sys.argv[3])

errors = []
checked = 0
total_discovered = 0
total_created = 0
total_files = 0

for design in designs_file.read_text(encoding="utf-8").splitlines():
    design = design.strip()
    if not design:
        continue

    checked += 1
    summary_path = full_dataset / "metadata" / "raw_logs" / design / "tier1" / algorithm / "summary.json"
    out_dir = full_dataset / "optimized_aigs" / algorithm / "tier1" / design
    out_zip = full_dataset / "optimized_aigs" / algorithm / "tier1" / f"{design}.zip"

    if not summary_path.is_file():
        errors.append(f"missing summary: {summary_path}")
        continue

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid json summary {summary_path}: {exc}")
        continue

    discovered = int(payload.get("discovered", -1))
    processed = int(payload.get("processed", -1))
    created = int(payload.get("created", -1))
    failed = int(payload.get("failed", -1))

    if discovered <= 0:
        errors.append(f"non-positive discovered count in {summary_path}: {discovered}")
        continue
    if processed != discovered:
        errors.append(f"processed!=discovered in {summary_path}: {processed}!={discovered}")
    if created != discovered:
        errors.append(f"created!=discovered in {summary_path}: {created}!={discovered}")
    if failed != 0:
        errors.append(f"failed!=0 in {summary_path}: {failed}")

    if out_dir.is_dir():
        aig_count = sum(1 for _ in out_dir.rglob("*.aig"))
    elif out_zip.is_file():
        with zipfile.ZipFile(out_zip, "r") as zf:
            aig_count = sum(
                1 for name in zf.namelist() if name.lower().endswith(".aig") and not name.endswith("/")
            )
    else:
        errors.append(f"missing output payload for design={design}: dir={out_dir} zip={out_zip}")
        continue
    if aig_count != discovered:
        errors.append(
            f"tier1 AIG count mismatch for design={design}, algorithm={algorithm}: "
            f"files={aig_count}, discovered={discovered}"
        )

    total_discovered += discovered
    total_created += created
    total_files += aig_count

if errors:
    print("ERROR: Tier-1 pre-check failed")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print(f"✓ Tier-1 pre-check passed for algorithm={algorithm}")
print(f"✓ Designs checked: {checked}")
print(f"✓ Total discovered from summaries: {total_discovered}")
print(f"✓ Total created from summaries: {total_created}")
print(f"✓ Total .aig files found: {total_files}")
PY

echo ""
echo "=========================================="
echo "STEP 2/7: Update dataset_manifest.json (Tier-1 confirmed)"
echo "=========================================="

python3 - "$FULL_DATASET" "$ALGORITHM" "$designs_file" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import zipfile

full_dataset = Path(sys.argv[1])
algorithm = sys.argv[2]
designs_file = Path(sys.argv[3])
manifest_path = full_dataset / "dataset_manifest.json"

if not manifest_path.is_file():
    raise SystemExit(f"ERROR: missing dataset manifest: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as fh:
    manifest = json.load(fh)

designs = [d.strip() for d in designs_file.read_text(encoding="utf-8").splitlines() if d.strip()]
if not designs:
    raise SystemExit("ERROR: no selected designs found for manifest update")

total_discovered = 0
total_output_aigs = 0
for design in designs:
    summary_path = full_dataset / "metadata" / "raw_logs" / design / "tier1" / algorithm / "summary.json"
    out_dir = full_dataset / "optimized_aigs" / algorithm / "tier1" / design
    out_zip = full_dataset / "optimized_aigs" / algorithm / "tier1" / f"{design}.zip"
    if not summary_path.is_file():
        raise SystemExit(f"ERROR: cannot update manifest, summary missing: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    total_discovered += int(payload.get("discovered", 0))
    if out_dir.is_dir():
        total_output_aigs += sum(1 for _ in out_dir.rglob("*.aig"))
    elif out_zip.is_file():
        with zipfile.ZipFile(out_zip, "r") as zf:
            total_output_aigs += sum(
                1
                for name in zf.namelist()
                if name.lower().endswith(".aig") and not name.endswith("/")
            )

opt_status = manifest.setdefault("optimization_status", {})
algo_status = opt_status.setdefault(algorithm, {})
algo_status["tier1"] = {
    "confirmed": True,
    "confirmed_at": datetime.now(timezone.utc).isoformat(),
    "selected_design_count": len(designs),
    "selected_designs": designs,
    "summary_discovered_total": total_discovered,
    "output_aig_count_total": total_output_aigs,
}

manifest["last_updated"] = datetime.now(timezone.utc).isoformat()

with manifest_path.open("w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")

print(f"✓ Updated manifest: {manifest_path}")
print(f"✓ optimization_status[{algorithm}][tier1].confirmed = true")
print(f"✓ selected_design_count = {len(designs)}")
print(f"✓ summary_discovered_total = {total_discovered}")
print(f"✓ output_aig_count_total = {total_output_aigs}")
PY

echo ""
echo "=========================================="
echo "STEP 3/7: Populate Tier-1 metadata CSV rows"
echo "=========================================="

active_jobs=0
started_jobs=0
for design in $(cat "$designs_file"); do
    [ -z "$design" ] && continue

    (
        python3 "$UPDATE_SCRIPT" \
            --full-dataset "$FULL_DATASET" \
            --design "$design" \
            --algorithm "$ALGORITHM" \
            --tier tier1
    ) &

    active_jobs=$((active_jobs + 1))
    started_jobs=$((started_jobs + 1))

    if [ "$active_jobs" -ge "$METADATA_WORKERS" ]; then
        wait -n
        active_jobs=$((active_jobs - 1))
    fi
done

while [ "$active_jobs" -gt 0 ]; do
    wait -n
    active_jobs=$((active_jobs - 1))
done

echo "✓ Metadata updates finished for $started_jobs design(s)"

echo ""
echo "=========================================="
echo "STEP 4/7: Verify CSV population (tier0 + tier1)"
echo "=========================================="

python3 - "$FULL_DATASET" "$ALGORITHM" "$designs_file" <<'PY'
import csv
import sys
import zipfile
from pathlib import Path

full_dataset = Path(sys.argv[1])
algorithm = sys.argv[2]
designs_file = Path(sys.argv[3])

errors = []
checked = 0

for design in designs_file.read_text(encoding="utf-8").splitlines():
    design = design.strip()
    if not design:
        continue
    checked += 1

    csv_path = full_dataset / "metadata" / "stats" / f"{design}.csv"
    tier1_dir = full_dataset / "optimized_aigs" / algorithm / "tier1" / design
    tier1_zip = full_dataset / "optimized_aigs" / algorithm / "tier1" / f"{design}.zip"

    if not csv_path.is_file():
        errors.append(f"missing metadata csv: {csv_path}")
        continue

    if tier1_dir.is_dir():
        tier1_aig_count = sum(1 for _ in tier1_dir.rglob("*.aig"))
    elif tier1_zip.is_file():
        with zipfile.ZipFile(tier1_zip, "r") as zf:
            tier1_aig_count = sum(
                1 for name in zf.namelist() if name.lower().endswith(".aig") and not name.endswith("/")
            )
    else:
        errors.append(f"missing tier1 output payload: dir={tier1_dir} zip={tier1_zip}")
        continue
    tier1_csv_count = 0
    tier0_csv_count = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        if "algorithm" not in cols or "tier_id" not in cols:
            errors.append(f"missing required columns in {csv_path}: algorithm/tier_id")
            continue

        for row in reader:
            row_algorithm = (row.get("algorithm") or "").strip()
            row_tier = (row.get("tier_id") or "").strip()
            if row_algorithm == algorithm and row_tier == "1":
                tier1_csv_count += 1

            if row_algorithm == "" and row_tier in {"", "0"}:
                tier0_csv_count += 1

    if tier1_csv_count != tier1_aig_count:
        errors.append(
            f"tier1 csv mismatch design={design} algorithm={algorithm}: "
            f"csv_rows={tier1_csv_count}, aig_files={tier1_aig_count}"
        )

    if tier0_csv_count == 0:
        errors.append(
            f"tier0 rows missing/undetected in {csv_path} (expected algorithm empty and tier_id in ['', '0'])"
        )

if errors:
    print("ERROR: CSV verification failed")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print(f"✓ CSV verification passed for algorithm={algorithm}")
print(f"✓ Designs checked: {checked}")
PY

echo ""
echo "=========================================="
echo "STEP 5/7: Dataset-wide stats report"
echo "=========================================="

python3 - "$FULL_DATASET" "$designs_file" <<'PY'
import csv
import json
from datetime import datetime
from pathlib import Path
import zipfile
import sys

full_dataset = Path(sys.argv[1])
designs_file = Path(sys.argv[2])
designs = [d.strip() for d in designs_file.read_text(encoding="utf-8").splitlines() if d.strip()]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_dir = full_dataset / "metadata" / "stats"
report_dir.mkdir(parents=True, exist_ok=True)
json_report = report_dir / f"job9_full_stats_{timestamp}.json"
txt_report = report_dir / f"job9_full_stats_{timestamp}.txt"

def count_base_graphs(design: str) -> tuple[int, int, int]:
    ddir = full_dataset / "base_aigs" / design
    if not ddir.is_dir():
        return (0, 0, 0)

    plain_aigs = sum(1 for p in ddir.glob("*.aig") if p.is_file())
    zip_files = sorted(ddir.glob("syn*.zip"))
    zipped_aigs = 0
    for zpath in zip_files:
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zipped_aigs += sum(1 for n in zf.namelist() if n.lower().endswith(".aig"))
        except Exception:
            pass

    return (plain_aigs + zipped_aigs, plain_aigs, len(zip_files))

algorithms = []
opt_root = full_dataset / "optimized_aigs"
if opt_root.is_dir():
    for p in sorted(opt_root.iterdir()):
        if p.is_dir() and p.name not in {"manifests", "done"}:
            algorithms.append(p.name)


def count_tier1_for_algorithm(alg: str, design: str) -> int:
    t1_dir = full_dataset / "optimized_aigs" / alg / "tier1" / design
    t1_zip = full_dataset / "optimized_aigs" / alg / "tier1" / f"{design}.zip"
    if t1_dir.is_dir():
        return sum(1 for _ in t1_dir.rglob("*.aig"))
    if t1_zip.is_file():
        with zipfile.ZipFile(t1_zip, "r") as zf:
            return sum(
                1
                for name in zf.namelist()
                if name.lower().endswith(".aig") and not name.endswith("/")
            )
    return 0

report = {
    "timestamp": datetime.now().isoformat(),
    "full_dataset": str(full_dataset),
    "designs_checked": len(designs),
    "designs": {},
    "totals": {
        "tier0_graphs": 0,
        "tier1_graphs": 0,
        "csv_rows": 0,
    },
    "algorithms": {},
}

for alg in algorithms:
    report["algorithms"][alg] = {
        "designs_with_tier1_dir": 0,
        "tier1_graphs": 0,
        "missing_designs": [],
    }

for design in designs:
    tier0_total, tier0_plain, tier0_zip_count = count_base_graphs(design)
    csv_path = full_dataset / "metadata" / "stats" / f"{design}.csv"
    csv_rows = 0
    if csv_path.is_file():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                csv_rows = sum(1 for _ in csv.DictReader(handle))
        except Exception:
            csv_rows = -1

    design_entry = {
        "tier0_graphs_total": tier0_total,
        "tier0_plain_aig_files": tier0_plain,
        "tier0_syn_zip_files": tier0_zip_count,
        "metadata_csv_rows": csv_rows,
        "tier1_by_algorithm": {},
    }

    report["totals"]["tier0_graphs"] += tier0_total
    if csv_rows > 0:
        report["totals"]["csv_rows"] += csv_rows

    for alg in algorithms:
        count = count_tier1_for_algorithm(alg, design)
        if count > 0:
            report["algorithms"][alg]["designs_with_tier1_dir"] += 1
            report["algorithms"][alg]["tier1_graphs"] += count
            report["totals"]["tier1_graphs"] += count
            design_entry["tier1_by_algorithm"][alg] = count
        else:
            report["algorithms"][alg]["missing_designs"].append(design)
            design_entry["tier1_by_algorithm"][alg] = 0

    report["designs"][design] = design_entry

json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = []
lines.append("JOB 9 FULL DATASET STATS")
lines.append(f"timestamp={report['timestamp']}")
lines.append(f"full_dataset={report['full_dataset']}")
lines.append(f"designs_checked={report['designs_checked']}")
lines.append(f"total_tier0_graphs={report['totals']['tier0_graphs']}")
lines.append(f"total_tier1_graphs={report['totals']['tier1_graphs']}")
lines.append(f"total_csv_rows={report['totals']['csv_rows']}")
lines.append("")
lines.append("Algorithms:")
for alg in algorithms:
    info = report["algorithms"][alg]
    lines.append(
        f"  {alg}: tier1_graphs={info['tier1_graphs']} designs_with_tier1_dir={info['designs_with_tier1_dir']} missing_designs={len(info['missing_designs'])}"
    )
    if info["missing_designs"]:
        lines.append("    missing=" + ",".join(info["missing_designs"]))

txt_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("✓ Wrote dataset-wide stats reports")
print(f"  JSON: {json_report}")
print(f"  TXT : {txt_report}")
print(f"✓ total tier0 graphs: {report['totals']['tier0_graphs']}")
print(f"✓ total tier1 graphs: {report['totals']['tier1_graphs']}")
print(f"✓ total metadata csv rows: {report['totals']['csv_rows']}")
PY

echo ""
echo "=========================================="
echo "STEP 6/7: Tier-0 completeness gate before backup ZIP"
echo "=========================================="

python3 - "$FULL_DATASET" "$ALGORITHM" "$designs_file" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

full_dataset = Path(sys.argv[1])
algorithm = sys.argv[2]
designs_file = Path(sys.argv[3])


def count_tier0_inputs(design_dir: Path) -> int:
    if not design_dir.is_dir():
        return 0
    plain_aigs = sum(1 for p in design_dir.glob("*.aig") if p.is_file())
    zipped_aigs = 0
    for zip_path in sorted(design_dir.glob("syn*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zipped_aigs += sum(1 for n in zf.namelist() if n.lower().endswith(".aig"))
        except Exception:
            pass
    return plain_aigs + zipped_aigs


errors = []
checked = 0

for design in designs_file.read_text(encoding="utf-8").splitlines():
    design = design.strip()
    if not design:
        continue
    checked += 1

    summary_path = full_dataset / "metadata" / "raw_logs" / design / "tier1" / algorithm / "summary.json"
    if not summary_path.is_file():
        errors.append(f"missing tier1 summary for gate check: {summary_path}")
        continue

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid summary json {summary_path}: {exc}")
        continue

    discovered = int(payload.get("discovered", -1))
    expected_from_tier0 = count_tier0_inputs(full_dataset / "base_aigs" / design)

    if discovered <= 0:
        errors.append(f"non-positive discovered count in {summary_path}: {discovered}")
        continue

    if expected_from_tier0 != discovered:
        errors.append(
            f"tier0 completeness mismatch for design={design}, algorithm={algorithm}: "
            f"tier0_inputs={expected_from_tier0}, tier1_summary_discovered={discovered}"
        )

if errors:
    print("ERROR: Tier-0 completeness gate failed. Skipping backup zip.")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print(f"✓ Tier-0 completeness gate passed for algorithm={algorithm}")
print(f"✓ Designs checked: {checked}")
PY

echo ""
echo "=========================================="
echo "STEP 7/7: Timestamped FULL_DATASET backup ZIP"
echo "=========================================="

if [ "$ARCHIVE_FULL_DATASET" = "true" ]; then
    mkdir -p "$BACKUP_DIR"

    timestamp="$(date +%Y%m%d_%H%M%S)"
    backup_zip="$BACKUP_DIR/FULL_DATASET_${timestamp}.zip"

    echo "Creating backup zip: $backup_zip"
    python3 - "$FULL_DATASET" "$backup_zip" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()

if not source.is_dir():
    raise SystemExit(f"ERROR: source dataset not found: {source}")

zip_path.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
    for root, _, files in os.walk(source):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            arcname = file_path.relative_to(source.parent)
            zf.write(file_path, arcname=str(arcname))

size_bytes = zip_path.stat().st_size if zip_path.exists() else 0
print(f"✓ backup zip created: {zip_path}")
print(f"✓ backup zip size bytes: {size_bytes}")
PY
else
    echo "ARCHIVE_FULL_DATASET=false, skipping backup creation"
fi

echo ""
echo "=========================================="
echo "JOB 9 Complete"
echo "=========================================="
echo "Algorithm: ${ALGORITHM}"
echo "End time: $(date)"

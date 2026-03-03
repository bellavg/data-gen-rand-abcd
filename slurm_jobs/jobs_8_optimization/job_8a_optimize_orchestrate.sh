#!/bin/bash
#SBATCH --job-name=8a_opt_orch
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=genoa
#SBATCH --output=logs/8a_opt_orchestrate_%j.out

set -euo pipefail

echo "STEP 8a start: job=${SLURM_JOB_ID:-local} host=$(hostname) time=$(date)"

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

ALGORITHM="Orchestrate"
SOURCE_LABEL="base"
OUTPUT_TIER="tier1"
QUIET_OUTPUT="${QUIET_OUTPUT:-true}"

FULL_DATASET="${FULL_DATASET:-/scratch-shared/$USER/FULL_DATASET}"
SCRIPT_ZIP_ROOT="$FULL_DATASET/synScripts/optimization"
MANIFEST_DIR="$FULL_DATASET/optimized_aigs/manifests"

if [ ! -d "$SCRIPT_ZIP_ROOT" ]; then
    echo "✗ ERROR: Missing generated script zip root: $SCRIPT_ZIP_ROOT"
    exit 1
fi

if ! command -v abc >/dev/null 2>&1; then
    echo "✗ ERROR: abc not found in PATH"
    exit 1
fi

latest_manifest=$(ls -1t "$MANIFEST_DIR"/bulk_scripts_manifest_*.json 2>/dev/null | head -n 1 || true)
if [ -z "$latest_manifest" ]; then
    echo "✗ ERROR: No manifest found under: $MANIFEST_DIR"
    exit 1
fi

expected_designs=$(python3 - "$latest_manifest" "$ALGORITHM" <<'PY'
import json
import sys

manifest_path, algorithm = sys.argv[1], sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as fh:
    m = json.load(fh)

if m.get("design_group") != "all":
    raise SystemExit("manifest design_group is not 'all'")
if algorithm not in m.get("algorithms", []):
    raise SystemExit(f"manifest missing algorithm: {algorithm}")
if "base_aigs" not in m.get("input_sources", []):
    raise SystemExit("manifest missing input_source: base_aigs")

designs = m.get("designs", [])
if not designs:
    raise SystemExit("manifest has zero designs")

print(len(designs))
PY
)

script_count=$(find "$SCRIPT_ZIP_ROOT" -type f -name '*.zip' -exec sh -c 'for z in "$@"; do unzip -Z1 "$z" "$0" 2>/dev/null; done' "optimizeBulk_${ALGORITHM}_*_${SOURCE_LABEL}.sh" {} + | wc -l | tr -d ' ')
if [ "$script_count" -ne "$expected_designs" ]; then
    echo "✗ ERROR: Expected $expected_designs ${ALGORITHM} shard scripts, found $script_count"
    exit 1
fi

if [ -n "${PARALLELISM:-}" ]; then
    :
elif [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
    PARALLELISM="$SLURM_CPUS_ON_NODE"
elif [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    PARALLELISM="$SLURM_CPUS_PER_TASK"
elif command -v nproc >/dev/null 2>&1; then
    PARALLELISM=$(nproc)
else
    PARALLELISM=4
fi

processed_designs=0

while IFS= read -r design_zip; do
    design_name="$(basename "$design_zip" .zip)"
    echo "STEP 8a: starting design ${design_name}"

    tmp_extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/opt_orch_${SLURM_JOB_ID:-local}_XXXXXX")"

    if unzip -q "$design_zip" "optimizeBulk_${ALGORITHM}_*_${SOURCE_LABEL}.sh" -d "$tmp_extract_dir" 2>/dev/null; then
        find "$tmp_extract_dir" -type f -name 'optimizeBulk_Orchestrate_*.sh' -exec chmod +x {} +
        mapfile -t script_list < <(find "$tmp_extract_dir" -type f -name 'optimizeBulk_Orchestrate_*.sh' | sort)

        if [ "${#script_list[@]}" -eq 0 ]; then
            rm -rf "$tmp_extract_dir"
            continue
        fi

        if command -v parallel >/dev/null 2>&1; then
            if [ "$QUIET_OUTPUT" = "true" ]; then
                printf "%s\n" "${script_list[@]}" | parallel -j "$PARALLELISM" 'bash {} >/dev/null 2>&1' || {
                    echo "✗ ERROR: ${ALGORITHM} shard execution failed for ${design_name}"
                    exit 1
                }
            else
                printf "%s\n" "${script_list[@]}" | parallel -j "$PARALLELISM" bash {} || {
                    echo "✗ ERROR: ${ALGORITHM} shard execution failed for ${design_name}"
                    exit 1
                }
            fi
        else
            for f in "${script_list[@]}"; do
                if [ "$QUIET_OUTPUT" = "true" ]; then
                    bash "$f" >/dev/null 2>&1 &
                else
                    bash "$f" &
                fi
                while [ "$(jobs -rp | wc -l)" -ge "$PARALLELISM" ]; do sleep 1; done
            done
            wait || {
                echo "✗ ERROR: ${ALGORITHM} shard execution failed for ${design_name}"
                exit 1
            }
        fi

        summary_path="$FULL_DATASET/metadata/raw_logs/${design_name}/${OUTPUT_TIER}/${ALGORITHM}/summary.json"
        if [ ! -f "$summary_path" ]; then
            echo "✗ ERROR: Missing per-design summary: $summary_path"
            exit 1
        fi

        python3 - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)

if int(payload.get("failed", 0)) != 0:
    raise SystemExit(f"failed>0 in {summary_path}")
PY

        processed_designs=$((processed_designs + 1))
        echo "STEP 8a: done design ${design_name}"
    fi

    rm -rf "$tmp_extract_dir"
done < <(find "$SCRIPT_ZIP_ROOT" -type f -name '*.zip' | sort)

if [ "$processed_designs" -ne "$expected_designs" ]; then
    echo "✗ ERROR: Expected to process $expected_designs designs, processed $processed_designs"
    exit 1
fi

echo "STEP 8a complete: verified=${processed_designs} time=$(date)"

#!/bin/bash
#SBATCH --job-name=4d_opt_C2RS
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --constraint=scratch-node
#SBATCH --array=0-54
#SBATCH --output=logs/opt_c2rs/4d_opt_C2RS_%A_%a.out

set -euo pipefail

if [[ -z "${TMPDIR:-}" ]]; then
    export TMPDIR="/scratch-shared/$USER/tmp"
fi
mkdir -p "$TMPDIR"

echo "STEP 4d: start job=${SLURM_JOB_ID:-local} host=$(hostname) time=$(date)"

module purge
module load 2025
module load foss/2025a
module load Python/3.13.1-GCCcore-14.2.0
export PATH="$HOME/abc:$PATH"

# ==========================================
# ALGORITHM SELECTION 
# ==========================================
ALGORITHM="C2RS"

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

DESIGN="${DESIGNS[$SLURM_ARRAY_TASK_ID]}"

# Paths
BASE_DIR="$HOME/data-gen-rand-abcd"
PERM_TIER0_DIR="$BASE_DIR/data/designs/$DESIGN"
PERM_TIER1_DIR="$BASE_DIR/data/designs/$DESIGN/tier1"
PERM_LOG_DIR="$BASE_DIR/data/designs/$DESIGN/design_metadata/raw_logs/optimization_logs/tier1/${ALGORITHM}"

# Scratch setup
JOB_SCRATCH="$(mktemp -d "$TMPDIR/tier1_${ALGORITHM}_${DESIGN}_XXXXXX")"
trap 'rm -rf "$JOB_SCRATCH"' EXIT

SCRATCH_IN="$JOB_SCRATCH/in"
SCRATCH_OUT="$JOB_SCRATCH/out"
SCRATCH_LOGS="$JOB_SCRATCH/logs"
SCRATCH_SCRIPTS="$JOB_SCRATCH/scripts"

mkdir -p "$SCRATCH_IN" "$SCRATCH_OUT" "$SCRATCH_LOGS" "$SCRATCH_SCRIPTS" "$PERM_LOG_DIR" "$PERM_TIER1_DIR"

echo "=================================================="
echo " Starting Tier-1 [${ALGORITHM}] for: $DESIGN"
echo " Cores: $SLURM_CPUS_PER_TASK | Scratch: $JOB_SCRATCH"
echo "=================================================="

# 1. GENERATE SCRIPTS ON THE FLY
echo ">>> Generating runner scripts for $DESIGN..."
python3 "$BASE_DIR/data/creation/automate_bulkOptimization.py" --home "$BASE_DIR" --design "$DESIGN"
unzip -q "$BASE_DIR/data/abc_scripts/optimization_scripts/${DESIGN}.zip" -d "$SCRATCH_SCRIPTS"
chmod +x "$SCRATCH_SCRIPTS/$DESIGN/"*.sh

# 2. STAGE AIGS
echo ">>> Staging inputs to local scratch..."
if [ -f "$PERM_TIER0_DIR/tier0.zip" ]; then
    unzip -q "$PERM_TIER0_DIR/tier0.zip" -d "$SCRATCH_IN"
elif [ -f "$PERM_TIER0_DIR/tier0/tier0.zip" ]; then
    unzip -q "$PERM_TIER0_DIR/tier0/tier0.zip" -d "$SCRATCH_IN"
fi

# Flatten any subfolders hidden inside the zip
find "$SCRATCH_IN" -mindepth 2 -name "*.aig" -exec mv -t "$SCRATCH_IN" {} +
find "$SCRATCH_IN" -mindepth 1 -type d -delete 2>/dev/null || true

INPUT_COUNT=$(find "$SCRATCH_IN" -maxdepth 1 -name "*.aig" | wc -l)
if [[ "$INPUT_COUNT" -eq 0 ]]; then
    echo "✗ ERROR: No input AIGs found for $DESIGN in $SCRATCH_IN!" >&2
    exit 1
fi

PERM_AIG_ZIP="$PERM_TIER1_DIR/${DESIGN}_${ALGORITHM}.zip"
PERM_LOG_ZIP="$PERM_LOG_DIR/optimize_${ALGORITHM}_${DESIGN}.zip"

# 3. EXECUTE ALGORITHM
echo ">>> Executing ${ALGORITHM} on all 192 cores..."
bash "$SCRATCH_SCRIPTS/$DESIGN/${ALGORITHM}.sh" "$SCRATCH_IN" "$SCRATCH_OUT" "$SCRATCH_LOGS" 192

# 4. VALIDATE AND ZIP OUT
echo ">>> Validating exact AIG count and Zipping..."
if ! python3 - "$SCRATCH_OUT" "$PERM_AIG_ZIP" "$INPUT_COUNT" "$ALGORITHM" <<'PY'
import sys
import zipfile
from pathlib import Path

out_dir = Path(sys.argv[1])
zip_dest = Path(sys.argv[2])
expected_count = int(sys.argv[3])
algo = sys.argv[4]

aig_files = list(out_dir.glob("*.aig"))
actual_count = len(aig_files)

if actual_count != expected_count:
    print(f"✗ ERROR [{algo}]: Count mismatch! Expected {expected_count}, got {actual_count}", file=sys.stderr)
    sys.exit(1)

zip_dest.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
    for aig in aig_files:
        zf.write(aig, arcname=aig.name)
sys.exit(0)
PY
then
    echo "✗ ERROR: Validation failed for ${ALGORITHM}. Halting." >&2
    exit 1
fi

echo ">>> Zipping Logs directly to permanent storage..."
(cd "$SCRATCH_LOGS" && zip -q -r "$PERM_LOG_ZIP" .)

echo ">>> Tier-1 [${ALGORITHM}] Complete for $DESIGN at $(date)"
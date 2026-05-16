#!/bin/bash
#SBATCH --job-name=venv
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --output=logs/rebuild_venv_%j.out



# Rebuild the scratch-shared venv from scratch using pyproject.toml.
# Run this on an interactive/login node after a HPC software update breaks the venv.
#
# Usage:
#   bash src/shell/rebuild_venv.sh
#
# Override the venv path:
#   VENV_PATH=/my/path bash src/shell/rebuild_venv.sh

set -euo pipefail

VENV_PATH="${VENV_PATH:-/scratch-shared/$USER/.venv}"
BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
TORCH_VERSION="2.11.0"
CUDA_TAG="cu130"   # matches nvidia-cuda-runtime 13.0.x on Snellius H100 nodes
PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "Rebuilding venv at: $VENV_PATH"
echo "PyG wheel index:    $PYG_WHEEL_INDEX"

# 1. Load the same modules used at runtime
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# 2. Remove old venv
if [[ -d "$VENV_PATH" ]]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_PATH"
fi

# 3. Create fresh venv
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# 4. Upgrade pip/setuptools first to avoid RECORD-file issues
pip install --upgrade pip setuptools wheel

# 5. Install torch (with bundled CUDA 13 wheels)
echo "Installing torch $TORCH_VERSION..."
pip install "torch==$TORCH_VERSION"

# 6. Install torch-scatter from the PyG prebuilt index (must come before -e install
#    so the pyg optional dep resolves correctly)
echo "Installing torch-scatter from PyG index..."
pip install "torch-scatter>=2.1" --find-links "$PYG_WHEEL_INDEX"

# 7. Install the project and all core deps from pyproject.toml
echo "Installing project deps from pyproject.toml..."
pip install -e "$BASE_DIR" --find-links "$PYG_WHEEL_INDEX"

# 8. Verify key imports
echo "Verifying imports..."
python -c "
import torch, torch_geometric, pytorch_lightning, optuna
from optuna.storages import RDBStorage
print(f'torch={torch.__version__}')
print(f'torch_geometric={torch_geometric.__version__}')
print(f'pytorch_lightning={pytorch_lightning.__version__}')
print(f'optuna={optuna.__version__}')
print('RDBStorage: ok')
print()
print('Rebuild successful.')
"

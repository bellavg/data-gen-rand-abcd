#!/bin/bash
#SBATCH --job-name=conda312
#SBATCH --time=00:45:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --output=logs/rebuild_conda_py312_%j.out

# Build a fresh conda environment for this project on Python 3.12.
# Usage:
#   bash src/shell/rebuild_conda_py312.sh
#
# Overrides:
#   CONDA_MODULE=Anaconda3/2025.06-1
#   CONDA_ENV_PREFIX=/scratch-shared/$USER/.conda/envs/data-gen-py312
#   PYTHON_VERSION=3.12
#   TORCH_VERSION=2.11.0
#   CUDA_TAG=cu130
#   RECREATE=true

set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/data-gen-rand-abcd}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3/2025.06-1}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch-shared/$USER/.conda/envs/data-gen-py312}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
CUDA_TAG="${CUDA_TAG:-cu130}"
RECREATE="${RECREATE:-true}"
PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "Building conda env at: $CONDA_ENV_PREFIX"
echo "Conda module:          $CONDA_MODULE"
echo "Python version:        $PYTHON_VERSION"
echo "Torch version:         $TORCH_VERSION"
echo "PyG wheel index:       $PYG_WHEEL_INDEX"

module purge
module load 2025
module load "$CONDA_MODULE"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -n "${EBROOTANACONDA3:-}" && -f "${EBROOTANACONDA3}/etc/profile.d/conda.sh" ]]; then
    source "${EBROOTANACONDA3}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda command not found after loading $CONDA_MODULE" >&2
    exit 1
fi

if [[ "$RECREATE" == "true" && -d "$CONDA_ENV_PREFIX" ]]; then
    echo "Removing existing env..."
    rm -rf "$CONDA_ENV_PREFIX"
fi

if [[ ! -d "$CONDA_ENV_PREFIX" ]]; then
    conda create -y -p "$CONDA_ENV_PREFIX" "python=${PYTHON_VERSION}" pip
fi

conda activate "$CONDA_ENV_PREFIX"
python --version

pip install --upgrade pip setuptools wheel

# Install torch first so PyG extension wheels resolve against the target torch build.
pip install "torch==${TORCH_VERSION}"

# Install torch-scatter from the PyG index before project install.
pip install "torch-scatter>=2.1" --find-links "$PYG_WHEEL_INDEX"

# Install project and dependencies from pyproject.toml
pip install -e "$BASE_DIR" --find-links "$PYG_WHEEL_INDEX"

python -c "
import torch, torch_geometric, pytorch_lightning, optuna
print(f'torch={torch.__version__}')
print(f'torch_geometric={torch_geometric.__version__}')
print(f'pytorch_lightning={pytorch_lightning.__version__}')
print(f'optuna={optuna.__version__}')
print('Conda py312 rebuild successful')
"

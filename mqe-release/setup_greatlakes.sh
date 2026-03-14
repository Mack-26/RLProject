#!/bin/bash
# setup_greatlakes.sh
# Run this ONCE on a Great Lakes login node to create the conda environment.
# Usage: bash setup_greatlakes.sh

set -e

ENV_NAME="mqe"

echo "=== Loading modules ==="
module load anaconda3/2023.09   # provides conda; adjust version if needed
module load cuda/12.1.1
module load cudnn/12.1-v8.9.0

# Initialize conda for this shell session
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

echo "=== Installing JAX (CUDA 12) ==="
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== Installing OGBench (local modified version from repo) ==="
# The repo ships its own ogbench/ with the custom antmaze-colossal environments.
# Install it in editable mode from the repo root — do NOT use 'pip install ogbench' from PyPI.
cd "$(dirname "$0")"
pip install -e ".[train]"   # installs ogbench + jax[cuda12] + flax + distrax + wandb etc.

echo "=== Installing remaining dependencies ==="
cd "$(dirname "$0")/impls"
pip install absl-py tqdm

echo "=== Verifying JAX sees GPU ==="
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Then submit jobs with: bash submit_visual_cube_triple_play.sh"

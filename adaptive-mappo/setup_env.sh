#!/bin/bash
# Run once on Great Lakes to create the benchmarl conda environment.
# Usage: bash setup_env.sh

set -e

module load python3.10-anaconda/2023.03

conda create -n benchmarl python=3.10 -y
source ~/.bashrc

PIP=/home/aromanan/.conda/envs/benchmarl/bin/pip

echo "Installing PyTorch (CUDA 12.1, works on V100)..."
$PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing TorchRL / TensorDict..."
$PIP install torchrl tensordict

echo "Installing BenchMARL + VMAS..."
$PIP install benchmarl vmas

echo "Done. Activate with: conda activate benchmarl"
echo "Test with: /home/aromanan/.conda/envs/benchmarl/bin/python -c 'import benchmarl; import vmas; print(\"OK\")'"

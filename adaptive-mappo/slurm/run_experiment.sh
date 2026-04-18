#!/bin/bash
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --qos=class
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/home/aromanan/RLProject/logs/adaptive_mappo_%A_%a.out
#SBATCH --error=/home/aromanan/RLProject/logs/adaptive_mappo_%A_%a.err
#SBATCH --array=0-3

# -----------------------------------------------------------------------
# Usage:
#   sbatch run_experiment.sh                        # adaptive, navigation
#   ALGO=scheduled sbatch run_experiment.sh
#   ALGO=adaptive FEATURES=global,local,disagree sbatch run_experiment.sh
#   ALGO=mappo ENV=wheel sbatch run_experiment.sh
#
# SLURM_ARRAY_TASK_ID is used as the random seed (0-3 for 4 seeds).
# -----------------------------------------------------------------------

ALGO=${ALGO:-adaptive}
ENV=${ENV:-navigation}
FEATURES=${FEATURES:-}
ALPHA_SCHEDULE=${ALPHA_SCHEDULE:-linear}
ALPHA_START=${ALPHA_START:-1.0}
ALPHA_END=${ALPHA_END:-0.0}

SEED=${SLURM_ARRAY_TASK_ID}

# Absolute paths — $(dirname $0) resolves to SLURM spool, so hardcode
REPO_DIR="$HOME/RLProject/adaptive-mappo"
PYTHON=/home/aromanan/.conda/envs/benchmarl/bin/python

mkdir -p "$HOME/RLProject/logs"

echo "=== Job info ==="
echo "ALGO=$ALGO  ENV=$ENV  SEED=$SEED"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  ARRAY_TASK=$SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES"
echo "================"

cd "$REPO_DIR"

EXTRA_ARGS=""
if [ -n "$FEATURES" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --features $FEATURES"
fi
if [ "$ALGO" = "scheduled" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --alpha_schedule $ALPHA_SCHEDULE --alpha_start $ALPHA_START --alpha_end $ALPHA_END"
fi

$PYTHON train.py \
    --algo "$ALGO" \
    --env "$ENV" \
    --seed "$SEED" \
    $EXTRA_ARGS

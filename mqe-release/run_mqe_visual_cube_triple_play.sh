#!/bin/bash
# run_mqe_visual_cube_triple_play.sh
# SLURM job script for MQE on visual-cube-triple-play-v0.
# Reproduces Table 4 of "Multistep Quasimetric Learning for Scalable GCRL" (ICLR 2026).
# Expected result: 19.8 (±0.9)% success rate (averaged over 4 seeds).
#
# Submit all 4 seeds at once:
#   sbatch --array=0-3 run_mqe_visual_cube_triple_play.sh
# Or submit a single seed:
#   sbatch --export=SEED=0 run_mqe_visual_cube_triple_play.sh

#SBATCH --job-name=mqe_visual_cube_triple
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --qos=class
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00                 # class account MaxWall=8h; 7.5h gives buffer
#SBATCH --output=/home/aromanan/RLProject/logs/mqe_vct_%A_%a.out
#SBATCH --error=/home/aromanan/RLProject/logs/mqe_vct_%A_%a.err
#SBATCH --array=0-3                     # 4 seeds, matching paper's pixel-based eval

# ── Seed: use SLURM array index if available, else fall back to $SEED ──────────
SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}

echo "=== Job info ==="
echo "  SLURM_JOB_ID : $SLURM_JOB_ID"
echo "  SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID"
echo "  Seed : $SEED"
echo "  Node : $(hostname)"
echo "  GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Environment ─────────────────────────────────────────────────────────────────
module load python3.10-anaconda/2023.03
module load cuda/12.1.1

PYTHON="/home/aromanan/.conda/envs/mqe/bin/python"

# EGL rendering for MuJoCo (headless GPU rendering)
export MUJOCO_GL=egl
export EGL_DEVICE_ID=${SLURM_STEP_GPUS:-0}

# ── Paths ───────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/RLProject/mqe-release"
IMPLS_DIR="$REPO_DIR/impls"
mkdir -p "$HOME/RLProject/logs"

# Path to the pre-downloaded dataset (run download_dataset.py first).
# Default: ~/ogbench_data/visual-cube-triple-play-v0.npz
DATASET_PATH="${DATASET_PATH:-$HOME/ogbench_data/visual-cube-triple-play-v0.npz}"

if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset not found at $DATASET_PATH"
    echo "Run: python $REPO_DIR/download_dataset.py --dataset_dir $HOME/ogbench_data"
    exit 1
fi
echo "  Dataset : $DATASET_PATH"

# ── Hyperparameters (from paper Table 2 & 3, pixel-based visual-cube environments)
# γ  = 0.995, λ = 0.95, p (next_state_sample) = 0.1, α = 3.0, p_aug = 0.5
# encoder: impala_small, latent_dim: 512, batch_size: 256, train_steps: 500k
# eval_episodes: 50 (matches paper), eval_on_cpu: 0 (GPU eval)
# ────────────────────────────────────────────────────────────────────────────────

cd "$IMPLS_DIR"

$PYTHON main.py \
    --run_group="visual_cube_triple_play_reproduce" \
    --seed=$SEED \
    --env_name=visual-cube-triple-play-v0 \
    --dataset_path="$DATASET_PATH" \
    --train_steps=500000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --eval_episodes=50 \
    --eval_on_cpu=0 \
    --video_episodes=0 \
    --agent=agents/mqe.py \
    --agent.batch_size=256 \
    --agent.encoder=impala_small \
    --agent.latent_dim=512 \
    --agent.actor_hidden_dims="(512,512,512)" \
    --agent.value_hidden_dims="(512,512,512)" \
    --agent.layer_norm=True \
    --agent.discount=0.995 \
    --agent.lambda_=0.95 \
    --agent.next_state_sample=0.1 \
    --agent.alpha=3.0 \
    --agent.p_aug=0.5 \
    --agent.components=8 \
    --agent.diag_backup=0.5 \
    --agent.normalize_q_loss=True \
    --agent.const_std=True

echo "=== Job finished (seed=$SEED) ==="

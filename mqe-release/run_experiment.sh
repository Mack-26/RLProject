#!/bin/bash
# run_experiment.sh — unified SLURM job script for MQE/HIQL on OGBench visual environments.
#
# Submit via submit_jobs.sh, or manually:
#   sbatch --export=AGENT=mqe,ENV=visual-cube-triple-play-v0,SEED=0 run_experiment.sh
#
# AGENT: mqe | hiql
# ENV:   visual-cube-triple-play-v0 | visual-scene-play-v0
# SEED:  0 | 1 | 2 | 3

#SBATCH --job-name=rl_exp
#SBATCH --account=<YOUR_ACCOUNT>        # change this
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aromanan/RLProject/mqe-release/logs/exp_%j.out
#SBATCH --error=/home/aromanan/RLProject/mqe-release/logs/exp_%j.err

# ── Defaults (overridden by --export in sbatch) ─────────────────────────────
AGENT=${AGENT:-mqe}
ENV=${ENV:-visual-cube-triple-play-v0}
SEED=${SEED:-0}

# ── Derived names ────────────────────────────────────────────────────────────
case "$ENV" in
    visual-cube-triple-play-v0) ENV_SHORT="visual_cube_triple" ;;
    visual-scene-play-v0)       ENV_SHORT="visual_scene_play"  ;;
    *) ENV_SHORT=$(echo "$ENV" | tr '-' '_' | sed 's/_v0//') ;;
esac
RUN_GROUP="${AGENT}_${ENV_SHORT}"

echo "=== Job info ==="
echo "  AGENT    : $AGENT"
echo "  ENV      : $ENV"
echo "  SEED     : $SEED"
echo "  RUN_GROUP: $RUN_GROUP"
echo "  JOB_ID   : $SLURM_JOB_ID"
echo "  NODE     : $(hostname)"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11.5
module load cuda/12.1.1
module load cudnn/12.1-v8.9.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mqe

export MUJOCO_GL=egl
export EGL_DEVICE_ID=${SLURM_STEP_GPUS:-0}

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/RLProject/mqe-release"
IMPLS_DIR="$REPO_DIR/impls"
mkdir -p "$REPO_DIR/logs"

DATASET_PATH="${DATASET_PATH:-$HOME/ogbench_data/${ENV}.npz}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run: python $REPO_DIR/download_dataset.py --dataset_dir $HOME/ogbench_data"
    exit 1
fi
echo "  Dataset  : $DATASET_PATH"

# ── Checkpoint resume ─────────────────────────────────────────────────────────
# If a previous run for this (agent, env, seed) exists with saved checkpoints,
# resume from the latest one.
EXP_BASE="$IMPLS_DIR/exp/OGBench/$RUN_GROUP"
SEED_PADDED=$(printf '%03d' $SEED)
EXISTING_DIR=$(ls -d ${EXP_BASE}/sd${SEED_PADDED}_* 2>/dev/null | head -1)

RESTORE_ARGS=""
TOTAL_STEPS=500000
TRAIN_STEPS=$TOTAL_STEPS

if [ -n "$EXISTING_DIR" ]; then
    LATEST_PKL=$(ls "$EXISTING_DIR"/params_*.pkl 2>/dev/null \
        | sed 's/.*params_\([0-9]*\)\.pkl/\1 &/' \
        | sort -n | tail -1 | awk '{print $2}')
    if [ -n "$LATEST_PKL" ]; then
        EPOCH=$(basename "$LATEST_PKL" .pkl | sed 's/params_//')
        REMAINING=$((TOTAL_STEPS - EPOCH))
        if [ "$REMAINING" -le 0 ]; then
            echo "  Already complete at step $EPOCH — nothing to do."
            exit 0
        fi
        # main.py's loop always runs from 1..train_steps regardless of restore epoch,
        # so pass only the REMAINING steps to avoid over-training.
        TRAIN_STEPS=$REMAINING
        RESTORE_ARGS="--restore_path=$EXISTING_DIR --restore_epoch=$EPOCH"
        echo "  Resuming : $LATEST_PKL (step $EPOCH, $REMAINING steps remaining)"
    else
        echo "  Found old run dir but no checkpoints — starting fresh"
    fi
fi

# ── Agent-specific hyperparameters ───────────────────────────────────────────
case "$AGENT" in
    mqe)
        AGENT_FLAGS="--agent=agents/mqe.py \
            --agent.batch_size=256 \
            --agent.encoder=impala_small \
            --agent.latent_dim=512 \
            --agent.discount=0.995 \
            --agent.lambda_=0.95 \
            --agent.next_state_sample=0.1 \
            --agent.alpha=3.0 \
            --agent.p_aug=0.5 \
            --agent.components=8 \
            --agent.diag_backup=0.5 \
            --agent.normalize_q_loss=True \
            --agent.const_std=True"
        ;;
    hiql)
        AGENT_FLAGS="--agent=agents/hiql.py \
            --agent.batch_size=256 \
            --agent.encoder=impala_small \
            --agent.high_alpha=3.0 \
            --agent.low_alpha=3.0 \
            --agent.low_actor_rep_grad=True \
            --agent.subgoal_steps=10 \
            --agent.p_aug=0.5"
        ;;
    *)
        echo "ERROR: Unknown AGENT=$AGENT (must be mqe or hiql)"
        exit 1
        ;;
esac

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$IMPLS_DIR"

python "$IMPLS_DIR/main.py" \
    --run_group="$RUN_GROUP" \
    --seed=$SEED \
    --env_name="$ENV" \
    --dataset_path="$DATASET_PATH" \
    --train_steps=$TRAIN_STEPS \
    --save_interval=100000 \
    --log_interval=5000 \
    --eval_interval=100000 \
    --eval_episodes=50 \
    --eval_on_cpu=0 \
    --video_episodes=0 \
    $AGENT_FLAGS \
    $RESTORE_ARGS

echo "=== Done (AGENT=$AGENT ENV=$ENV SEED=$SEED) ==="

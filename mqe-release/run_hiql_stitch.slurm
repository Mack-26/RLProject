#!/bin/bash
#SBATCH --job-name=hiql_stitch
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch/ece567w26_class_root/ece567w26_class/adityg/logs/hiql_stitch_%A_%a.out
#SBATCH --error=/scratch/ece567w26_class_root/ece567w26_class/adityg/logs/hiql_stitch_%A_%a.err

set -euo pipefail

cd /home/adityg/mqe-release
module purge
module load python/3.10.4
source mqe-env/bin/activate
cd impls

unset JAX_PLATFORMS
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SAVE_ROOT=/scratch/ece567w26_class_root/ece567w26_class/adityg/mqe_runs/exp_hiql_stitch
LOG_ROOT=/scratch/ece567w26_class_root/ece567w26_class/adityg/logs

ENVS=(
  antmaze-medium-stitch-v0
  antmaze-large-stitch-v0
  antmaze-giant-stitch-v0
)

RUNS=(
  hiql_medium_stitch
  hiql_large_stitch
  hiql_giant_stitch
)

NUM_ENVS=${#ENVS[@]}
NUM_SEEDS=8

IDX=${SLURM_ARRAY_TASK_ID}
ENV_ID=$((IDX / NUM_SEEDS))
SEED=$((IDX % NUM_SEEDS))

if [ "${ENV_ID}" -ge "${NUM_ENVS}" ]; then
  echo "Invalid ENV_ID=${ENV_ID} for IDX=${IDX}"
  exit 1
fi

ENV_NAME=${ENVS[$ENV_ID]}
RUN_NAME=${RUNS[$ENV_ID]}

EXTRA_ARGS="--agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5"
if [ "$ENV_NAME" = "antmaze-giant-stitch-v0" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --agent.discount=0.995"
fi

echo "======================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "ENV_ID=${ENV_ID}"
echo "SEED=${SEED}"
echo "Starting HIQL on ${ENV_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Save dir: ${SAVE_ROOT}/${RUN_NAME}_seed${SEED}"
echo "======================================"

python main.py \
  --seed="${SEED}" \
  --eval_episodes=50 \
  --eval_on_cpu=0 \
  --env_name="${ENV_NAME}" \
  --agent=agents/hiql.py \
  --train_steps=1000000 \
  --eval_interval=100000 \
  --log_interval=1000 \
  --save_interval=1000000 \
  --save_dir="${SAVE_ROOT}/${RUN_NAME}_seed${SEED}" \
  --agent.high_alpha=3.0 \
  --agent.low_alpha=3.0 \
  ${EXTRA_ARGS}

echo "Finished HIQL on ${ENV_NAME} with seed ${SEED}"
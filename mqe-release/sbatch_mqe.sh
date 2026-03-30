#!/bin/bash
#SBATCH --job-name=ogbench_mqe
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --output=slurm_log/%x-%j.log

TRAIN_STEPS=1000000
EVAL_START=700000
BATCH_SIZE=256
ENVS=("cube-double-play-v0" "scene-play-v0" "pointmaze-giant-navigate-v0")

for ENV in "${ENVS[@]}"; do
    # Determine environment-specific MQE hyperparameters
    if [ "$ENV" == "pointmaze-giant-navigate-v0" ]; then
        MQE_FLAGS="--agent.diag_backup=0.3"
    else
        MQE_FLAGS="--agent.discount=0.99"
    fi

    LOG_FILE="logs/LOG_SEQ_${ENV}_MQE_SEED0.log"

    # Check if experiment is already complete
    if [ -f "$LOG_FILE" ] && grep -q "### EXPERIMENT_COMPLETE ###" "$LOG_FILE"; then
        echo ">>> Skipping MQE (Seed 0) for $ENV (Already completed)"
        continue
    fi

    echo ">>> Starting MQE (Seed 0) for $ENV"
    python main.py \
      --env_name=$ENV \
      --seed=0 \
      --agent=agents/mqe.py \
      $MQE_FLAGS \
      --train_steps=$TRAIN_STEPS \
      --eval_start=$EVAL_START \
      --eval_episodes=50 \
      --agent.batch_size=$BATCH_SIZE \
      --run_group="Sequential_${ENV}_MQE" \
      --dataset_path=../data/$ENV.npz > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "### EXPERIMENT_COMPLETE ###" >> "$LOG_FILE"
        echo ">>> Finished MQE (Seed 0) for $ENV"
    else
        echo ">>> ERROR: MQE (Seed 0) for $ENV failed!"
        exit 1
    fi
done

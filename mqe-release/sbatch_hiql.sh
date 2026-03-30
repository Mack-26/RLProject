#!/bin/bash
#SBATCH --job-name=ogbench_hiql
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
    # Environment-specific flags for HIQL
    if [ "$ENV" == "pointmaze-giant-navigate-v0" ]; then
        HIQL_FLAGS="--agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0"
    else
        HIQL_FLAGS="--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10"
    fi

    for SEED in {0..7}; do
        LOG_FILE="logs/LOG_SEQ_${ENV}_HIQL_SEED${SEED}.log"

        # Check if experiment is already complete
        if [ -f "$LOG_FILE" ] && grep -q "### EXPERIMENT_COMPLETE ###" "$LOG_FILE"; then
            echo ">>> Skipping HIQL (Seed $SEED) for $ENV (Already completed)"
            continue
        fi

        echo ">>> Starting HIQL (Seed $SEED) for $ENV"
        python main.py \
          --env_name=$ENV \
          --seed=$SEED \
          --agent=agents/hiql.py \
          $HIQL_FLAGS \
          --train_steps=$TRAIN_STEPS \
          --eval_start=$EVAL_START \
          --eval_episodes=50 \
          --agent.batch_size=$BATCH_SIZE \
          --run_group="Sequential_${ENV}_HIQL" \
          --dataset_path=../data/$ENV.npz > "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "### EXPERIMENT_COMPLETE ###" >> "$LOG_FILE"
            echo ">>> Finished HIQL (Seed $SEED) for $ENV"
        else
            echo ">>> ERROR: HIQL (Seed $SEED) for $ENV failed!"
            exit 1
        fi
    done
done

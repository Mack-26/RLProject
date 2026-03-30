#!/bin/bash
#SBATCH --job-name=ogbench_baselines
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
ENVS=("pointmaze-giant-navigate-v0" "cube-double-play-v0" "scene-play-v0" "scene-noisy-v0")

# Download data using absolute path (only if missing)
DATA_DIR="/home/minsukc/MQE/data"
mkdir -p "$DATA_DIR"
for ENV in "${ENVS[@]}"; do
    if [ ! -f "$DATA_DIR/$ENV.npz" ]; then
        echo ">>> Dataset $ENV not found, downloading..."
        python -c "from ogbench.utils import download_datasets; download_datasets(['$ENV'], '$DATA_DIR')"
    else
        echo ">>> Dataset $ENV already exists, skipping download check."
    fi
done

for ENV in "${ENVS[@]}"; do
    # Environment-specific agents and flags (consistent with hyperparameters.sh)
    if [ "$ENV" == "pointmaze-giant-navigate-v0" ]; then
        AGENT="agents/qrl.py"
        AGENT_NAME="QRL"
        AGENT_FLAGS="--agent.alpha=0.0003 --agent.discount=0.995"
    else
        AGENT="agents/gciql.py"
        AGENT_NAME="GCIQL"
        if [ "$ENV" == "scene-noisy-v0" ]; then
            AGENT_FLAGS="--agent.alpha=0.03"
        else
            AGENT_FLAGS="--agent.alpha=1.0"
        fi
    fi

    for SEED in {0..7}; do
        LOG_FILE="logs/LOG_SEQ_${ENV}_${AGENT_NAME}_SEED${SEED}.log"

        # Check if experiment is already complete (Resume logic)
        if [ -f "$LOG_FILE" ] && grep -q "### EXPERIMENT_COMPLETE ###" "$LOG_FILE"; then
            echo ">>> Skipping $AGENT_NAME (Seed $SEED) for $ENV (Already completed)"
            continue
        fi

        echo ">>> Starting $AGENT_NAME (Seed $SEED) for $ENV"
        python main.py \
          --env_name=$ENV \
          --seed=$SEED \
          --agent=$AGENT \
          $AGENT_FLAGS \
          --train_steps=$TRAIN_STEPS \
          --eval_start=$EVAL_START \
          --eval_episodes=50 \
          --agent.batch_size=$BATCH_SIZE \
          --run_group="Sequential_${ENV}_${AGENT_NAME}" \
          --dataset_path=$DATA_DIR/$ENV.npz > "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "### EXPERIMENT_COMPLETE ###" >> "$LOG_FILE"
            echo ">>> Finished $AGENT_NAME (Seed $SEED) for $ENV"
        else
            echo ">>> ERROR: $AGENT_NAME (Seed $SEED) for $ENV failed!"
            exit 1
        fi
    done
done

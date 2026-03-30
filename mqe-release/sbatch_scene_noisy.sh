#!/bin/bash
#SBATCH --job-name=scene_noisy_all
#SBATCH --account=ece567w26_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --output=/home/minsukc/MQE/impls/slurm_log/%x-%j.log

TRAIN_STEPS=1000000
EVAL_START=700000
BATCH_SIZE=256
ENV="scene-noisy-v0"

# ---------------------------------------------------------
# 1. Run MQE (Seed 0 Only)
# ---------------------------------------------------------
MQE_LOG="logs/LOG_SEQ_${ENV}_MQE_SEED0.log"
MQE_FLAGS="--agent.discount=0.99"

if [ -f "$MQE_LOG" ] && grep -q "### EXPERIMENT_COMPLETE ###" "$MQE_LOG"; then
    echo ">>> Skipping MQE (Seed 0) for $ENV (Already completed)"
else
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
      --dataset_path=../data/$ENV.npz > "$MQE_LOG" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "### EXPERIMENT_COMPLETE ###" >> "$MQE_LOG"
        echo ">>> Finished MQE (Seed 0) for $ENV"
    else
        echo ">>> ERROR: MQE (Seed 0) for $ENV failed!"
        exit 1
    fi
fi

# ---------------------------------------------------------
# 2. Run HIQL (Seeds 0 through 7)
# ---------------------------------------------------------
HIQL_FLAGS="--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=10"

for SEED in {0..7}; do
    HIQL_LOG="logs/LOG_SEQ_${ENV}_HIQL_SEED${SEED}.log"

    if [ -f "$HIQL_LOG" ] && grep -q "### EXPERIMENT_COMPLETE ###" "$HIQL_LOG"; then
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
      --dataset_path=../data/$ENV.npz > "$HIQL_LOG" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "### EXPERIMENT_COMPLETE ###" >> "$HIQL_LOG"
        echo ">>> Finished HIQL (Seed $SEED) for $ENV"
    else
        echo ">>> ERROR: HIQL (Seed $SEED) for $ENV failed!"
        exit 1
    fi
done

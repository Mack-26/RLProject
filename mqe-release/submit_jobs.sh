#!/bin/bash
# submit_jobs.sh — submit MQE and HIQL jobs for both visual environments.
#
# Usage:
#   bash submit_jobs.sh pilot   # seed 0 only — validate setup (4 jobs)
#   bash submit_jobs.sh full    # all 4 seeds (16 jobs total)

MODE=${1:-pilot}

AGENTS=(mqe hiql)
ENVS=(visual-cube-triple-play-v0 visual-scene-play-v0)

case "$MODE" in
    pilot) SEEDS=(0) ;;
    full)  SEEDS=(0 1 2 3) ;;
    *)
        echo "Usage: bash submit_jobs.sh [pilot|full]"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

echo "=== Submitting $MODE jobs ==="
echo "  Agents : ${AGENTS[*]}"
echo "  Envs   : ${ENVS[*]}"
echo "  Seeds  : ${SEEDS[*]}"
echo ""

for AGENT in "${AGENTS[@]}"; do
    for ENV in "${ENVS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB=$(sbatch \
                --export=AGENT=$AGENT,ENV=$ENV,SEED=$SEED \
                --job-name="${AGENT}_${SEED}" \
                "$SCRIPT_DIR/run_experiment.sh")
            echo "  Submitted: AGENT=$AGENT ENV=$ENV SEED=$SEED → $JOB"
        done
    done
done

echo ""
echo "Monitor with: squeue -u \$USER"

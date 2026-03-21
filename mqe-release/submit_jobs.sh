#!/bin/bash
# submit_jobs.sh — submit MQE and HIQL jobs for both visual environments.
#
# Usage:
#   bash submit_jobs.sh pilot   # seed 0 only — validate setup
#   bash submit_jobs.sh full    # all 4 seeds
#
# Skip specific combos by editing SKIP below (format: "agent:env"):
SKIP=(
    "mqe:visual-cube-triple-play-v0"   # already done
)

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
        # Check if this combo is in the skip list
        SKIP_THIS=0
        for S in "${SKIP[@]}"; do
            if [ "$S" = "${AGENT}:${ENV}" ]; then
                SKIP_THIS=1
                break
            fi
        done
        if [ "$SKIP_THIS" -eq 1 ]; then
            echo "  Skipping : AGENT=$AGENT ENV=$ENV (in SKIP list)"
            continue
        fi

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

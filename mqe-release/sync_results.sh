#!/bin/bash
# sync_results.sh — pull eval/train CSVs from Great Lakes to local machine.
# Run this FROM YOUR LOCAL MAC (not from the server).
#
# Usage:
#   bash sync_results.sh              # dry run (shows what would be copied)
#   bash sync_results.sh --apply      # actually copy

REMOTE_USER="aromanan"
REMOTE_HOST="greatlakes-xfer.arc-ts.umich.edu"
REMOTE_DIR="/home/aromanan/RLProject/mqe-release/impls/exp"
LOCAL_DIR="$(cd "$(dirname "$0")/impls/exp" && pwd)"   # mirrors server structure locally

DRY=""
if [ "${1}" != "--apply" ]; then
    DRY="--dry-run"
    echo "=== DRY RUN — pass --apply to actually sync ==="
fi

# Sync only small CSV logs — skip large checkpoint .pkl files
rsync -avz $DRY \
    --include="*/" \
    --include="*.csv" \
    --exclude="*" \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/ \
    "${LOCAL_DIR}/"

echo ""
echo "Once synced, run:"
echo "  cd $(dirname "$0") && python aggregate_results.py"

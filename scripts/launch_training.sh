#!/usr/bin/env bash
# Launch FixProtoGPT multi-version training in the background.
# Safe to close the IDE — the process continues via nohup.
#
# Usage:
#   ./scripts/launch_training.sh                       # Full pipeline (ingest + train combined)
#   ./scripts/launch_training.sh --resume-only         # Resume combined training (skip data prep)
#   ./scripts/launch_training.sh --per-version         # Combined + per-version training
#   ./scripts/launch_training.sh --only-per-version    # Per-version training only
#   ./scripts/launch_training.sh --only-per-version --versions 4.2 4.4 Latest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs pids

LOG_FILE="logs/training_all_versions.log"
PID_FILE="pids/training_all_versions.pid"

# Collect extra args to forward to the Python script
EXTRA_ARGS="${*}"

echo "=================================================================="
echo "  FixProtoGPT — Multi-Version Training Launcher"
echo "  Project root: $PROJECT_ROOT"
echo "  Log file:     $LOG_FILE"
echo "  Args:         ${EXTRA_ARGS:-<default: full pipeline>}"
echo "=================================================================="

# Kill any existing training process
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "  Stopping existing training (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Launch training in background with nohup
nohup python3 -u scripts/train_all_versions.py $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo ""
echo "  Training started! PID: $TRAIN_PID"
echo "  Log: tail -f $LOG_FILE"
echo "  Stop: kill $TRAIN_PID"
echo ""
echo "  Safe to close this IDE — training continues in background."
echo "=================================================================="

# Show first few seconds of output
sleep 5
echo ""
echo "  --- First output ---"
head -40 "$LOG_FILE" 2>/dev/null || echo "  (waiting for output...)"

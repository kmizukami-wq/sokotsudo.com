#!/bin/bash
set -euo pipefail

# Weekly XGBoost model retrain script
# Cron: 0 18 * * 0 (Sunday 18:00 UTC = Monday 03:00 JST)

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="/home/ubuntu/xgb_model.pkl"
BACKUP_PATH="/home/ubuntu/xgb_model.pkl.bak"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

echo "$LOG_PREFIX Starting weekly ML retrain..."

# Backup current model
if [ -f "$MODEL_PATH" ]; then
    cp "$MODEL_PATH" "$BACKUP_PATH"
    echo "$LOG_PREFIX Backed up existing model."
fi

# Retrain
cd "$BOT_DIR"
if python3 trainer.py --retrain --train-months 6; then
    echo "$LOG_PREFIX Retrain SUCCESS. Model updated at $MODEL_PATH"

    # Verify model file was created/updated
    if [ -f "$MODEL_PATH" ]; then
        MODEL_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
        echo "$LOG_PREFIX Model size: ${MODEL_SIZE} bytes"
    fi
else
    echo "$LOG_PREFIX Retrain FAILED. Restoring backup..."
    if [ -f "$BACKUP_PATH" ]; then
        cp "$BACKUP_PATH" "$MODEL_PATH"
        echo "$LOG_PREFIX Backup restored."
    else
        echo "$LOG_PREFIX No backup available. Bot will run without ML."
    fi
fi

echo "$LOG_PREFIX Retrain job complete."

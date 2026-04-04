#!/bin/bash

# BTC/JPY Bot Health Check
# Usage: bash health_check.sh
# Exit 0 = healthy, Exit 1 = unhealthy

ERRORS=0
LOG_FILE="/home/ubuntu/momentum_bot_v2.log"
MODEL_FILE="/home/ubuntu/xgb_model.pkl"

echo "=== BTC/JPY Bot Health Check ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# 1. Process check
echo "[Process]"
if systemctl is-active --quiet btcbot 2>/dev/null; then
    echo "  Status: RUNNING"
    UPTIME=$(systemctl show btcbot --property=ActiveEnterTimestamp --value 2>/dev/null || echo "unknown")
    echo "  Since: $UPTIME"
else
    echo "  Status: NOT RUNNING"
    ERRORS=$((ERRORS + 1))
fi

# 2. Log freshness
echo ""
echo "[Log]"
if [ -f "$LOG_FILE" ]; then
    LOG_MOD=$(stat -c %Y "$LOG_FILE" 2>/dev/null || stat -f %m "$LOG_FILE" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    AGE=$(( NOW - LOG_MOD ))
    echo "  File: $LOG_FILE"
    echo "  Last update: ${AGE}s ago"
    if [ "$AGE" -gt 600 ]; then
        echo "  WARNING: Log not updated in 10+ minutes!"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK"
    fi
    # Last 3 lines
    echo "  Recent:"
    tail -3 "$LOG_FILE" | sed 's/^/    /'
else
    echo "  WARNING: Log file not found ($LOG_FILE)"
    ERRORS=$((ERRORS + 1))
fi

# 3. ML model
echo ""
echo "[ML Model]"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo "?")
    MODEL_MOD=$(stat -c %Y "$MODEL_FILE" 2>/dev/null || stat -f %m "$MODEL_FILE" 2>/dev/null || echo 0)
    MODEL_AGE=$(( $(date +%s) - MODEL_MOD ))
    MODEL_DAYS=$(( MODEL_AGE / 86400 ))
    echo "  File: $MODEL_FILE (${MODEL_SIZE} bytes)"
    echo "  Age: ${MODEL_DAYS} days"
    if [ "$MODEL_DAYS" -gt 14 ]; then
        echo "  WARNING: Model older than 2 weeks! Retrain recommended."
    else
        echo "  OK"
    fi
else
    echo "  WARNING: No ML model found. Bot runs in V3-only mode."
fi

# 4. System resources
echo ""
echo "[System]"
MEM_AVAIL=$(awk '/MemAvailable/ {printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null || echo "?")
DISK_AVAIL=$(df -h / 2>/dev/null | awk 'NR==2 {print $4}' || echo "?")
LOAD=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}' || echo "?")
echo "  Memory available: ${MEM_AVAIL} MB"
echo "  Disk available: ${DISK_AVAIL}"
echo "  Load (1m): ${LOAD}"

if [ "${MEM_AVAIL:-0}" -lt 100 ] 2>/dev/null; then
    echo "  WARNING: Low memory!"
    ERRORS=$((ERRORS + 1))
fi

# Result
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "=== HEALTHY ==="
    exit 0
else
    echo "=== UNHEALTHY ($ERRORS issues) ==="
    exit 1
fi

#!/bin/bash
set -euo pipefail

# BTC/JPY Bot V4 — EC2 Deployment Script
# Usage: sudo bash deploy.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BOT_DIR="$REPO_DIR/bot"
ENV_FILE="$REPO_DIR/.env"
SERVICE_SRC="$BOT_DIR/btcbot.service"
CRON_SCRIPT="$BOT_DIR/retrain_cron.sh"

# Detect deploy user (whoever owns the repo)
DEPLOY_USER="$(stat -c '%U' "$REPO_DIR")"
DEPLOY_HOME="$(eval echo "~$DEPLOY_USER")"

echo "=== BTC/JPY Bot V4 Deploy ==="
echo "Repo: $REPO_DIR"
echo "User: $DEPLOY_USER"
echo "Home: $DEPLOY_HOME"

# 1. System packages
echo ""
echo "[1/6] Checking system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip > /dev/null 2>&1
echo "  Python3: $(python3 --version)"

# 2. Python dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip3 install -r "$REPO_DIR/requirements.txt" --quiet
echo "  Done."

# 3. Environment file check
echo ""
echo "[3/6] Checking environment file..."
if [ ! -f "$ENV_FILE" ]; then
    echo "  ERROR: $ENV_FILE not found!"
    echo "  Copy .env.example and fill in your API keys:"
    echo "    cp $REPO_DIR/.env.example $ENV_FILE"
    echo "    nano $ENV_FILE"
    exit 1
fi

if ! grep -q "BITFLYER_API_KEY=" "$ENV_FILE" || ! grep -q "BITFLYER_API_SECRET=" "$ENV_FILE"; then
    echo "  ERROR: .env must contain BITFLYER_API_KEY and BITFLYER_API_SECRET"
    exit 1
fi

if grep -q "your_api_key_here" "$ENV_FILE"; then
    echo "  ERROR: .env contains placeholder values. Update with real API keys."
    exit 1
fi
echo "  .env OK"

# 4. Initial ML model training
echo ""
echo "[4/6] Training XGBoost model (initial, 6 months)..."
MODEL_PATH="$DEPLOY_HOME/xgb_model.pkl"
if [ -f "$MODEL_PATH" ]; then
    echo "  Model already exists at $MODEL_PATH, skipping."
    echo "  To force retrain: bash $CRON_SCRIPT"
else
    cd "$BOT_DIR"
    sudo -u "$DEPLOY_USER" python3 trainer.py --retrain --train-months 6
    echo "  Model saved."
fi

# 5. Install systemd service (replace placeholder paths)
echo ""
echo "[5/6] Installing systemd service..."
sed \
    -e "s|/home/ubuntu|$DEPLOY_HOME|g" \
    -e "s|User=ubuntu|User=$DEPLOY_USER|g" \
    "$SERVICE_SRC" > /etc/systemd/system/btcbot.service
systemctl daemon-reload
systemctl enable btcbot
systemctl restart btcbot
echo "  Service installed and started."
sleep 2
systemctl status btcbot --no-pager || true

# 6. Install weekly retrain cron
echo ""
echo "[6/6] Installing weekly retrain cron job..."
chmod +x "$CRON_SCRIPT"
# Sunday 18:00 UTC = Monday 03:00 JST
CRON_LINE="0 18 * * 0 $CRON_SCRIPT >> $DEPLOY_HOME/retrain.log 2>&1"
(crontab -u "$DEPLOY_USER" -l 2>/dev/null | grep -v "retrain_cron.sh"; echo "$CRON_LINE") | crontab -u "$DEPLOY_USER" -
echo "  Cron installed: weekly Sunday 18:00 UTC (Mon 03:00 JST)"

echo ""
echo "=== Deploy Complete ==="
echo ""
echo "Useful commands:"
echo "  systemctl status btcbot        # Check bot status"
echo "  journalctl -u btcbot -f        # Live logs"
echo "  tail -f $DEPLOY_HOME/momentum_bot_v2.log"
echo "  bash $BOT_DIR/health_check.sh  # Health check"
echo "  bash $CRON_SCRIPT              # Force ML retrain"

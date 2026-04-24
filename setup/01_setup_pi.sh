#!/bin/bash
# ============================================================
# PHASE 1: Raspberry Pi 5 Setup for Kronos Trader
# Run this script on your fresh Pi 5
# ============================================================

set -e
echo "🚀 Starting Kronos Trader Pi Setup..."

# --- System Update ---
sudo apt update && sudo apt upgrade -y

# --- Install Python 3.11+ ---
sudo apt install -y python3 python3-pip python3-venv python3-dev \
  git build-essential libssl-dev libffi-dev \
  sqlite3 libsqlite3-dev curl wget

# --- Install Tailscale ---
echo "📡 Installing Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
echo "✅ Tailscale installed. Authenticate in the browser link above."
echo "   After auth, your Pi's Tailscale IP will be shown with: tailscale ip -4"

# --- Create project directory ---
mkdir -p ~/kronos-trader
cd ~/kronos-trader

# --- Create Python virtual environment ---
python3 -m venv venv
source venv/bin/activate

# --- Install Python dependencies ---
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub
pip install fastapi uvicorn[standard]
pip install apscheduler
pip install pandas numpy scipy
pip install ccxt alpaca-trade-api
pip install python-dotenv aiohttp websockets
pip install sqlalchemy alembic

echo ""
echo "✅ Pi setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run: tailscale ip -4   (note your Pi's Tailscale IP)"
echo "  2. Run: 02_install_kronos.sh"

#!/bin/bash
# ============================================================
# PHASE 6: Start everything and install as a system service
# Run from ~/kronos-trader
# ============================================================

set -e

# --- Serve the dashboard as static files via the FastAPI backend ---
# (FastAPI will serve index.html at / automatically via StaticFiles)
# First, copy dashboard into backend's static folder
mkdir -p backend/static
cp dashboard/index.html backend/static/index.html

echo "📝 Creating systemd service..."
sudo tee /etc/systemd/system/kronos-trader.service > /dev/null <<EOF
[Unit]
Description=Kronos Trader Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/kronos-trader/backend
Environment="PATH=$HOME/kronos-trader/venv/bin"
ExecStart=$HOME/kronos-trader/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable kronos-trader
sudo systemctl start kronos-trader

echo ""
echo "✅ Kronos Trader is running as a system service!"
echo ""
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "YOUR_TAILSCALE_IP")
echo "🌐 Dashboard URL (from any device on Tailscale):"
echo "   http://$TAILSCALE_IP:8000"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status kronos-trader   # check status"
echo "  sudo journalctl -u kronos-trader -f   # live logs"
echo "  sudo systemctl restart kronos-trader  # restart"

#!/bin/bash
sudo systemctl restart kronos-trader
echo "Restarted. Tailing log (Ctrl+C to stop)..."
sudo journalctl -u kronos-trader -f

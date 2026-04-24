# Kronos Trader — Claude Code Context

## What This Project Is
A paper trading monitor running the Kronos AI foundation model on a Raspberry Pi 5.
Predicts future price candles for Crypto, US Stocks, and Forex using candlestick (OHLCV)
sequences, then automatically paper trades based on model confidence.

## Hardware
- Raspberry Pi 5 (64-bit, aarch64)
- Running VS Code directly on the Pi
- Tailscale installed for remote access

## Project Structure
~/kronos-trader/
├── setup/
│   ├── 01_setup_pi.sh           # System deps + Tailscale install
│   ├── 02_install_kronos.sh     # Clones Kronos repo + downloads model weights
│   └── 03_start_service.sh      # Installs systemd service (runs on boot)
├── backend/
│   ├── main.py                  # FastAPI app + APScheduler (hourly predictions)
│   ├── database.py              # SQLAlchemy models: Portfolio, Position, Trade, Prediction
│   ├── data_fetcher.py          # Binance (crypto), Alpaca (stocks), OANDA (forex)
│   ├── kronos_engine.py         # Kronos model inference wrapper
│   ├── trading_engine.py        # Paper trading logic: open/close/PnL
│   ├── serve_static.py          # Appended to main.py — serves dashboard at /
│   ├── .env.example             # Config template
│   └── .env                     # Your actual config (never commit this)
├── dashboard/
│   └── index.html               # React dashboard (no build step needed)
├── Kronos/                      # Auto-created by 02_install_kronos.sh
├── models/                      # Auto-created by 02_install_kronos.sh
├── venv/                        # Auto-created by 01_setup_pi.sh
└── CLAUDE.md                    # This file

## Markets & Symbols
- **Crypto**: BTC/USDT, ETH/USDT (Binance — no API key needed)
- **Stocks**: AAPL, TSLA, NVDA, SPY (Alpaca paper trading)
- **Forex**: EUR_USD, GBP_USD, USD_JPY (OANDA practice account)

## Paper Trading Config
- Starting balance: $1,000 per market ($3,000 total)
- Position size: 10% of available cash per trade
- Min confidence to enter: 55%
- Stop loss: -3% | Take profit: +5%
- Predictions: every 60 min (scheduled) + manual trigger from dashboard

## Model
- Kronos-small (24.7M params) from HuggingFace: NeoQuasar/Kronos-small
- Tokenizer: NeoQuasar/Kronos-Tokenizer-base
- Context: 512 candles max | Lookback: 400 candles | Horizon: 24h
- Monte Carlo samples: 5 per prediction

## Current Setup Status
- [ ] 01_setup_pi.sh completed
- [ ] 02_install_kronos.sh completed
- [ ] .env configured
- [ ] serve_static.py appended to main.py
- [ ] backend/static/index.html created
- [ ] 03_start_service.sh completed
- [ ] Dashboard accessible at http://TAILSCALE_IP:8000

## Key Commands
```bash
# Activate virtualenv
source ~/kronos-trader/venv/bin/activate

# Run backend manually (without systemd)
cd ~/kronos-trader/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Live service logs
sudo journalctl -u kronos-trader -f

# Restart service
sudo systemctl restart kronos-trader

# Manually trigger prediction via API
curl -X POST http://localhost:8000/api/predict/trigger \
  -H "Content-Type: application/json" \
  -d '{"market": "crypto"}'

# Check all portfolios
curl http://localhost:8000/api/overview

# SQLite direct access
sqlite3 ~/kronos-trader/backend/kronos_trader.db ".tables"
```

## API Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | /api/overview | All 3 portfolios summary |
| GET | /api/portfolio/{market} | Single market portfolio |
| GET | /api/trades | Trade history |
| GET | /api/predictions | Prediction history |
| GET | /api/prices | Latest price snapshots |
| POST | /api/predict/trigger | Manual prediction trigger |
| GET | /api/scheduler/status | Scheduler jobs + next run |
| GET | /api/health | Health check |

## Dependencies
All installed into ~/kronos-trader/venv via 01_setup_pi.sh:
- fastapi, uvicorn, apscheduler
- sqlalchemy, pandas, numpy
- ccxt (Binance), alpaca-trade-api, aiohttp
- torch (CPU), transformers, huggingface_hub
- python-dotenv

## Notes for Claude Code
- Always activate venv before running Python: `source ~/kronos-trader/venv/bin/activate`
- Kronos repo lives at ~/kronos-trader/Kronos/ (added to sys.path in kronos_engine.py)
- Model weights live at ~/kronos-trader/models/ (downloaded from HuggingFace)
- SQLite DB is at ~/kronos-trader/backend/kronos_trader.db (auto-created on first run)
- serve_static.py must be appended to main.py before starting the service
- Dashboard auto-refreshes every 30 seconds via polling
- Pi is aarch64 — use CPU-only PyTorch, no CUDA
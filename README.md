# Kronos Trader

A self-hosted paper-trading system that ran the [Kronos](https://github.com/shiyu-coder/Kronos) candlestick foundation model on a Raspberry Pi 5. It forecast hourly price candles for crypto and US stocks, opened and closed simulated positions based on those forecasts, and gradually grew a measurement layer to answer whether the model's signals made money.

**Status: retired.** Live paper trading ran from **2026-04-16 to 2026-10-14** and was then shut down. It was never profitable, and six months of measurement found no usable directional edge in the model output. See [Results](#results) and [Why it was shut down](#why-it-was-shut-down).

No real money was ever traded. Every trade was simulated against live market prices.

---

## Contents

- [How it started](#how-it-started)
- [What it became](#what-it-became)
- [Architecture](#architecture)
- [Feature reference](#feature-reference)
- [Results](#results)
- [Why it was shut down](#why-it-was-shut-down)
- [Lessons](#lessons)
- [Running it](#running-it)
- [Repository layout](#repository-layout)

---

## How it started

The first version (April 2026) was a weekend-sized idea: point a pretrained time-series foundation model at live candles and see whether its forecasts could drive a simple trading loop.

| | Original design |
|---|---|
| Hardware | Raspberry Pi 5, CPU-only PyTorch, reachable over Tailscale |
| Model | Kronos-small (24.7M params), 400-candle lookback, 24h horizon, 5 Monte Carlo samples |
| Markets | Crypto (BTC, ETH via Binance), stocks (AAPL, TSLA, NVDA, SPY via Alpaca), forex (EUR/USD, GBP/USD, USD/JPY via OANDA) |
| Capital | $1,000 paper balance per market |
| Entry rule | Model "confidence" ≥ 55% → buy 10% of available cash |
| Exit rule | Stop-loss −3%, take-profit +5%, checked once per hourly prediction cycle |
| Stack | FastAPI + APScheduler + SQLite, single-file React dashboard (no build step) |

"Confidence" was the fraction of bars in the model's averaged 24h forecast path that sat above the current price. That definition turned out to matter a great deal later on.

## What it became

Over six months the project shifted from "run the model and trade" to "measure honestly whether the trading is any good". By the end, roughly two-thirds of the code was measurement, diagnostics and test infrastructure rather than trading logic.

| Area | Start | End |
|---|---|---|
| Models | Kronos-small only | Kronos-small **and** Kronos-mini (4.1M). Optional dual-model mode where both predict every cycle and one `active_model` trades |
| Universe | 4 stocks, 2 crypto, 3 forex | 17 stocks (high-beta names such as MARA, RIOT, HUT, AMD, SPCE, GDX), 2 crypto. Forex dropped and never traded. 9 symbols disabled after per-symbol review |
| Entry gate | confidence ≥ 55% | confidence ≥ 75% **and** predicted move ≥ 2% |
| Exits | SL/TP checked hourly | Intraday price monitor every 3 minutes, plus an exit when a later prediction turns bearish |
| Take-profit | +5% | +6%, then +3% (2026-09-03) |
| P&L | Gross price difference | Side-aware fee and slippage cost model, recording gross and net separately |
| Oversight | Manual | Weekly Claude-driven audit that proposes parameter changes into a human approval queue |
| Evaluation | Dashboard P&L | Random-entry Monte Carlo benchmarks, three-arm entry/execution decomposition, block-bootstrap confidence intervals, exit-policy replay, walk-forward backtest harness, 93 tests |

### Timeline

- **2026-04-14 to 04-24:** Initial build: data fetchers, Kronos wrapper, paper-trading engine, dashboard, Pi setup scripts and systemd service. Live trading started **04-16**.
- **May to July:** Feature growth: intraday price monitor, outcome recorder (scores every prediction after 24h), Discord notifications, hourly candle backfill, chart views, Kronos-mini, dual-model mode, settings UI.
- **July to August:** First measurement layer: cost model, buy-and-hold / random-entry / inverse-signal benchmarks, MFE/MAE and exit-policy replay. Weekly AI audit with an approve/reject queue. Model switched small → mini (07-21) → small (08-09).
- **September:** Rigour pass. The diagnostics were themselves found to be biased in the simulator's favour, and fixed:
  - Two separate exit simulators merged into one gap-aware simulator (`exit_sim.py`, semantics ported from vectorbt). A bar that opens past a level now fills at the open, and a bar that touches both SL and TP now counts as the stop.
  - The random benchmark now compounds against available cash instead of using flat notional.
  - Results split into per-model, per-threshold "eras" instead of being pooled.
  - Vendored `arch` block bootstrap for confidence intervals on autocorrelated series.
  - A three-arm benchmark separates *entry skill* from *execution cost*.
  - A golden test pins the live P&L arithmetic.
  - Take-profit cut to 3% (09-03). Model switched back to mini (09-06).
- **Late September to October:** Probabilistic-signal experiment (see below). It found no improvement over the existing signal.
- **2026-10-14:** Scheduler shut down and the systemd service disabled.

## Architecture

```
               ┌────────────── APScheduler (inside FastAPI) ──────────────┐
               │ hourly   predictions per market (crypto, stocks)         │
               │ 3 min    price_monitor: intraday SL/TP enforcement       │
               │ 5 min    price snapshots                                 │
               │ hourly   outcome_recorder: score predictions after 24h   │
               │ hourly   candle_backfill: incremental OHLCV append       │
               │ weekly   audit_engine (Sun 02:00 UTC)                    │
               │ weekly   benchmarks (Sun 04:00 UTC)                      │
               └──────────────────────────────────────────────────────────┘
                                         │
 Binance / Alpaca ──▶ data_fetcher ──▶ kronos_engine ──▶ trading_engine ──▶ SQLite
                                        (small / mini)    (+ cost_model)       │
                                                                               ▼
               static/index.html (React) ◀── FastAPI /api/* ◀── diagnostics, benchmarks,
                                                                 symbol_fit, audit_engine
```

Everything runs in one process on the Pi. Inference costs about 15 seconds per forecast path on the Pi 5's CPU.

## Feature reference

### Trading core (`backend/`)

| Module | Purpose |
|---|---|
| `kronos_engine.py` | Loads and caches Kronos-small and Kronos-mini; produces forecast high/low/close and confidence |
| `data_fetcher.py` | OHLCV from Binance (ccxt), Alpaca, and OANDA |
| `trading_engine.py` | Opens and closes paper positions; applies entry gates, cooldowns, sizing; exits on bearish signal |
| `price_monitor.py` | 3-minute stop-loss / take-profit enforcement between prediction cycles |
| `cost_model.py` | Side-aware fees and adverse slippage, versioned so historical trades stay comparable |
| `outcome_recorder.py` | After 24h, records the actual close and whether each prediction's direction was right |
| `candle_backfill.py` | Persists hourly candles for all traded symbols, used by every simulator |
| `notifications.py` | Discord embeds on trade open/close and audit results (webhook configured in settings, not in the repo) |
| `database.py` | SQLAlchemy models: portfolios, positions, trades, predictions, candles, audit recommendations |

### Oversight and measurement

| Module | Purpose |
|---|---|
| `audit_engine.py` | Weekly and monthly review that runs Claude Code headless (`claude -p`) over recent performance and writes **proposed** changes, kept within hard bounds, to an approval queue. Nothing is applied without a click in the dashboard. 16 applied, 3 reverted over the project's life |
| `benchmarks.py` | Buy-and-hold, 1,000-run random-entry Monte Carlo, inverse-signal. Sharpe, drawdown, profit factor, bootstrap CIs |
| `diagnostics.py` | MFE/MAE analysis and replay of alternative exit policies (wider stops, longer holds, horizon-only, ATR-based) |
| `exit_sim.py` | The single, gap-aware exit simulator every analysis shares |
| `symbol_fit.py` | Per-symbol EV with bootstrap CIs, volatility heuristics, BTC-correlation check for crypto-miner stocks |
| `stats_bootstrap.py` + `vendor/arch_bootstrap/` | Block bootstrap (vendored from `arch`, NCSA licence) |
| `diagnose_confidence.py`, `backfill_prediction_linkage.py` | Found and fixed SELL trades linked to their exit prediction instead of their entry prediction |
| `scripts/rerun_diagnostics.py` | Regenerates the dated reports in `backend/reports/` |
| `experiments/` | Probabilistic-signal experiment and its analysis |
| `backtest/` | Walk-forward replay harness: snapshots, fill models (exact vs empirical overshoot), reconciliation against live fills. Read-only against the live DB |

### Dashboard

A single React page served at `/`, which auto-refreshes every 30 seconds. Tabs: **Overview** (portfolios, equity curve, open positions), **Charts** (candles with prediction overlays), **Predictions**, **Symbols** (per-symbol performance), **Audit** (approve/reject queue, config eras, diagnostics), **Settings** (risk parameters, symbols, model choice, notifications), **Features**.

### API

About 35 endpoints under `/api/`, including `overview`, `portfolio/{market}`, `trades`, `predictions`, `predict/trigger`, `settings`, `chart/{market}/{symbol}`, `accuracy`, `equity-curve`, `symbol-performance`, `audit/*`, `benchmarks`, `diagnostics/*` (confidence, mfe-mae, exit-replay, lag, symbol-fit, three-arm), `config-eras`, `logs`, `system`, `health`.

---

## Results

All figures come from the live SQLite database at shutdown and from [`backend/reports/diagnostics_2026-09-24.md`](backend/reports/diagnostics_2026-09-24.md).

### Headline

| Metric | Value |
|---|---|
| Live period | 2026-04-16 → 2026-10-14 (~6 months) |
| Predictions made / scored | 22,427 / 22,331 |
| Directional accuracy | **51.4%** (small 51.5%, mini 51.2%) |
| Closed round-trip trades | 425 (340 stocks, 85 crypto) |
| Win rate | 41% (stocks 43.5%, crypto 31.8%) |
| Gross P&L | −$110.61 |
| Fees + slippage | −$154.49 |
| **Net P&L** | **−$265.11** on $2,000 of active capital (≈ −13%) |

### P&L by exit type

| Exit | Trades | Net P&L |
|---|---|---|
| Stop-loss | 138 | **−$593.88** |
| Bearish-signal flip | 242 | +$111.09 |
| Take-profit | 45 | +$217.67 |

Stop-losses did the damage. Live stops averaged **−3.93%** against a −3% target, and 100% of them filled worse than the level. That slippage comes from 3-minute polling, overnight gaps in stocks (26.5% of stock stops gapped, 0% of crypto) and price-feed differences.

### Net P&L by month

| Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|
| −36.80 | −26.44 | **−184.09** | −3.96 | +1.10 | −4.46 | −10.46 |

Once the stricter entry gate and per-symbol pruning were in place, losses shrank to roughly breakeven. That was the cost of trading less, not evidence of edge.

### What went right

- **The engineering held up.** Six months of unattended hourly inference on a Pi 5, with an intraday monitor, backfill, notifications and a dashboard, without losing data.
- **The measurement got honest.** Each correction in September (gap fills, stop-first on same-bar ties, compounding benchmarks, era separation, entry-vs-exit linkage) moved the numbers *against* the strategy, and each was kept. Reports state that "nothing was tuned to match recorded history", and the golden test means recorded P&L can't drift silently.
- **Entry timing showed a hint of signal.** In the three-arm benchmark, live entries replayed through the simulator beat the random median in three of four eras, with percentiles 26, 73, **95 (p = 0.047)** and 89. The 2026-08-09 → 09-03 small-model era had positive simulated P&L (+$130.75).
- **The human-in-the-loop audit worked as designed.** It caught clearly bad symbols (SIDU was the only symbol with a bootstrap CI entirely below zero), and its worst proposals (for example, raising the minimum predicted move to 10%) were rejected or reverted.
- **Costs were modelled before they were believed.** Fees and slippage were 58% of the total loss. Without the cost model this would have looked like a near-breakeven strategy.

### What failed

- **The model's direction calls were a coin flip.** 51.4% across 22k scored predictions, and neither model size changed that. The audit's "mini is 17 points better" and "small is 9 points better" findings were week-to-week noise. Those flips are visible in the era table.
- **"Confidence" carried no information.** Across 22,051 scored predictions, `corr(confidence, correct) = −0.019`. Accuracy was flat across the whole range, so raising the gate from 55% to 75% only cut trade count.
- **Execution ate whatever entry edge existed.** Execution cost was negative in every populated era, about −$215 in total. In the best era, a +$131 simulated result became +$17 live.
- **Asymmetric exits.** Stops filled worse than −3% while take-profits were capped, so a 72% win rate in one period still had negative expected value. Cutting take-profit to 3% traded fewer large wins for more small ones without fixing the asymmetry.
- **Forex never ran.** The OANDA integration was built, but that portfolio never placed a trade.

### The final experiment: probabilistic signal

The last open question was whether the problem was the *signal definition* rather than the model. The live "confidence" averaged the Monte Carlo samples away before measuring anything. The Kronos paper instead recommends a probability across independent sample paths. [`experiments/prob_signal.py`](backend/experiments/prob_signal.py) computed `p_up` (the fraction of 8 independent paths ending above entry) side by side with the legacy statistic, on the same 184 points and the same model runs.

| Signal | Directional accuracy | 95% CI |
|---|---|---|
| `p_up` (sample paths) | 52.7% | [45.1, 59.2] |
| Legacy confidence | 53.3% | [45.7, 60.3] |
| Base rate (price simply went up) | 56.0% | |

McNemar's exact test gave p = 1.00, so there was no detectable difference between the two. **Both signals did worse than always guessing "up".** A related `p_tp_first` estimate (will take-profit be hit before the stop?) was 42% accurate against a 55.5% base rate.

## Why it was shut down

On 2026-10-14 the scheduler was stopped and the service disabled. The evidence collected by then pointed the same way from every angle:

1. **No directional edge.** About 51% accuracy over 22k predictions from both model sizes, with a confidence score uncorrelated with being right.
2. **The best alternative signal also failed.** The probabilistic-output experiment (the approach the model's own authors recommend) did no better than the existing signal or the market's base rate.
3. **Losses were structural, not tunable.** 19 approved audit changes over five months (3 later reverted) moved the result from losing to roughly breakeven by trading less. They never produced a positive, statistically distinguishable result.
4. **Remaining upside was within noise.** The one promising era (p = 0.047, one of four eras, before execution costs) is the kind of result that appears by chance when enough slices are examined.
5. **Ongoing cost.** Hourly CPU inference on the Pi and weekly AI audits continued with nothing left to learn from more of the same data.

The conclusion: a general-purpose candlestick foundation model, used zero-shot on hourly bars of volatile retail-favourite stocks and crypto, did not produce tradeable direction calls. Building out the measurement layer is what made it possible to say that with confidence instead of blaming bad luck or bad parameters.

## Lessons

- Build the benchmark before trusting the strategy. A random-entry arm with identical sizing, costs and exits is the minimum bar.
- Audit your simulator for optimism. Every bug found here flattered the strategy.
- Check whether a "confidence" number actually predicts correctness before gating on it.
- Separate entry skill from execution cost. They fail for different reasons and need different fixes.
- An automated tuner reading weekly accuracy will chase noise. Keep a human in the approval loop and segment results by config era.

---

## Running it

> The project is retired. These steps are kept for reference.

**Requirements:** Raspberry Pi 5 (or any 64-bit Linux) with Python 3 and CPU-only PyTorch. Alpaca paper keys for stocks. Optionally OANDA practice keys for forex and a Discord webhook.

```bash
# 1. System deps, virtualenv, Tailscale
bash setup/01_setup_pi.sh

# 2. Clone Kronos and download model weights from Hugging Face
bash setup/02_install_kronos.sh

# 3. Configure secrets (never committed; .env is gitignored)
cp backend/.env.example backend/.env && $EDITOR backend/.env

# 4a. Run manually
source venv/bin/activate
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 4b. ...or install as a systemd service that starts on boot
bash setup/03_start_service.sh
```

The dashboard is served at `http://<host>:8000/`. Runtime settings (risk parameters, symbols, model, notifications) live in `backend/settings.json`, which the Settings tab writes. That file is gitignored because it can hold the Discord webhook URL.

**Useful commands**

```bash
# Tests (93)
cd backend && ../venv/bin/python -m pytest -q tests ../backtest/tests

# Regenerate the diagnostics report
venv/bin/python backend/scripts/rerun_diagnostics.py

# Re-run the probabilistic-signal analysis
venv/bin/python backend/experiments/analyse_prob_signal.py

# Trigger a prediction cycle manually
curl -X POST localhost:8000/api/predict/trigger -H 'Content-Type: application/json' -d '{"market":"crypto"}'

# Service logs
sudo journalctl -u kronos-trader -f
```

## Repository layout

```
setup/          Pi provisioning, Kronos install, systemd service
backend/        FastAPI app, trading engine, measurement modules
  static/       React dashboard (single file, no build step)
  tests/        Unit and golden tests
  reports/      Dated diagnostics reports
  experiments/  Probabilistic-signal experiment
  scripts/      Report regeneration
  vendor/       Vendored arch block bootstrap (NCSA licence)
backtest/       Walk-forward replay and fill-model harness (read-only)
start.sh, restart.sh
```

Not in the repo (gitignored): `backend/.env`, `backend/settings.json`, SQLite databases, logs, `Kronos/`, `models/`, `venv/`.

## Credits

- [Kronos](https://github.com/shiyu-coder/Kronos): the foundation model for financial candlesticks used for all forecasts.
- [`arch`](https://github.com/bashtage/arch) by Kevin Sheppard: block bootstrap, vendored under the NCSA licence.
- [vectorbt](https://github.com/polakowo/vectorbt): stop-price semantics ported (not vendored) into `exit_sim.py`.

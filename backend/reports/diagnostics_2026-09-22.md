# Diagnostics re-run — 2026-09-22

Exit simulation: `backend/exit_sim.py` (ported from vectorbt `get_stop_price_nb`). Bootstrap: vendored `arch`. 1000 simulations, seed 0. Read-only on the live DB.

## What changed vs previous numbers

| Area | Before | Now |
|---|---|---|
| Stop/TP fill | Always filled exactly at the level | Gap rule: fills at `open` when the bar opened through the level |
| Same-bar SL+TP | Take-profit won | **Stop-loss wins** (vectorbt order) |
| Exit implementations | 2 independent copies (`_walk_candles`, `_simulate_exit`) | 1 shared `simulate_exit`, both are now adapters |
| Random-arm sizing | Flat notional; `cash` set once, never decremented | **15% of available cash, compounding**, with cash returned at exit |
| Random-arm TP | Hardcoded 6% for all history | Per-era threshold (6% → 3% on 2026-09-03) |
| Segmentation | Pooled across all history | Split per model/threshold era, never pooled |
| Bootstrap | IID resample, duplicated in 2 modules | Block bootstrap via vendored `arch` |
| Benchmark arms | random vs live only | **Third arm: live entries + simulated exits**, separating entry skill from execution cost |
| Era keying | Models that emitted predictions | **Model that drove entries** (dual-mode split at the active_model flip) |
| `apply_costs` | Long-sided gross P&L regardless of `side` | Side-aware; `side='buy'` unchanged, so live is bit-identical |

The first four all pushed in the same direction: they made the simulator look better than reality, or the random benchmark look better than it should. Numbers below are expected to move against the simulator.

## Config eras

Boundaries are the recorded `active_model` changes in `audit_recommendations` plus the recorded take-profit change, at timestamp granularity (both landed mid-day). `settings.json` is gitignored and has never been committed, so these records are the only recoverable history. Membership is then verified against entry linkage below rather than assumed.

| Era | Entry model | TP | Entries |
|---|---|---|---|
| 2026-04-16 22:21 .. 2026-07-21 23:13 | small | 6% | 282 |
| 2026-07-21 23:13 .. 2026-08-09 02:01 | mini | 6% | 48 |
| 2026-08-09 02:01 .. 2026-09-03 18:32 | small | 6% | 57 |
| 2026-09-03 18:32 .. 2026-09-06 02:18 | - | 3% | 0 |
| 2026-09-06 02:18 .. 2026-09-21 19:47 | mini | 3% | 25 |

**Eras are keyed on the model that drove entries, not the model set that emitted predictions.** In dual-model mode both models predicted every cycle, but `process_prediction` only acts when `pred_model == active_model`, so only one model ever generated entries. Keying on the emitting set would have pooled two different entry generators into a single `mini+small` era.

- Entry-model purity: **PASS** — every era has exactly one entry model
- Era entry counts sum to total: **PASS** — 412 assigned + 0 unassigned = 412 linked = 412 total BUY trades

## 1. Stop-fill validation

- Live stop-loss exits found: **133**
- Compared against `simulate_exit`: **126**
- Unreproducible (fill outside its own bar): **6**
- Simulator did not produce a stop at all: **1**

| Metric | Live | Simulated |
|---|---|---|
| Mean stop return | -3.927% | -3.392% |
| Median stop return | -3.316% | -3.000% |
| Worse than −3.00% | 100.0% | 75.4% |

Simulated fills flagged as gapped: **21.4%**. Mean sim−live difference: **+0.535pp** (median +0.220pp). A positive number means the simulator still reports a better fill than reality.

### Gap rate by market

Equities close overnight and at weekends; crypto never does. Pooling the two hides a real structural difference in how often price can jump a level.

| Market | n | Gapped | Mean sim−live |
|---|---|---|---|
| crypto | 24 | 0.0% | +0.330pp |
| stocks | 102 | 26.5% | +0.583pp |

### Unreproducible fills (not fixed, not excluded silently)

These filled outside their own backfilled bar's range. `price_monitor` samples live Alpaca IEX bars while `candles` is written by a separate backfill, so the price the exit actually filled at is not in the candle table. No candle-based simulator can reach them.

| Symbol | Exit | Fill | Bar low | Bar high | Route |
|---|---|---|---|---|---|
| MARA | 2026-08-10 16:54 | 9.4650 | 9.5300 | 9.6100 | price_monitor |
| RIOT | 2026-08-14 14:48 | 18.8200 | 18.8450 | 19.2200 | price_monitor |
| MARA | 2026-08-19 13:42 | 8.6900 | 8.9000 | 9.0200 | price_monitor |
| GDX | 2026-09-01 12:39 | 95.2500 | 95.8300 | 95.8500 | price_monitor |
| MARA | 2026-09-01 13:39 | 10.2000 | 10.3900 | 10.5200 | price_monitor |
| MARA | 2026-09-15 18:48 | 10.9850 | 11.0800 | 11.3900 | price_monitor |

## 2. Three-arm benchmark (compounding, per era)

Three arms, differing in exactly one thing each:

| Arm | Entries | Exits |
|---|---|---|
| random | random bars | `simulate_exit` |
| sim-live | **live entry timestamps** | `simulate_exit` |
| live | live entry timestamps | what actually filled |

So **sim-live vs random isolates entry selection** (exits identical), and **live vs sim-live isolates execution** (entries identical). Sizing, cooldown and costs are shared by construction — all three run through the same `_run_portfolio`.

**Caveat on execution cost.** The sim-live arm exits via `simulate_exit`, which section 1 shows is still optimistic by roughly +0.5pp per stop even after the gap rule. So part of every `execution cost` figure is the simulator being too generous rather than live execution being bad. Treat it as an upper bound on the true execution drag, not a measurement of it.

### 2026-04-16 22:21..2026-07-21 23:13 [small] TP=6%

- Live net P&L: **-218.44** over 282 trades
- Sim-live net P&L: **-155.05** over 282 replayed entries
- Random mean: -92.43 | median: -94.99 | p5–p95: -237.69 … +68.99

- **Entry skill — sim-live percentile vs random: 26.0** (p = 0.7400)
- **Execution cost — live minus sim-live: -63.39** (upper bound — see caveat below)
- Live percentile vs random (both effects combined): 8.6 (p = 0.9140)
- Starting capital: 2000

### 2026-07-21 23:13..2026-08-09 02:01 [mini] TP=6%

- Live net P&L: **-45.45** over 48 trades
- Sim-live net P&L: **-45.43** over 48 replayed entries
- Random mean: -73.26 | median: -73.06 | p5–p95: -149.31 … +4.53

- **Entry skill — sim-live percentile vs random: 73.3** (p = 0.2670)
- **Execution cost — live minus sim-live: -0.01** (upper bound — see caveat below)
- Live percentile vs random (both effects combined): 73.3 (p = 0.2670)
- Starting capital: 2000

### 2026-08-09 02:01..2026-09-03 18:32 [small] TP=6%

- Live net P&L: **+17.17** over 57 trades
- Sim-live net P&L: **+130.75** over 57 replayed entries
- Random mean: +35.61 | median: +35.58 | p5–p95: -52.59 … +129.39

- **Entry skill — sim-live percentile vs random: 95.3** (p = 0.0470)
- **Execution cost — live minus sim-live: -113.57** (upper bound — see caveat below)
- Live percentile vs random (both effects combined): 38.4 (p = 0.6160)
- Starting capital: 2000

### 2026-09-03 18:32..2026-09-06 02:18 [-] TP=3%

Skipped — no live trades in era.

### 2026-09-06 02:18..2026-09-21 19:47 [mini] TP=3%

- Live net P&L: **-4.35** over 25 trades
- Sim-live net P&L: **+33.78** over 25 replayed entries
- Random mean: +10.46 | median: +9.52 | p5–p95: -47.28 … +67.43

- **Entry skill — sim-live percentile vs random: 74.9** (p = 0.2510)
- **Execution cost — live minus sim-live: -38.13** (upper bound — see caveat below)
- Live percentile vs random (both effects combined): 33.3 (p = 0.6670)
- Starting capital: 2000

## 3. Per-symbol expected value

Trade-level return % per symbol, block-bootstrap 95% CI. Symbols with n < 15 are flagged insufficient and get no CI.

### 2026-04-16 22:21..2026-07-21 23:13 [small] TP=6%

| Symbol | n | Mean % | 95% CI | Block | Method |
|---|---|---|---|---|---|
| AAPL | 2 | +1.486 | insufficient | — | — |
| AMD | 22 | +0.984 | [-0.208, +2.302] | 2 | bca |
| BTC/USDT | 37 | -0.412 | [-1.287, +0.305] | 3 | bca |
| CAR | 2 | -21.202 | insufficient | — | — |
| ETH/USDT | 40 | -0.266 | [-1.104, +0.769] | 2 | bca |
| GME | 14 | +0.803 | insufficient | — | — |
| HOOD | 2 | +3.237 | insufficient | — | — |
| HUT | 4 | +2.612 | insufficient | — | — |
| MARA | 4 | +3.306 | insufficient | — | — |
| MSTR | 24 | -0.662 | [-1.859, +0.999] | 2 | bca |
| NVDA | 15 | +0.019 | [-0.882, +1.081] | 1 | bca |
| RIOT | 1 | +2.796 | insufficient | — | — |
| SIDU | 29 | -1.045 | [-1.229, -0.642] | 10 | bca |
| SMCI | 5 | +0.412 | insufficient | — | — |
| SMR | 26 | +0.193 | [-1.277, +1.805] | 1 | bca |
| SNDK | 4 | +1.584 | insufficient | — | — |
| SPCE | 32 | -0.523 | [-2.075, +0.786] | 6 | bca |
| SPY | 2 | +3.435 | insufficient | — | — |
| TSLA | 16 | -0.421 | [-1.358, +0.398] | 1 | bca |
| UPRO | 1 | +1.400 | insufficient | — | — |

### 2026-07-21 23:13..2026-08-09 02:01 [mini] TP=6%

| Symbol | n | Mean % | 95% CI | Block | Method |
|---|---|---|---|---|---|
| AMD | 4 | -0.165 | insufficient | — | — |
| ETH/USDT | 2 | -0.282 | insufficient | — | — |
| HOOD | 6 | -1.472 | insufficient | — | — |
| HUT | 6 | -1.436 | insufficient | — | — |
| MARA | 7 | -1.922 | insufficient | — | — |
| NVDA | 2 | +2.677 | insufficient | — | — |
| RIOT | 5 | -1.068 | insufficient | — | — |
| SMCI | 3 | -0.361 | insufficient | — | — |
| SNDK | 3 | -2.448 | insufficient | — | — |
| SPCE | 3 | -0.834 | insufficient | — | — |
| TSLA | 3 | -0.580 | insufficient | — | — |
| UPRO | 4 | +1.733 | insufficient | — | — |

### 2026-08-09 02:01..2026-09-03 18:32 [small] TP=6%

| Symbol | n | Mean % | 95% CI | Block | Method |
|---|---|---|---|---|---|
| AMD | 5 | +1.352 | insufficient | — | — |
| BTC/USDT | 2 | -0.198 | insufficient | — | — |
| ETH/USDT | 3 | +0.360 | insufficient | — | — |
| GDX | 5 | +0.400 | insufficient | — | — |
| HOOD | 3 | +1.073 | insufficient | — | — |
| HUT | 6 | +2.251 | insufficient | — | — |
| MARA | 7 | +0.666 | insufficient | — | — |
| NVDA | 5 | +0.360 | insufficient | — | — |
| RIOT | 15 | -0.140 | [-1.748, +2.119] | 1 | bca |
| SPCE | 4 | +0.631 | insufficient | — | — |
| UPRO | 2 | +1.614 | insufficient | — | — |

### 2026-09-06 02:18..2026-09-21 19:47 [mini] TP=3%

| Symbol | n | Mean % | 95% CI | Block | Method |
|---|---|---|---|---|---|
| AMD | 2 | +1.376 | insufficient | — | — |
| ETH/USDT | 1 | +2.138 | insufficient | — | — |
| GDX | 3 | -0.378 | insufficient | — | — |
| HOOD | 3 | -2.860 | insufficient | — | — |
| HUT | 5 | +0.617 | insufficient | — | — |
| MARA | 6 | -0.636 | insufficient | — | — |
| NVDA | 1 | -3.156 | insufficient | — | — |
| SPCE | 2 | +1.476 | insufficient | — | — |
| UPRO | 2 | +1.932 | insufficient | — | — |

---

Nothing here was tuned to match recorded history. Gaps between simulated and live results are reported as found.
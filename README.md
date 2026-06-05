# equity-signal-research

A cross-sectional equity **signal-research** study: build price/volume signals
that try to predict the cross-section of US equity returns, turn them into
dollar-neutral long–short portfolios, and evaluate them the way a quant
researcher would — Information Coefficients with Newey–West t-stats, transaction
costs, factor-neutral alpha, purged/embargoed walk-forward CV, and a
**multiple-testing-corrected (deflated) Sharpe ratio**. A contextual-bandit /
offline-RL allocator combines the signals as an extension.

**The point is not "I found alpha."** The point is honest evaluation. The
headline finding below is a *qualified/negative* result, and that is the
intended outcome: it demonstrates the discipline (look-ahead control, cost
accounting, multiple-testing correction) rather than an over-fit backtest.

All numbers in this README are produced by the commands below from freshly
downloaded free data. Nothing is hardcoded.

---

## Headline results (real, reproducible)

Run: **402 liquid US large/mid-caps** (current S&P 500 snapshot), daily data
**2010-01-04 → 2024-12-30**, monthly rebalance, 21-day forward-return target,
**10 bps/side** transaction costs. Selection across **12 configurations**
(6 signals × {decile, rank}).

| Metric | Value |
|---|---|
| Best signal (by net Sharpe) | **illiquidity** (decile) |
| Mean IC | **+0.0153** (Newey–West t = **+1.93**, IR = +0.137) |
| Long–short Sharpe, gross | **+0.61** |
| Long–short Sharpe, **net of costs** | **+0.53** |
| **Deflated Sharpe Ratio** (corrects for 12 trials) | **0.22** |
| FF5 + momentum alpha | **+8.0%/yr** (t = **+3.60**) |
| Avg turnover / rebalance | 0.70 |
| Max drawdown (net) | −35.1% |
| Baseline: equal-weight market (excess) Sharpe | +0.90 |
| Baseline: no-skill random signal Sharpe | −0.79 |

**How to read this honestly:**

1. The best signal (an illiquidity / low-dollar-volume tilt) has a positive mean
   IC and a positive FF5+momentum alpha (t≈3.6) on this universe.
2. **But the deflated Sharpe is only 0.22.** After accounting for the 12
   configurations tried, the probability that the *true* Sharpe exceeds the
   multiple-testing benchmark (SR₀ ≈ 0.73 annualized) is ~22% — i.e. **not**
   significant once you correct for selection.
3. The edge is **regime-dependent**: net Sharpe is **+1.27 pre-2020** but
   **−0.20 post-2020** (`results/robustness.csv`).
4. Simply **owning the equal-weight market beat every long–short signal**
   (market excess Sharpe +0.90 vs best L–S net +0.53).
5. The "illiquidity premium" inside the S&P 500 is largely a size/liquidity tilt
   and is **heavily contaminated by survivorship bias** (we use today's
   constituents). See `LIMITATIONS.md`.

The other five classic signals (12–1 momentum, 1-month reversal, low volatility,
idiosyncratic volatility, 52-week-high proximity) have **net Sharpes near zero or
negative** on this survivorship-biased large-cap universe — see
`results/signal_summary.csv`.

#### Independent verification & robustness

The headline statistics are re-derived from scratch (without trusting
`src/evaluation.py`) by `python -m experiments.verify_headline`, which confirms:

- IC Newey–West t-stat matches a hand-rolled Bartlett-kernel HAC estimator;
- the deflated Sharpe matches a from-scratch computation;
- the FF5+momentum alpha matches a plain numpy least-squares fit (and carries an
  **SMB beta ≈ +0.45**, i.e. a real small-cap tilt — the residual alpha *beyond*
  SMB is consistent with survivorship bias, not a new factor);
- decile weights are exactly dollar-neutral and cost accounting reconciles to
  machine precision;
- signals are point-in-time (identical when recomputed on truncated data);
- the conclusion is **frequency-robust**: the deflated Sharpe is 0.22 (daily) vs
  0.23 (monthly per-rebalance returns), and the alpha t-stat is stable across HAC
  lag lengths (5 → 42). So "not significant after multiple testing" is not an
  artifact of daily-return autocorrelation.

### RL / contextual-bandit allocator (extension)

Out-of-sample net Sharpe of signal-combination methods (`results/rl_allocator_summary.csv`):

| Method | OOS net Sharpe |
|---|---|
| Mean–variance combination (point-in-time) | **+0.65** |
| Equal-weight combination | −0.52 |
| Learned LinUCB contextual bandit | −0.87 |

**The learned allocator did NOT beat the baselines.** A simple point-in-time
mean–variance combination of the signals was the best combiner; the contextual
bandit, which chases recently-strong signals, underperformed (signal strength
mean-reverts across regimes). Reported plainly, as intended.

---

## Reproduce

```bash
pip install -r requirements.txt

# End-to-end signal study -> results/  (downloads + caches data on first run)
python -m experiments.run_baseline --start 2010-01-01 --end 2024-12-31

# RL / contextual-bandit allocator vs baselines -> results/
python -m experiments.run_rl --start 2010-01-01 --end 2024-12-31 --learner linucb

# Independently re-derive & stress-test the headline stats (prints PASS/FAIL)
python -m experiments.verify_headline

# Tests (no-look-ahead, cost accounting, CV purging, IC sign conventions)
pytest -q
```

Useful flags: `--max-tickers N` (smaller/faster universe), `--cost-bps`,
`--horizon`, `--standardize {zscore,rank}`, `--learner {linucb,thompson}`,
`--force` (re-download). First run downloads from Yahoo Finance + the Ken French
Data Library and caches to `data/cache/` (git-ignored); later runs are fast and
offline-friendly.

Outputs written to `results/`: `signal_summary.csv`, `baselines.csv`,
`robustness.csv`, `rl_allocator_summary.csv`, `headline_*.json`, and per-signal
equity-curve / IC figures (`equity_*.png`, `ic_*.png`, `rl_allocator_curves.png`).

---

## Data (free, no credentials)

- **Prices:** daily adjusted OHLCV via `yfinance` (`auto_adjust=True`).
- **Universe:** a curated snapshot of current S&P 500 constituents
  (`src/universe.py`). **Not point-in-time** — survivorship caveat documented in
  `LIMITATIONS.md`.
- **Factors:** Fama–French 5 factors + momentum + RF from the **Ken French Data
  Library**, downloaded as direct CSV (more robust than `pandas-datareader`
  against pandas version churn). See `src/data.py`.

---

## Methodology

**Signals** (`src/signals.py`), all strictly point-in-time (backward-looking
windows only) and oriented long-high (higher value = ex-ante hypothesis for
higher forward return — the academic prior, decided before looking at the data):
12–1 momentum, 1-month reversal, realized volatility (low-vol), idiosyncratic
volatility vs market (low-idio), 52-week-high proximity, and an illiquidity /
turnover proxy. Each is cross-sectionally z-scored (or ranked) **within each
date** — no full-sample scaling.

**Point-in-time discipline.** At each rebalance date `t`, features use only data
≤ `t`. The target is the *forward* 21-day return (never a feature). The backtest
applies weights with a one-day lag (`weights.shift(1)`) so a weight set at `t`
cannot capture `t`'s own return — asserted by `tests/test_backtest.py`.

**Portfolio construction & costs** (`src/backtest.py`). Monthly rebalance; long
top decile / short bottom decile, dollar-neutral; also a rank-weighted variant.
Transaction costs = `bps/side × turnover`, charged on the rebalance date;
results reported gross **and net**.

**Evaluation** (`src/evaluation.py`). Spearman IC per period with Newey–West
(HAC) t-stats and information ratio; annualized return/vol/Sharpe, max drawdown,
turnover; **Deflated Sharpe Ratio** (López de Prado) accounting for the number of
configurations tried; **factor-neutral alpha** from regressing the long–short
return on FF5 + momentum (HAC errors); baselines = equal-weight market, a
no-skill random signal, and each single signal.

**Purged & embargoed CV** (`src/cv.py`). López de Prado purged + embargoed
walk-forward / k-fold splits; `tests/test_cv.py` asserts no train index falls
within `horizon + embargo` of any test index.

**RL extension** (`src/rl_allocator.py`). Signal combination framed as a
contextual bandit: state = market-regime features + lagged realized per-signal
returns; action = which signal to tilt toward; reward = next-period net return.
LinUCB and linear Thompson sampling are provided, trained online walk-forward
(a reward is only used after it is realized — no leakage), and benchmarked
against equal-weight and point-in-time mean–variance combination.

---

## Repo structure

```
equity-signal-research/
  README.md                 # this file (honest results + reproduce commands)
  LIMITATIONS.md            # where the edge is weak/absent; biases; caveats
  requirements.txt
  src/
    universe.py             # S&P 500 snapshot (with survivorship caveat)
    data.py                 # yfinance prices + Ken French factors, cached
    signals.py              # point-in-time cross-sectional signals
    cv.py                   # purged + embargoed walk-forward splits
    backtest.py             # portfolio construction, cost model, simulator
    evaluation.py           # IC, NW t-stats, deflated Sharpe, FF alpha, plots
    rl_allocator.py         # LinUCB / Thompson bandit + EW/MV baselines
    pipeline.py             # shared data-prep used by the experiments
  experiments/
    run_baseline.py         # end-to-end signal study -> results/
    run_rl.py               # RL allocator vs baselines -> results/
  results/                  # CSV tables + PNG figures (reproducible)
  tests/                    # look-ahead, cost, CV purge, IC sign
  data/                     # cached parquet (git-ignored)
```

## Resume line (fill with the numbers above)

> Built a cross-sectional return-prediction study over 402 liquid US equities,
> evaluated as a dollar-neutral long–short portfolio with explicit transaction
> costs; reported out-of-sample mean IC ≈ 0.015 (Newey–West t ≈ 1.9) and a
> **deflated** (multiple-testing-corrected) Sharpe of ≈ 0.22 — i.e. the apparent
> edge is not significant after selection and survivorship bias. Combined
> signals via a LinUCB contextual-bandit / mean–variance allocator and reported,
> honestly, that the learned allocator did not beat a mean–variance baseline.

See `LIMITATIONS.md` for the full list of caveats before trusting any number.

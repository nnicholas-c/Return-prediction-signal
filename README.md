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

Run: **point-in-time S&P 500 membership** — 614 of 805 ever-members usable
(191 missing/unrecoverable, see coverage note), daily data
**2010-01-04 → 2024-12-30**, monthly rebalance, 21-day forward-return target,
**10 bps/side** transaction costs. Selection across **12 configurations**
(6 signals × {decile, rank}).

| Metric | Value |
|---|---|
| Best signal (by net Sharpe) | **12–1 momentum** (decile) |
| Mean IC | **+0.0052** (Newey–West t = **+0.37**, IR = +0.024) |
| Long–short Sharpe, gross | **+0.07** |
| Long–short Sharpe, **net of costs** | **+0.01** |
| **Deflated Sharpe Ratio** (corrects for 12 trials) | **0.35** |
| FF5 + momentum alpha | **−3.3%/yr** (t = **−1.21**) |
| Avg turnover / rebalance | 1.06 |
| Max drawdown (net) | −64.4% |
| Baseline: equal-weight market (excess) Sharpe | +0.86 |
| Baseline: no-skill random signal Sharpe | −0.70 |

**How to read this honestly:**

1. Under point-in-time membership, **no signal survives**. The best-by-net-Sharpe
   signal (12–1 momentum) has a net Sharpe of ≈ **0.01** and a deflated Sharpe of
   **0.35** — well below 0.5, i.e. **not significant** after the multiple-testing
   correction. Every long–short signal's net Sharpe is ≈ 0 or negative
   (`results/signal_summary.csv`).
2. Simply **owning the equal-weight market beat every long–short signal**
   (market excess Sharpe +0.86).
3. The headline does not depend on the daily/monthly choice (verified): the
   deflated Sharpe is 0.35 (daily) vs 0.30 (monthly), both < 0.5.

### The survivorship before/after (the key result)

Moving from today's S&P 500 snapshot to **point-in-time membership** removes the
look-back bias the original study flagged — and it **erases the one signal that
looked good**. The previously "best" signal, an illiquidity/low-dollar-volume
tilt, was a survivorship artifact (`results/survivorship_comparison.csv`,
`results/survivorship_comparison.png`):

| illiquidity signal | Snapshot (biased) | Point-in-time | Δ |
|---|---|---|---|
| Mean IC | +0.0153 | **−0.0060** | −0.0213 |
| IC Newey–West t | +1.92 | **−0.84** | −2.77 |
| Long–short Sharpe (net) | +0.53 | **−0.21** | −0.74 |
| FF5+momentum alpha | +8.0%/yr | **−0.7%/yr** | −8.7 pp |
| alpha t-stat | +3.59 | **−0.33** | −3.92 |

The snapshot universe even produced a strong *pre-2020* illiquidity Sharpe of
+1.27; point-in-time, the selected signal's pre-2020 Sharpe is +0.05. In other
words, the apparent edge was almost entirely the bias. This is the expected,
correct outcome and the central finding of the project.

#### Independent verification & robustness

The headline statistics are re-derived from scratch (without trusting
`src/evaluation.py`) by `python -m experiments.verify_headline`, which confirms
(under the point-in-time universe):

- IC Newey–West t-stat matches a hand-rolled Bartlett-kernel HAC estimator;
- the deflated Sharpe matches a from-scratch computation;
- the FF5+momentum alpha matches a plain numpy least-squares fit;
- decile weights are exactly dollar-neutral and cost accounting reconciles to
  machine precision;
- signals are point-in-time (identical when recomputed on truncated data), and
  at each rebalance the portfolio only holds names that were index members
  **as of** that date (`tests/test_universe.py`);
- the conclusion is **frequency-robust**: deflated Sharpe 0.35 (daily) vs 0.30
  (monthly per-rebalance returns), and the alpha t-stat is stable across HAC lag
  lengths (5 → 42).

### RL / contextual-bandit allocator (extension)

Out-of-sample net Sharpe of signal-combination methods, point-in-time universe
(`results/rl_allocator_summary.csv`):

| Method | OOS net Sharpe |
|---|---|
| Learned LinUCB contextual bandit | **−0.05** |
| Equal-weight combination | −0.23 |
| Mean–variance combination (point-in-time) | −0.56 |

**All three allocators lose money out-of-sample** under point-in-time data. The
learned LinUCB has the highest (least-negative) OOS Sharpe, but a negative Sharpe
is not a "win" — there is simply no profitable combination of signals to find
once the survivorship bias is removed. Reported plainly, as intended.

---

## Reproduce

```bash
pip install -r requirements.txt

# End-to-end signal study -> results/  (default --universe pit; caches on first run)
python -m experiments.run_baseline --universe pit
python -m experiments.run_baseline --universe snapshot     # old survivorship-biased run

# Survivorship before/after: snapshot vs point-in-time -> survivorship_comparison.{csv,png}
python -m experiments.run_comparison

# RL / contextual-bandit allocator vs baselines -> results/
python -m experiments.run_rl --universe pit --learner linucb

# Independently re-derive & stress-test the headline stats (prints PASS/FAIL)
python -m experiments.verify_headline --universe pit

# Tests (no-look-ahead, cost accounting, CV purging, IC sign, PIT membership)
pytest -q
```

Useful flags: `--universe {pit,snapshot}`, `--max-tickers N` (smaller/faster
universe), `--cost-bps`, `--horizon`, `--standardize {zscore,rank}`,
`--learner {linucb,thompson}`, `--force` (re-download). The first point-in-time
run downloads the historical-constituents list and the wider price set (the union
of all ever-members, ~800 tickers) from Yahoo Finance + the Ken French Data
Library and caches to `data/cache/` (git-ignored); later runs are fast and
offline-friendly.

Outputs written to `results/`: `signal_summary.csv`, `baselines.csv`,
`robustness.csv`, `rl_allocator_summary.csv`, `survivorship_comparison.csv`,
`headline_*.json`, and figures (`equity_*.png`, `ic_*.png`,
`survivorship_comparison.png`, `rl_allocator_curves.png`).

---

## Data (free, no credentials)

- **Prices:** daily adjusted OHLCV via `yfinance` (`auto_adjust=True`). A
  data-quality filter (`src/data.py: clean_prices`) drops tickers with corrupted
  Yahoo series (e.g. an 8000× one-day "return") while keeping legitimate large
  moves (e.g. GameStop's real +135% squeeze day).
- **Universe (two modes):**
  - `pit` (**default**) — **point-in-time S&P 500 membership** from the free
    `fja05680/sp500` historical-constituents list (`src/universe_pit.py`). At
    each rebalance date the tradable set is the constituents *as of* that date,
    including names later removed. This is the survivorship-bias reduction.
  - `snapshot` — today's S&P 500 list (`src/universe.py`), kept for the
    before/after comparison. **Not point-in-time** (survivorship-biased).
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
    universe_pit.py         # point-in-time index membership (members_asof)
    data.py                 # yfinance prices + Ken French factors, quality filter, cached
    signals.py              # point-in-time cross-sectional signals
    cv.py                   # purged + embargoed walk-forward splits
    backtest.py             # portfolio construction, cost model, simulator
    evaluation.py           # IC, NW t-stats, deflated Sharpe, FF alpha, plots
    rl_allocator.py         # LinUCB / Thompson bandit + EW/MV baselines
    pipeline.py             # shared data-prep (snapshot/pit), membership masking
  experiments/
    run_baseline.py         # end-to-end signal study -> results/
    run_comparison.py       # snapshot vs point-in-time survivorship comparison
    run_rl.py               # RL allocator vs baselines -> results/
    verify_headline.py      # independent re-derivation of the headline stats
  report/                   # short written report (markdown, + PDF if pandoc)
  results/                  # CSV tables + PNG figures (reproducible)
  tests/                    # look-ahead, cost, CV purge, IC sign, PIT membership
  data/                     # cached parquet (git-ignored)
```

## Resume line (uses the point-in-time numbers above)

> Built a cross-sectional return-prediction study over a **point-in-time** S&P
> 500 universe (≈614 of 805 historical constituents, survivorship-controlled),
> evaluated as a dollar-neutral long–short portfolio with explicit transaction
> costs, purged/embargoed walk-forward CV, Newey–West IC t-stats and a
> **deflated** (multiple-testing-corrected) Sharpe ratio. Found that **no signal
> survives**: best out-of-sample mean IC ≈ 0.005 (t ≈ 0.4), net Sharpe ≈ 0,
> deflated Sharpe ≈ 0.35 (< 0.5). Demonstrated that a prior "illiquidity alpha"
> (+8%/yr, t ≈ 3.6 on a current-snapshot universe) was a **survivorship
> artifact** that vanished (alpha ≈ 0, t ≈ −0.3) under point-in-time membership.
> Added a LinUCB contextual-bandit allocator and reported honestly that no
> signal combination is profitable out of sample once the bias is removed.

See `report/equity_signal_research_report.md` for the full write-up and
`LIMITATIONS.md` for the caveats before trusting any number.

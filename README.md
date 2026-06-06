# equity-signal-research

**A cross-sectional equity signal-research study built to quant-research review
standards — and an honest null result.** Six well-documented price/volume signals
are turned into dollar-neutral long–short portfolios and evaluated on a
**point-in-time** S&P 500 universe with transaction costs, purged/embargoed
walk-forward CV, Newey–West IC t-stats, factor-neutral alpha, and a
multiple-testing-corrected **deflated Sharpe ratio**. A contextual-bandit
allocator is included as an RL extension.

> **The headline isn't a Sharpe — it's judgment.** On a current-snapshot
> universe one signal showed a **+8.0%/yr factor alpha (t ≈ 3.6)**. Rebuilt on
> **point-in-time index membership**, that alpha **vanishes to −0.7%/yr
> (t ≈ −0.3)** — it was survivorship bias, not skill. After costs,
> multiple-testing correction, and survivorship control, **no signal survives**
> (best deflated Sharpe **0.35 < 0.5**). Every number is reproducible from a
> command and independently re-derived; nothing is hardcoded.

---

## About this project

I started this project to understand how quant researchers actually decide
whether a trading signal is real, rather than just whether it looks good in a
backtest. I took six classic price/volume signals (momentum, reversal, low
volatility, idiosyncratic volatility, 52-week-high proximity, and an illiquidity
proxy), turned each into a dollar-neutral long–short portfolio, and tried to
evaluate them as honestly as I could.

The most useful thing I learned came from a mistake I almost made. On a universe
of *today's* S&P 500 members, the illiquidity signal looked genuinely good — a
+8%/yr factor alpha with a t-stat around 3.6. It would have been easy to write
that down as a finding. But that universe only contains companies that survived
and stayed in the index, so I rebuilt the whole study on **point-in-time index
membership** (the constituents as they actually were on each date, including the
ones that were later dropped or delisted). The alpha disappeared — it dropped to
roughly −0.7%/yr with a t-stat near zero. The "edge" was survivorship bias, not
skill. That single before/after comparison taught me more about backtesting than
any positive result would have.

The other lesson was about being honest with statistics. Because I tried twelve
configurations and picked the best one, I learned to discount the Sharpe with a
**deflated Sharpe ratio** that accounts for how many things I tried — and once I
did, nothing was significant. So the final result is a clean null: after
transaction costs, multiple-testing correction, and survivorship control, none of
these simple signals earns a credible return. I think being able to reach and
report that conclusion is the actual skill here.

Along the way I implemented, mostly from first principles, the pieces a quant
researcher would expect to see:

- a **point-in-time universe** layer so the backtest never trades a name before
  it joined or after it left the index;
- **strict point-in-time features and labels** — backward-looking signals only,
  weights lagged one day so a weight set at *t* can't earn *t*'s return, and
  membership masking applied *before* cross-sectional standardization;
- **purged + embargoed walk-forward cross-validation** (López de Prado) to avoid
  train/test leakage across the label horizon;
- a **dollar-neutral long–short backtester** with a transaction-cost model
  (bps/side × turnover), reported gross *and* net;
- **Information Coefficients with Newey–West (HAC) t-stats**, a **deflated Sharpe
  ratio**, and **factor-neutral alpha** against Fama–French 5 factors + momentum;
- sensible **baselines** (equal-weight market, a no-skill random signal); and
- an **RL extension** — a LinUCB / Thompson-sampling contextual bandit that tries
  to allocate across the signals, trained walk-forward with the same no-leakage
  discipline (it didn't beat the simple baselines, and I say so).

To keep myself honest I wrote 50 unit tests (look-ahead, cost accounting, CV
purging, IC sign conventions, membership) and a separate script that re-derives
every headline number from scratch without trusting my own library code.

**Stack:** Python · NumPy · pandas · SciPy · scikit-learn · statsmodels ·
matplotlib · yfinance · Ken French Data Library.

---

## Headline results (point-in-time, real & reproducible)

Point-in-time S&P 500 membership · **614 of 805 ever-members usable** (191
missing/unrecoverable) · daily data **2010-01-04 → 2024-12-30** · monthly
rebalance · 21-day forward target · **10 bps/side** costs · selection across
**12 configurations** (6 signals × {decile, rank}).

| Metric | Value |
|---|---|
| Best signal (by net Sharpe) | **12–1 momentum** (decile) |
| Mean IC | **+0.0052** (Newey–West t = **+0.37**) |
| Long–short Sharpe, gross / **net** | +0.07 / **+0.01** |
| **Deflated Sharpe** (12 trials) | **0.35**  → not significant |
| FF5 + momentum alpha | **−3.3%/yr** (t = −1.21) |
| Avg turnover / rebalance | 1.06 |
| Max drawdown (net) | −64.4% |
| Baseline — equal-weight market (excess) Sharpe | **+0.86** |
| Baseline — no-skill random signal Sharpe | −0.70 |

**Read:** every signal's net Sharpe is ≈ 0 or negative; the deflated Sharpe is
below 0.5 (probability the true Sharpe beats the multiple-testing benchmark is
worse than a coin flip); and simply holding the market beat every long–short.
The conclusion is frequency-robust (deflated Sharpe 0.35 daily vs 0.30 monthly)
and stable across HAC lag choices.

### The survivorship before/after — the centerpiece

Holding the methodology fixed and changing **only** the universe (today's
snapshot → point-in-time membership) removes the look-back bias and erases the
one signal that looked good. The previously "best" signal — an illiquidity /
low-dollar-volume tilt — was a survivorship artifact
(`results/survivorship_comparison.csv`, `results/survivorship_comparison.png`):

| illiquidity signal | Snapshot (biased) | Point-in-time | Δ |
|---|---|---|---|
| Mean IC | +0.0153 | **−0.0060** | −0.0213 |
| IC Newey–West t | +1.92 | **−0.84** | −2.77 |
| Long–short Sharpe (net) | +0.53 | **−0.21** | −0.74 |
| FF5 + momentum alpha | +8.0%/yr | **−0.7%/yr** | −8.7 pp |
| alpha t-stat | +3.59 | **−0.33** | −3.92 |

The snapshot even manufactured a +1.27 *pre-2020* illiquidity Sharpe; point-in-
time, the selected signal's pre-2020 Sharpe is +0.05. The "edge" was the bias.

### RL / contextual-bandit allocator (extension)

Out-of-sample net Sharpe of signal-combination methods, point-in-time
(`results/rl_allocator_summary.csv`):

| Method | OOS net Sharpe |
|---|---|
| Learned LinUCB contextual bandit | **−0.05** |
| Equal-weight combination | −0.23 |
| Mean–variance combination (point-in-time) | −0.56 |

All three allocators **lose money out-of-sample**. The learned bandit is the
least-negative, but a negative Sharpe is not a win — there is no profitable
combination once the bias is removed. Reported plainly, as intended.

---

## Reproduce

```bash
pip install -r requirements.txt

# Headline study (default --universe pit) -> results/
python -m experiments.run_baseline --universe pit
python -m experiments.run_baseline --universe snapshot     # old survivorship-biased run

# Survivorship before/after -> survivorship_comparison.{csv,png}
python -m experiments.run_comparison

# RL / contextual-bandit allocator vs baselines -> results/
python -m experiments.run_rl --universe pit --learner linucb

# Independently re-derive & stress-test every headline stat (prints PASS/FAIL)
python -m experiments.verify_headline --universe pit

# Rigor tests (look-ahead, cost accounting, CV purge, IC sign, PIT membership)
pytest -q
```

The first point-in-time run downloads the historical-constituents list and the
union of all ever-members (~800 tickers) from Yahoo Finance + the Ken French Data
Library, caching to `data/cache/` (git-ignored); later runs are fast and offline.
Useful flags: `--universe {pit,snapshot}`, `--cost-bps`, `--horizon`,
`--standardize {zscore,rank}`, `--learner {linucb,thompson}`, `--force`.

---

## How it works

**Signals** (`src/signals.py`) — six, strictly point-in-time, each oriented
long-high using the academic prior fixed *before* looking at the data: 12–1
momentum, 1-month reversal, low realized volatility, low idiosyncratic volatility
(residual vs market), 52-week-high proximity, and an illiquidity proxy.
Cross-sectionally z-scored/ranked within each date.

**Universe** (`src/universe_pit.py`, `src/universe.py`) — point-in-time S&P 500
membership (`members_asof(t)`, default) from the free `fja05680/sp500` list, or a
current snapshot for the comparison. A data-quality filter (`src/data.py:
clean_prices`) drops corrupted Yahoo series (e.g. an 8000× one-day "return")
while keeping legitimate moves (e.g. GameStop's real +135% squeeze day).

**Backtest** (`src/backtest.py`) — monthly rebalance; long top decile / short
bottom decile (and a rank-weighted variant), dollar-neutral; weights lagged one
day; costs = bps/side × turnover.

**Evaluation** (`src/evaluation.py`) — Spearman IC + Newey–West t-stats; Sharpe,
drawdown, turnover; deflated Sharpe; FF5+momentum alpha; baselines; figures.

**CV** (`src/cv.py`) — purged + embargoed walk-forward / k-fold splits.

**RL** (`src/rl_allocator.py`) — LinUCB / Thompson bandit over the signals;
state = market-regime + lagged realized per-signal returns; reward = next-period
net return; benchmarked vs equal-weight and mean–variance.

---

## Repo structure

```
src/
  universe.py        # current S&P 500 snapshot (survivorship caveat)
  universe_pit.py    # point-in-time index membership (members_asof)
  data.py            # yfinance prices + Ken French factors, quality filter, cached
  signals.py         # point-in-time cross-sectional signals
  cv.py              # purged + embargoed walk-forward splits
  backtest.py        # portfolio construction, cost model, simulator
  evaluation.py      # IC, NW t-stats, deflated Sharpe, FF alpha, plots
  rl_allocator.py    # LinUCB / Thompson bandit + EW/MV baselines
  pipeline.py        # shared data-prep (snapshot/pit), membership masking
experiments/
  run_baseline.py    # end-to-end signal study -> results/
  run_comparison.py  # snapshot vs point-in-time survivorship comparison
  run_rl.py          # RL allocator vs baselines -> results/
  verify_headline.py # independent re-derivation of the headline stats
report/              # written report (markdown + PDF)
results/             # CSV tables + PNG figures (reproducible)
tests/               # 50 tests: look-ahead, cost, CV purge, IC sign, PIT
```

See `report/equity_signal_research_report.md` for the full write-up and
`LIMITATIONS.md` for every caveat (residual survivorship from the price-coverage
gap, unmodelled delisting returns, cost-model simplifications) before trusting
any number.

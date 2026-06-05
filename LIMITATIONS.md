# Limitations & honest caveats

This project is graded on rigor and honesty, not on backtest performance. This
file states plainly where the results are weak, biased, or could be wrong. Read
it **before** trusting any number in `README.md` or `results/`.

## 1. Survivorship / universe bias (the big one)

- The universe (`src/universe.py`) is a snapshot of **today's** S&P 500
  constituents. This is **not point-in-time**: we only include firms that
  survived and were promoted into the index over 2010–2024. Failed, delisted,
  acquired, and demoted names are absent.
- This biases results **upward**, and it specifically inflates the one signal
  that looked best here — **illiquidity** (low dollar volume). Within a
  survivorship-filtered large-cap set, "less liquid" largely means "smaller,
  more recently added" names that, conditional on still being in the index
  today, did well. The +8%/yr factor alpha (t≈3.6) should be read with heavy
  skepticism for exactly this reason.
- A point-in-time historical-constituent membership snapshot would materially
  reduce this bias. We did not have a free one wired in; this is the single most
  important fix for any follow-up.

## 2. Multiple testing — already corrected, and it matters

- We tried 12 configurations (6 signals × {decile, rank}) and selected the best
  by net Sharpe. The **deflated Sharpe ratio = 0.22** says: after correcting for
  that selection, the probability the true Sharpe exceeds the benchmark
  (SR₀ ≈ 0.73 annualized) is only ~22%. **The apparent edge is not significant
  once you account for the search.**
- The deflated-Sharpe `n_trials` only counts configurations evaluated *inside a
  single run*. The true number of researcher degrees of freedom (window choices,
  cost assumptions, signal definitions explored during development) is larger, so
  even 0.22 is an optimistic upper bound on significance.

## 3. Regime dependence

- The selected signal's net Sharpe is **+1.27 pre-2020** and **−0.20
  post-2020** (`results/robustness.csv`). The edge essentially disappears (or
  reverses) in the more recent regime. Any single headline Sharpe hides this.

## 4. The market beat the long–shorts

- Equal-weight market excess Sharpe (+0.90) exceeds the best long–short net
  Sharpe (+0.53). On this universe and period, simple market exposure dominated
  the cross-sectional signals net of costs. The long–short framing is about
  *factor-neutral* alpha, but the practical takeaway is sobering.

## 5. Transaction-cost model is simplified

- Costs = `bps/side × turnover`, charged on the rebalance date. This ignores the
  bid–ask spread term structure, market impact (nonlinear in size), borrow
  fees / short-availability for the short leg, and financing. Turnover is
  measured **target-to-target** (it ignores intra-period drift), which slightly
  misstates true turnover. Real-world costs for the short leg of an illiquidity
  signal would likely be **higher** than modeled, further eroding the edge.

## 6. Idiosyncratic-vol and turnover approximations

- Idiosyncratic volatility is the residual std from a **single-factor** (market)
  model via a closed-form rolling formula, not a full FF5+momentum residual.
- "Turnover" uses **average dollar volume** as a liquidity proxy because we lack
  point-in-time shares outstanding for true turnover (volume / shares). It is
  therefore really an (il)liquidity / size proxy, not classic turnover.

## 7. Point-in-time fidelity

- Signals are backward-looking and verified point-in-time
  (`tests/test_signals.py` recomputes on truncated data and asserts equality),
  and weights are applied with a one-day lag (`tests/test_backtest.py`). However:
  - Yahoo Finance **adjusted** prices are restated over time (dividend/split
    adjustments use the full history), so the *adjusted* close at an old date is
    not exactly what was observable then. This is a common, usually small,
    source of subtle look-ahead in `yfinance`-based studies.
  - Fama–French factors are reindexed to the trading calendar and forward-filled
    on rare mismatched days.

## 8. RL / bandit extension is deliberately small

- The contextual bandit selects a single signal per period (a 100% tilt), not a
  fully continuous allocation; rewards used for learning are per-signal net
  returns, while the realized combined P&L additionally pays allocation-switching
  costs (a minor mismatch). The honest result is that the learned LinUCB
  allocator **underperformed** both equal-weight and mean–variance combination
  out of sample. We did not tune it to look good.

## 9. Statistical caveats

- IC t-stats use Newey–French/Newey–West HAC errors, but the IC series is short
  (~170 monthly periods); small-sample inference is noisy.
- Sharpe ratios assume returns are stationary within the sample; drawdowns show
  they are not.

## Where this signal is weak or fails (summary)

- **Fails** the multiple-testing bar (deflated Sharpe 0.22).
- **Fails** out-of-sample in the post-2020 regime (Sharpe −0.20).
- **Fails** to beat simply holding the market.
- The one positive headline (illiquidity alpha) is **most exposed** to the
  survivorship bias we cannot fully remove with a free, current-constituent
  universe.
- The RL allocator **does not** improve on a plain mean–variance combination.

Net assessment: a clean, honest **null-to-weak** result. That is the intended
deliverable for this project.

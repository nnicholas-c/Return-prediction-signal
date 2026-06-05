# Limitations & honest caveats

This project is graded on rigor and honesty, not on backtest performance. This
file states plainly where the results are weak, biased, or could be wrong. Read
it **before** trusting any number in `README.md` or `results/`.

## 1. Survivorship / universe bias — now REDUCED, not eliminated

The default universe is now **point-in-time S&P 500 membership**
(`src/universe_pit.py`, source: `fja05680/sp500`): at each rebalance date the
tradable set is the constituents *as of* that date, including names later
removed. This directly attacks the look-back bias the original snapshot run had —
and, as expected, it **erased the one signal that looked good**: the illiquidity
"alpha" went from +8.0%/yr (t≈3.6) on the snapshot to −0.7%/yr (t≈−0.3)
point-in-time (`results/survivorship_comparison.csv`). The bias was the edge.

The `snapshot` mode (today's constituents) is retained only for that comparison;
it remains survivorship-biased and should not be read as a result.

**Residual bias still present — do not over-claim "bias-free":**

- **Price-coverage gap.** Of 805 tickers that were ever S&P 500 members in
  2010–2024, only **614 have usable Yahoo Finance prices**; **191 are
  missing/unrecoverable** (delisted, renamed beyond our remaps, or data so
  corrupted the quality filter dropped them). Missing names skew toward
  failures/acquisitions, so a *residual* survivorship bias remains — the true
  effect of removing bias is, if anything, slightly understated.
- **Delisting returns are not modelled.** When a name leaves the index (or
  delists) it simply stops contributing to the portfolio; we do not apply a
  terminal delisting return. For bankruptcies this understates short-leg gains /
  long-leg losses. Standard CRSP-style delisting returns would fix this but are
  not free.
- **Membership dates are end-of-event, not intraday**, and the source list
  itself may contain small dating errors.
- **Data-quality filter** (`src/data.py: clean_prices`) drops tickers with a
  max |daily return| > 200% or a min adjusted close < $0.05. Thresholds are
  loose (they keep e.g. GameStop's real +135% squeeze day) but are still a
  judgment call that removes a handful of historical members.

## 2. Multiple testing — already corrected, and it matters

- We tried 12 configurations (6 signals × {decile, rank}) and selected the best
  by net Sharpe. On the point-in-time universe the **deflated Sharpe ratio =
  0.35** (< 0.5): after correcting for the selection, the probability the true
  Sharpe exceeds the multiple-testing benchmark is below even chance. **No edge
  is significant once you account for the search.**
- The deflated-Sharpe `n_trials` only counts configurations evaluated *inside a
  single run*. The true number of researcher degrees of freedom (window choices,
  cost assumptions, signal definitions explored during development) is larger, so
  even 0.35 is an optimistic upper bound on significance.
- We report the deflated Sharpe on **daily** returns, which is statistically
  aggressive for a monthly-rebalanced strategy (daily returns inside a holding
  month are autocorrelated, inflating the effective sample size). As a check,
  `experiments/verify_headline.py` recomputes it on **monthly** per-rebalance
  returns and gets essentially the same answer (0.30 vs 0.35), and the FF-alpha
  t-stat is stable across HAC lag lengths (5–42). The null conclusion is
  therefore not an artifact of the return frequency.

## 3. Regime dependence

- The point-in-time selected signal is weak in both regimes (net Sharpe ≈ +0.05
  pre-2020, ≈ −0.07 post-2020; `results/robustness.csv`). The snapshot universe
  had manufactured a strong pre-2020 illiquidity Sharpe (+1.27) that was a
  survivorship artifact — it does not survive point-in-time membership.

## 4. The market beat the long–shorts

- Equal-weight market excess Sharpe (+0.86) exceeds every long–short net Sharpe
  (best ≈ +0.01). On this universe and period, simple market exposure dominated
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
  costs (a minor mismatch). The honest point-in-time result is that **all three
  allocators lose money out of sample** (learned LinUCB ≈ −0.05, equal-weight
  ≈ −0.23, mean–variance ≈ −0.56). The learned allocator is the least-negative
  but a negative Sharpe is not a win — there is no profitable combination to find.

## 9. Statistical caveats

- IC t-stats use Newey–West HAC errors, but the IC series is short (~170 monthly
  periods); small-sample inference is noisy.
- Sharpe ratios assume returns are stationary within the sample; drawdowns show
  they are not.

## Where this signal is weak or fails (summary)

- **Fails** the multiple-testing bar (point-in-time deflated Sharpe 0.35 < 0.5).
- The prior "illiquidity alpha" was a **survivorship artifact** that vanished
  point-in-time (+8%/yr → −0.7%/yr).
- **Fails** to beat simply holding the market (best L–S net Sharpe ≈ 0 vs market
  excess +0.86).
- **No signal combination is profitable** out of sample (all allocators have a
  negative net Sharpe).
- A **residual** survivorship bias remains via the 191/805 price-coverage gap and
  the unmodelled delisting returns — so the true picture is, if anything, even
  weaker than reported.

Net assessment: a clean, honest **null** result, strengthened (not rescued) by
removing survivorship bias. That is the intended deliverable for this project.

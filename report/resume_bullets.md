# Résumé material — equity-signal-research

Copy whichever fits your résumé's length. All numbers are reproducible from the
repo (`python -m experiments.run_baseline --universe pit`,
`run_comparison`, `verify_headline`) and were double-checked.

---

## One-line project title

**Cross-Sectional Equity Signal Research — point-in-time backtesting, deflated
Sharpe, factor-neutral alpha, RL allocator (Python)**

---

## Short version (2 bullets)

- Built a cross-sectional equity return-prediction study over a **point-in-time**
  S&P 500 universe (~614 of 805 historical constituents, survivorship-controlled),
  evaluating six price/volume signals as dollar-neutral long–short portfolios with
  transaction costs, **purged & embargoed walk-forward CV**, **Newey–West IC
  t-stats**, factor-neutral (Fama–French 5 + momentum) alpha, and a
  **deflated (multiple-testing-corrected) Sharpe ratio**.
- Demonstrated rigor and intellectual honesty: showed a **+8%/yr factor alpha
  (t ≈ 3.6)** from a current-index universe was a **survivorship artifact** that
  vanished to **−0.7%/yr (t ≈ −0.3)** under point-in-time membership, and reported
  the honest null — **no signal survives costs + multiple testing + survivorship
  correction** (deflated Sharpe ≈ 0.35 < 0.5).

---

## Detailed version (3–4 bullets)

- Engineered an end-to-end cross-sectional equity research pipeline in **Python
  (NumPy/pandas/SciPy/statsmodels/scikit-learn)**: free data ingestion (yfinance
  prices + Ken French factors) with on-disk caching, a **point-in-time S&P 500
  membership** layer (`members_asof(t)`), six point-in-time signals (12–1
  momentum, short-term reversal, low/idiosyncratic volatility, 52-week-high
  proximity, illiquidity), and a dollar-neutral long–short backtester with a
  transaction-cost model (reported gross **and** net).
- Implemented quant-grade evaluation from first principles: **Information
  Coefficient with Newey–West (HAC) t-stats**, **purged & embargoed walk-forward
  cross-validation** (López de Prado), a **multiple-testing-corrected deflated
  Sharpe ratio**, and **factor-neutral alpha** vs Fama–French 5 + momentum.
- Isolated and quantified **survivorship bias** by re-running the identical study
  on point-in-time vs current-snapshot universes: a headline illiquidity signal’s
  factor alpha collapsed from **+8.0%/yr (t ≈ 3.6) to −0.7%/yr (t ≈ −0.3)**,
  proving the “edge” was look-back bias; concluded honestly that **no signal earns
  a statistically credible net return** (best deflated Sharpe ≈ 0.35).
- Extended to sequential decision-making: built a **LinUCB / Thompson-sampling
  contextual-bandit allocator** for signal combination, trained online
  walk-forward with no leakage, and reported it did not beat simple baselines —
  backed by **50 unit tests** (look-ahead, cost accounting, CV purging, IC sign,
  membership) and an independent re-derivation of every headline statistic.

---

## Skills / keywords line (for a skills section)

Quantitative research · cross-sectional alpha signals · point-in-time data &
survivorship-bias control · purged/embargoed walk-forward CV · Information
Coefficient (Newey–West HAC) · deflated Sharpe ratio (multiple-testing) ·
Fama–French factor-neutral alpha · transaction-cost backtesting · contextual
bandits (LinUCB/Thompson) · Python (NumPy, pandas, SciPy, statsmodels,
scikit-learn) · reproducible research, unit testing

---

## Why a quant-research recruiter should care (talking points for interviews)

- **You won’t fool yourself.** The single most valued trait in QR is not finding
  alpha — it’s not *claiming* alpha that isn’t there. This project’s headline is a
  worked example of catching a real-looking +8%/yr alpha (t ≈ 3.6) and proving it
  was survivorship bias.
- **You know the honesty toolkit.** Deflated Sharpe (multiple testing), Newey–West
  t-stats (autocorrelation), purged/embargoed CV (leakage), point-in-time data
  (look-ahead), and cost-net reporting — all implemented, not just named.
- **You can falsify your own result.** A separate script re-derives every headline
  number from scratch and a 50-test suite enforces no-look-ahead, correct cost
  accounting, and correct CV purging.
- **You can frame a problem as sequential decision-making** (the contextual-bandit
  allocator) and still report a negative outcome plainly.

---

## Honest framing note

This is a **null result**, and that is the point. Present it as: “I built the
evaluation a quant researcher would trust, and used it to show that the obvious
signals don’t work net of costs once survivorship bias and multiple testing are
handled correctly.” Do not present any positive Sharpe from the snapshot universe
as real — it is the bias, and the repo says so.

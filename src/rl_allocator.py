"""Contextual-bandit signal allocator and baselines.

We frame *signal combination* as a sequential decision problem solved online,
walk-forward, with the same purge/embargo discipline used elsewhere:

* **State / context** ``x_t``: simple, point-in-time market-regime features
  (trailing market return and volatility over a couple of horizons) plus the
  most recently *realized* per-signal period returns.
* **Action**: which candidate signal to tilt the portfolio toward this period
  (a disjoint-arm contextual bandit selects one signal; the action is a 100%
  tilt to that signal's dollar-neutral long-short portfolio).
* **Reward**: the selected signal's next-period long-short return, net of costs.

Two learners are provided: **LinUCB** (Li et al. 2010) and **linear Thompson
sampling**.  Crucially, the bandit is updated with a reward only *after it has
been realized* (period ``j``'s reward is known at ``j+1``), so when acting at
period ``i`` it has only used information available at ``i`` -- no leakage.

Baselines: equal-weight signal combination and a point-in-time mean-variance
combination estimated from past realized per-signal returns.

This is a research extension: the honest question is whether the learned
allocator improves *out-of-sample* net Sharpe over these simple baselines.  We
report the answer plainly, whatever it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest as bt


# --------------------------------------------------------------------------- #
# Bandit learners
# --------------------------------------------------------------------------- #
class LinUCB:
    """Disjoint linear UCB contextual bandit (one weight vector per arm)."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, seed: int = 0):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray) -> int:
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta @ x)
            bonus = self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
            scores[a] = mean + bonus
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


class LinThompson:
    """Linear Thompson sampling contextual bandit."""

    def __init__(self, n_arms: int, dim: int, v: float = 0.25, seed: int = 0):
        self.n_arms = n_arms
        self.dim = dim
        self.v = v
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray) -> int:
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            cov = (self.v ** 2) * A_inv
            theta = self.rng.multivariate_normal(theta_hat, cov)
            scores[a] = float(theta @ x)
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


# --------------------------------------------------------------------------- #
# Context features (point-in-time)
# --------------------------------------------------------------------------- #
def build_context(
    factors: pd.DataFrame,
    rebal: pd.DatetimeIndex,
    signal_period_rets: pd.DataFrame,
) -> pd.DataFrame:
    """Build regime + lagged-performance context for each rebalance date.

    All features use only information available at the rebalance date:
    trailing market return/vol (from realized factor returns up to ``t``) and the
    most recently realized per-signal period returns (shifted by one period).
    Features are standardized with an *expanding* (shifted) mean/std so scaling
    itself never peeks ahead.
    """
    market = (factors["Mkt-RF"] + factors["RF"]).sort_index()
    feats = pd.DataFrame(index=rebal)
    feats["mkt_ret_21"] = market.rolling(21).sum().reindex(rebal)
    feats["mkt_ret_63"] = market.rolling(63).sum().reindex(rebal)
    feats["mkt_vol_21"] = market.rolling(21).std().reindex(rebal)
    feats["mkt_vol_63"] = market.rolling(63).std().reindex(rebal)

    # Lagged realized per-signal performance (known at t because shifted by 1).
    lagged = signal_period_rets.reindex(rebal).shift(1)
    for c in lagged.columns:
        feats[f"perf_{c}"] = lagged[c]

    feats = feats.sort_index()
    # Expanding standardization using stats through the previous period only.
    mean = feats.expanding(min_periods=3).mean().shift(1)
    std = feats.expanding(min_periods=3).std().shift(1)
    z = (feats - mean) / std.replace(0.0, np.nan)
    z = z.clip(-3, 3).fillna(0.0)
    z["intercept"] = 1.0
    return z


# --------------------------------------------------------------------------- #
# Allocator driver
# --------------------------------------------------------------------------- #
@dataclass
class AllocatorResult:
    name: str
    daily_net: pd.Series
    weights_over_signals: pd.DataFrame  # rebalance date x signal
    combined_result: object             # BacktestResult


def _signal_weight_matrices(
    std_signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    config: bt.BacktestConfig,
    rebal: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Per-signal rebalance-date x ticker target-weight matrices."""
    mats: dict[str, pd.DataFrame] = {}
    for name, panel in std_signals.items():
        rows = {}
        for t in rebal:
            if t in panel.index:
                w = bt._build_weights(panel.loc[t], config)
                rows[t] = w.reindex(returns.columns).fillna(0.0)
        mats[name] = pd.DataFrame(rows).T.reindex(columns=returns.columns).fillna(0.0)
    return mats


def _combined_weights(
    alloc: pd.DataFrame, sig_mats: dict[str, pd.DataFrame], returns: pd.DataFrame
) -> pd.DataFrame:
    """Combine per-signal weight matrices using per-period signal allocations."""
    signals = list(sig_mats.keys())
    combined = None
    for s in signals:
        w_s = alloc[s].reindex(sig_mats[s].index).fillna(0.0)
        contrib = sig_mats[s].mul(w_s, axis=0)
        combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)
    return combined.reindex(columns=returns.columns).fillna(0.0)


def mean_variance_weights(
    history: pd.DataFrame, ridge: float = 1e-3
) -> np.ndarray:
    """Long-short mean-variance signal weights from past per-signal returns.

    ``w ~ Sigma^{-1} mu`` with ridge regularization, normalized so abs weights
    sum to 1.  Estimated only from realized history (point-in-time).
    """
    mu = history.mean().values
    cov = history.cov().values + ridge * np.eye(history.shape[1])
    try:
        w = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        w = mu
    s = np.abs(w).sum()
    if s == 0 or not np.isfinite(s):
        return np.ones(history.shape[1]) / history.shape[1]
    return w / s


def run_allocators(
    std_signals: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    config: bt.BacktestConfig | None = None,
    *,
    warmup_periods: int = 24,
    learner: str = "linucb",
    alpha: float = 1.0,
    seed: int = 0,
) -> dict[str, AllocatorResult]:
    """Run the learned allocator plus equal-weight and mean-variance baselines.

    Returns a dict of method name -> AllocatorResult.  Metrics should be computed
    on the out-of-sample window (rebalance index >= ``warmup_periods``).
    """
    config = config or bt.BacktestConfig()
    returns = returns.sort_index()
    rebal = bt.get_rebalance_dates(returns.index, config.rebalance_freq)
    # Keep rebalance dates where all signals exist.
    rebal = pd.DatetimeIndex([t for t in rebal if all(t in p.index for p in std_signals.values())])

    signals = list(std_signals.keys())
    n_arms = len(signals)
    sig_mats = _signal_weight_matrices(std_signals, returns, config, rebal)

    # Per-signal per-period net returns (reward signal for the bandit).
    sig_period_rets = {}
    for name in signals:
        res_s = bt.simulate_weights(sig_mats[name], returns, cost_bps=config.cost_bps,
                                    config=config, rebalance_dates=rebal)
        sig_period_rets[name] = bt.period_returns_from_daily(res_s.daily_net, rebal)
    sig_period_rets = pd.DataFrame(sig_period_rets).reindex(rebal)

    context = build_context(factors, rebal, sig_period_rets)
    feat_cols = list(context.columns)
    dim = len(feat_cols)

    # --- Learned allocator (one-hot tilt per period) ---
    if learner == "linucb":
        bandit = LinUCB(n_arms, dim, alpha=alpha, seed=seed)
    elif learner == "thompson":
        bandit = LinThompson(n_arms, dim, seed=seed)
    else:
        raise ValueError(f"Unknown learner: {learner}")

    alloc_learned = pd.DataFrame(0.0, index=rebal, columns=signals)
    prev = None  # (arm, x, period)
    for i, t in enumerate(rebal):
        x = context.loc[t, feat_cols].values.astype(float)
        # Update with the previously realized reward (period i-1 realized by i).
        if prev is not None:
            p_arm, p_x, p_t = prev
            r = sig_period_rets.loc[p_t, signals[p_arm]]
            if np.isfinite(r):
                bandit.update(p_arm, p_x, float(r))
        arm = bandit.select(x)
        alloc_learned.loc[t, signals[arm]] = 1.0
        prev = (arm, x, t)

    # --- Equal-weight baseline ---
    alloc_ew = pd.DataFrame(1.0 / n_arms, index=rebal, columns=signals)

    # --- Mean-variance baseline (point-in-time) ---
    alloc_mv = pd.DataFrame(0.0, index=rebal, columns=signals)
    for i, t in enumerate(rebal):
        hist = sig_period_rets.iloc[:i].dropna()
        if len(hist) >= max(6, warmup_periods // 2):
            w = mean_variance_weights(hist[signals])
        else:
            w = np.ones(n_arms) / n_arms
        alloc_mv.loc[t, signals] = w

    results: dict[str, AllocatorResult] = {}
    for name, alloc in [
        (f"learned_{learner}", alloc_learned),
        ("equal_weight", alloc_ew),
        ("mean_variance", alloc_mv),
    ]:
        comb_W = _combined_weights(alloc, sig_mats, returns)
        res = bt.simulate_weights(comb_W, returns, cost_bps=config.cost_bps,
                                  config=config, rebalance_dates=rebal)
        results[name] = AllocatorResult(
            name=name,
            daily_net=res.daily_net,
            weights_over_signals=alloc,
            combined_result=res,
        )

    # Attach the OOS start date for downstream slicing.
    oos_start = rebal[min(warmup_periods, len(rebal) - 1)]
    for r in results.values():
        r.oos_start = oos_start  # type: ignore[attr-defined]
    return results

"""Backtest engine: portfolio construction, costs, walk-forward P&L.

Conventions (point-in-time, no look-ahead)
------------------------------------------
* Weights are decided at the **close of rebalance date ``t``** using only the
  signal value at ``t``.  They earn returns starting on ``t+1``.  Concretely the
  daily P&L on day ``d`` uses weights that were known no later than ``d-1``
  (``weights.shift(1)``).  A unit test asserts this lag.
* Between rebalances, target weights are held constant (no intraperiod drift
  rebalancing).  Turnover is measured target-to-target:
  ``turnover_t = sum_i |w_t,i - w_{t-1},i|``.
* Transaction costs: ``cost_t = (cost_bps / 1e4) * turnover_t`` charged on the
  rebalance date.  ``cost_bps`` is *per side* and ``turnover`` already counts
  both buy and sell notionals, so this is the round-trip cost.  Results are
  reported both gross and net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    rebalance_freq: str = "ME"          # month-end (pandas offset alias)
    horizon: int = 21                   # forward-return horizon (trading days) for IC
    quantile: float = 0.1               # decile = 0.1
    cost_bps: float = 10.0              # transaction cost per side, in bps
    weighting: str = "decile"          # 'decile' or 'rank'
    min_names: int = 20                 # min cross-section size to trade a date


@dataclass
class BacktestResult:
    daily_gross: pd.Series
    daily_net: pd.Series
    turnover: pd.Series                  # indexed by rebalance date
    weights: pd.DataFrame               # rebalance date x ticker
    rebalance_dates: pd.DatetimeIndex
    config: BacktestConfig = field(repr=False, default_factory=BacktestConfig)

    @property
    def avg_turnover(self) -> float:
        return float(self.turnover.mean())


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Last available trading day in each period of ``freq``."""
    s = pd.Series(index, index=index)
    grouped = s.groupby(index.to_period(_freq_to_period(freq))).last()
    return pd.DatetimeIndex(grouped.values)


def _freq_to_period(freq: str) -> str:
    f = freq.upper()
    if f.startswith("M"):
        return "M"
    if f.startswith("W"):
        return "W"
    if f.startswith("Q"):
        return "Q"
    if f.startswith("A") or f.startswith("Y"):
        return "Y"
    return "M"


def build_weights_decile(
    signal_row: pd.Series, *, quantile: float = 0.1, min_names: int = 20
) -> pd.Series:
    """Dollar-neutral decile long-short: +1 split over top decile, -1 over bottom."""
    s = signal_row.dropna()
    if len(s) < min_names:
        return pd.Series(dtype=float)
    n = len(s)
    k = max(1, int(round(n * quantile)))
    ranked = s.sort_values()
    short_names = ranked.index[:k]
    long_names = ranked.index[-k:]
    w = pd.Series(0.0, index=s.index)
    w.loc[long_names] = 1.0 / len(long_names)
    w.loc[short_names] = -1.0 / len(short_names)
    return w


def build_weights_rank(signal_row: pd.Series, *, min_names: int = 20) -> pd.Series:
    """Dollar-neutral rank-weighted long-short (long sum +1, short sum -1)."""
    s = signal_row.dropna()
    if len(s) < min_names:
        return pd.Series(dtype=float)
    r = s.rank()
    c = r - r.mean()
    pos = c.clip(lower=0.0)
    neg = c.clip(upper=0.0)
    w = pd.Series(0.0, index=s.index)
    if pos.sum() > 0:
        w = w.add(pos / pos.sum(), fill_value=0.0)
    if neg.sum() < 0:
        w = w.add(neg / (-neg.sum()), fill_value=0.0)
    return w


def _build_weights(signal_row: pd.Series, config: BacktestConfig) -> pd.Series:
    if config.weighting == "decile":
        return build_weights_decile(
            signal_row, quantile=config.quantile, min_names=config.min_names
        )
    if config.weighting == "rank":
        return build_weights_rank(signal_row, min_names=config.min_names)
    raise ValueError(f"Unknown weighting: {config.weighting}")


def run_backtest(
    signal_panel: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a walk-forward long-short backtest on a standardized signal panel."""
    config = config or BacktestConfig()
    returns = returns.sort_index()
    signal_panel = signal_panel.reindex_like(returns)

    rebal = get_rebalance_dates(returns.index, config.rebalance_freq)
    rebal = rebal[rebal.isin(signal_panel.index)]

    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    turnover_vals: dict[pd.Timestamp, float] = {}
    prev_w = pd.Series(0.0, index=returns.columns)

    for t in rebal:
        s = signal_panel.loc[t]
        w = _build_weights(s, config).reindex(returns.columns).fillna(0.0)
        turnover_vals[t] = float((w - prev_w).abs().sum())
        weight_rows[t] = w
        prev_w = w

    if not weight_rows:
        raise ValueError("No rebalance dates produced any tradable weights.")

    W = pd.DataFrame(weight_rows).T.reindex(columns=returns.columns).fillna(0.0)
    W.index = pd.DatetimeIndex(W.index)

    return simulate_weights(W, returns, cost_bps=config.cost_bps, config=config,
                            rebalance_dates=rebal)


def simulate_weights(
    W: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
    config: BacktestConfig | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> BacktestResult:
    """Simulate daily P&L from a rebalance-date x ticker target-weight matrix.

    Weights set on rebalance date ``t`` earn returns from ``t+1`` (one-day lag,
    so no look-ahead).  Turnover is target-to-target; cost is charged on the
    rebalance date.  Reused by both ``run_backtest`` and the RL allocator.
    """
    returns = returns.sort_index()
    W = W.reindex(columns=returns.columns).fillna(0.0).sort_index()

    # Turnover at each rebalance vs the previous target weights.
    prev = W.shift(1).fillna(0.0)
    turnover = (W - prev).abs().sum(axis=1)

    # Daily weights known at each date (most recent rebalance <= date), lagged 1d.
    W_daily = W.reindex(returns.index, method="ffill").fillna(0.0)
    active = W_daily.shift(1).fillna(0.0)
    daily_gross = (active * returns).sum(axis=1)

    cost_rate = cost_bps / 1e4
    cost_daily = pd.Series(0.0, index=returns.index)
    cost_daily.loc[turnover.index] = turnover.values * cost_rate
    daily_net = daily_gross - cost_daily.reindex(returns.index).fillna(0.0)

    first_active = active.abs().sum(axis=1)
    nz = first_active[first_active > 0].index
    if len(nz) > 0:
        start = nz.min()
        daily_gross = daily_gross.loc[start:]
        daily_net = daily_net.loc[start:]

    return BacktestResult(
        daily_gross=daily_gross,
        daily_net=daily_net,
        turnover=turnover,
        weights=W,
        rebalance_dates=rebalance_dates
        if rebalance_dates is not None
        else pd.DatetimeIndex(W.index),
        config=config or BacktestConfig(cost_bps=cost_bps),
    )


def period_returns_from_daily(
    daily: pd.Series, rebalance_dates: pd.DatetimeIndex
) -> pd.Series:
    """Compound a daily return series into per-rebalance-period returns.

    Period ``i`` covers ``(rebal[i], rebal[i+1]]`` and is indexed by ``rebal[i]``
    (the date the position was put on).  Used to feed the RL allocator's reward.
    """
    daily = daily.sort_index()
    rd = pd.DatetimeIndex(sorted(rebalance_dates))
    out = {}
    for i in range(len(rd) - 1):
        a, b = rd[i], rd[i + 1]
        window = daily.loc[(daily.index > a) & (daily.index <= b)]
        if len(window) > 0:
            out[a] = float(np.expm1(np.log1p(window).sum()))
    return pd.Series(out).sort_index()


def forward_returns(
    returns: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int
) -> pd.DataFrame:
    """Forward compounded return over the next ``horizon`` trading days.

    For each date in ``dates`` (must be in ``returns.index``), compute the
    product of (1 + daily return) over the *following* ``horizon`` days, minus 1.
    Strictly forward-looking by construction; used only as the prediction target,
    never as a feature.
    """
    returns = returns.sort_index()
    idx = returns.index
    log1p = np.log1p(returns)
    out = {}
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in dates:
        i = pos_of.get(d)
        if i is None or i + horizon >= len(idx):
            continue
        window = log1p.iloc[i + 1 : i + 1 + horizon]
        out[d] = np.expm1(window.sum(axis=0))
    fwd = pd.DataFrame(out).T
    fwd.index = pd.DatetimeIndex(fwd.index)
    return fwd

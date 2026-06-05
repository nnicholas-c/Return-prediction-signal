"""Backtest correctness: one-day lag (no look-ahead) and cost accounting."""

import numpy as np
import pandas as pd

from src import backtest as bt


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_weights_are_lagged_no_lookahead():
    """A weight set on date t must NOT capture t's own return."""
    d = _dates(8)
    cols = ["A", "B"]
    returns = pd.DataFrame(0.0, index=d, columns=cols)
    returns.loc[d[4], "A"] = 0.50  # big spike on d4

    # Hold B from the start; switch to A exactly on d4 (the spike day).
    W = pd.DataFrame(0.0, index=[d[1], d[4]], columns=cols)
    W.loc[d[1], "B"] = 1.0
    W.loc[d[4], "A"] = 1.0

    res = bt.simulate_weights(W, returns, cost_bps=0.0)
    # On d4 the active weights are those known by d3 (B=1), so the A spike is NOT
    # captured -- proving no look-ahead.
    assert res.daily_gross.loc[d[4]] == 0.0


def test_lagged_weight_earns_next_day():
    d = _dates(8)
    cols = ["A"]
    returns = pd.DataFrame(0.0, index=d, columns=cols)
    returns.loc[d[5], "A"] = 0.10
    W = pd.DataFrame(0.0, index=[d[1]], columns=cols)
    W.loc[d[1], "A"] = 1.0
    res = bt.simulate_weights(W, returns, cost_bps=0.0)
    assert np.isclose(res.daily_gross.loc[d[5]], 0.10)


def test_turnover_and_cost_accounting():
    d = _dates(8)
    cols = ["A", "B"]
    returns = pd.DataFrame(0.0, index=d, columns=cols)  # zero returns isolate cost
    W = pd.DataFrame(0.0, index=[d[1], d[3]], columns=cols)
    W.loc[d[1], "A"] = 0.5
    W.loc[d[3], "A"] = 1.0  # change of 0.5 -> turnover 0.5

    bps = 20.0
    res = bt.simulate_weights(W, returns, cost_bps=bps)
    assert np.isclose(res.turnover.loc[d[1]], 0.5)   # from 0 to 0.5
    assert np.isclose(res.turnover.loc[d[3]], 0.5)   # from 0.5 to 1.0

    expected_cost = bps / 1e4 * 0.5
    # gross is zero, so net on the rebalance date equals -cost.
    assert np.isclose(res.daily_net.loc[d[3]], -expected_cost)


def test_decile_weights_dollar_neutral():
    s = pd.Series(np.arange(50, dtype=float), index=[f"T{i}" for i in range(50)])
    w = bt.build_weights_decile(s, quantile=0.1, min_names=20)
    assert np.isclose(w.sum(), 0.0)            # dollar neutral
    assert np.isclose(w[w > 0].sum(), 1.0)     # long leg sums to +1
    assert np.isclose(w[w < 0].sum(), -1.0)    # short leg sums to -1
    # Highest-signal names are long, lowest are short.
    assert w["T49"] > 0 and w["T0"] < 0


def test_rank_weights_dollar_neutral():
    s = pd.Series(np.random.default_rng(0).standard_normal(40),
                  index=[f"T{i}" for i in range(40)])
    w = bt.build_weights_rank(s, min_names=20)
    assert np.isclose(w.sum(), 0.0, atol=1e-9)
    assert np.isclose(w[w > 0].sum(), 1.0)
    assert np.isclose(w[w < 0].sum(), -1.0)

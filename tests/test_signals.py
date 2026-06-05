"""Signal correctness: IC sign convention and point-in-time (no look-ahead)."""

import numpy as np
import pandas as pd

from src import data, evaluation as ev, signals


def _make_panel(seed=0, n_days=400, n_assets=30):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_assets)]
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.02, size=(n_days, n_assets)), index=dates, columns=tickers
    )
    close = 100 * (1 + rets).cumprod()
    volume = pd.DataFrame(
        rng.integers(1e6, 5e6, size=(n_days, n_assets)), index=dates, columns=tickers
    ).astype(float)
    panels = {
        "Close": close, "Open": close, "High": close, "Low": close, "Volume": volume,
    }
    factors = pd.DataFrame({
        "Mkt-RF": rng.normal(0.0002, 0.01, n_days),
        "RF": np.full(n_days, 0.00005),
    }, index=dates)
    return panels, data.compute_returns(close), factors


def test_ic_sign_convention():
    """A signal equal to the forward return has IC ~ +1; its negation ~ -1."""
    panels, rets, _ = _make_panel()
    dates = rets.index[50:-30:10]
    fwd = pd.DataFrame(
        {d: (panels["Close"].shift(-21).loc[d] / panels["Close"].loc[d] - 1)
         for d in dates}
    ).T
    perfect = fwd.copy()  # signal == forward return
    ic_pos = ev.spearman_ic(perfect, fwd, min_names=10).dropna()
    ic_neg = ev.spearman_ic(-perfect, fwd, min_names=10).dropna()
    assert ic_pos.mean() > 0.99
    assert ic_neg.mean() < -0.99


def test_perfect_signal_long_short_positive():
    """If a signal perfectly ranks forward returns, the L-S spread is positive."""
    from src import backtest as bt

    panels, rets, _ = _make_panel(seed=1)
    rebal = bt.get_rebalance_dates(rets.index, "ME")
    fwd = bt.forward_returns(rets, rebal, 21)
    # Oracle signal = forward return, standardized cross-sectionally.
    oracle = signals.cross_sectional_zscore(fwd.reindex(rets.index), min_names=10)
    res = bt.run_backtest(oracle, rets, bt.BacktestConfig(cost_bps=0.0, min_names=10))
    assert res.daily_gross.mean() > 0


def test_signals_point_in_time():
    """Signals computed on truncated data match the full-data values up to the cut.

    This is the look-ahead guard: a backward-looking signal cannot change when
    future data is removed.
    """
    panels, rets, factors = _make_panel(seed=2)
    full = signals.compute_signals(panels, rets, factors)

    cut = rets.index[300]
    trunc_panels = {k: v.loc[:cut] for k, v in panels.items()}
    trunc_rets = rets.loc[:cut]
    trunc_factors = factors.loc[:cut]
    trunc = signals.compute_signals(trunc_panels, trunc_rets, trunc_factors)

    check_date = rets.index[280]  # well inside both
    for name in signals.SIGNAL_NAMES:
        a = full[name].loc[check_date]
        b = trunc[name].loc[check_date]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10)


def test_zscore_is_cross_sectional():
    """Z-scored rows have ~zero mean and unit-ish std, independent across dates."""
    panels, rets, factors = _make_panel(seed=3)
    raw = signals.compute_signals(panels, rets, factors)
    z = signals.cross_sectional_zscore(raw["mom_12_1"], winsor=None, min_names=10)
    row = z.dropna(how="all").iloc[-1].dropna()
    assert abs(row.mean()) < 1e-9
    assert abs(row.std(ddof=1) - 1.0) < 1e-6

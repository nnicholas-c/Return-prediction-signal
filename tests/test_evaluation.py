"""Evaluation correctness: deflated Sharpe and Newey-West behave sensibly."""

import numpy as np
import pandas as pd

from src import evaluation as ev


def test_newey_west_tstat_matches_mean_sign():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.05, 1.0, 500))
    mean, tstat, se = ev.newey_west_tstat(s)
    assert mean > 0
    assert tstat > 0
    assert se > 0


def test_deflated_sharpe_penalizes_many_trials():
    """More trials => higher SR0 benchmark => lower deflated Sharpe."""
    rng = np.random.default_rng(1)
    daily = pd.Series(rng.normal(0.0005, 0.01, 1000))  # modest positive SR
    few = [0.0, 0.02, -0.01]
    many = list(rng.normal(0.0, 0.05, 200))
    dsr_few = ev.deflated_sharpe_ratio(daily, few + [daily.mean() / daily.std()])
    dsr_many = ev.deflated_sharpe_ratio(daily, many + [daily.mean() / daily.std()])
    assert dsr_many["sr0_daily"] > dsr_few["sr0_daily"]
    assert dsr_many["deflated_sharpe"] <= dsr_few["deflated_sharpe"]


def test_deflated_sharpe_in_unit_interval():
    rng = np.random.default_rng(2)
    daily = pd.Series(rng.normal(0.0, 0.01, 500))
    out = ev.deflated_sharpe_ratio(daily, list(rng.normal(0, 0.05, 50)))
    assert 0.0 <= out["deflated_sharpe"] <= 1.0


def test_max_drawdown_known_case():
    # +100% then -50% -> back to start; drawdown should be -50%.
    daily = pd.Series([1.0, -0.5])
    assert np.isclose(ev.max_drawdown(daily), -0.5)


def test_sharpe_zero_for_zero_mean():
    s = pd.Series([0.01, -0.01, 0.01, -0.01] * 50)
    assert abs(ev.sharpe_ratio(s)) < 1e-9

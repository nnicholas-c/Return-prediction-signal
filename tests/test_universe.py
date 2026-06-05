"""Point-in-time universe correctness.

(a) ``members_asof`` respects add/drop dates: no ticker appears before its add
    date or after its removal date (and multiple stints are honoured).
(b) at each rebalance the backtest only holds names that are index members
    *as of* that date (membership masking flows into the portfolio).

These use synthetic membership + prices (no network).
"""

import numpy as np
import pandas as pd

from src import backtest as bt
from src.universe_pit import MembershipProvider, _normalize


def _provider():
    intervals = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "A"],
        "start": pd.to_datetime(["2018-01-01", "2018-06-01", "2018-01-01",
                                 "2019-01-01", "2020-01-01"]),
        "end": pd.to_datetime(["2018-12-31", None, "2018-09-30", None, None]),
    })
    return MembershipProvider(intervals=intervals)


def test_members_asof_respects_add_and_drop():
    m = _provider()
    assert m.asof("2018-03-01") == {"A", "C"}        # B not yet in; D far future
    assert m.asof("2018-07-01") == {"A", "B", "C"}   # B added June
    assert m.asof("2018-10-15") == {"A", "B"}        # C removed end of Sep
    assert m.asof("2019-02-01") == {"B", "D"}        # A's first stint ended 2018
    assert m.asof("2020-02-01") == {"A", "B", "D"}   # A re-added in 2020 (2nd stint)


def test_no_ticker_before_add_or_after_drop():
    m = _provider()
    # C is only a member Jan-Sep 2018.
    assert "C" not in m.asof("2017-12-31")
    assert "C" in m.asof("2018-05-01")
    assert "C" not in m.asof("2018-10-01")
    # D only from 2019.
    assert "D" not in m.asof("2018-12-31")
    assert "D" in m.asof("2019-01-01")


def test_union_and_mask_shapes():
    m = _provider()
    assert set(m.union("2018-01-01", "2018-03-01")) == {"A", "C"}
    assert set(m.union("2018-01-01", "2020-12-31")) == {"A", "B", "C", "D"}
    dates = pd.bdate_range("2018-01-01", "2018-12-31")
    mask = m.daily_mask(dates, ["A", "B", "C", "D"])
    assert mask.loc["2018-03-01", "A"] and not mask.loc["2018-03-01", "B"]
    assert mask.loc["2018-07-02", "B"]
    assert not mask.loc["2018-10-15", "C"]


def test_normalize_remap():
    assert _normalize("brk.b") == "BRK-B"
    assert _normalize("FB") == "META"
    assert _normalize("AAPL") == "AAPL"


def test_backtest_only_holds_members_asof():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2017-01-01", "2019-12-31")
    tickers = [f"T{i}" for i in range(30)]

    # Each ticker is a member on a deterministic sub-window; ensure breadth.
    intervals = []
    for i, t in enumerate(tickers):
        start = dates[0] if i % 5 else dates[120]          # a few late entrants
        end = None if i % 7 else dates[400]                # a few early exits
        intervals.append({"ticker": t, "start": pd.Timestamp(start),
                          "end": (pd.Timestamp(end) if end is not None else pd.NaT)})
    provider = MembershipProvider(intervals=pd.DataFrame(intervals))

    returns = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(tickers))),
                           index=dates, columns=tickers)
    mask = provider.daily_mask(dates, tickers)
    signal = pd.DataFrame(rng.normal(0, 1, (len(dates), len(tickers))),
                          index=dates, columns=tickers).where(mask)

    res = bt.run_backtest(signal, returns,
                          bt.BacktestConfig(cost_bps=5.0, weighting="decile",
                                            min_names=5, quantile=0.2))
    for t in res.weights.index:
        held = set(res.weights.loc[t][res.weights.loc[t] != 0].index)
        members = provider.asof(t)
        assert held.issubset(members), f"{t}: held non-members {held - members}"

"""Shared data-preparation pipeline used by the experiment scripts.

Supports two universe modes:

* ``snapshot`` -- today's S&P 500 list (``src/universe.py``); fast but
  survivorship-biased (the original study).
* ``pit`` -- point-in-time index membership (``src/universe_pit.py``): at each
  date the tradable set is the constituents *as of* that date, including names
  later removed.  This is the survivorship-bias reduction.

In ``pit`` mode the membership mask is applied to the raw signals **before**
cross-sectional standardization, so z-scores/ranks are computed only among the
as-of-date members -- never using a future-membership cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import data, signals, universe, universe_pit


@dataclass
class Dataset:
    universe: list[str]
    panels: dict[str, pd.DataFrame]
    returns: pd.DataFrame
    factors: pd.DataFrame
    raw_signals: dict[str, pd.DataFrame]
    std_signals: dict[str, pd.DataFrame]
    start: str
    end: str
    universe_mode: str = "snapshot"
    coverage: dict | None = None
    membership_mask: pd.DataFrame | None = None
    membership: object | None = field(default=None, repr=False)


def prepare(
    start: str,
    end: str,
    *,
    universe_mode: str = "snapshot",
    max_tickers: int | None = None,
    cache_dir: str | Path = data.DEFAULT_CACHE_DIR,
    standardize: str = "zscore",
    winsor: float | None = 3.0,
    force: bool = False,
) -> Dataset:
    """Download/cache data and compute standardized point-in-time signals."""
    membership = None
    coverage = None
    mask = None

    if universe_mode == "snapshot":
        tickers = universe.get_universe(max_tickers)
    elif universe_mode == "pit":
        membership = universe_pit.load_membership(cache_dir=cache_dir, force=force)
        tickers = membership.union(start, end)
        if max_tickers is not None:  # only for quick/dev runs
            tickers = tickers[:max_tickers]
    else:
        raise ValueError(f"Unknown universe_mode: {universe_mode!r}")

    panels = data.load_prices(tickers, start, end, cache_dir=cache_dir, force=force)
    # Drop tickers with corrupted adjusted-price series (Yahoo data errors).
    panels, dropped_quality = data.clean_prices(panels)
    factors = data.load_ff_factors(start, end, cache_dir=cache_dir, force=force)
    returns = data.compute_returns(panels["Close"])

    # Align the factor frame to the trading calendar of the price data.
    factors = factors.reindex(returns.index).ffill()

    raw = signals.compute_signals(panels, returns, factors)

    if universe_mode == "pit":
        coverage = data.coverage_report(tickers, panels)
        coverage["n_dropped_quality"] = len(dropped_quality)
        coverage["dropped_quality"] = dropped_quality
        have = list(panels["Close"].columns)
        mask = membership.daily_mask(returns.index, have)
        # Restrict each raw signal to as-of-date members BEFORE standardizing.
        raw = {name: panel.where(mask) for name, panel in raw.items()}

    std = signals.standardize_signals(raw, method=standardize, winsor=winsor)

    return Dataset(
        universe=list(panels["Close"].columns),
        panels=panels,
        returns=returns,
        factors=factors,
        raw_signals=raw,
        std_signals=std,
        start=start,
        end=end,
        universe_mode=universe_mode,
        coverage=coverage,
        membership_mask=mask,
        membership=membership,
    )

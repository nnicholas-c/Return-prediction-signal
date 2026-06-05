"""Shared data-preparation pipeline used by the experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data, signals, universe


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


def prepare(
    start: str,
    end: str,
    *,
    max_tickers: int | None = None,
    cache_dir: str | Path = data.DEFAULT_CACHE_DIR,
    standardize: str = "zscore",
    winsor: float | None = 3.0,
    force: bool = False,
) -> Dataset:
    """Download/cache data and compute standardized point-in-time signals."""
    tickers = universe.get_universe(max_tickers)
    panels = data.load_prices(tickers, start, end, cache_dir=cache_dir, force=force)
    factors = data.load_ff_factors(start, end, cache_dir=cache_dir, force=force)
    returns = data.compute_returns(panels["Close"])

    # Align the factor frame to the trading calendar of the price data.
    factors = factors.reindex(returns.index).ffill()

    raw = signals.compute_signals(panels, returns, factors)
    std = signals.standardize_signals(raw, method=standardize, winsor=winsor)

    have = list(panels["Close"].columns)
    return Dataset(
        universe=have,
        panels=panels,
        returns=returns,
        factors=factors,
        raw_signals=raw,
        std_signals=std,
        start=start,
        end=end,
    )

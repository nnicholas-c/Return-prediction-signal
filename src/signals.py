"""Cross-sectional equity signals, computed strictly point-in-time.

Every signal at date ``t`` is a function of information available **at the close
of ``t``** (backward-looking rolling windows only -- no centered windows, no
full-sample statistics).  Each signal is *oriented* so that a **higher value is
the ex-ante hypothesis for a higher forward return**.  That orientation is the
academic prior (e.g. the low-volatility anomaly), decided before looking at the
data -- it is NOT fit to the sample.  The honest test is whether the data agree.

Signals implemented (all long-high oriented):

==================  =========================================================
name                definition (oriented)
==================  =========================================================
mom_12_1            12-1 momentum: return from t-252 to t-21 (skip last month)
st_reversal         short-term reversal: -1 * (return over last 21 days)
low_volatility      -1 * realized daily-return vol over last 63 days
low_idio_vol        -1 * idiosyncratic vol (residual std vs market, 63 days)
near_52w_high       close / trailing 252-day max close (proximity to 52w high)
illiquidity         -1 * log(mean daily dollar volume over last 21 days)
==================  =========================================================

``illiquidity`` is a turnover/liquidity proxy.  Without point-in-time shares
outstanding we cannot compute true turnover (volume/shares), so we use average
dollar volume as a liquidity proxy (the illiquidity premium hypothesis: less
liquid names earn a premium, hence the negative sign).  This approximation is
documented in ``LIMITATIONS.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_NAMES = [
    "mom_12_1",
    "st_reversal",
    "low_volatility",
    "low_idio_vol",
    "near_52w_high",
    "illiquidity",
]

# All signals are pre-oriented long-high, so direction is uniformly +1.
SIGNAL_DIRECTION = {name: 1 for name in SIGNAL_NAMES}


def _idiosyncratic_vol(
    returns: pd.DataFrame, market: pd.Series, window: int
) -> pd.DataFrame:
    """Rolling idiosyncratic volatility = std of residual vs the market.

    Uses the closed form  resid_var = var(r) - cov(r, m)^2 / var(m)  over a
    trailing window, which equals the residual variance of the OLS market model
    (an intercept is implicitly absorbed by the demeaning inside cov/var).  All
    quantities are trailing, so the result is point-in-time.
    """
    market = market.reindex(returns.index)
    m = market.to_numpy()

    var_m = market.rolling(window).var(ddof=1)
    var_r = returns.rolling(window).var(ddof=1)

    # Rolling cov(r_i, m) = E[r_i m] - E[r_i] E[m], computed columnwise.
    mean_m = market.rolling(window).mean()
    mean_r = returns.rolling(window).mean()
    rm = returns.mul(market, axis=0)
    mean_rm = rm.rolling(window).mean()
    cov = mean_rm.sub(mean_r.mul(mean_m, axis=0))

    resid_var = var_r.sub((cov ** 2).div(var_m, axis=0))
    resid_var = resid_var.clip(lower=0.0)
    return np.sqrt(resid_var)


def compute_signals(
    panels: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    mom_skip: int = 21,
    mom_lookback: int = 252,
    rev_window: int = 21,
    vol_window: int = 63,
    high_window: int = 252,
    turn_window: int = 21,
) -> dict[str, pd.DataFrame]:
    """Compute all raw (oriented) signal panels (date x ticker), point-in-time."""
    close = panels["Close"].sort_index()
    volume = panels["Volume"].reindex_like(close)
    returns = returns.reindex_like(close)

    # Market daily return for the idiosyncratic-vol regression.
    market = (factors["Mkt-RF"] + factors["RF"]).reindex(close.index)

    signals: dict[str, pd.DataFrame] = {}

    # 12-1 momentum: return from t-mom_lookback to t-mom_skip (skip last month).
    signals["mom_12_1"] = close.shift(mom_skip) / close.shift(mom_lookback) - 1.0

    # Short-term (1-month) reversal: negative of last-month return.
    signals["st_reversal"] = -(close / close.shift(rev_window) - 1.0)

    # Low-volatility: negative trailing realized vol.
    signals["low_volatility"] = -returns.rolling(vol_window).std(ddof=1)

    # Low idiosyncratic volatility (residual vs market).
    signals["low_idio_vol"] = -_idiosyncratic_vol(returns, market, vol_window)

    # 52-week-high proximity: close relative to trailing max close.
    signals["near_52w_high"] = close / close.rolling(high_window).max()

    # Illiquidity (turnover/liquidity proxy): negative log avg dollar volume.
    dvol = (close * volume).rolling(turn_window).mean()
    signals["illiquidity"] = -np.log(dvol.where(dvol > 0))

    return signals


# --------------------------------------------------------------------------- #
# Cross-sectional standardization (within each date)
# --------------------------------------------------------------------------- #
def cross_sectional_zscore(
    panel: pd.DataFrame, *, winsor: float | None = 3.0, min_names: int = 10
) -> pd.DataFrame:
    """Z-score each row (date) across tickers; optionally winsorize.

    Rows with fewer than ``min_names`` valid observations are set to NaN (too
    thin a cross-section to standardize reliably).  This is a per-date operation,
    so it never uses information from other dates -- no temporal leakage.
    """
    valid = panel.notna().sum(axis=1)
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=1)
    z = panel.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    if winsor is not None:
        z = z.clip(lower=-winsor, upper=winsor)
    z = z.mask((valid < min_names), other=np.nan)
    return z


def cross_sectional_rank(panel: pd.DataFrame, *, min_names: int = 10) -> pd.DataFrame:
    """Rank each row to the interval [-0.5, 0.5] (robust, scale-free)."""
    valid = panel.notna().sum(axis=1)
    ranks = panel.rank(axis=1, method="average")
    counts = panel.notna().sum(axis=1)
    centered = ranks.sub((counts + 1) / 2.0, axis=0).div(counts, axis=0)
    centered = centered.mask((valid < min_names), other=np.nan)
    return centered


def standardize_signals(
    signals: dict[str, pd.DataFrame],
    *,
    method: str = "zscore",
    winsor: float | None = 3.0,
    min_names: int = 10,
) -> dict[str, pd.DataFrame]:
    """Apply cross-sectional standardization to every signal panel."""
    out: dict[str, pd.DataFrame] = {}
    for name, panel in signals.items():
        if method == "zscore":
            out[name] = cross_sectional_zscore(panel, winsor=winsor, min_names=min_names)
        elif method == "rank":
            out[name] = cross_sectional_rank(panel, min_names=min_names)
        else:
            raise ValueError(f"Unknown standardization method: {method}")
    return out

"""Point-in-time S&P 500 index membership.

The base study (``src/universe.py``) uses *today's* S&P 500 snapshot, which
introduces survivorship/look-back bias: it only contains firms that survived and
were promoted into the index.  This module instead provides **dated historical
membership** so that, at each rebalance date ``t``, the tradable universe is the
set of index constituents *as of* ``t`` -- including names that were later
removed (delisted, acquired, demoted).

Data source (free, no credentials)
-----------------------------------
``fja05680/sp500`` historical components, specifically the tidy
``sp500_ticker_start_end.csv`` file, which lists, per ticker, the date it
entered and (optionally) left the index.  A ticker may have several stints
(e.g. ``AAL`` 1996-1997 and again 2015-2024); all are honoured.

    https://github.com/fja05680/sp500
    https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv

Residual bias caveat
--------------------
Even with point-in-time membership, two gaps remain (see ``LIMITATIONS.md``):
1. some historical constituents are unrecoverable from Yahoo Finance (delisted /
   ticker-changed), so they are excluded -- and missing names skew toward
   failures, leaving a *residual* survivorship bias;
2. delisting returns are not modelled (a removed name simply stops contributing).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import urllib.request
except Exception:  # pragma: no cover
    urllib = None  # type: ignore

from .data import DEFAULT_CACHE_DIR, DataUnavailableError, _http_get

SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "sp500_ticker_start_end.csv"
)
SOURCE_CITATION = "fja05680/sp500 historical components (sp500_ticker_start_end.csv)"

# Conservative, clearly-correct symbol remaps (same listed entity, renamed/
# re-tickered) to improve Yahoo Finance coverage of historical members.
TICKER_REMAP = {
    "FB": "META",     # Facebook -> Meta (rename)
    "BBT": "TFC",     # BB&T -> Truist (rename)
    "RTN": "RTX",     # Raytheon -> Raytheon Technologies
    "WLTW": "WTW",    # Willis Towers Watson re-ticker
    "ANTM": "ELV",    # Anthem -> Elevance Health (rename)
    "FISV": "FI",     # Fiserv re-ticker
    "HFC": "DINO",    # HollyFrontier -> HF Sinclair
    "STI": "TFC",     # SunTrust merged into Truist
    "CBE": "ETN",     # Cooper Industries -> Eaton (approx.)
}


def _normalize(ticker: str) -> str:
    """Map a raw constituent ticker to a Yahoo Finance symbol."""
    t = ticker.strip().upper().replace(".", "-")
    return TICKER_REMAP.get(t, t)


@dataclass
class MembershipProvider:
    """Holds index-membership intervals and answers as-of queries."""

    intervals: pd.DataFrame  # columns: ticker, start, end (end NaT = current)

    # ------------------------------------------------------------------ #
    def asof(self, date) -> set[str]:
        """Set of (normalized) tickers that are index members as of ``date``."""
        d = pd.Timestamp(date)
        df = self.intervals
        mask = (df["start"] <= d) & (df["end"].isna() | (d <= df["end"]))
        return set(df.loc[mask, "ticker"])

    def union(self, start, end) -> list[str]:
        """All (normalized) tickers that were members at any point in [start, end]."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        df = self.intervals
        # Interval [start_i, end_i] overlaps [s, e] iff start_i <= e and end_i >= s.
        end_filled = df["end"].fillna(pd.Timestamp.max)
        mask = (df["start"] <= e) & (end_filled >= s)
        return sorted(set(df.loc[mask, "ticker"]))

    def daily_mask(self, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
        """Boolean (date x ticker) membership mask, True where a member that day."""
        dates = pd.DatetimeIndex(dates)
        cols = list(tickers)
        col_pos = {t: i for i, t in enumerate(cols)}
        arr = np.zeros((len(dates), len(cols)), dtype=bool)
        dvals = dates.values
        for _, row in self.intervals.iterrows():
            t = row["ticker"]
            j = col_pos.get(t)
            if j is None:
                continue
            s = np.datetime64(row["start"])
            e = (np.datetime64(row["end"]) if pd.notna(row["end"])
                 else np.datetime64(dates.max()))
            sel = (dvals >= s) & (dvals <= e)
            arr[sel, j] = True
        return pd.DataFrame(arr, index=dates, columns=cols)

    @property
    def n_members_ever(self) -> int:
        return int(self.intervals["ticker"].nunique())


def _parse_table(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    cols = {c.lower().strip(): c for c in raw.columns}
    tcol = cols.get("ticker")
    scol = cols.get("start_date") or cols.get("start")
    ecol = cols.get("end_date") or cols.get("end")
    if not (tcol and scol):
        raise DataUnavailableError(
            f"Unexpected membership CSV columns: {list(raw.columns)}"
        )
    df = pd.DataFrame({
        "ticker": raw[tcol].astype(str).map(_normalize),
        "start": pd.to_datetime(raw[scol], errors="coerce"),
        "end": pd.to_datetime(raw[ecol], errors="coerce") if ecol else pd.NaT,
    })
    df = df.dropna(subset=["ticker", "start"])
    # Collapse remap collisions (e.g. STI & BBT -> TFC) by keeping all intervals.
    return df.reset_index(drop=True)


def load_membership(
    cache_dir: str | Path = DEFAULT_CACHE_DIR, *, force: bool = False
) -> MembershipProvider:
    """Download (and cache) point-in-time S&P 500 membership intervals."""
    cache = Path(cache_dir) / "universe"
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / "sp500_ticker_start_end.csv"

    if fp.exists() and not force:
        text = fp.read_text()
    else:
        text = _http_get(SOURCE_URL).decode("utf-8")
        fp.write_text(text)

    table = _parse_table(text)
    if table.empty:
        raise DataUnavailableError("Parsed empty S&P 500 membership table.")
    return MembershipProvider(intervals=table)

"""Data layer: price and factor loaders with on-disk caching.

Design goals
------------
* **Free, no credentials.** Prices come from Yahoo Finance via ``yfinance``;
  Fama-French + momentum factors come from the Ken French Data Library as direct
  CSV downloads (more robust than ``pandas-datareader``, which frequently breaks
  against new pandas releases).
* **Point-in-time friendly.** Loaders return clean panels indexed by date.  The
  *use* of the data point-in-time (only information <= t at rebalance t) is
  enforced in ``signals.py`` and ``backtest.py``; this module only fetches and
  caches.
* **Reproducible.** Raw downloads are cached to ``data/cache`` so repeated runs
  are fast and offline-friendly, but the cache is reconstructable purely from
  code (and is git-ignored).

Survivorship caveat
-------------------
The universe (``src/universe.py``) is today's S&P 500 snapshot, which is *not*
point-in-time.  Backtests on it are biased upward because we only include firms
that survived and were promoted into the index.  This is documented and quanti-
fied qualitatively in ``LIMITATIONS.md``.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:  # urllib is stdlib; requests not required
    import urllib.request
except Exception:  # pragma: no cover
    urllib = None  # type: ignore

PRICE_FIELDS = ("Open", "High", "Low", "Close", "Volume")

# Ken French Data Library direct CSV (zip) endpoints.
_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_DAILY_URL = _FF_BASE + "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_DAILY_URL = _FF_BASE + "F-F_Momentum_Factor_daily_CSV.zip"

DEFAULT_CACHE_DIR = Path("data/cache")


class DataUnavailableError(RuntimeError):
    """Raised when a required data source cannot be reached.

    Per the project's prime directive, we STOP rather than fabricate data.
    """


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def _chunks(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull a single ticker's OHLCV out of a yfinance multi-ticker download."""
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by='ticker' -> level 0 is the ticker.
        if ticker not in raw.columns.get_level_values(0):
            return None
        sub = raw[ticker].copy()
    else:
        sub = raw.copy()
    keep = [c for c in PRICE_FIELDS if c in sub.columns]
    if not keep:
        return None
    sub = sub[keep]
    sub = sub.dropna(how="all")
    if sub.empty:
        return None
    sub.index = pd.to_datetime(sub.index)
    sub.index.name = "Date"
    return sub


def _download_prices_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,  # adjusted OHLC (splits + dividends)
                progress=False,
                group_by="ticker",
                threads=True,
                actions=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:  # network / rate-limit
            last_err = exc
        time.sleep(2 ** attempt)
    if last_err is not None:
        raise DataUnavailableError(
            f"Failed to download prices for {tickers[:3]}... : {last_err}"
        )
    return pd.DataFrame()


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    batch_size: int = 40,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load adjusted daily OHLCV for ``tickers`` between ``start`` and ``end``.

    Returns a dict mapping each price field (``Open/High/Low/Close/Volume``) to a
    ``DataFrame`` indexed by date with one column per ticker.  Adjusted prices
    (``auto_adjust=True``) so ``Close`` already accounts for splits/dividends.

    Per-ticker parquet caching makes reruns fast.  A cached ticker is reused
    only if its cached date range covers the request; otherwise it is
    re-downloaded.
    """
    cache = Path(cache_dir) / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for t in tickers:
        fp = cache / f"{t}.parquet"
        if fp.exists() and not force:
            df = pd.read_parquet(fp)
            if not df.empty and df.index.min() <= start_ts and df.index.max() >= (
                end_ts - pd.Timedelta(days=7)
            ):
                frames[t] = df
                continue
        missing.append(t)

    if missing:
        for batch in _chunks(missing, batch_size):
            raw = _download_prices_batch(batch, start, end)
            if raw.empty:
                continue
            for t in batch:
                sub = _extract_ticker_frame(raw, t)
                if sub is not None:
                    sub.to_parquet(cache / f"{t}.parquet")
                    frames[t] = sub

    if not frames:
        raise DataUnavailableError(
            "No price data could be downloaded for any requested ticker. "
            "Check internet access to Yahoo Finance."
        )

    # Assemble per-field panels (date x ticker), restricted to the request window.
    panels: dict[str, pd.DataFrame] = {}
    for field in PRICE_FIELDS:
        cols = {}
        for t, df in frames.items():
            if field in df.columns:
                cols[t] = df[field]
        if cols:
            panel = pd.DataFrame(cols).sort_index()
            panel = panel.loc[(panel.index >= start_ts) & (panel.index <= end_ts)]
            panels[field] = panel
    return panels


def dollar_volume(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily dollar volume = Close * Volume (a liquidity proxy)."""
    close = panels["Close"]
    vol = panels["Volume"].reindex_like(close)
    return close * vol


# --------------------------------------------------------------------------- #
# Fama-French + momentum factors (direct CSV)
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: int = 60) -> bytes:
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise DataUnavailableError(f"Could not download {url}: {last_err}")


def _parse_ff_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV into a date-indexed DataFrame (in percent).

    The files have a free-text header, then rows beginning with an 8-digit
    ``YYYYMMDD`` date, then (sometimes) trailing annual/footnote sections.  We
    keep only the contiguous block of daily rows.
    """
    lines = text.splitlines()
    header_idx = None
    cols: list[str] = []
    for i, line in enumerate(lines):
        # The column header line starts with a comma then the factor names.
        if line.startswith(",") and ("Mkt-RF" in line or "Mom" in line):
            header_idx = i
            cols = [c.strip() for c in line.split(",")[1:] if c.strip()]
            break
    if header_idx is None:
        raise DataUnavailableError("Unexpected Ken French CSV format (no header row).")

    records: list[tuple] = []
    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        token = parts[0]
        if len(token) == 8 and token.isdigit():
            vals = parts[1 : 1 + len(cols)]
            if len(vals) != len(cols):
                continue
            try:
                row = [float(v) for v in vals]
            except ValueError:
                continue
            records.append((token, *row))
        else:
            # Reached a non-daily section (e.g. annual averages); stop.
            if records:
                break

    if not records:
        raise DataUnavailableError("No daily rows parsed from Ken French CSV.")

    df = pd.DataFrame(records, columns=["Date", *cols])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index()
    # Ken French uses -99.99 / -999 as missing sentinels.
    df = df.mask(df <= -99.0)
    return df


def load_ff_factors(
    start: str,
    end: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Load daily Fama-French 5 factors + momentum + RF, in **decimal** returns.

    Columns: ``Mkt-RF, SMB, HML, RMW, CMA, RF, Mom``.  Values are divided by 100
    so they are directly comparable to simple daily asset returns.
    """
    cache = Path(cache_dir) / "factors"
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / "ff5_mom_daily.parquet"

    if fp.exists() and not force:
        ff = pd.read_parquet(fp)
    else:
        ff5_bytes = _http_get(FF5_DAILY_URL)
        mom_bytes = _http_get(MOM_DAILY_URL)

        def _read_zip(b: bytes) -> str:
            z = zipfile.ZipFile(io.BytesIO(b))
            name = z.namelist()[0]
            return z.read(name).decode("latin-1")

        ff5 = _parse_ff_csv(_read_zip(ff5_bytes))
        mom = _parse_ff_csv(_read_zip(mom_bytes))
        mom.columns = ["Mom" if c.lower().startswith("mom") else c for c in mom.columns]
        ff = ff5.join(mom[["Mom"]], how="inner")
        ff = ff / 100.0  # percent -> decimal
        ff.to_parquet(fp)

    ff = ff.loc[(ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))]
    if ff.empty:
        raise DataUnavailableError(
            "Fama-French factor frame is empty for the requested window."
        )
    return ff


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from an adjusted close panel (date x ticker)."""
    return close.sort_index().pct_change(fill_method=None)

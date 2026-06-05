"""Static equity universe.

We use a curated list of large/mid-cap US equities drawn from *current* S&P 500
constituents.  This is deliberately simple and free of credentials, but it is
**not point-in-time**: using today's index membership introduces survivorship
and look-back bias (we only see companies that survived and were promoted into
the index).  This caveat is documented in ``LIMITATIONS.md`` and re-stated in
code wherever it matters.

The list is intentionally a few hundred names so that each cross-section has
enough breadth for decile portfolios.  Tickers use Yahoo Finance conventions
(e.g. ``BRK-B`` rather than ``BRK.B``).
"""

from __future__ import annotations

# A curated snapshot of liquid US large/mid-caps (current S&P 500 members).
# Not point-in-time; see module docstring and LIMITATIONS.md.
SP500_SNAPSHOT: list[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "PG", "MA", "JNJ", "HD", "COST",
    "MRK", "ABBV", "CVX", "CRM", "WMT", "BAC", "KO", "PEP", "ADBE", "NFLX",
    "AMD", "TMO", "MCD", "CSCO", "ACN", "ABT", "LIN", "WFC", "DHR", "DIS",
    "TXN", "INTC", "PM", "VZ", "CMCSA", "INTU", "QCOM", "NKE", "AMGN", "IBM",
    "NOW", "CAT", "UNP", "HON", "SPGI", "GE", "LOW", "BA", "AMAT", "GS",
    "ELV", "PFE", "ISRG", "BKNG", "SYK", "DE", "BLK", "PLD", "TJX", "MDT",
    "AXP", "ADP", "GILD", "VRTX", "MMC", "C", "LRCX", "REGN", "CB", "ETN",
    "MU", "ADI", "SCHW", "ZTS", "BSX", "FI", "MO", "BX", "SO", "CI", "DUK",
    "SLB", "EOG", "BDX", "AON", "ITW", "CME", "TGT", "EQIX", "MPC", "APD",
    "CSX", "ICE", "KLAC", "PYPL", "WM", "SHW", "PNC", "FCX", "USB", "NOC",
    "CL", "EMR", "GD", "MCK", "ORLY", "MMM", "PSX", "MSI", "FDX", "MAR",
    "PH", "GM", "ROP", "APH", "NXPI", "TT", "AJG", "CARR", "PCAR", "ECL",
    "HUM", "MCHP", "AZO", "CTAS", "TDG", "AIG", "F", "COF", "OXY", "VLO",
    "HCA", "MET", "WELL", "SRE", "PSA", "ADM", "AEP", "TRV", "STZ", "NEM",
    "DXCM", "EW", "O", "JCI", "GIS", "D", "KMB", "CCI", "NSC", "MNST",
    "EXC", "FTNT", "AFL", "ROST", "IDXX", "PAYX", "CMG", "ODFL", "MRNA",
    "KHC", "DOW", "BK", "SPG", "WMB", "AMP", "HLT", "TEL", "PRU", "A",
    "CPRT", "HSY", "ON", "FAST", "KMI", "DLR", "OKE", "KR", "GWW", "VRSK",
    "ALL", "PEG", "CTSH", "EA", "GEHC", "AME", "DD", "MSCI", "PCG", "IQV",
    "CSGP", "XEL", "OTIS", "YUM", "DG", "KDP", "BIIB", "EXR", "FANG", "ED",
    "RSG", "ANSS", "VICI", "CDW", "ACGL", "HES", "EFX", "DFS", "WST", "AVB",
    "DLTR", "GLW", "FITB", "WEC", "MTD", "CHD", "WBD", "KEYS", "TROW",
    "CBRE", "ZBH", "WTW", "EBAY", "APTV", "RMD", "HPQ", "DAL", "STT", "TSCO",
    "ULTA", "FTV", "GPN", "PPG", "EIX", "AWK", "MTB", "NUE", "HIG", "BR",
    "ETR", "VMC", "ROK", "DTE", "MLM", "HPE", "WAB", "GPC", "EQR", "INVH",
    "IFF", "WY", "PWR", "FE", "DOV", "AEE", "CAH", "VRSN", "TTWO", "STE",
    "FICO", "EXPE", "BALL", "PFG", "ES", "K", "TDY", "CNP", "DRI", "CMS",
    "CLX", "MOH", "HBAN", "RF", "WAT", "MKC", "COO", "NTAP", "TYL", "ARE",
    "OMC", "SYF", "CFG", "NVR", "BAX", "LH", "HOLX", "J", "ATO", "AVY",
    "LVS", "CINF", "EXPD", "JBHT", "TRGP", "CE", "MAA", "DGX", "WRB",
    "AKAM", "TXT", "IP", "VTR", "LUV", "SWKS", "POOL", "BG", "NDAQ", "PKG",
    "GEN", "L", "EG", "SNA", "FDS", "AES", "DPZ", "JBL", "ESS", "MAS",
    "ALB", "UAL", "KIM", "LDOS", "STX", "RVTY", "AMCR", "VLTO", "ZBRA",
    "BBY", "CF", "LNT", "WDC", "TER", "IEX", "BF-B", "HST", "MRO", "EQT",
    "NTRS", "TFX", "CPB", "ROL", "PNR", "PODD", "JKHY", "EVRG", "CAG",
    "GL", "WBA", "DOC", "INCY", "UDR", "AVTR", "SWK", "DVN", "ALLE", "BXP",
    "PAYC", "CHRW", "FFIV", "EMN", "SJM", "MGM", "APA", "HRL", "TAP",
    "KMX", "BWA", "TPR", "AOS", "NDSN", "LKQ", "MOS", "CPT", "RL", "MKTX",
    "HSIC", "REG", "PNW", "WYNN", "JNPR", "AIZ", "CZR", "NI", "FRT", "GNRC",
    "PHM", "DVA", "QRVO", "IPG", "TFC", "HAS", "CTLT", "BEN", "FOXA", "FOX",
    "NWSA", "NWS", "PARA", "MHK", "IVZ", "FMC", "BIO", "ETSY", "HII",
]


def get_universe(max_tickers: int | None = None) -> list[str]:
    """Return the equity universe, optionally truncated to ``max_tickers``.

    Duplicates (e.g. dual share classes) are de-duplicated while preserving
    order.  Truncation keeps the most liquid names first because the list is
    roughly ordered by market cap.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for t in SP500_SNAPSHOT:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    if max_tickers is not None:
        ordered = ordered[:max_tickers]
    return ordered

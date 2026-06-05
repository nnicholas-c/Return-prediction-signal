"""Evaluation: IC, Newey-West t-stats, deflated Sharpe, factor alpha, plots.

All statistics are designed to be *honest*: IC t-stats use Newey-West (HAC)
standard errors to account for autocorrelation; the Sharpe ratio is reported in
both raw and **deflated** (multiple-testing-corrected) form; portfolio alpha is
measured net of the Fama-French five factors plus momentum.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Information Coefficient
# --------------------------------------------------------------------------- #
def spearman_ic(
    signal_panel: pd.DataFrame, fwd_returns: pd.DataFrame, *, min_names: int = 20
) -> pd.Series:
    """Per-date Spearman rank correlation between signal and forward return."""
    common = signal_panel.index.intersection(fwd_returns.index)
    ics = {}
    for d in common:
        s = signal_panel.loc[d]
        f = fwd_returns.loc[d]
        pair = pd.concat([s, f], axis=1).dropna()
        if len(pair) < min_names:
            continue
        rho, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        if not math.isnan(rho):
            ics[d] = rho
    out = pd.Series(ics).sort_index()
    out.name = "IC"
    return out


def newey_west_tstat(series: pd.Series, lags: int | None = None) -> tuple[float, float, float]:
    """Return (mean, HAC t-stat, HAC std-error) for the mean of ``series``.

    Regresses the series on a constant with Newey-West (HAC) covariance, so the
    t-stat is robust to autocorrelation and heteroskedasticity.
    """
    import statsmodels.api as sm

    x = series.dropna().astype(float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    if lags is None:
        lags = int(round(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West rule of thumb
        lags = max(lags, 1)
    model = sm.OLS(x.values, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean = float(model.params[0])
    se = float(model.bse[0])
    tstat = float(model.tvalues[0])
    return mean, tstat, se


def ic_summary(ic: pd.Series) -> dict:
    """Summarize an IC series: mean, std, NW t-stat, information ratio, n."""
    ic = ic.dropna()
    mean, tstat, se = newey_west_tstat(ic)
    std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    ir = mean / std if std and not math.isnan(std) else float("nan")
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_tstat_nw": tstat,
        "ic_ir": ir,          # per-period information ratio (mean/std)
        "n_periods": int(len(ic)),
    }


# --------------------------------------------------------------------------- #
# Portfolio statistics
# --------------------------------------------------------------------------- #
def sharpe_ratio(daily: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = daily.dropna()
    sd = r.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods))


def annualized_return(daily: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(daily.dropna().mean() * periods)


def annualized_vol(daily: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(daily.dropna().std(ddof=1) * math.sqrt(periods))


def max_drawdown(daily: pd.Series) -> float:
    r = daily.dropna()
    if r.empty:
        return float("nan")
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def portfolio_summary(result) -> dict:
    """Summarize a BacktestResult: gross/net annualized return, vol, Sharpe, DD."""
    g, n = result.daily_gross, result.daily_net
    return {
        "ann_return_gross": annualized_return(g),
        "ann_return_net": annualized_return(n),
        "ann_vol": annualized_vol(n),
        "sharpe_gross": sharpe_ratio(g),
        "sharpe_net": sharpe_ratio(n),
        "max_drawdown_net": max_drawdown(n),
        "avg_turnover": result.avg_turnover,
        "n_days": int(n.dropna().shape[0]),
    }


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio (Lopez de Prado)
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float
) -> float:
    """PSR: probability the true SR exceeds ``sr_benchmark`` (per-period units)."""
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark) * math.sqrt(max(1, n_obs - 1)) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sr_trials_std: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null across ``n_trials`` (per-period)."""
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_trials_std * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    daily_returns: pd.Series, all_trial_sharpes_daily: list[float]
) -> dict:
    """Deflated Sharpe Ratio of ``daily_returns`` given all trials' daily SRs.

    ``all_trial_sharpes_daily`` is the list of per-day Sharpe ratios of *every*
    configuration evaluated (signals x weightings, etc.).  The variance across
    those trials sets the multiple-testing benchmark ``SR0``; DSR is the
    probability the selected strategy's true SR exceeds that benchmark.
    """
    r = daily_returns.dropna().astype(float)
    n_obs = len(r)
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))  # normal = 3
    trials = np.asarray(all_trial_sharpes_daily, dtype=float)
    trials = trials[np.isfinite(trials)]
    sr0 = expected_max_sharpe(float(trials.std(ddof=1)) if len(trials) > 1 else 0.0,
                              len(trials))
    dsr = probabilistic_sharpe_ratio(sr, sr0, n_obs, sk, ku)
    return {
        "sr_daily": sr,
        "sr_annual": sr * math.sqrt(TRADING_DAYS),
        "sr0_daily": sr0,
        "sr0_annual": sr0 * math.sqrt(TRADING_DAYS),
        "deflated_sharpe": dsr,
        "n_trials": int(len(trials)),
        "n_obs": n_obs,
        "skew": sk,
        "kurtosis": ku,
    }


# --------------------------------------------------------------------------- #
# Factor-neutral alpha
# --------------------------------------------------------------------------- #
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def factor_alpha(
    portfolio_daily: pd.Series, factors: pd.DataFrame, lags: int | None = None
) -> dict:
    """Regress long-short returns on FF5 + momentum; report alpha and t-stat.

    The long-short portfolio is self-financing (dollar-neutral), so its return is
    already an excess return -- we regress it directly on the factor returns with
    Newey-West (HAC) standard errors.
    """
    import statsmodels.api as sm

    cols = [c for c in FACTOR_COLS if c in factors.columns]
    df = pd.concat([portfolio_daily.rename("y"), factors[cols]], axis=1).dropna()
    if len(df) < 30:
        return {"alpha_daily": float("nan"), "alpha_annual": float("nan"),
                "alpha_tstat": float("nan"), "n_obs": int(len(df))}
    y = df["y"].values
    X = sm.add_constant(df[cols].values)
    n = len(df)
    if lags is None:
        lags = max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    alpha = float(model.params[0])
    betas = {c: float(b) for c, b in zip(cols, model.params[1:])}
    return {
        "alpha_daily": alpha,
        "alpha_annual": alpha * TRADING_DAYS,
        "alpha_tstat": float(model.tvalues[0]),
        "betas": betas,
        "r_squared": float(model.rsquared),
        "n_obs": int(n),
    }


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def equal_weight_market(returns: pd.DataFrame) -> pd.Series:
    """Long-only equal-weight portfolio of the universe (daily returns)."""
    return returns.mean(axis=1)


def random_signal_panel(
    template: pd.DataFrame, seed: int = 0
) -> pd.DataFrame:
    """A no-skill random signal with the same NaN footprint as ``template``."""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(template.shape)
    rand = pd.DataFrame(vals, index=template.index, columns=template.columns)
    return rand.where(template.notna())


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_equity_curve(result, path: str, title: str = "Long-short equity curve") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = (1.0 + result.daily_gross.dropna()).cumprod()
    n = (1.0 + result.daily_net.dropna()).cumprod()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g.index, g.values, label="gross", lw=1.2)
    ax.plot(n.index, n.values, label="net of costs", lw=1.2)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ic_series(ic: pd.Series, path: str, title: str = "Information coefficient") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ic = ic.dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ic.index, ic.values, width=20, color="steelblue", alpha=0.7)
    roll = ic.rolling(12).mean()
    ax.plot(roll.index, roll.values, color="darkorange", lw=1.5, label="12-period mean")
    ax.axhline(ic.mean(), color="black", ls="--", lw=1, label=f"mean = {ic.mean():.3f}")
    ax.set_title(title)
    ax.set_ylabel("Spearman IC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drawdown(result, path: str, title: str = "Drawdown (net)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = result.daily_net.dropna()
    curve = (1.0 + r).cumprod()
    dd = curve / curve.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

"""End-to-end cross-sectional signal study.

Runs every signal as a dollar-neutral long-short portfolio under realistic
transaction costs, evaluates each with Information Coefficient (Newey-West
t-stats), portfolio Sharpe (gross and net), factor-neutral alpha, and a
multiple-testing-corrected **deflated Sharpe ratio**.  Also runs no-skill and
market baselines plus regime / cost robustness.

Usage::

    python -m experiments.run_baseline                 # full default run
    python -m experiments.run_baseline --max-tickers 60 --start 2012-01-01

Every number printed and written to ``results/`` is computed from freshly
downloaded data -- nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as bt
from src import evaluation as ev
from src import pipeline

RESULTS = Path("results")


def _signal_daily_sr(daily: pd.Series) -> float:
    r = daily.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def run(args) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = pipeline.prepare(
        args.start, args.end, max_tickers=args.max_tickers,
        standardize=args.standardize, force=args.force,
    )
    print(f"Universe: {len(ds.universe)} tickers | "
          f"{ds.returns.index.min().date()} -> {ds.returns.index.max().date()} "
          f"({len(ds.returns)} trading days)")

    cfg = bt.BacktestConfig(cost_bps=args.cost_bps, weighting="decile",
                            horizon=args.horizon)

    per_signal_rows = []
    trial_daily_sharpes: list[float] = []
    trial_records: list[tuple[str, str, pd.Series]] = []  # (signal, weighting, daily_net)

    for name, panel in ds.std_signals.items():
        rebal = bt.get_rebalance_dates(ds.returns.index, cfg.rebalance_freq)
        fwd = bt.forward_returns(ds.returns, rebal, cfg.horizon)
        ic = ev.spearman_ic(panel, fwd)
        ic_stats = ev.ic_summary(ic)

        for weighting in ("decile", "rank"):
            c = bt.BacktestConfig(cost_bps=args.cost_bps, weighting=weighting,
                                  horizon=args.horizon)
            res = bt.run_backtest(panel, ds.returns, c)
            psum = ev.portfolio_summary(res)
            alpha = ev.factor_alpha(res.daily_net, ds.factors)
            trial_daily_sharpes.append(_signal_daily_sr(res.daily_net))
            trial_records.append((name, weighting, res.daily_net))

            row = {
                "signal": name,
                "weighting": weighting,
                **ic_stats,
                **psum,
                "alpha_annual": alpha["alpha_annual"],
                "alpha_tstat": alpha["alpha_tstat"],
            }
            per_signal_rows.append(row)

            if weighting == "decile":
                # Save figures for the decile version of each signal.
                ev.plot_equity_curve(res, str(RESULTS / f"equity_{name}.png"),
                                     title=f"{name} long-short (decile)")
                ev.plot_ic_series(ic, str(RESULTS / f"ic_{name}.png"),
                                  title=f"{name} IC (mean={ic_stats['mean_ic']:.3f}, "
                                        f"t={ic_stats['ic_tstat_nw']:.2f})")

    table = pd.DataFrame(per_signal_rows)
    table.to_csv(RESULTS / "signal_summary.csv", index=False)

    # --- Selection + deflated Sharpe (multiple-testing correction) ---
    best_idx = int(np.nanargmax([r["sharpe_net"] for r in per_signal_rows]))
    best = per_signal_rows[best_idx]
    best_name, best_weighting, best_daily = trial_records[best_idx]
    dsr = ev.deflated_sharpe_ratio(best_daily, trial_daily_sharpes)

    # --- Baselines ---
    mkt = ev.equal_weight_market(ds.returns)
    mkt_excess = mkt - ds.factors["RF"].reindex(mkt.index).fillna(0.0)
    baseline_rows = [{
        "baseline": "equal_weight_market_excess",
        "sharpe": ev.sharpe_ratio(mkt_excess),
        "ann_return": ev.annualized_return(mkt_excess),
        "ann_vol": ev.annualized_vol(mkt_excess),
    }]
    rand_sharpes = []
    for seed in range(args.n_random):
        rand_panel = ev.random_signal_panel(ds.std_signals[best_name], seed=seed)
        rres = bt.run_backtest(rand_panel, ds.returns,
                               bt.BacktestConfig(cost_bps=args.cost_bps,
                                                 weighting=best_weighting,
                                                 horizon=args.horizon))
        rand_sharpes.append(ev.sharpe_ratio(rres.daily_net))
    baseline_rows.append({
        "baseline": f"random_signal_mean({args.n_random})",
        "sharpe": float(np.mean(rand_sharpes)),
        "ann_return": float("nan"),
        "ann_vol": float(np.std(rand_sharpes)),
    })
    pd.DataFrame(baseline_rows).to_csv(RESULTS / "baselines.csv", index=False)

    # --- Robustness: regime subsamples + cost sensitivity (on selected config) ---
    robustness = []
    splits = {
        "pre_2020": (ds.returns.index.min(), pd.Timestamp("2019-12-31")),
        "post_2020": (pd.Timestamp("2020-01-01"), ds.returns.index.max()),
    }
    sel_cfg = bt.BacktestConfig(cost_bps=args.cost_bps, weighting=best_weighting,
                               horizon=args.horizon)
    for label, (a, b) in splits.items():
        sub = ds.returns.loc[a:b]
        if len(sub) < 60:
            continue
        rsub = bt.run_backtest(ds.std_signals[best_name].loc[a:b], sub, sel_cfg)
        robustness.append({"slice": label, "sharpe_net": ev.sharpe_ratio(rsub.daily_net),
                           "n_days": len(rsub.daily_net.dropna())})
    for bps in args.cost_grid:
        rc = bt.run_backtest(ds.std_signals[best_name], ds.returns,
                             bt.BacktestConfig(cost_bps=bps, weighting=best_weighting,
                                               horizon=args.horizon))
        robustness.append({"slice": f"cost_{bps}bps", "sharpe_net": ev.sharpe_ratio(rc.daily_net),
                           "n_days": len(rc.daily_net.dropna())})
    pd.DataFrame(robustness).to_csv(RESULTS / "robustness.csv", index=False)

    headline = {
        "universe_size": len(ds.universe),
        "period": f"{ds.returns.index.min().date()}..{ds.returns.index.max().date()}",
        "cost_bps_per_side": args.cost_bps,
        "n_trials": len(trial_daily_sharpes),
        "selected_signal": best_name,
        "selected_weighting": best_weighting,
        "mean_ic": best["mean_ic"],
        "ic_tstat_nw": best["ic_tstat_nw"],
        "ic_ir": best["ic_ir"],
        "sharpe_gross": best["sharpe_gross"],
        "sharpe_net": best["sharpe_net"],
        "deflated_sharpe": dsr["deflated_sharpe"],
        "sr0_annual_benchmark": dsr["sr0_annual"],
        "alpha_annual": best["alpha_annual"],
        "alpha_tstat": best["alpha_tstat"],
        "avg_turnover": best["avg_turnover"],
        "max_drawdown_net": best["max_drawdown_net"],
        "market_excess_sharpe": baseline_rows[0]["sharpe"],
        "random_signal_mean_sharpe": baseline_rows[1]["sharpe"],
    }
    with open(RESULTS / "headline_baseline.json", "w") as f:
        json.dump(headline, f, indent=2)

    _print_headline(headline, table)
    return headline


def _print_headline(h: dict, table: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BASELINE STUDY — HEADLINE METRICS (real, reproducible)")
    print("=" * 70)
    print(f"Universe / period : {h['universe_size']} stocks, {h['period']}")
    print(f"Transaction cost  : {h['cost_bps_per_side']} bps/side")
    print(f"Configs tried     : {h['n_trials']} (signals x weightings)")
    print(f"Selected (best net Sharpe): {h['selected_signal']} [{h['selected_weighting']}]")
    print("-" * 70)
    print(f"Mean IC           : {h['mean_ic']:+.4f}  (NW t = {h['ic_tstat_nw']:+.2f}, "
          f"IR = {h['ic_ir']:+.3f})")
    print(f"Sharpe (gross)    : {h['sharpe_gross']:+.3f}")
    print(f"Sharpe (net)      : {h['sharpe_net']:+.3f}")
    print(f"Deflated Sharpe   : {h['deflated_sharpe']:.3f}  "
          f"(prob. true SR > multiple-testing benchmark "
          f"SR0_annual={h['sr0_annual_benchmark']:.2f})")
    print(f"FF5+Mom alpha     : {h['alpha_annual']:+.3%}/yr  (t = {h['alpha_tstat']:+.2f})")
    print(f"Avg turnover/rebal: {h['avg_turnover']:.2f}   Max DD (net): {h['max_drawdown_net']:.1%}")
    print("-" * 70)
    print(f"Baseline market (excess) Sharpe : {h['market_excess_sharpe']:+.3f}")
    print(f"No-skill random signal Sharpe   : {h['random_signal_mean_sharpe']:+.3f}")
    print("=" * 70)
    print("\nPer-signal (decile) summary:")
    show = table[table["weighting"] == "decile"][
        ["signal", "mean_ic", "ic_tstat_nw", "sharpe_net", "alpha_annual", "alpha_tstat"]
    ]
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(show.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-sectional signal study.")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--standardize", default="zscore", choices=["zscore", "rank"])
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--cost-grid", type=float, nargs="*", default=[0.0, 5.0, 10.0, 20.0])
    p.add_argument("--force", action="store_true", help="force re-download")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

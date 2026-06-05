"""RL / contextual-bandit signal allocator vs simple baselines.

Trains a contextual bandit (LinUCB or Thompson sampling) online, walk-forward,
to tilt across the candidate signals, and compares its **out-of-sample** net
Sharpe against equal-weight and point-in-time mean-variance signal combination.

Usage::

    python -m experiments.run_rl
    python -m experiments.run_rl --learner thompson --warmup 36

The honest question: does the learned allocator beat the baselines out of
sample?  The script reports the answer plainly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import backtest as bt  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import pipeline  # noqa: E402
from src import rl_allocator as rl  # noqa: E402

RESULTS = Path("results")


def run(args) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = pipeline.prepare(
        args.start, args.end, max_tickers=args.max_tickers,
        standardize=args.standardize, force=args.force,
    )
    cfg = bt.BacktestConfig(cost_bps=args.cost_bps, weighting=args.weighting,
                            horizon=args.horizon)
    print(f"Universe: {len(ds.universe)} tickers | "
          f"{ds.returns.index.min().date()} -> {ds.returns.index.max().date()}")

    results = rl.run_allocators(
        ds.std_signals, ds.returns, ds.factors, cfg,
        warmup_periods=args.warmup, learner=args.learner, alpha=args.alpha,
        seed=args.seed,
    )

    oos_start = list(results.values())[0].oos_start
    rows = []
    curves = {}
    for name, r in results.items():
        daily_oos = r.daily_net.loc[oos_start:]
        rows.append({
            "method": name,
            "oos_sharpe_net": ev.sharpe_ratio(daily_oos),
            "oos_ann_return": ev.annualized_return(daily_oos),
            "oos_ann_vol": ev.annualized_vol(daily_oos),
            "oos_max_drawdown": ev.max_drawdown(daily_oos),
            "oos_n_days": int(daily_oos.dropna().shape[0]),
        })
        curves[name] = (1.0 + daily_oos.dropna()).cumprod()

    table = pd.DataFrame(rows).sort_values("oos_sharpe_net", ascending=False)
    table.to_csv(RESULTS / "rl_allocator_summary.csv", index=False)

    # Equity-curve comparison figure.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, lw=1.3, label=name)
    ax.set_title(f"OOS allocator comparison (net of {args.cost_bps:.0f}bps)")
    ax.set_ylabel("Growth of $1 (out-of-sample)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "rl_allocator_curves.png", dpi=130)
    plt.close(fig)

    learned_name = f"learned_{args.learner}"
    learned_sr = table.set_index("method").loc[learned_name, "oos_sharpe_net"]
    ew_sr = table.set_index("method").loc["equal_weight", "oos_sharpe_net"]
    mv_sr = table.set_index("method").loc["mean_variance", "oos_sharpe_net"]
    beat = bool(learned_sr > ew_sr and learned_sr > mv_sr)

    headline = {
        "learner": args.learner,
        "oos_start": str(pd.Timestamp(oos_start).date()),
        "learned_oos_sharpe": float(learned_sr),
        "equal_weight_oos_sharpe": float(ew_sr),
        "mean_variance_oos_sharpe": float(mv_sr),
        "learned_beats_baselines": beat,
    }
    with open(RESULTS / "headline_rl.json", "w") as f:
        json.dump(headline, f, indent=2)

    print("\n" + "=" * 70)
    print("RL / CONTEXTUAL-BANDIT ALLOCATOR — OUT-OF-SAMPLE (net) Sharpe")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(table.to_string(index=False))
    print("-" * 70)
    print(f"Learned ({args.learner}) OOS Sharpe : {learned_sr:+.3f}")
    print(f"Equal-weight OOS Sharpe       : {ew_sr:+.3f}")
    print(f"Mean-variance OOS Sharpe      : {mv_sr:+.3f}")
    print(f"Learned beats BOTH baselines  : {beat}")
    print("=" * 70)
    return headline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL/bandit signal allocator study.")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--weighting", default="decile", choices=["decile", "rank"])
    p.add_argument("--standardize", default="zscore", choices=["zscore", "rank"])
    p.add_argument("--learner", default="linucb", choices=["linucb", "thompson"])
    p.add_argument("--warmup", type=int, default=24)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

"""Survivorship before/after: snapshot vs point-in-time universe.

Runs the full baseline study under BOTH ``--universe snapshot`` (current S&P 500
list, survivorship-biased) and ``--universe pit`` (point-in-time membership), and
writes a head-to-head comparison of the headline metrics -- with special focus on
the snapshot-selected signal (the one most suspected of being a survivorship
artifact, per the README).

Outputs::

    results/survivorship_comparison.csv
    results/survivorship_comparison.png

Usage::

    python -m experiments.run_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments import run_baseline  # noqa: E402

RESULTS = Path("results")


def _focus_row(table: pd.DataFrame, signal: str) -> pd.Series:
    sub = table[(table["signal"] == signal) & (table["weighting"] == "decile")]
    return sub.iloc[0]


def run(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = run_baseline.build_parser().parse_args([])
    base.start, base.end = args.start, args.end
    base.cost_bps, base.horizon = args.cost_bps, args.horizon

    print("### Running SNAPSHOT (survivorship-biased) study ...")
    snap_args = argparse.Namespace(**vars(base)); snap_args.universe = "snapshot"
    snap = run_baseline.run(snap_args, write=False)

    print("### Running POINT-IN-TIME study ...")
    pit_args = argparse.Namespace(**vars(base)); pit_args.universe = "pit"
    pit = run_baseline.run(pit_args, write=False)

    # Focus on the snapshot-selected signal (apples-to-apples survivorship delta).
    focus = snap["headline"]["selected_signal"]
    s_row, p_row = _focus_row(snap["table"], focus), _focus_row(pit["table"], focus)

    focus_metrics = ["mean_ic", "ic_tstat_nw", "sharpe_gross", "sharpe_net",
                     "alpha_annual", "alpha_tstat", "avg_turnover", "max_drawdown_net"]
    rows = []
    for m in focus_metrics:
        sv, pv = float(s_row[m]), float(p_row[m])
        rows.append({"metric": f"{focus}: {m}", "snapshot": sv, "pit": pv,
                     "delta": pv - sv})

    sh, ph = snap["headline"], pit["headline"]
    for m in ["selected_signal", "deflated_sharpe", "sr0_annual_benchmark",
              "sharpe_net_pre2020", "sharpe_net_post2020",
              "market_excess_sharpe", "random_signal_mean_sharpe"]:
        sv, pv = sh.get(m), ph.get(m)
        delta = (pv - sv) if isinstance(sv, (int, float)) and isinstance(pv, (int, float)) else ""
        rows.append({"metric": f"selected_best: {m}", "snapshot": sv, "pit": pv,
                     "delta": delta})

    rows.append({"metric": "universe_size", "snapshot": sh["universe_size"],
                 "pit": ph["universe_size"], "delta": ph["universe_size"] - sh["universe_size"]})
    if "coverage_n_requested" in ph:
        rows.append({"metric": "pit_coverage(usable/requested)",
                     "snapshot": "", "pit": f"{ph['coverage_n_usable']}/{ph['coverage_n_requested']}",
                     "delta": f"{ph['coverage_n_missing']} missing"})

    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "survivorship_comparison.csv", index=False)

    _plot(focus, s_row, p_row, RESULTS / "survivorship_comparison.png")

    print("\n" + "=" * 72)
    print(f"SURVIVORSHIP BEFORE/AFTER — focus signal: {focus}")
    print("=" * 72)
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}",
                           "display.max_colwidth", 40):
        print(comp.to_string(index=False))
    print("=" * 72)
    print(f"Snapshot selected={sh['selected_signal']} (net Sharpe {sh['sharpe_net']:+.2f}, "
          f"deflated {sh['deflated_sharpe']:.2f}); "
          f"PIT selected={ph['selected_signal']} (net Sharpe {ph['sharpe_net']:+.2f}, "
          f"deflated {ph['deflated_sharpe']:.2f}).")


def _plot(focus, s_row, p_row, path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    levels = ["mean_ic", "sharpe_net", "alpha_annual"]
    llabels = ["mean IC", "net Sharpe", "FF alpha/yr"]
    x = np.arange(len(levels)); w = 0.38
    ax1.bar(x - w / 2, [float(s_row[m]) for m in levels], w, label="snapshot", color="#c44")
    ax1.bar(x + w / 2, [float(p_row[m]) for m in levels], w, label="point-in-time", color="#48a")
    ax1.set_xticks(x); ax1.set_xticklabels(llabels)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_title(f"{focus}: levels"); ax1.legend(); ax1.grid(alpha=0.3, axis="y")

    tstats = ["ic_tstat_nw", "alpha_tstat"]
    tlabels = ["IC NW t", "alpha t"]
    x2 = np.arange(len(tstats))
    ax2.bar(x2 - w / 2, [float(s_row[m]) for m in tstats], w, label="snapshot", color="#c44")
    ax2.bar(x2 + w / 2, [float(p_row[m]) for m in tstats], w, label="point-in-time", color="#48a")
    ax2.set_xticks(x2); ax2.set_xticklabels(tlabels)
    ax2.axhline(2, color="g", ls="--", lw=1, label="t=2")
    ax2.axhline(-2, color="g", ls="--", lw=1)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title(f"{focus}: t-stats"); ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("Survivorship bias: snapshot vs point-in-time universe")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snapshot vs point-in-time comparison.")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--horizon", type=int, default=21)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

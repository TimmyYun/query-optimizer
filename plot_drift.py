#!/usr/bin/env python3
"""
Line charts (Median / P95 / Avg q-error vs. drift %) for every
results/<experiment>/combined_<prefix>.csv written by merge_drift.py.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["Median", "P95", "Avg"]
# One colour per model, one line style per stage, so 11+ lines stay distinguishable
STAGE_STYLE = {"": "-", "RB": "-", "Rebuild": "-", "Online": "-",
               "FT": "--", "Post": "--", "Shock": ":", "Static": ":"}


def plot_combined(csv_path: Path, labels=None):
    df = pd.read_csv(csv_path)
    if labels:
        df = df[df["label"].isin(labels)]
    if df.empty:
        print(f"WARNING: nothing to plot in {csv_path}")
        return

    prefix = csv_path.stem[len("combined_"):]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Drift Analysis: {prefix}", fontsize=15, fontweight="bold", y=1.02)

    df["stage"] = df["stage"].fillna("")
    models = sorted(df["model"].unique())
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(models)}

    for ax, metric in zip(axes, METRICS):
        for (model, stage, label), sub in df.groupby(["model", "stage", "label"], sort=True):
            sub = sub.sort_values("drift")
            ax.plot(sub["drift"], sub[metric], marker="o", markersize=3, linewidth=1.8,
                    color=colors[model], linestyle=STAGE_STYLE.get(stage, "-."), label=label)
        ax.set_title(f"{metric} Q-Error", fontsize=13)
        ax.set_xlabel("Drift %", fontsize=11)
        ax.set_ylabel("Q-Error", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    out_path = csv_path.with_name(f"drift_chart_{prefix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot merged drift results (run merge_drift.py first)")
    parser.add_argument("--experiment-name", required=True, nargs="+", help="One or more folders under --out-dir")
    parser.add_argument("--out-dir", default="results", help="Root of the results tree")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Only these lines, e.g. EquiWidth Hybrid_RB STHist_Online (default: all)")
    args = parser.parse_args()

    for exp in args.experiment_name:
        files = sorted((Path(args.out_dir) / exp).glob("combined_*.csv"))
        if not files:
            print(f"No combined_*.csv in {Path(args.out_dir) / exp}; run merge_drift.py first")
        for f in files:
            plot_combined(f, args.labels)


if __name__ == "__main__":
    main()

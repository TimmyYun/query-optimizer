#!/usr/bin/env python3
"""
Charts of <out-dir>/<scenario>/combined.csv (written by merge_drift.py):
median and P95 q-error per step, one line per model; a marker where the model was rebuilt.
Saved as <out-dir>/<scenario>/drift_chart.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_combined(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Drift: {csv_path.parent.name}", fontsize=14, fontweight="bold")
    for i, (model, sub) in enumerate(sorted(df.groupby("model"), key=lambda kv: kv[0])):
        sub = sub.sort_values("step")
        color = plt.cm.tab10(i % 10)
        for ax, col in zip(axes, ["median_qerror", "p95_qerror"]):
            ax.plot(sub["drift_pct"], sub[col], marker="o", markersize=3, linewidth=1.8, color=color, label=model)
            rb = sub[sub["action"].str.contains("rebuild")]
            ax.scatter(rb["drift_pct"], rb[col], marker="s", s=40, facecolors="none", edgecolors=color)
    for ax, title in zip(axes, ["Median q-error", "P95 q-error"]):
        ax.set_title(f"{title} (square = rebuild at this step)")
        ax.set_xlabel("Rows changed since start, %")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    out = csv_path.with_name("drift_chart.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot merged drift results (run merge_drift.py first)")
    parser.add_argument("--experiment-name", required=True, nargs="+", help="Scenario folders under --out-dir")
    parser.add_argument("--out-dir", default="results", help="Folder that holds the scenario folders")
    args = parser.parse_args()
    for exp in args.experiment_name:
        csv = Path(args.out_dir) / exp / "combined.csv"
        if csv.exists():
            plot_combined(csv)
        else:
            print(f"No {csv}; run merge_drift.py first")


if __name__ == "__main__":
    main()

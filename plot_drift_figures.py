#!/usr/bin/env python3
"""
Six figures per drift scenario, one metric each, x = drift %, one line per method:

    median_qerror.png    median q-error          (after the step's action; Hybrid = RB stage)
    p95_qerror.png       P95 q-error
    avg_qerror.png       average q-error
    inference_time.png   median inference time per query, µs
    adaptation_time.png  cumulative adaptation time, s: initial build + every FT/feedback
                         update + every rebuild up to this step (action_time_s summed)
    memory.png           model size, KB (stored numbers x 8 bytes)

Hybrid is labelled Hybrid_RB: its row of a step is the score after Shock -> FT -> RB.

    poetry run python plot_drift_figures.py --results-dir results/exp1_100_buckets
writes <results-dir>/figures/<scenario>/*.png.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from merge_drift import collect_drift  # noqa: E402

# Fixed order = fixed color: a method keeps its color in every figure and scenario
METHODS = ["Hybrid", "EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist"]
LABELS = {"Hybrid": "Hybrid_RB"}
# Validated categorical palette (dataviz reference, light surface), slots 1-6
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# Secondary encoding, so identity never rests on color alone
MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1))]

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#52514e", "#e4e3df"

FIGURES = [
    ("median_qerror", "Median Q-error", "Q-error", lambda d: d["median_qerror"]),
    ("p95_qerror", "P95 Q-error", "Q-error", lambda d: d["p95_qerror"]),
    ("avg_qerror", "Avg Q-error", "Q-error", lambda d: d["avg_qerror"]),
    ("inference_time", "Inference time (median per query)", "Inference time, µs", lambda d: d["infer_us_median"]),
    ("adaptation_time", "Adaptation time (cumulative: initial build + FT + rebuild)", "Adaptation time, s",
     lambda d: d["action_time_s"].fillna(0.0).cumsum()),
    ("memory", "Memory (model size)", "Memory, KB", lambda d: d["size_bytes"] / 1024.0),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)


def plot_scenario(df, scenario, out_dir: Path, bucket_note: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in METHODS if m in set(df["model"])] + sorted(set(df["model"]) - set(METHODS))
    for name, title, ylabel, value in FIGURES:
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
        style_axes(ax)
        for m in models:
            i = METHODS.index(m) if m in METHODS else len(METHODS) - 1
            sub = df[df["model"] == m].sort_values("step")
            y = value(sub)
            ax.plot(sub["drift_pct"], y, label=LABELS.get(m, m), color=COLORS[i], linestyle=DASHES[i],
                    linewidth=2.6 if m == "Hybrid" else 2.0, marker=MARKERS[i], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3 if m == "Hybrid" else 2)
        ax.set_yscale("log")
        # plain numbers on the log axis (1, 1.5, 2, 3, 5, 10, ...), not 10^k
        lo, hi = ax.get_ylim()
        decades = np.log10(hi / max(lo, 1e-300))
        subs = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0) if decades <= 1.2 else (1.0, 2.0, 5.0) if decades <= 3 else (1.0,)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Drift, %", color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)
        ax.set_title(f"{title}\n{scenario} · {bucket_note}", color=INK, fontsize=11, loc="left")
        ax.legend(title="Method", frameon=False, fontsize=9, title_fontsize=9, labelcolor=INK,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description="Per-scenario drift figures: q-error, inference time, adaptation time, memory")
    ap.add_argument("--results-dir", required=True, help="Experiment folder with drift/<scenario>/...")
    args = ap.parse_args()
    root = Path(args.results_dir)
    note = "Freedman–Diaconis buckets" if "freedman" in root.name else "100 buckets" if "100" in root.name else root.name
    for scen in sorted(p for p in (root / "drift").iterdir() if p.is_dir()):
        df = collect_drift(scen)
        if df.empty:
            continue
        plot_scenario(df, scen.name, root / "figures" / scen.name, note)


if __name__ == "__main__":
    main()

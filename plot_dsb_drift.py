#!/usr/bin/env python3
"""Plot line charts for DSB drift experiment results: Median, P95, Avg Q-Error."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

COLUMNS = ["cr_item_sk", "cr_returned_time_sk", "ws_item_sk"]
RESULTS_DIR = Path("results/dsb_drift")

# Use Hybrid_RB as the final hybrid result (best after all adaptation stages)
MODEL_MAP = {
    "EquiWidth": "EquiWidth",
    "EquiHist": "EquiHist",
    "Hybrid_RB": "Hybrid",
}

COLORS = {
    "EquiWidth": "#e74c3c",
    "EquiHist": "#3498db",
    "Hybrid": "#2ecc71",
}

METRICS = ["Median", "P95", "Avg"]


def plot_column(col_name: str):
    csv_path = RESULTS_DIR / f"combined_{col_name}.csv"
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping.")
        return

    df = pd.read_csv(csv_path)
    df["drift_int"] = df["drift"].str.replace("%", "").astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Drift Analysis: {col_name}", fontsize=15, fontweight="bold", y=1.02)

    for ax, metric in zip(axes, METRICS):
        for src_model, label in MODEL_MAP.items():
            sub = df[df["model"] == src_model].sort_values("drift_int")
            ax.plot(
                sub["drift_int"], sub[metric],
                marker="o", markersize=4, linewidth=2,
                color=COLORS[label], label=label,
            )

        ax.set_title(f"{metric} Q-Error", fontsize=13)
        ax.set_xlabel("Drift %", fontsize=11)
        ax.set_ylabel("Q-Error", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 101, 10))

    plt.tight_layout()
    out_path = RESULTS_DIR / f"drift_chart_{col_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    for col in COLUMNS:
        plot_column(col)
    print("\nAll charts saved.")


if __name__ == "__main__":
    main()

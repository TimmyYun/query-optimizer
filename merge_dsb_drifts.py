#!/usr/bin/env python3
"""Merge DSB drift experiment results per column into combined CSVs."""
import pandas as pd
from pathlib import Path


COLUMNS = ["cr_item_sk", "cr_returned_time_sk", "ws_item_sk"]


def merge_column(results_dir: Path, column: str, output_path: Path):
    """Merge equiwidth, equihist, and hybrid results for a single column."""
    ew_path = results_dir / f"dsb_{column}_EquiWidth" / "equiwidth_drift_analysis.csv"
    eh_path = results_dir / f"dsb_{column}_EquiHist" / "equihist_drift_analysis.csv"
    hy_path = results_dir / f"dsb_{column}_Hybrid" / "adaptation_ft_rb.csv"

    for p in [ew_path, eh_path, hy_path]:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping column {column}")
            return

    df_width = pd.read_csv(ew_path)
    df_hist = pd.read_csv(eh_path)
    df_hybrid = pd.read_csv(hy_path)

    all_data = []

    # EquiHist
    for _, row in df_hist.iterrows():
        all_data.append({
            'drift': row['Shift %'],
            'model': 'EquiHist',
            'Median': row['Median Q-Error'],
            'P95': row['P95 Q-Error'],
            'Avg': row['Avg Q-Error'],
            'Time': row['Time'],
            'Count': None,
        })

    # EquiWidth
    for _, row in df_width.iterrows():
        all_data.append({
            'drift': row['Shift %'],
            'model': 'EquiWidth',
            'Median': row['Median Q-Error'],
            'P95': row['P95 Q-Error'],
            'Avg': row['Avg Q-Error'],
            'Time': row['Inference Time'],
            'Count': None,
        })

    # Hybrid (Shock, FT, RB)
    for _, row in df_hybrid.iterrows():
        all_data.append({
            'drift': row['Shift %'],
            'model': 'Hybrid_Shock',
            'Median': row['Shock_Med'],
            'P95': row['Shock_P95'],
            'Avg': row['Shock_Avg'],
            'Time': row['Evaluation Time'],
            'Count': None,
        })
        all_data.append({
            'drift': row['Shift %'],
            'model': 'Hybrid_FT',
            'Median': row['FT_Med'],
            'P95': row['FT_P95'],
            'Avg': row['FT_Avg'],
            'Time': row['FT_Time'],
            'Count': row['Finetuned_Count'],
        })
        all_data.append({
            'drift': row['Shift %'],
            'model': 'Hybrid_RB',
            'Median': row['RB_Med'],
            'P95': row['RB_P95'],
            'Avg': row['RB_Avg'],
            'Time': row['RB_Time'],
            'Count': row['Rebuilt_Count'],
        })

    df_combined = pd.DataFrame(all_data)

    # Sort by drift %, then model
    df_combined['drift_int'] = df_combined['drift'].str.replace('%', '').astype(int)
    model_order = ['EquiHist', 'EquiWidth', 'Hybrid_Shock', 'Hybrid_FT', 'Hybrid_RB']
    df_combined['model'] = pd.Categorical(df_combined['model'], categories=model_order, ordered=True)
    df_combined = df_combined.sort_values(by=['drift_int', 'model']).drop(columns=['drift_int'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Saved {column} -> {output_path} ({len(df_combined)} rows)")


def main():
    results_dir = Path("results/dsb_drift")

    for col in COLUMNS:
        output_path = results_dir / f"combined_{col}.csv"
        merge_column(results_dir, col, output_path)

    print("\nDone. All combined files saved.")


if __name__ == "__main__":
    main()

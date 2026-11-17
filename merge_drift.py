#!/usr/bin/env python3
"""
Merge the drift runs of one scenario into one table.

Reads <out-dir>/<scenario>/<prefix>_<Model>/drift_steps.csv (written by drift_harness.py),
where <prefix> is gradual_<init>_to_<target>, and writes <out-dir>/<scenario>/combined.csv:
the same columns, plus "model", one row per (model, step).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# Model names have no "_", so the model is everything after the last "_"
MODEL_DIR_RE = re.compile(r"^(?P<prefix>.+)_(?P<model>[^_]+)$")


def collect_drift(scenario_dir: Path) -> pd.DataFrame:
    """All drift_steps.csv of one scenario folder, with a 'model' column; empty if none."""
    frames = []
    for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
        m = MODEL_DIR_RE.match(run_dir.name)
        csv = run_dir / "drift_steps.csv"
        if not m or not csv.exists():
            continue
        df = pd.read_csv(csv)
        df.insert(0, "model", m["model"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_scenario(scenario_dir: Path):
    df = collect_drift(scenario_dir)
    if df.empty:
        print(f"No drift runs in {scenario_dir}")
        return None
    out = scenario_dir / "combined.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df)} rows, models: {sorted(df['model'].unique())})")
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge drift runs of a scenario into <scenario>/combined.csv")
    parser.add_argument("--experiment-name", required=True, nargs="+", help="Scenario folders under --out-dir")
    parser.add_argument("--out-dir", default="results", help="Folder that holds the scenario folders")
    args = parser.parse_args()
    for exp in args.experiment_name:
        merge_scenario(Path(args.out_dir) / exp)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge every drift run of one experiment into long-format tables.

Reads results/<experiment>/<prefix>_<Model>/*.csv, where <prefix> is
gradual_<init>_to_<target> (synthetic) or dsb_<column> (DSB), and writes one
results/<experiment>/combined_<prefix>.csv per prefix with one row per
(step, model, stage):

    drift, step, model, stage, label, Median, P95, Avg, Infer_Med, Infer_Avg, Adapt_Time, Rebuilt_Buckets, Initial_Build

Adapt_Time = seconds the model spent at this step to reach this stage, cumulative
within the step (e.g. Hybrid_RB = sample refresh + FT + bucket retraining).

Single-stage CSVs ("Median Q-Error", ...) get stage "" and label = model.
Multi-stage CSVs (<Stage>_Med, <Stage>_P95, ...) get one row per stage and
label = <model>_<stage>, e.g. Hybrid_Shock / Hybrid_FT / Hybrid_RB.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# Model names have no "_" except variant suffixes (e.g. EntropyHist-MEA-Eq_x), so anchor on the base name
MODEL_DIR_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<model>(?:EquiWidth|EquiHist|Hybrid|TKHist|EquiDepth|STHist|EntropyHist)\S*)$"
)
DRIFT_CSV_GLOBS = ["*_drift_analysis.csv", "adaptation_ft_rb.csv"]


def stage_cost(r, stage):
    """Adapt_Time of one stage; the source column names differ per runner."""
    def g(key):
        return float(r[key]) if key in r.index and pd.notna(r[key]) else 0.0
    if stage == "":  # single-stage: ANALYZE rebuild (histograms) or feedback (Equi-Hist)
        return g("Build Time") + g("Update Time")
    return {
        "Static": 0.0,                                              # EntropyHist: never updated
        "Rebuild": g("Rebuild_Time"),                               # EntropyHist: rebuilt from the step's sample
        "Shock": g("Refresh_Time"),                                 # Hybrid: bucket masses from the sample; STHist: 0
        "FT": g("Refresh_Time") + g("FT_Time"),
        "RB": g("Refresh_Time") + g("FT_Time") + g("RB_Time"),
        "Online": g("Update_Time"),                                 # STHist: feedback on the adapt half
        "Post": g("Update_Time"),
    }.get(stage)


def stage_names(columns):
    """Stages are the prefixes of <Stage>_Med columns (Shock, FT, RB, Online, Post, Static, Rebuild)."""
    return [c[: -len("_Med")] for c in columns if c.endswith("_Med") and not c.endswith("_Infer_Med")]


def to_long(df, model):
    rows = []
    stages = stage_names(df.columns)
    for _, r in df.iterrows():
        drift = int(str(r["Shift %"]).rstrip("%"))
        base = {
            "drift": drift,
            "step": r.get("Step", ""),
            "model": model,
            # Building the model on step 0 (Hybrid: training every bucket's model); not part of any stage
            "Initial_Build": (r.get("Build_Time", r.get("Build Time")) if drift == 0 else None),
        }
        if stages:
            for st in stages:
                rows.append({
                    **base,
                    "stage": st,
                    "label": f"{model}_{st}",
                    "Median": r[f"{st}_Med"],
                    "P95": r.get(f"{st}_P95"),
                    "Avg": r.get(f"{st}_Avg"),
                    "Infer_Med": r.get(f"{st}_Infer_Med"),
                    "Infer_Avg": r.get(f"{st}_Infer_Avg"),
                    "Adapt_Time": stage_cost(r, st),
                    "Rebuilt_Buckets": r.get("Rebuilt_Count") if st == "RB" else None,
                })
        else:
            rows.append({
                **base,
                "stage": "",
                "label": model,
                "Median": r["Median Q-Error"],
                "P95": r.get("P95 Q-Error"),
                "Avg": r.get("Avg Q-Error"),
                "Infer_Med": r.get("Median Inference Time"),
                "Infer_Avg": r.get("Avg Inference Time"),
                "Adapt_Time": stage_cost(r, ""),
                "Rebuilt_Buckets": None,
            })
    return rows


def collect_drift_rows(exp_dir: Path):
    """{prefix: [long-format rows]} for every <prefix>_<Model>/ drift run in exp_dir."""
    groups = {}
    for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        m = MODEL_DIR_RE.match(run_dir.name)
        if not m:
            continue
        csvs = [c for g in DRIFT_CSV_GLOBS for c in run_dir.glob(g)]
        if not csvs:
            print(f"WARNING: no drift CSV in {run_dir}, skipping")
            continue
        df = pd.read_csv(csvs[0])
        groups.setdefault(m["prefix"], []).extend(to_long(df, m["model"]))
    return groups


def merge_experiment(exp_dir: Path):
    groups = collect_drift_rows(exp_dir)
    if not groups:
        print(f"No drift runs found in {exp_dir}")
        return []

    outputs = []
    for prefix, rows in groups.items():
        out = exp_dir / f"combined_{prefix}.csv"
        pd.DataFrame(rows).sort_values(["drift", "label"]).to_csv(out, index=False)
        print(f"Saved {out} ({len(rows)} rows, models: {sorted({r['label'] for r in rows})})")
        outputs.append(out)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Merge drift results of one experiment into combined_<prefix>.csv")
    parser.add_argument("--experiment-name", required=True, nargs="+",
                        help="One or more folders under --out-dir (e.g. dsb_drift drift_5_ds_zipf_wl_normal)")
    parser.add_argument("--out-dir", default="results", help="Root of the results tree")
    args = parser.parse_args()

    for exp in args.experiment_name:
        merge_experiment(Path(args.out_dir) / exp)


if __name__ == "__main__":
    main()

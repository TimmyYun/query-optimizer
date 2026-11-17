#!/usr/bin/env python3
"""
Readable tables for one experiment folder, e.g. results/exp1_100_buckets:

    <root>/static/<benchmark>/<dataset>/summary_<Model>.json      (run_experiments.py --out-dir <root>/static)
    <root>/drift/<scenario>/<prefix>_<Model>/drift_steps.csv      (run_drift_experiments.py --out-dir <root>/drift)

Writes into <root>:
    static_results.csv        one row per benchmark x dataset x model, every metric
    static_median_qerror.csv  median q-error: rows = benchmark/dataset, columns = models
    static_p95_qerror.csv     the same for P95
    static_ranking.csv        per benchmark x model: mean place by median (1 = best), wins
    histogram_size.csv        size of each model in bytes (8 bytes per stored number)
    hybrid_cdf_choice.csv     how many buckets got each local CDF in Hybrid (static + drift)
    drift_results.csv         one row per scenario x model x step
    drift_summary.csv         per scenario x model: error over the steps, rebuilds and their real time;
                              for Hybrid also the mean Shock / FT / RB error and FT / RB time separately
    results.xlsx              all of the above, one sheet each
and for every drift scenario <root>/drift/<scenario>/combined.csv + drift_chart.png.

    poetry run python export_results.py --results-dir results/exp1_100_buckets
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from merge_drift import collect_drift
from plot_drift import plot_combined

MODEL_ORDER = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid"]


def ordered(models):
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))


def static_table(static_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(static_dir.rglob("summary_*.json")):
        s = json.loads(path.read_text())
        m = s["metrics"]
        rel = path.parent.relative_to(static_dir).parts
        rows.append({
            "benchmark": rel[0],
            "dataset": "/".join(rel[1:]),
            "model": s["model"],
            "queries": m.get("n_queries"),
            "median_qerror": m.get("median_q_error"),
            "p95_qerror": m.get("p95_q_error"),
            "avg_qerror": m.get("avg_q_error"),
            "buckets": m.get("n_buckets"),
            "stored_numbers": m.get("n_values"),
            "size_bytes": m.get("size_bytes"),
            "build_s": m.get("build_time"),
            "feedback_updates_s": m.get("update_time_total"),
            "infer_us_median": None if m.get("median_infer_time") is None else m["median_infer_time"] * 1e6,
            "infer_us_mean": None if m.get("avg_infer_time") is None else m["avg_infer_time"] * 1e6,
            "cdf_choice": m.get("cdf_choice"),
        })
    return pd.DataFrame(rows)


def pivot(df, value):
    p = df.pivot_table(index=["benchmark", "dataset"], columns="model", values=value, aggfunc="first")
    return p[ordered(p.columns)].reset_index()


def ranking(df):
    df = df.copy()
    df["place"] = df.groupby(["benchmark", "dataset"])["median_qerror"].rank(method="min")
    g = df.groupby(["benchmark", "model"])
    res = pd.DataFrame({
        "mean_place": g["place"].mean().round(2),
        "wins": g["place"].apply(lambda r: int((r == 1).sum())),
        "datasets": g["place"].size(),
        "mean_median_qerror": g["median_qerror"].mean(),
        "mean_p95_qerror": g["p95_qerror"].mean(),
    }).reset_index()
    return res.sort_values(["benchmark", "mean_place"])


def cdf_rows(static_df, drift_df):
    rows = []
    static_hybrid = static_df[static_df["model"] == "Hybrid"] if "model" in static_df else static_df
    for _, r in static_hybrid.iterrows():
        if isinstance(r["cdf_choice"], dict):
            rows.append({"benchmark": r["benchmark"], "dataset": r["dataset"], "when": "static", **r["cdf_choice"]})
    if not drift_df.empty and "cdf_choice" in drift_df:
        is_build = (drift_df["action"] == "initial build") | drift_df["action"].str.contains("rebuild")
        hyb = drift_df[(drift_df["model"] == "Hybrid") & is_build]
        for _, r in hyb.iterrows():
            choice = dict(kv.split(":") for kv in str(r["cdf_choice"]).split())
            rows.append({"benchmark": r["scenario"], "dataset": f"step {int(r['step'])} ({r['drift_pct']:g}% changed)",
                         "when": r["action"], **{k: int(v) for k, v in choice.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fixed = ["benchmark", "dataset", "when"]
    types = [c for c in df.columns if c not in fixed]
    types = sorted(types, key=lambda c: (c == "Empty", -df[c].fillna(0).sum()))
    df = df[fixed + types]
    df[types] = df[types].fillna(0).astype(int)
    return df


def drift_table(drift_dir: Path) -> pd.DataFrame:
    frames = []
    for scen in sorted(p for p in drift_dir.iterdir() if p.is_dir()) if drift_dir.exists() else []:
        df = collect_drift(scen)
        if df.empty:
            continue
        df.to_csv(scen / "combined.csv", index=False)
        plot_combined(scen / "combined.csv")
        df.insert(0, "scenario", scen.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def drift_summary(df):
    rows = []
    for (scen, model), sub in df.groupby(["scenario", "model"]):
        sub = sub.sort_values("step")
        later = sub[sub["step"] > 0]
        rb = later[later["action"].str.contains("rebuild")]
        fb = later[later["action"] == "feedback update"]
        staged = "ft_time_s" in later and later["ft_time_s"].notna().any()  # Hybrid: Shock -> FT -> RB
        rows.append({
            "scenario": scen,
            "model": model,
            "mean_median_qerror": later["median_qerror"].mean(),
            "worst_median_qerror": later["median_qerror"].max(),
            "last_step_median_qerror": sub["median_qerror"].iloc[-1],
            "mean_p95_qerror": later["p95_qerror"].mean(),
            "initial_build_s": sub["action_time_s"].iloc[0],
            "rebuilds": len(rb),
            "rebuild_steps_pct": " ".join(f"{p:g}" for p in rb["drift_pct"]),
            "rebuild_time_total_s": rb["rb_time_s" if staged else "action_time_s"].sum(),
            "rebuild_time_mean_s": rb["rb_time_s" if staged else "action_time_s"].mean() if len(rb) else 0.0,
            "feedback_time_total_s": later["ft_time_s"].sum() if staged else fb["action_time_s"].sum(),
            "mean_shock_median_qerror": later["before_median_qerror"].mean(),
            "mean_ft_median_qerror": later["ft_median_qerror"].mean() if staged else None,
            "mean_rb_median_qerror": later["rb_median_qerror"].mean() if staged else None,
            "rebuilt_buckets_total": int(later["rebuilt_buckets"].sum()) if staged else None,
            "size_bytes": sub["size_bytes"].iloc[0],
            "buckets": sub["n_buckets"].iloc[0],
        })
    res = pd.DataFrame(rows)
    res["place"] = res.groupby("scenario")["mean_median_qerror"].rank(method="min").astype(int)
    return res.sort_values(["scenario", "place"])


def main():
    parser = argparse.ArgumentParser(description="Readable tables + xlsx for one experiment folder")
    parser.add_argument("--results-dir", required=True, help="Experiment folder with static/ and drift/")
    args = parser.parse_args()
    root = Path(args.results_dir)

    tables = {}
    static_df = static_table(root / "static") if (root / "static").exists() else pd.DataFrame()
    drift_df = drift_table(root / "drift")

    if not static_df.empty:
        tables["static_results"] = static_df.drop(columns=["cdf_choice"])
        tables["static_median_qerror"] = pivot(static_df, "median_qerror")
        tables["static_p95_qerror"] = pivot(static_df, "p95_qerror")
        tables["static_ranking"] = ranking(static_df)
        size = pivot(static_df, "size_bytes")
        buckets = pivot(static_df, "buckets")
        tables["histogram_size"] = size.merge(buckets, on=["benchmark", "dataset"], suffixes=("", " buckets"))
    cdf = cdf_rows(static_df, drift_df) if not static_df.empty or not drift_df.empty else pd.DataFrame()
    if not cdf.empty:
        tables["hybrid_cdf_choice"] = cdf
    if not drift_df.empty:
        tables["drift_results"] = drift_df
        tables["drift_summary"] = drift_summary(drift_df)

    for name, df in tables.items():
        df.to_csv(root / f"{name}.csv", index=False, float_format="%.6g")
        print(f"Saved {root / (name + '.csv')} ({len(df)} rows)")

    with pd.ExcelWriter(root / "results.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 50)
    print(f"Saved {root / 'results.xlsx'}")


if __name__ == "__main__":
    main()

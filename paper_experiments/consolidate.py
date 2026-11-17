#!/usr/bin/env python3
"""
Consolidated tables over every part of results/paper_experiments (re-runnable on partial results).

    poetry run python -m paper_experiments.consolidate

Reads (never writes) methodology/reproduction/static, dsb_expanded/{static,displacement,additive_200},
fd_reduction/fd_reduction_all.csv, memory_analysis/, and the synthetic drift runs of
results/exp1_100_buckets and results/exp2_freedman_diaconis (read-only, run on 2026-11-10 before the
instrumentation; their estimates are bit-identical to the instrumented code, PLAN.md §0).

Writes into results/paper_experiments/consolidated/:
  all_static.csv       every static configuration, schema.ID_COLUMNS + unified metrics (stage STATIC)
  all_drift_long.csv   every drift run, one row per (step, stage), via schema.drift_steps_to_long
  completeness.csv     expected configuration -> OK / PARTIAL / FAILED / NOT_RUN (no method disappears silently)
  fd_vs_fixed100.csv   paired FD vs 100-bucket comparison (static and drift)
  best_by_accuracy.csv / best_by_memory.csv / best_by_latency.csv   separate rankings with runner-up and margin
  checks.csv           consistency / leakage-related checks
  consolidated.xlsx    all of the above (paper_experiments.excel_export)
  CONSOLIDATED.md      text generated only from those tables
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.schema import ID_COLUMNS, drift_steps_to_long, final_stage  # noqa: E402

BASE = ROOT / "results" / "paper_experiments"
OUT = BASE / "consolidated"
EXP_ROOTS = {"fixed100": ROOT / "results" / "exp1_100_buckets", "fd": ROOT / "results" / "exp2_freedman_diaconis"}
BIN_RX = re.compile(r"^(fd|fixed\d+|fd_x[\d.]+|fd_r\d+|fd_merge_r\d+|mem_match_.+)$")
MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid"]
# labelled MODIFICATION of Hybrid FD CDF (models/hybrid_corrected.py): expected wherever Hybrid is expected;
# its diagnostic variant -PredictOnly is reported only where it exists (not expected)
CORRECTED = "Hybrid-Corrected"
PREDICT_ONLY = "Hybrid-Corrected-PredictOnly"
RUN_RX = re.compile(r"gradual_(.+)_to_(.+?)_(EquiWidth|EquiDepth|TKHist|EntropyHist-MEA|STHist|Hybrid-Corrected-PredictOnly|Hybrid-Corrected|Hybrid)$")
REPRO_SCENARIOS = {  # registry.STATIC_SCENARIOS experiment names of the thesis' synthetic tables
    "synth_simple_wide": ("simple", "uniform_wide"), "synth_simple_narrow": ("simple", "uniform_narrow"),
    "synth_hard_wide": ("hard", "uniform_wide"), "synth_hard_narrow": ("hard", "uniform_narrow"),
    "synth_hard_narrow_data_driven": ("hard", "data_driven_narrow"),
}
SYNTH_DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
DSB_WORKLOADS = ["dd_narrow", "uniform_narrow", "uniform_wide"]
DSB_SCENARIOS = {"displacement": 21, "additive_200": 41}
UNIFIED = ID_COLUMNS + [
    "n_queries", "median_qerror", "p25_qerror", "p75_qerror", "p95_qerror", "avg_qerror", "mae_rows",
    "build_s", "feedback_update_s", "infer_us_median", "infer_us_mean", "infer_us_p95", "latency_us_quiet",
    "n_buckets", "n_values", "size_bytes", "cdf_choice", "family", "reduction_pct", "K_FD", "matched_to_K", "source",
]


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _model_version(run_dir: Path, model: str):
    """model_version from the manifest written next to the run (manifest_<Model>.json / manifest.json)."""
    for d in (run_dir, *run_dir.parents):
        if d == BASE.parent:
            break
        for name in (f"manifest_{model}.json", "manifest.json"):
            m = _read_json(d / name)
            if m:
                cfg = m.get("config", {})
                if cfg.get("model") == model and cfg.get("model_version"):
                    return cfg["model_version"]
                mv = m.get("model_versions", {}).get(model)
                if mv:
                    return mv
    return None


def _bin_of(parts):
    return next((p for p in parts if BIN_RX.match(p)), None)


# ----------------------------------------------------------------------------- static

def _summary_row(path: Path, ids: dict) -> dict:
    s = _read_json(path)
    m = s["metrics"]
    us = lambda k: None if m.get(k) is None else m[k] * 1e6  # noqa: E731
    model = s["model"]
    return {**{c: ids.get(c) for c in ID_COLUMNS}, "model": model,
            "model_version": ids.get("model_version") or _model_version(path.parent, model),
            "seed": 42, "stage": "STATIC", "drift_type": "none", "drift_pct": 0.0,
            "workload": ids.get("workload") or s.get("wl_name"),
            "bucket_count": m.get("n_buckets"), "n_queries": m.get("n_queries"),
            "median_qerror": m.get("median_q_error"), "p25_qerror": m.get("p25_q_error"),
            "p75_qerror": m.get("p75_q_error"), "p95_qerror": m.get("p95_q_error"), "avg_qerror": m.get("avg_q_error"),
            "mae_rows": m.get("mae_rows"), "build_s": m.get("build_time"), "feedback_update_s": m.get("update_time_total"),
            "infer_us_median": us("median_infer_time"), "infer_us_mean": us("avg_infer_time"),
            "infer_us_p95": us("p95_infer_time"), "n_buckets": m.get("n_buckets"), "n_values": m.get("n_values"),
            "size_bytes": m.get("size_bytes"),
            "cdf_choice": json.dumps(m["cdf_choice"]) if isinstance(m.get("cdf_choice"), dict) else m.get("cdf_choice"),
            "source": str(path.relative_to(ROOT))}


def static_rows() -> pd.DataFrame:
    rows = []
    rep = BASE / "methodology" / "reproduction" / "static"
    for f in sorted(rep.rglob("summary_*.json")) if rep.exists() else []:
        rel = f.relative_to(rep).parts  # <bin>/<experiment>/<dist>/summary_<M>.json
        if len(rel) < 4:
            continue
        variant, wl = REPRO_SCENARIOS.get(rel[1], (rel[1], rel[1]))
        rows.append(_summary_row(f, {"experiment_id": f"methodology_repro/{rel[1]}", "dataset": f"{variant}/{rel[2]}",
                                     "table": None, "column": rel[2], "workload": wl, "bin_strategy": rel[0]}))
    dsb = BASE / "dsb_expanded" / "static"
    for f in sorted(dsb.rglob("summary_*.json")) if dsb.exists() else []:
        rel = f.relative_to(dsb).parts  # <bin>/<workload>/<table.column>/summary_<M>.json
        if len(rel) < 4:
            continue
        col = rel[2]
        table, _, column = col.partition(".")
        rows.append(_summary_row(f, {"experiment_id": "dsb_expanded_static", "dataset": f"dsb_expanded/{col}",
                                     "table": table, "column": column, "workload": rel[1], "bin_strategy": rel[0]}))
    df = pd.DataFrame(rows)
    fd = BASE / "fd_reduction" / "fd_reduction_all.csv"
    if fd.exists():
        f = pd.read_csv(fd, low_memory=False).rename(columns={"build_time_s": "build_s", "lat_us_median": "latency_us_quiet"})
        f["source"] = str(fd.relative_to(ROOT))
        df = pd.concat([df, f[[c for c in UNIFIED if c in f.columns]]], ignore_index=True)
    for c in UNIFIED:
        if c not in df.columns:
            df[c] = None
    return df[UNIFIED]


# ----------------------------------------------------------------------------- drift

def _legacy_flags(df: pd.DataFrame, staged: bool) -> pd.DataFrame:
    """Runs made before checkpoint_triggered / rebuild_performed existed: derive them from action / rebuilt_buckets."""
    df = df.copy()
    if "checkpoint_triggered" not in df.columns:
        df["checkpoint_triggered"] = df["action"].isin(["FT + rebuild", "rebuild"])
    if "rebuild_performed" not in df.columns:
        if staged:
            df["rebuild_performed"] = df.get("rebuilt_buckets", pd.Series(0, index=df.index)).fillna(0) > 0
        else:
            df["rebuild_performed"] = df["action"] == "rebuild"
    return df


def drift_runs():
    """-> [(run_dir, ids, staged, df)] for every drift_steps.csv we consolidate."""
    runs = []
    for scen in DSB_SCENARIOS:
        root = BASE / "dsb_expanded" / scen
        for f in sorted(root.rglob("drift_steps.csv")) if root.exists() else []:
            m = RUN_RX.match(f.parent.name)
            if not m:
                continue
            init, model = m.group(1), m.group(3)
            table, _, column = init.partition(".")
            ids = {"experiment_id": f"dsb_expanded_{scen}", "dataset": f"dsb_expanded/{init}", "table": table,
                   "column": column, "workload": "per_step_dd_1000000", "drift_type": None,
                   "bin_strategy": _bin_of(f.relative_to(root).parts), "model": model,
                   "model_version": _model_version(f.parent, model), "seed": 42}
            runs.append((f.parent, ids, model.startswith("Hybrid"), pd.read_csv(f), scen))
    try:
        from registry import DRIFT_SCENARIOS
    except Exception:
        DRIFT_SCENARIOS = {}

    def synth_ids(scen, model, bin_):
        cfg = DRIFT_SCENARIOS.get(scen, {})
        dsb = scen.startswith("dsb_drift_")
        dataset = f"dsb/{scen.replace('dsb_drift_', '')}" if dsb else f"hard/{Path(cfg.get('shift_dir', scen)).name}"
        wl = "official" if dsb else ("per_step_dd_1000000" if cfg.get("static_workload") is None
                                     else Path(cfg["static_workload"]).stem)
        return {"dataset": dataset, "table": "web_sales" if dsb else None,
                "column": scen.replace("dsb_drift_", "") if dsb else None, "workload": wl,
                "drift_type": "churn_official" if dsb else ("additive" if "_add_" in scen else "displacement"),
                "bin_strategy": bin_, "model": model, "seed": 42}

    meth = BASE / "methodology"
    for f in sorted(meth.rglob("drift_steps.csv")) if meth.exists() else []:
        m = RUN_RX.match(f.parent.name)
        rel = f.parent.relative_to(BASE).parts
        scen = next((p for p in rel if p in DRIFT_SCENARIOS), None)
        if not m or scen is None:
            continue
        df = pd.read_csv(f)
        bin_ = _bin_of(rel) or ("fixed100" if int(df["n_buckets"].iloc[0]) == 100 else "fd")
        ids = {**synth_ids(scen, m.group(3), bin_), "experiment_id": "/".join(rel[:-1]),
               "model_version": _model_version(f.parent, m.group(3))}
        runs.append((f.parent, ids, m.group(3).startswith("Hybrid"), df, scen))
    for bin_, root in EXP_ROOTS.items():
        droot = root / "drift"
        for f in sorted(droot.rglob("drift_steps.csv")) if droot.exists() else []:
            scen = f.relative_to(droot).parts[0]
            m = RUN_RX.match(f.parent.name)
            if not m:
                continue
            model = m.group(3)
            cfg = DRIFT_SCENARIOS.get(scen, {})
            dsb = scen.startswith("dsb_drift_")
            dataset = (f"dsb/{scen.replace('dsb_drift_', '')}" if dsb else
                       f"hard/{Path(cfg.get('shift_dir', scen)).name}")
            wl = "official" if dsb else ("per_step_dd_1000000" if cfg.get("static_workload") is None
                                         else Path(cfg["static_workload"]).stem)
            ids = {"experiment_id": f"synthetic_drift/{scen}" if not dsb else f"dsb_official_drift/{scen}",
                   "dataset": dataset, "table": "web_sales" if dsb else None,
                   "column": scen.replace("dsb_drift_", "") if dsb else None, "workload": wl,
                   "drift_type": "churn_official" if dsb else ("additive" if "_add_" in scen else "displacement"),
                   "bin_strategy": bin_, "model": model, "model_version": f"{model}:run-2026-11-10-pre-instrumentation",
                   "seed": 42}
            runs.append((f.parent, ids, model.startswith("Hybrid"), pd.read_csv(f), scen))
    return runs


def drift_long(runs) -> pd.DataFrame:
    parts = []
    for run_dir, ids, staged, df, scen in runs:
        df = _legacy_flags(df, staged)
        if ids.get("drift_type") is None and "drift_type" in df.columns:
            ids = {**ids, "drift_type": df["drift_type"].iloc[0]}
        lg = drift_steps_to_long(df, {**ids, "bucket_count": None}, staged=staged)
        lg["source"] = str(run_dir.relative_to(ROOT))
        lg["scenario"] = scen
        parts.append(lg)
    parts = [p.dropna(axis=1, how="all") for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ID_COLUMNS)


# ----------------------------------------------------------------------------- completeness

def _jobs_status():
    p = BASE / "dsb_expanded" / "jobs_status.csv"
    if not p.exists():
        return {}
    js = pd.read_csv(p)
    out = {}
    for _, r in js.iterrows():
        key = (str(r.get("kind")), str(r.get("bin")), str(r.get("workload_or_scenario")), str(r.get("column")), str(r.get("model")))
        out[key] = r
    return out


def _dsb_columns():
    ready = _read_json(ROOT / "data" / "dsb_expanded" / "READY_STATIC.json")
    if ready and isinstance(ready.get("columns"), dict):
        return list(ready["columns"])
    man = BASE / "dsb_expanded" / "dataset_manifest.csv"
    if man.exists():
        d = pd.read_csv(man)
        col = next((c for c in ("dataset", "column_id", "name") if c in d.columns), None)
        if col:
            return d[col].astype(str).tolist()
    return []


def completeness(static: pd.DataFrame, runs) -> pd.DataFrame:
    rows = _completeness_base(static, runs)
    df = pd.DataFrame(rows)
    # Hybrid-Corrected: expected wherever Hybrid is expected (same part, same config); NOT_RUN until present
    have = set(zip(static["experiment_id"], static["dataset"], static["workload"], static["model"], static["bin_strategy"]))
    found_runs = {(scen, ids["dataset"], ids["model"], ids["bin_strategy"]): len(d) for _, ids, _, d, scen in runs}
    extra = []
    for _, r in df[df["model"] == "Hybrid"].iterrows():
        key = r.get("key")
        if not isinstance(key, tuple):  # parts that list their own models (memory_analysis, fd_reduction)
            continue
        for model, expected in ((CORRECTED, True), (PREDICT_ONLY, False)):
            ok = False
            if isinstance(key, tuple) and key[0] == "static":
                ok = (key[1], key[2], key[3], model, key[4]) in have
            elif isinstance(key, tuple) and key[0] == "drift":
                n = found_runs.get((key[1], key[2], model, key[3]))
                ok = n == key[4] if key[4] else bool(n)
                if n and not ok:
                    extra.append({**r.to_dict(), "model": model, "status": f"PARTIAL ({n}/{key[4]} steps)"})
                    continue
            if ok:
                extra.append({**r.to_dict(), "model": model, "status": "OK"})
            elif expected:
                extra.append({**r.to_dict(), "model": model, "status": "NOT_RUN"})
    out = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return out.drop(columns=["key"], errors="ignore")


def _completeness_base(static, runs):
    rows = []
    have = set(zip(static["experiment_id"], static["dataset"], static["workload"], static["model"], static["bin_strategy"]))
    rep = BASE / "methodology" / "reproduction" / "static"
    for bin_ in ("fd", "fixed100"):
        for exp, (variant, wl) in REPRO_SCENARIOS.items():
            for d in SYNTH_DISTS:
                for m in MODELS:
                    ok = (f"methodology_repro/{exp}", f"{variant}/{d}", wl, m, bin_) in have
                    started = (rep / bin_ / exp / d).exists()
                    # MISSING: the scenario ran (or is running) but this model's summary is absent — failed or not yet done
                    rows.append({"part": "methodology_repro_static", "config": f"{exp}|{d}|{bin_}", "model": m,
                                 "status": "OK" if ok else ("MISSING" if started else "NOT_RUN"),
                                 "key": ("static", f"methodology_repro/{exp}", f"{variant}/{d}", wl, bin_)})
    jobs = _jobs_status()
    cols = _dsb_columns()
    for bin_ in ("fd", "fixed100"):
        # 'official' only where the benchmark's templates filter the column (Agent 2 ran it there)
        offc = {c for c in cols if (BASE / "dsb_expanded" / "static" / bin_ / "official" / c).exists()}
        for wl in [*DSB_WORKLOADS, "official"]:
            for col in (cols if wl != "official" else sorted(offc)):
                for m in MODELS:
                    ok = ("dsb_expanded_static", f"dsb_expanded/{col}", wl, m, bin_) in have
                    j = jobs.get(("static", bin_, wl, col, m))
                    st = "OK" if ok else ("FAILED" if j is not None and (j.get("rc") not in (0, "0") or str(j.get("outputs_ok")) == "False")
                                          else "NOT_RUN")
                    rows.append({"part": "dsb_expanded_static", "config": f"{col}|{wl}|{bin_}", "model": m, "status": st,
                                 "key": ("static", "dsb_expanded_static", f"dsb_expanded/{col}", wl, bin_)})
    found = {}
    for run_dir, ids, staged, df, scen in runs:
        found[(scen, ids["dataset"], ids["model"], ids["bin_strategy"])] = len(df)
    for scen, n_exp in DSB_SCENARIOS.items():
        for bin_ in ("fd", "fixed100"):
            for col in cols:
                for m in MODELS:
                    n = found.get((scen, f"dsb_expanded/{col}", m, bin_))
                    j = jobs.get(("drift", bin_, scen, col, m))
                    if n == n_exp:
                        st = "OK"
                    elif n:
                        st = f"PARTIAL ({n}/{n_exp} steps)"
                    elif j is not None and j.get("rc") not in (0, "0"):
                        st = "FAILED"
                    else:
                        st = "NOT_RUN"
                    rows.append({"part": f"dsb_expanded_{scen}", "config": f"{col}|{bin_}", "model": m, "status": st,
                                 "key": ("drift", scen, f"dsb_expanded/{col}", bin_, n_exp)})
    try:
        from registry import DRIFT_SCENARIOS
    except Exception:
        DRIFT_SCENARIOS = {}
    for scen in DRIFT_SCENARIOS:
        for bin_ in EXP_ROOTS:
            for m in MODELS:
                hit = [k for k in found if k[0] == scen and k[2] == m and k[3] == bin_]
                rows.append({"part": "exp1_exp2_drift (read-only)", "config": f"{scen}|{bin_}", "model": m,
                             "status": "OK" if hit else "NOT_RUN"})  # read-only trees: no Hybrid-Corrected expected
    fdc = BASE / "fd_reduction" / "completeness.csv"
    if fdc.exists():
        for _, r in pd.read_csv(fdc).iterrows():
            missing = [] if pd.isna(r.get("missing")) else str(r["missing"]).split(";")
            rows.append({"part": "fd_reduction", "config": f"{r['dataset']}", "model": "(all)",
                         "status": "OK" if not missing else f"PARTIAL ({int(r['present'])}/{int(r['expected'])})",
                         "detail": ";".join(missing[:20]) + (" ..." if len(missing) > 20 else "")})
    else:
        rows.append({"part": "fd_reduction", "config": "-", "model": "(all)", "status": "NOT_RUN"})
    ms = BASE / "memory_analysis" / "run_status.csv"
    if ms.exists():
        for _, r in pd.read_csv(ms).iterrows():
            rows.append({"part": "memory_analysis", "config": f"{r['dataset']}|{r['bin']}", "model": r["model"],
                         "status": "OK" if str(r["status"]).startswith("OK") or ": OK" in str(r["status"]) else str(r["status"])})
    else:
        rows.append({"part": "memory_analysis", "config": "-", "model": "(all)", "status": "NOT_RUN"})
    return rows


# ----------------------------------------------------------------------------- comparisons

def drift_aggregate(long: pd.DataFrame) -> pd.DataFrame:
    """Per run: mean over drift steps >= 1 of the final-stage metrics (the state each step ends in)."""
    if long.empty:
        return pd.DataFrame()
    fin = final_stage(long[long["step"] > 0])
    keys = ["experiment_id", "dataset", "workload", "drift_type", "bin_strategy", "model"]
    agg = fin.groupby(keys, dropna=False).agg(
        steps=("step", "nunique"), median_qerror=("median_qerror", "mean"), p95_qerror=("p95_qerror", "mean"),
        max_p95_qerror=("p95_qerror", "max"), size_bytes=("size_bytes", "mean"), infer_us_median=("infer_us_median", "mean"),
        rebuilds=("rebuild_performed", "sum"),
        shock_median_qerror=("median_qerror", lambda s: np.nan)).reset_index()
    shock = long[(long["stage"] == "SHOCK")].groupby(keys, dropna=False)["median_qerror"].mean().rename("shock_median_qerror")
    agg = agg.drop(columns="shock_median_qerror").merge(shock.reset_index(), on=keys, how="left")
    return agg


def fd_vs_fixed(static: pd.DataFrame, dagg: pd.DataFrame) -> pd.DataFrame:
    out = []
    metrics = ["median_qerror", "p95_qerror", "size_bytes", "infer_us_median"]
    s = static[static["bin_strategy"].isin(["fd", "fixed100"])]
    keys = ["experiment_id", "dataset", "workload", "model"]
    for kind, df in (("static", s), ("drift_mean_final_stage", dagg)):
        if df.empty:
            continue
        a = df[df["bin_strategy"] == "fd"].set_index(keys)[metrics + (["n_buckets"] if "n_buckets" in df else [])]
        b = df[df["bin_strategy"] == "fixed100"].set_index(keys)[metrics + (["n_buckets"] if "n_buckets" in df else [])]
        j = a.join(b, lsuffix="_fd", rsuffix="_fixed100", how="inner")
        for mt in metrics:
            j[f"{mt}_ratio_fd_over_100"] = j[f"{mt}_fd"] / j[f"{mt}_fixed100"]
        j = j.reset_index()
        j.insert(0, "kind", kind)
        out.append(j)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def comparison_groups(static: pd.DataFrame) -> pd.DataFrame:
    """
    compare_group = bin_strategy, except FD reduction: all models at the same K (canonical + merged variants)
    form one group 'same K r=<pct>%', and the memory-matched baselines form 'same memory as Hybrid K=<K>'
    together with that Hybrid configuration (the row is repeated in both groups).
    """
    s = static.copy()
    s["compare_group"] = s["bin_strategy"]
    fd = s["experiment_id"] == "fd_reduction"
    if not fd.any():
        return s
    sk = fd & s["family"].isin(["same_K", "merge"]) & s["reduction_pct"].notna()
    s.loc[sk, "compare_group"] = "same K r=" + s.loc[sk, "reduction_pct"].astype(float).astype(int).astype(str) + "%"
    mm = fd & (s["family"] == "memory_matched") & s["matched_to_K"].notna()
    s.loc[mm, "compare_group"] = "same memory as Hybrid K=" + s.loc[mm, "matched_to_K"].astype(float).astype(int).astype(str)
    extra = []
    for (ds, k), _ in s[mm].groupby(["dataset", "matched_to_K"]):
        h = s[fd & (s["family"] == "same_K") & (s["model"] == "Hybrid") & (s["dataset"] == ds) & (s["bucket_count"] == k)]
        if len(h):
            extra.append(h.head(1).assign(compare_group=f"same memory as Hybrid K={int(k)}"))
    return pd.concat([s, *extra], ignore_index=True)


def best_tables(static: pd.DataFrame, dagg: pd.DataFrame):
    groups = []
    s = comparison_groups(static)
    s["kind"] = "static"
    groups.append((s, ["experiment_id", "dataset", "workload", "compare_group"]))
    if not dagg.empty:
        d = dagg.copy()
        d["kind"] = "drift (mean over steps, final stage)"
        d["compare_group"] = d["bin_strategy"]
        groups.append((d, ["experiment_id", "dataset", "workload", "drift_type", "compare_group"]))
    acc, mem, lat = [], [], []

    def ranked(g, col):
        g = g.dropna(subset=[col]).sort_values(col)
        return g

    for df, keys in groups:
        for k, g in df.groupby(keys, dropna=False):
            base = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
            base["kind"] = g["kind"].iloc[0]
            base["n_models"] = g["model"].nunique()
            row = dict(base)
            for col in ("median_qerror", "p95_qerror"):
                r = ranked(g, col)
                if r.empty:
                    continue
                row[f"best_by_{col}"] = r["model"].iloc[0]
                row[f"best_{col}"] = r[col].iloc[0]
                if len(r) > 1:
                    row[f"runner_up_by_{col}"] = r["model"].iloc[1]
                    row[f"runner_up_{col}"] = r[col].iloc[1]
                    row[f"margin_{col}_ratio"] = r[col].iloc[1] / r[col].iloc[0]
                hy = r.reset_index(drop=True)
                pos = hy.index[hy["model"] == "Hybrid"]
                row[f"hybrid_rank_by_{col}"] = int(pos[0]) + 1 if len(pos) else None
                pos = hy.index[hy["model"] == CORRECTED]
                row[f"hybrid_corrected_rank_by_{col}"] = int(pos[0]) + 1 if len(pos) else None
            row["same_winner_median_and_p95"] = row.get("best_by_median_qerror") == row.get("best_by_p95_qerror")
            acc.append(row)
            for col, sink, ctx in (("size_bytes", mem, ["median_qerror", "p95_qerror"]),
                                   ("infer_us_median", lat, ["median_qerror", "p95_qerror"])):
                r = ranked(g, col)
                if r.empty:
                    continue
                row2 = dict(base)
                row2[f"best_by_{col}"] = r["model"].iloc[0]
                row2[f"best_{col}"] = r[col].iloc[0]
                for c in ctx:
                    row2[f"best_{c}"] = r[c].iloc[0]
                if len(r) > 1:
                    row2[f"runner_up_by_{col}"] = r["model"].iloc[1]
                    row2[f"runner_up_{col}"] = r[col].iloc[1]
                    row2[f"margin_{col}_ratio"] = r[col].iloc[1] / r[col].iloc[0] if r[col].iloc[0] else None
                pos = r.reset_index(drop=True).index[r.reset_index(drop=True)["model"] == "Hybrid"]
                row2[f"hybrid_rank_by_{col}"] = int(pos[0]) + 1 if len(pos) else None
                if col == "infer_us_median":
                    row2["note"] = "latency measured with other experiments running; compare within the group only"
                sink.append(row2)
    return pd.DataFrame(acc), pd.DataFrame(mem), pd.DataFrame(lat)


# ----------------------------------------------------------------------------- ablation: original vs corrected

ABLATION_PAIRS = [("Hybrid", CORRECTED), ("Hybrid", PREDICT_ONLY),
                  ("Hybrid-Merge", "Hybrid-Corrected-Merge"), ("EquiWidth-Merge", "EquiWidth-Merge-Corrected")]
ABLATION_METRICS = ["median_qerror", "p95_qerror", "avg_qerror", "mae_rows", "size_bytes", "infer_us_median"]


def cdf_shares(cc):
    """cdf_choice as JSON dict ('{"Uniform": 3}') or drift string ('Uniform:3 Isotonic:2') -> (uniform, isotonic) share."""
    if cc is None or (isinstance(cc, float) and np.isnan(cc)):
        return np.nan, np.nan
    d = {}
    try:
        d = json.loads(cc) if str(cc).strip().startswith("{") else {k: int(v) for k, v in
                                                                    (t.rsplit(":", 1) for t in str(cc).split())}
    except Exception:
        return np.nan, np.nan
    tot = sum(d.values())
    return (d.get("Uniform", 0) / tot, d.get("Isotonic", 0) / tot) if tot else (np.nan, np.nan)


def ablation(static: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """
    Paired original vs labelled modification, same configuration:
      static: experiment_id x dataset x workload x bin_strategy;
      drift:  scenario x dataset x workload x drift_type x bin_strategy x drift_pct x stage
              (experiment_id ignored: a modified run may live in another tree than its original).
    delta = modified - original (q-error: < 0 means the modification is more accurate); ratio = modified / original.
    """
    out = []
    for kind, df, keys in (("static", static, ["experiment_id", "dataset", "workload", "bin_strategy"]),
                           ("drift", long, ["scenario", "dataset", "workload", "drift_type", "bin_strategy", "drift_pct", "stage"])):
        if df.empty or "model" not in df:
            continue
        df = df.copy()
        sh = df["cdf_choice"].map(cdf_shares) if "cdf_choice" in df else pd.Series([(np.nan, np.nan)] * len(df), index=df.index)
        df["uniform_share"] = [a for a, _ in sh]
        df["isotonic_share"] = [b for _, b in sh]
        cols = [c for c in ABLATION_METRICS + ["uniform_share", "isotonic_share"] if c in df.columns]
        for orig, mod in ABLATION_PAIRS:
            a = df[df["model"] == orig].drop_duplicates(keys).set_index(keys)[cols]
            b = df[df["model"] == mod].drop_duplicates(keys).set_index(keys)[cols]
            if a.empty or b.empty:
                continue
            j = a.join(b, lsuffix="_original", rsuffix="_modified", how="inner")
            if j.empty:
                continue
            for c in cols:
                j[f"{c}_delta"] = j[f"{c}_modified"] - j[f"{c}_original"]
                if c in ABLATION_METRICS:
                    j[f"{c}_ratio"] = j[f"{c}_modified"] / j[f"{c}_original"]
            j = j.reset_index()
            j.insert(0, "modified_model", mod)
            j.insert(0, "original_model", orig)
            j.insert(0, "kind", kind)
            out.append(j)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- checks

def checks(static: pd.DataFrame, long: pd.DataFrame, runs) -> pd.DataFrame:
    out = []

    def add(check, scope, ok, detail=""):
        out.append({"check": check, "scope": scope, "ok": bool(ok), "detail": detail})

    if not long.empty:
        bad = long[(long["stage"] == "RB") & (~long["rebuild_performed"].astype(bool))]
        add("RB rows only where rebuild_performed", "all_drift_long", bad.empty, f"{len(bad)} violating rows")
        hy = long[long["model"] == "Hybrid"]
        cp = hy[(hy["stage"] == "FT") & (hy["checkpoint_triggered"]) & (~hy["rebuild_performed"].astype(bool))]
        add("info: Hybrid steps with checkpoint but no rebuild (0 marked buckets)", "all_drift_long", True, f"{len(cp)} steps")
    for run_dir, ids, staged, df, scen in runs:
        if not staged or not str(run_dir).startswith(str(BASE)):
            continue
        bt = run_dir / "bucket_tables"
        names = sorted(p.stem for p in bt.glob("*.csv")) if bt.exists() else []
        expected = sorted(f"p{int(round(p)):03d}" if abs(p - round(p)) < 1e-9 else f"step_{int(s):03d}"
                          for s, p in zip(df["step"], df["drift_pct"]))
        add("every drift point has a Hybrid CDF table", str(run_dir.relative_to(ROOT)), names == expected,
            f"{len(names)} tables / {len(expected)} steps" + ("" if names == expected else f"; missing {sorted(set(expected) - set(names))[:5]}"))
        bad = []
        for p, (s, nv) in zip(sorted(bt.glob("*.csv")) if bt.exists() else [], zip(df["step"], df["n_values"])):
            t = pd.read_csv(p)
            st = int(t["step"].iloc[0]) if "step" in t and len(t) else None
            row = df[df["step"] == st]
            if len(row) and int(2 + t["bucket_values"].sum()) != int(row["n_values"].iloc[0]):
                bad.append(p.stem)
        add("drift bucket tables: 2 + sum(bucket_values) == n_values", str(run_dir.relative_to(ROOT)), not bad,
            ", ".join(bad[:5]))
    hy = static[static["model"].astype(str).str.startswith("Hybrid") & static["source"].astype(str).str.endswith(".json")]
    bad = []
    for _, r in hy.iterrows():
        bt = ROOT / Path(r["source"]).parent / f"bucket_table_{r['model']}.csv"
        if bt.exists() and pd.notna(r["n_values"]):
            t = pd.read_csv(bt)
            if int(2 + t["bucket_values"].sum()) != int(r["n_values"]):
                bad.append(r["source"])
    add("static Hybrid*: 2 + sum(bucket_values) == n_values", f"{len(hy)} static Hybrid / Hybrid-Corrected runs", not bad,
        "; ".join(bad[:5]))
    mm = BASE / "memory_analysis" / "fd_reduction_memory.csv"
    if mm.exists():
        m = pd.read_csv(mm)
        add("FD-reduction n_values reproduced from pickled models", "memory_analysis/fd_reduction_memory.csv",
            bool(m["n_values_matches_reported"].all()), f"{int((~m['n_values_matches_reported']).sum())} mismatches of {len(m)}")
    t = ROOT / "tests" / "test_drift_stages.py"
    add("train/eval leakage (adapt/eval split, FT/mark/RB see adapt queries only)", "tests/test_drift_stages.py",
        t.exists(), "covered by unit tests" if t.exists() else "test file missing")
    for (exp, ds, wl, bs), g in static.groupby(["experiment_id", "dataset", "workload", "bin_strategy"], dropna=False):
        if str(exp).startswith(("methodology_repro", "dsb_expanded_static")):
            missing = sorted(set(MODELS) - set(g["model"]))
            if missing:
                add("all six models present per static configuration", f"{exp}|{ds}|{wl}|{bs}", False, ",".join(missing))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- markdown

def _fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_md(static, long, comp, fdv, acc, mem, lat, chk, abl=None):
    L = ["# Сводка paper_experiments (CONSOLIDATED)", "",
         "Файл сгенерирован `python -m paper_experiments.consolidate` только из таблиц этой папки; "
         "выводы о методах — в отчётах частей (METHODOLOGY / DSB README / FD_REDUCTION_ANALYSIS / MEMORY_ANALYSIS).", "",
         "## Полнота", "",
         "MISSING — сценарий запускался (или ещё идёт), но результата этой модели нет; FAILED — задача завершилась с ошибкой "
         "(по jobs_status); NOT_RUN — не запускалось.", "",
         "| part | OK | PARTIAL | FAILED | MISSING | NOT_RUN |", "|---|---|---|---|---|---|"]
    if not comp.empty:
        c = comp.assign(st=comp["status"].str.extract(r"^(OK|PARTIAL|FAILED|MISSING|NOT_RUN)")[0].fillna("other"))
        for part, g in c.groupby("part"):
            v = g["st"].value_counts()
            L.append(f"| {part} | {v.get('OK', 0)} | {v.get('PARTIAL', 0)} | {v.get('FAILED', 0)} | {v.get('MISSING', 0)} | "
                     f"{v.get('NOT_RUN', 0)} |")
    L += ["", f"Статических конфигураций: {len(static)}; строк дрейфа (step × stage): {len(long)}.", ""]
    if not acc.empty:
        L += ["## Победители по критериям (число групп, где модель лучшая)", "",
              "Группа = эксперимент × датасет × workload × bin_strategy (FD-reduction: все модели при одном K, "
              "и отдельно — при одном бюджете памяти с соответствующим Hybrid; дрейф: среднее по шагам, финальная стадия шага). "
              "Точность оценивается по медиане И по P95 отдельно; память и латентность — отдельные таблицы.", "",
              "| kind | experiment | criterion | winners (model: groups) |", "|---|---|---|---|"]
        acc2 = acc.assign(exp=acc["experiment_id"].astype(str).str.split("/").str[0])
        for (kind, exp), g in acc2.groupby(["kind", "exp"]):
            for col in ("best_by_median_qerror", "best_by_p95_qerror"):
                if col in g:
                    v = g[col].value_counts()
                    L.append(f"| {kind} | {exp} | {col.replace('best_by_', '')} | " + ", ".join(f"{m}: {n}" for m, n in v.items()) + " |")
        for name, t, col in (("memory", mem, "best_by_size_bytes"), ("latency", lat, "best_by_infer_us_median")):
            if t.empty or col not in t:
                continue
            t2 = t.assign(exp=t["experiment_id"].astype(str).str.split("/").str[0])
            for (kind, exp), g in t2.groupby(["kind", "exp"]):
                v = g[col].value_counts()
                L.append(f"| {kind} | {exp} | {name} | " + ", ".join(f"{m}: {n}" for m, n in v.items()) + " |")
        L.append("")
        if "hybrid_rank_by_median_qerror" in acc:
            L += ["Место Hybrid (1 = лучший) — медиана по группам:", ""]
            for (kind, exp), g in acc2.groupby(["kind", "exp"]):
                rm = g["hybrid_rank_by_median_qerror"].dropna()
                rp = g["hybrid_rank_by_p95_qerror"].dropna() if "hybrid_rank_by_p95_qerror" in g else pd.Series(dtype=float)
                if len(rm):
                    L.append(f"- {kind} / {exp}: по медиане место {rm.median():.0f} (1-е в {int((rm == 1).sum())} из {len(rm)}), "
                             f"по P95 место {rp.median():.0f} (1-е в {int((rp == 1).sum())} из {len(rp)})")
            L.append("")
    if not fdv.empty:
        L += ["## FD против 100 бакетов (отношение FD / 100; < 1 — FD лучше / меньше)", "",
              "| kind | model | pairs | median q-err ratio (median) | P95 ratio (median) | size ratio (median) | latency ratio (median) |",
              "|---|---|---|---|---|---|---|"]
        for (kind, m), g in fdv.groupby(["kind", "model"]):
            L.append(f"| {kind} | {m} | {len(g)} | {_fmt(g['median_qerror_ratio_fd_over_100'].median())} | "
                     f"{_fmt(g['p95_qerror_ratio_fd_over_100'].median())} | {_fmt(g['size_bytes_ratio_fd_over_100'].median(), 2)} | "
                     f"{_fmt(g['infer_us_median_ratio_fd_over_100'].median(), 2)} |")
        L.append("")
    if abl is not None and not abl.empty:
        L += ["## Original против Hybrid-Corrected (модификация; ablation_original_vs_corrected.csv)", "",
              "Пары одной конфигурации; Δ = модификация − оригинал (по q-error < 0 — модификация точнее). "
              "Hybrid-Corrected — помеченная МОДИФИКАЦИЯ канонического Hybrid FD CDF (models/hybrid_corrected.py).", "",
              "| kind | pair | pairs | median ratio (median) | P95 ratio (median) | modified better by median, % pairs | "
              "modified better by P95, % pairs | size ratio (median) | Δ Uniform share (median) | Δ Isotonic share (median) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for (kind, o, m), g in abl.groupby(["kind", "original_model", "modified_model"]):
            L.append(f"| {kind} | {o} → {m} | {len(g)} | {_fmt(g['median_qerror_ratio'].median())} | "
                     f"{_fmt(g['p95_qerror_ratio'].median())} | {100 * (g['median_qerror_delta'] < 0).mean():.0f} | "
                     f"{100 * (g['p95_qerror_delta'] < 0).mean():.0f} | "
                     f"{_fmt(g['size_bytes_ratio'].median(), 2) if 'size_bytes_ratio' in g else '—'} | "
                     f"{_fmt(g['uniform_share_delta'].median()) if 'uniform_share_delta' in g else '—'} | "
                     f"{_fmt(g['isotonic_share_delta'].median()) if 'isotonic_share_delta' in g else '—'} |")
        L.append("")
    if not chk.empty:
        L += ["## Проверки согласованности", "", "| check | scope | ok | detail |", "|---|---|---|---|"]
        g = chk.groupby(["check", "ok"]).agg(n=("scope", "size"), detail=("detail", "first"), scope=("scope", "first")).reset_index()
        for _, r in g.iterrows():
            L.append(f"| {r['check']} | {r['scope'] if r['n'] == 1 else str(r['n']) + ' scopes'} | {'OK' if r['ok'] else 'FAIL'} | {r['detail']} |")
        L.append("")
    L += ["Латентность во всех таблицах измерена, пока параллельно шли другие эксперименты (12 ядер, до ~12 процессов); "
          "сравнивайте её только внутри одной группы. Для FD-reduction есть отдельный последовательный проход "
          "(`latency_us_quiet`).", ""]
    (OUT / "CONSOLIDATED.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    static = static_rows()
    runs = drift_runs()
    long = drift_long(runs)
    comp = completeness(static, runs)
    dagg = drift_aggregate(long)
    fdv = fd_vs_fixed(static, dagg)
    acc, mem, lat = best_tables(static, dagg)
    chk = checks(static, long, runs)
    abl = ablation(static, long)
    abl.to_csv(OUT / "ablation_original_vs_corrected.csv", index=False)
    static.to_csv(OUT / "all_static.csv", index=False)
    long.to_csv(OUT / "all_drift_long.csv", index=False)
    dagg.to_csv(OUT / "drift_run_summary.csv", index=False)
    comp.to_csv(OUT / "completeness.csv", index=False)
    fdv.to_csv(OUT / "fd_vs_fixed100.csv", index=False)
    acc.to_csv(OUT / "best_by_accuracy.csv", index=False)
    mem.to_csv(OUT / "best_by_memory.csv", index=False)
    lat.to_csv(OUT / "best_by_latency.csv", index=False)
    chk.to_csv(OUT / "checks.csv", index=False)
    write_md(static, long, comp, fdv, acc, mem, lat, chk, abl)
    from paper_experiments.excel_export import export_dir
    res = export_dir(OUT, OUT / "consolidated.xlsx", max_rows=300_000)
    from paper_experiments.provenance import write_manifest
    write_manifest(OUT / "manifest.json", part="consolidated", static_rows=len(static), drift_long_rows=len(long),
                   drift_runs=len(runs), completeness=comp["status"].str.extract(r"^(\w+)")[0].value_counts().to_dict())
    counts = comp["status"].str.extract(r"^(\w+)")[0].value_counts().to_dict()
    print(f"static {len(static)} rows, drift long {len(long)} rows ({len(runs)} runs), completeness {counts}, xlsx {res}")


if __name__ == "__main__":
    main()

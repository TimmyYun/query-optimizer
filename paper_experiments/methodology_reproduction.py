"""
Static reproduction of the thesis (Tables 4.8-4.12, one-by-one inference) with the current code,
plus the drift instrumentation check. Re-runnable on partial results (missing configs -> NOT_RUN).

Inputs (results/paper_experiments/methodology/reproduction/):
  static/<fd|fixed100>/synth_<scenario>/<dist>/summary_<Model>.json             registry workloads (W <= 999)
  static_w100_thesis/<fd|fixed100>/synth_<scenario>_w100/<dist>/summary_*.json  thesis-width workloads (W <= 100)
  thesis_tables_4_8_to_4_13_one_by_one.csv                                      numbers extracted from the thesis
  drift_check/drift_5_ds_zipf_wl_normal/gradual_normal_to_zipf_Hybrid/drift_steps.csv
Outputs (same folder):
  static_long.csv            unified schema (paper_experiments.schema.ID_COLUMNS, stage STATIC), every model
  reproduction_vs_thesis.csv EquiWidth / Hybrid next to the thesis numbers
  drift_check.csv            column-by-column comparison with results/exp1_100_buckets
  REPRODUCTION.md

    poetry run python -m paper_experiments.methodology_reproduction
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_experiments.provenance import SEED, model_version
from paper_experiments.schema import ID_COLUMNS, fmt_bytes

BASE = Path("results/paper_experiments/methodology/reproduction")
MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid", "Hybrid-Corrected"]
EXTRA_SOURCES = {"Hybrid-Corrected": ["models/hybrid_corrected.py", "run_hybrid_corrected.py"]}
DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
# experiment folder -> (dataset kind, workload id, thesis table, thesis workload label)
SCENARIOS = {
    ("static", "synth_simple_wide"): ("simple", "wide", "4.8"),
    ("static", "synth_simple_narrow"): ("simple", "narrow_w999", "4.9"),
    ("static", "synth_hard_wide"): ("hard", "wide", "4.10"),
    ("static", "synth_hard_narrow"): ("hard", "narrow_w999", "4.11"),
    ("static", "synth_hard_narrow_data_driven"): ("hard", "driven_w999", "4.12"),
    ("static_w100_thesis", "synth_simple_narrow_w100"): ("simple", "narrow_w100_thesis", "4.9"),
    ("static_w100_thesis", "synth_hard_narrow_w100"): ("hard", "narrow_w100_thesis", "4.11"),
    ("static_w100_thesis", "synth_hard_narrow_data_driven_w100"): ("hard", "driven_w100_thesis", "4.12"),
}
THESIS_MODEL = {"EquiWidth": "Equi-Width", "Hybrid": "Hybrid"}
TIMING_COLS = ("time_s", "infer_us", "action_time")


def load_static():
    rows = []
    for (root, exp), (kind, wl, table) in SCENARIOS.items():
        for bins in ("fd", "fixed100"):
            for d in DISTS:
                for m in MODELS:
                    p = BASE / root / bins / exp / d / f"summary_{m}.json"
                    base = {"experiment_id": f"repro/{root}/{bins}/{exp}", "dataset": f"{kind}/{d}", "table": None,
                            "column": None, "workload": wl, "drift_type": None, "drift_pct": 0.0, "bin_strategy": bins,
                            "model": m, "model_version": model_version(m, EXTRA_SOURCES.get(m, ())), "seed": SEED, "stage": "STATIC",
                            "thesis_table": table}
                    if not p.exists():
                        rows.append({**base, "bucket_count": None, "status": "NOT_RUN"})
                        continue
                    mt = json.loads(p.read_text())["metrics"]
                    rows.append({**base, "bucket_count": mt.get("n_buckets"), "status": "RUN",
                                 "n_queries": mt.get("n_queries"),
                                 "median_qerror": mt.get("median_q_error"), "p95_qerror": mt.get("p95_q_error"),
                                 "avg_qerror": mt.get("avg_q_error"), "mae_rows": mt.get("mae_rows"),
                                 "mae_selectivity": mt.get("mae_selectivity"),
                                 "build_s": mt.get("build_time"), "update_s": mt.get("update_time_total"),
                                 "infer_us_median": None if mt.get("median_infer_time") is None else mt["median_infer_time"] * 1e6,
                                 "infer_us_p95": None if mt.get("p95_infer_time") is None else mt["p95_infer_time"] * 1e6,
                                 "n_values": mt.get("n_values"), "size_bytes": mt.get("size_bytes"),
                                 "cdf_choice": json.dumps(mt.get("cdf_choice")) if mt.get("cdf_choice") else None})
    df = pd.DataFrame(rows)
    return df[ID_COLUMNS + [c for c in df.columns if c not in ID_COLUMNS]]


def with_thesis(df):
    th = pd.read_csv(BASE / "thesis_tables_4_8_to_4_13_one_by_one.csv", dtype={"thesis_table": str})
    th = th.rename(columns={c: f"thesis_{c}" for c in ("median_qerror", "p95_qerror", "avg_qerror", "train_s", "infer_us_median")})
    th["model"] = th["model"].map({v: k for k, v in THESIS_MODEL.items()})
    th = th.dropna(subset=["model"])
    th["dist"] = th["dataset"]
    d = df[df["model"].isin(THESIS_MODEL)].copy()
    d["dist"] = d["dataset"].str.split("/").str[1]
    keep = ["thesis_table", "dist", "model", "thesis_median_qerror", "thesis_p95_qerror", "thesis_avg_qerror",
            "thesis_train_s", "thesis_infer_us_median"]
    return d.merge(th[keep], on=["thesis_table", "dist", "model"], how="left")


def drift_check():
    a = Path("results/exp1_100_buckets/drift/drift_5_ds_zipf_wl_normal/gradual_normal_to_zipf_Hybrid/drift_steps.csv")
    b = BASE / "drift_check/drift_5_ds_zipf_wl_normal/gradual_normal_to_zipf_Hybrid/drift_steps.csv"
    if not (a.exists() and b.exists()):
        return None, "NOT_RUN"
    A, B = pd.read_csv(a), pd.read_csv(b)
    out = []
    for c in A.columns:
        if c not in B.columns:
            out.append({"column": c, "kind": "missing_in_new", "identical": None, "max_abs_diff": None})
            continue
        timing = any(t in c for t in TIMING_COLS)
        if A[c].dtype == object or B[c].dtype == object:
            same = A[c].fillna("").astype(str).equals(B[c].fillna("").astype(str))
            diff = None
        else:
            same = np.array_equal(A[c].to_numpy(), B[c].to_numpy(), equal_nan=True)
            diff = float(np.nanmax(np.abs(A[c].to_numpy(dtype=float) - B[c].to_numpy(dtype=float)))) if len(A) == len(B) else None
        out.append({"column": c, "kind": "timing" if timing else "result", "identical": same, "max_abs_diff": diff})
    res = pd.DataFrame(out)
    bad = res[(res["kind"] == "result") & (res["identical"] != True)]  # noqa: E712
    return res, ("IDENTICAL" if bad.empty and len(A) == len(B) else f"DIFFERS in {bad['column'].tolist()}")


def uniform_share(cdf_json):
    """Share of all buckets whose local CDF is Uniform (Empty buckets count in the denominator)."""
    if not isinstance(cdf_json, str):
        return None
    c = json.loads(cdf_json)
    total = sum(c.values())
    return c.get("Uniform", 0) / total if total else None


def corrected_table(df):
    run = df[df["status"] == "RUN"]
    o = run[run["model"] == "Hybrid"].set_index(["experiment_id", "dataset", "workload", "bin_strategy"])
    c = run[run["model"] == "Hybrid-Corrected"].set_index(["experiment_id", "dataset", "workload", "bin_strategy"])
    both = o.join(c, lsuffix="_orig", rsuffix="_corr", how="inner")
    if both.empty:
        return None, ["NOT_RUN (no Hybrid-Corrected summaries yet)"]
    both = both.reset_index()
    for k in ("orig", "corr"):
        both[f"uniform_share_{k}"] = both[f"cdf_choice_{k}"].map(uniform_share)
    cols = ["median_qerror", "p95_qerror", "avg_qerror", "mae_rows", "size_bytes", "infer_us_median"]
    lines = ["| bins | workload | dataset | median O / C | P95 O / C | avg O / C | MAE rows O / C | size O / C | µs O / C | Uniform share O / C |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in both.sort_values(["bin_strategy", "workload", "dataset"]).iterrows():
        lines.append(f"| {r['bin_strategy']} | {r['workload']} | {r['dataset']} | {fmt(r['median_qerror_orig'])} / {fmt(r['median_qerror_corr'])} | "
                     f"{fmt(r['p95_qerror_orig'])} / {fmt(r['p95_qerror_corr'])} | {fmt(r['avg_qerror_orig'])} / {fmt(r['avg_qerror_corr'])} | "
                     f"{fmt(r['mae_rows_orig'], 1)} / {fmt(r['mae_rows_corr'], 1)} | {fmt_bytes(r['size_bytes_orig'])} / {fmt_bytes(r['size_bytes_corr'])} | "
                     f"{fmt(r['infer_us_median_orig'], 2)} / {fmt(r['infer_us_median_corr'], 2)} | "
                     f"{fmt(r['uniform_share_orig'], 2)} / {fmt(r['uniform_share_corr'], 2)} |")
    keep = ["experiment_id", "dataset", "workload", "bin_strategy"] + [f"{c}_{k}" for c in cols + ["uniform_share"] for k in ("orig", "corr")]
    return both[keep], lines


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(d, workload_ids, bins="fd"):
    lines = ["| table | dataset | workload | k | EW median ours / thesis | EW P95 ours / thesis | Hybrid median ours / thesis | Hybrid P95 ours / thesis | Hybrid size | Hybrid µs ours / thesis |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    sel = d[(d["bin_strategy"] == bins) & (d["workload"].isin(workload_ids))]
    for (tab, ds, wl), g in sel.groupby(["thesis_table", "dataset", "workload"], sort=False):
        ew = g[g["model"] == "EquiWidth"].iloc[0]
        hy = g[g["model"] == "Hybrid"].iloc[0]
        lines.append(f"| {tab} | {ds} | {wl} | {fmt(hy.get('bucket_count'), 0)} | {fmt(ew.get('median_qerror'))} / {fmt(ew.get('thesis_median_qerror'))} | "
                     f"{fmt(ew.get('p95_qerror'))} / {fmt(ew.get('thesis_p95_qerror'))} | "
                     f"{fmt(hy.get('median_qerror'))} / {fmt(hy.get('thesis_median_qerror'))} | "
                     f"{fmt(hy.get('p95_qerror'))} / {fmt(hy.get('thesis_p95_qerror'))} | {fmt_bytes(hy.get('size_bytes'))} | "
                     f"{fmt(hy.get('infer_us_median'), 2)} / {fmt(hy.get('thesis_infer_us_median'), 2)} |")
    return lines


def all_models_table(df, bins):
    run = df[(df["status"] == "RUN") & (df["bin_strategy"] == bins)]
    if run.empty:
        return ["NOT_RUN"]
    g = run.groupby(["workload", "model"]).agg(
        datasets=("dataset", "nunique"), median=("median_qerror", "mean"), p95=("p95_qerror", "mean"),
        size=("size_bytes", "mean"), us=("infer_us_median", "mean"), build=("build_s", "mean")).reset_index()
    lines = ["| workload | model | datasets | mean of median q | mean of P95 q | mean size | mean µs/query (median) | mean build s |",
             "|---|---|---|---|---|---|---|---|"]
    order = {m: i for i, m in enumerate(MODELS)}
    for _, r in g.sort_values(["workload", "model"], key=lambda s: s.map(order) if s.name == "model" else s).iterrows():
        lines.append(f"| {r['workload']} | {r['model']} | {r['datasets']} | {r['median']:.3f} | {r['p95']:.3f} | "
                     f"{fmt_bytes(r['size'])} | {r['us']:.2f} | {r['build']:.2f} |")
    return lines


def main():
    df = load_static()
    df.to_csv(BASE / "static_long.csv", index=False)
    cmp_ = with_thesis(df)
    cmp_.to_csv(BASE / "reproduction_vs_thesis.csv", index=False)
    dc, dc_status = drift_check()
    if dc is not None:
        dc.to_csv(BASE / "drift_check.csv", index=False)
    n_run, n_all = int((df["status"] == "RUN").sum()), len(df)
    missing = df[df["status"] == "NOT_RUN"].groupby(["experiment_id"]).size()
    L = ["# Репродукция статических таблиц тезиса (генерируется `paper_experiments/methodology_reproduction.py`)", "",
         f"Конфигураций: {n_run}/{n_all} RUN. Тезисные числа — one-by-one Tables 4.8–4.12 "
         "(`thesis_tables_4_8_to_4_13_one_by_one.csv`); латентность сравнима только с ними (batch Tables 4.2–4.6 у нас нет).",
         "Латентность у нас измерена при параллельной нагрузке других агентов (load average 6–12 на 12 ядрах).", ""]
    if not missing.empty:
        L += ["NOT_RUN по экспериментам:", ""] + [f"- `{k}`: {v} конфигураций" for k, v in missing.items()] + [""]
    for bins in ("fd", "fixed100"):
        L += [f"## EquiWidth и Hybrid против тезиса — {bins}", "",
              "W ≤ 999 (зарегистрированные workload'ы репозитория):", ""]
        L += md_table(cmp_, ["wide", "narrow_w999", "driven_w999"], bins) + [""]
        L += ["W ≤ 100 (тезисная ширина):", ""]
        L += md_table(cmp_, ["narrow_w100_thesis", "driven_w100_thesis"], bins) + [""]
        L += [f"## Все шесть моделей — {bins} (среднее по 5 распределениям)", ""] + all_models_table(df, bins) + [""]
    ctab, clines = corrected_table(df)
    if ctab is not None:
        ctab.to_csv(BASE / "original_vs_corrected.csv", index=False)
    L += ["## Hybrid (Original, канонический) vs Hybrid-Corrected (помеченная модификация, models/hybrid_corrected.py)", "",
          "O = Original, C = Corrected; Uniform share — доля всех бакетов с локальной CDF Uniform.", ""] + clines + [""]
    L += ["## Проверка инструментирования на полном масштабе (дрейф)", "",
          f"`drift_5_ds_zipf_wl_normal`, Hybrid, 100 бакетов, против `results/exp1_100_buckets`: **{dc_status}**"]
    if dc is not None:
        res = dc[dc["kind"] == "result"]
        L += [f"(сравнено {len(res)} столбцов результатов; столбцы времени ожидаемо отличаются: "
              f"{', '.join(dc[(dc['kind'] == 'timing') & (dc['identical'] != True)]['column'])})"]  # noqa: E712
    (BASE / "REPRODUCTION_TABLES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()

"""
DSB expansion, part 5: tables, workbooks and figures from whatever results exist (re-runnable).

    poetry run python -m paper_experiments.dsb_export [--no-plots] [--no-cdf-workbooks]

Reads   results/paper_experiments/dsb_expanded/{static,displacement,additive_200}/** and jobs_status.csv
Writes  results/paper_experiments/dsb_expanded/
  static/static_results.csv (+ .xlsx)        one row per bin x workload x column x model (stage STATIC)
  static/static_cdf_selection__<bin>.xlsx    Hybrid per-bucket CDF table per column + CDF/tier summary
  <scenario>/drift_long.csv                  SHOCK / FT / RB rows (schema.drift_steps_to_long)
  <scenario>/drift_summary__<scenario>.xlsx  long table, pivots, Hybrid stage/rebuild stats
  <scenario>/cdf_workbooks/cdf_selection__<column>__<scenario>__<bin>.xlsx   one sheet per drift point
  <scenario>/figures/<bin>/<column>.png      median and P95 q-error vs drift %
  completeness.csv                           expected job x status (DONE / FAILED / NOT_RUN)
Units: *_bytes bytes (+ *_kib), *_s seconds, *_us microseconds, q-error dimensionless.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from paper_experiments.dsb_columns import ROOT, SELECTED_ALL
from paper_experiments.dsb_run import RES, SCENARIOS, STATUS, TARGET_LABEL, drift_jobs, static_jobs
from paper_experiments.schema import ID_COLUMNS, drift_steps_to_long
from registry import DEFAULT_MODELS

MODEL_ORDER = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid",
               "Hybrid-Corrected", "Hybrid-Corrected-PredictOnly"]
# Hybrid-Corrected*: labelled MODIFICATIONS of the canonical Hybrid (inclusive-interval fix), never baselines
HYBRIDS = ["Hybrid", "Hybrid-Corrected", "Hybrid-Corrected-PredictOnly"]


def is_hybrid(model: str) -> bool:
    return model in HYBRIDS


def _suffix(model: str) -> str:
    return "" if model == "Hybrid" else f"__{model}"
SEED = 42


def _mv(manifest_path: Path, model: str):
    try:
        return json.loads(manifest_path.read_text())["config"]["model_version"]
    except Exception:
        return None


def _ids(**kw):
    split = kw["dataset"].split(".", 1)
    base = {"experiment_id": kw.pop("experiment_id"), "dataset": kw["dataset"], "table": split[0], "column": split[1],
            "seed": SEED}
    base.update(kw)
    return base


# ---------------------------------------------------------------- static

def static_table() -> pd.DataFrame:
    rows = []
    for path in sorted((RES / "static").glob("*/*/*/summary_*.json")):
        b, wl, col = path.parts[-4], path.parts[-3], path.parts[-2]
        s = json.loads(path.read_text())
        m, model = s["metrics"], s["model"]
        bt = path.parent / f"bucket_table_{model}.csv"
        r = _ids(experiment_id=f"dsb_expanded/static/{b}/{wl}", dataset=col, workload=wl, drift_type="none",
                 drift_pct=0.0, bin_strategy=b, bucket_count=m.get("n_buckets"), model=model,
                 model_version=_mv(path.parent / f"manifest_{model}.json", model), stage="STATIC")
        r.update({
            "n_queries": m.get("n_queries"), "median_qerror": m.get("median_q_error"),
            "p25_qerror": m.get("p25_q_error"), "p75_qerror": m.get("p75_q_error"),
            "p95_qerror": m.get("p95_q_error"), "avg_qerror": m.get("avg_q_error"),
            "mae_rows": m.get("mae_rows"), "mae_selectivity": m.get("mae_selectivity"),
            "build_time_s": m.get("build_time"), "feedback_update_time_s": m.get("update_time_total"),
            "infer_us_median": None if m.get("median_infer_time") is None else m["median_infer_time"] * 1e6,
            "infer_us_mean": None if m.get("avg_infer_time") is None else m["avg_infer_time"] * 1e6,
            "infer_us_p95": None if m.get("p95_infer_time") is None else m["p95_infer_time"] * 1e6,
            "infer_total_s": m.get("infer_time"),
            "n_buckets": m.get("n_buckets"), "n_values": m.get("n_values"), "size_bytes": m.get("size_bytes"),
            "size_kib": None if m.get("size_bytes") is None else m["size_bytes"] / 1024,
            "cdf_choice": json.dumps(m["cdf_choice"]) if "cdf_choice" in m else None,
        })
        if bt.exists():  # Hybrid: equal-width grid shared with EquiWidth at the same bucket count
            t = pd.read_csv(bt)
            r.update({"empty_buckets": int((t["count"] <= 0).sum()), "bucket_width_min": int(t["width"].min()),
                      "bucket_width_median": float(t["width"].median()), "bucket_width_max": int(t["width"].max()),
                      "cdf_tier_counts": json.dumps(t["tier"].fillna(-1).astype(int).value_counts().sort_index()
                                                    .rename(index={-1: "Empty"}).to_dict())})
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_o"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values(["bin_strategy", "workload", "dataset", "_o"]).drop(columns="_o")


def static_cdf_workbooks():
    for bdir in sorted((RES / "static").glob("*")):
        if bdir.is_dir():
            for model in HYBRIDS:
                _static_cdf_workbook(bdir, model)


def _static_cdf_workbook(bdir: Path, model: str):
    if True:
        tables = {}
        for col in SELECTED_ALL:
            for wl in ("dd_narrow", "uniform_narrow", "uniform_wide", "official"):  # same model for every workload
                p = bdir / wl / col / f"bucket_table_{model}.csv"
                if p.exists():
                    tables[col] = pd.read_csv(p)
                    break
        if not tables:
            return
        summ = []
        for col, t in tables.items():
            g = t.assign(tier=t["tier"].fillna(-1)).groupby(["cdf_model", "tier"])
            for (cdf, tier), d in g:
                summ.append({"column": col, "cdf_model": cdf, "tier": None if tier == -1 else int(tier),
                             "buckets": len(d), "pct_buckets": 100 * len(d) / len(t),
                             "bucket_bytes_total": int(d["bucket_bytes"].sum()),
                             "rows_covered_pct": 100 * d["count"].clip(lower=0).sum() / max(t["count"].clip(lower=0).sum(), 1)})
        with pd.ExcelWriter(bdir / f"static_cdf_selection__{bdir.name}{_suffix(model)}.xlsx") as xw:
            pd.DataFrame(summ).to_excel(xw, sheet_name="summary", index=False)
            for col, t in tables.items():
                t.to_excel(xw, sheet_name=col.split(".")[1][:31], index=False)


# ---------------------------------------------------------------- drift

def drift_runs():
    for sc in SCENARIOS:
        for path in sorted((RES / sc).glob(f"*/*/gradual_*_to_{TARGET_LABEL}_*/drift_steps.csv")):
            b, col, run = path.parts[-4], path.parts[-3], path.parent.name
            model = run[len(f"gradual_{col}_to_{TARGET_LABEL}_"):]
            yield sc, b, col, model, path


def drift_long() -> dict:
    out = {}
    for sc, b, col, model, path in drift_runs():
        df = pd.read_csv(path)
        ids = _ids(experiment_id=f"dsb_expanded/{sc}/{b}", dataset=col, workload="per_step_dd_narrow_1000000",
                   drift_type=SCENARIOS[sc]["drift_type"], drift_pct=None, bin_strategy=b, bucket_count=None,
                   model=model, model_version=_mv(path.parent / f"manifest_{model}.json", model), stage=None)
        long = drift_steps_to_long(df, ids, staged=is_hybrid(model))
        out.setdefault(sc, []).append(long)
    return {sc: pd.concat(v, ignore_index=True) for sc, v in out.items()}


def hybrid_stage_table(long: pd.DataFrame) -> pd.DataFrame:
    """Hybrid per column x bin: mean SHOCK / FT / RB (RB only where performed) and rebuild counts."""
    h = long[long["model"].isin(HYBRIDS)]
    rows = []
    for (model, b, col), d in h.groupby(["model", "bin_strategy", "dataset"]):
        steps = d[d["stage"] != "STATIC"]
        so = ("SHOCK", "FT", "RB")  # stage order: pivot sorts columns alphabetically
        piv = steps.pivot_table(index="step", columns="stage", values="median_qerror")
        piv = piv[[c for c in so if c in piv.columns]]
        p95 = steps.pivot_table(index="step", columns="stage", values="p95_qerror")
        p95 = p95[[c for c in so if c in p95.columns]]
        rb_steps = steps[steps["stage"] == "RB"]["step"]
        first = steps.drop_duplicates("step")
        rows.append({
            "model": model, "bin_strategy": b, "dataset": col, "steps": int(first["step"].nunique()),
            "checkpoint_triggered_steps": int(first["checkpoint_triggered"].sum()),
            "rebuild_performed_steps": int(first["rebuild_performed"].sum()),
            "rebuilt_buckets_total": float(first["rebuilt_buckets"].fillna(0).sum()),
            "ft_buckets_mean": float(first["ft_buckets"].astype(float).mean()),
            "shock_median_mean": float(piv.get("SHOCK", pd.Series(dtype=float)).mean()),
            "ft_median_mean": float(piv.get("FT", pd.Series(dtype=float)).mean()),
            "final_median_mean": float(piv.ffill(axis=1).iloc[:, -1].mean()) if not piv.empty else None,
            "final_p95_mean": float(p95.ffill(axis=1).iloc[:, -1].mean()) if not p95.empty else None,
            "shock_p95_mean": float(p95.get("SHOCK", pd.Series(dtype=float)).mean()),
            "ft_p95_mean": float(p95.get("FT", pd.Series(dtype=float)).mean()),
            # Only steps where RB actually ran: FT vs RB on the same steps (no credit to RB elsewhere)
            "ft_median_on_rb_steps": float(piv.loc[rb_steps, "FT"].mean()) if len(rb_steps) else None,
            "rb_median_on_rb_steps": float(piv.loc[rb_steps, "RB"].mean()) if len(rb_steps) else None,
            "ft_time_s_total": float(steps[steps["stage"] == "FT"]["stage_time_s"].sum()),
            "rb_time_s_total": float(steps[steps["stage"] == "RB"]["stage_time_s"].sum()),
            "size_bytes_final": float(d.sort_values("step")["size_bytes"].dropna().iloc[-1]),
            "final_mae_rows_mean": float(steps.assign(_o=steps["stage"].map({"SHOCK": 1, "FT": 2, "RB": 3}))
                                         .sort_values("_o").groupby("step").tail(1)["mae_rows"].mean()),
            "final_infer_us_median": float(steps[steps["stage"].isin(["FT", "RB"])]["infer_us_median"].median()),
        })
    return pd.DataFrame(rows)


def final_by_model(long: pd.DataFrame) -> pd.DataFrame:
    """Per bin x column x model: mean over steps >= 1 of the final state (last stage of each step)."""
    order = {"STATIC": 0, "SHOCK": 1, "FT": 2, "RB": 3}
    d = long[long["stage"] != "STATIC"].assign(_o=lambda x: x["stage"].map(order))
    fin = d.sort_values("_o").groupby(["bin_strategy", "dataset", "model", "step"]).tail(1)
    agg = fin.groupby(["bin_strategy", "dataset", "model"]).agg(
        final_median_mean=("median_qerror", "mean"), final_p95_mean=("p95_qerror", "mean"),
        final_median_max=("median_qerror", "max"), final_p95_max=("p95_qerror", "max"),
        final_infer_us_median=("infer_us_median", "median"),
        size_bytes_mean=("size_bytes", "mean"), action_time_s_total=("stage_time_s", "sum")).reset_index()
    shock = d[d["stage"] == "SHOCK"].groupby(["bin_strategy", "dataset", "model"]).agg(
        shock_median_mean=("median_qerror", "mean"), shock_p95_mean=("p95_qerror", "mean")).reset_index()
    return agg.merge(shock, on=["bin_strategy", "dataset", "model"], how="left")


def cdf_workbooks(sc: str, long: pd.DataFrame):
    out_dir = RES / sc / "cdf_workbooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    n_points = SCENARIOS[sc]["steps"] - 1  # "steps" = number of drift points (21 / 41)
    for scx, b, col, model, path in drift_runs():
        if scx != sc or model not in ("Hybrid", "Hybrid-Corrected"):
            continue
        tdir = path.parent / "bucket_tables"
        steps = pd.read_csv(path)
        index, sheets = [], {}
        for _, r in steps.iterrows():
            pct = float(r["drift_pct"])
            name = f"p{int(round(pct)):03d}"
            f = tdir / f"{name}.csv"
            if not f.exists():
                index.append({"sheet": name, "step": int(r["step"]), "drift_pct": pct, "status": "MISSING"})
                continue
            t = pd.read_csv(f)
            sheets[name] = t
            cnt = t["cdf_model"].value_counts().to_dict()
            index.append({"sheet": name, "step": int(r["step"]), "drift_pct": pct, "status": "OK",
                          "action": r["action"], "checkpoint_triggered": bool(r.get("checkpoint_triggered", False)),
                          "rebuild_performed": bool(r.get("rebuild_performed", False)),
                          "ft_buckets": r.get("ft_buckets"), "marked_buckets": r.get("marked_buckets"),
                          "rebuilt_buckets": r.get("rebuilt_buckets"), "n_rows": r.get("n_rows"),
                          "n_values": r["n_values"], "size_bytes": r["size_bytes"],
                          **{f"buckets_{k}": v for k, v in sorted(cnt.items())}})
        p = out_dir / f"cdf_selection__{col}__{sc}__{b}{_suffix(model)}.xlsx"
        with pd.ExcelWriter(p) as xw:
            pd.DataFrame(index).to_excel(xw, sheet_name="index", index=False)
            for name, t in sheets.items():
                t.to_excel(xw, sheet_name=name, index=False)
        report.append({"workbook": str(p.relative_to(ROOT)), "model": model, "column": col, "bin": b,
                       "expected_points": n_points + 1,
                       "sheets": len(sheets), "complete": len(sheets) == n_points + 1 == len(steps)})
    return report


def plots(sc: str, long: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"EquiWidth": "#4C72B0", "EquiDepth": "#55A868", "TKHist": "#8172B2", "EntropyHist-MEA": "#CCB974",
              "STHist": "#64B5CD", "Hybrid": "#C44E52", "Hybrid-Corrected": "#8C2D04"}
    order = {"STATIC": 0, "SHOCK": 1, "FT": 2, "RB": 3}
    for (b, col), d in long.groupby(["bin_strategy", "dataset"]):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, metric, title in ((axes[0], "median_qerror", "Median q-error"), (axes[1], "p95_qerror", "P95 q-error")):
            for model in [m for m in MODEL_ORDER if m in set(d["model"])]:
                dm = d[d["model"] == model].assign(_o=lambda x: x["stage"].map(order))
                fin = dm.sort_values("_o").groupby("step").tail(1).sort_values("step")
                if model in ("Hybrid", "Hybrid-Corrected"):
                    shock = dm[dm["stage"].isin(["STATIC", "SHOCK"])].sort_values("step")
                    ft = dm[dm["stage"].isin(["STATIC", "FT"])].sort_values("step")
                    ax.plot(shock["drift_pct"], shock[metric], ls=":", color=colors[model], lw=1.4, label=f"{model} SHOCK")
                    ax.plot(ft["drift_pct"], ft[metric], ls="--", color=colors[model], lw=1.4, label=f"{model} FT")
                    ax.plot(fin["drift_pct"], fin[metric], color=colors[model], lw=2.2,
                            label=f"{model} final (RB if performed)")
                    rb = dm[dm["stage"] == "RB"]
                    ax.scatter(rb["drift_pct"], rb[metric], marker="o", s=34, color=colors[model], zorder=5,
                               label=f"{model} RB performed")
                    if model == "Hybrid":
                        ck = dm[(dm["stage"] == "SHOCK") & dm["checkpoint_triggered"]]["drift_pct"]
                        for x in ck:
                            ax.axvline(x, color="#999999", lw=0.6, ls=":", zorder=0)
                else:
                    ax.plot(fin["drift_pct"], fin[metric], color=colors.get(model, "k"), lw=1.3, label=f"{model} (final)")
            ax.set_xlabel("drift % of initial table")
            ax.set_ylabel(f"{title} (log)")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.grid(alpha=0.3, which="both")
        axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle(f"{col} — {sc} ({b}); dotted verticals: checkpoint triggered", fontsize=10)
        fig.tight_layout()
        out = RES / sc / "figures" / b
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{col}.png", dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------- completeness

def completeness() -> pd.DataFrame:
    bins, cols = ["fd", "fixed100"], list(SELECTED_ALL)
    expected = static_jobs(bins, cols, DEFAULT_MODELS) + drift_jobs(bins, cols, DEFAULT_MODELS, list(SCENARIOS)) + \
        static_jobs(bins, cols, ["Hybrid-Corrected"]) + \
        static_jobs(bins, cols, ["Hybrid-Corrected-PredictOnly"], ["dd_narrow"]) + \
        drift_jobs(bins, cols, ["Hybrid-Corrected"], list(SCENARIOS))
    st = pd.read_csv(STATUS) if STATUS.exists() else pd.DataFrame(columns=["job_id", "rc", "outputs_ok"])
    last = st.groupby("job_id").tail(1).set_index("job_id")
    rows = []
    for j in expected:
        if j["job_id"] in last.index:
            r = last.loc[j["job_id"]]
            status = "DONE" if (int(r["rc"]) == 0 and str(r["outputs_ok"]) == "True") else "FAILED"
            note = r.get("note")
        else:
            status, note = "NOT_RUN", ""
        rows.append({"job_id": j["job_id"], "kind": j["kind"], "bin": j["bin"],
                     "workload_or_scenario": j["workload_or_scenario"], "column": j["column"], "model": j["model"],
                     "status": status, "note": note if isinstance(note, str) else "",
                     "command": "poetry run python " + " ".join(j["cmd"][1:])})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "completeness.csv", index=False)
    return df


def headline(df: pd.DataFrame, keys, med="median_qerror", p95="p95_qerror", size="size_bytes",
             lat="infer_us_median") -> pd.DataFrame:
    """Per group and Hybrid variant: vs the best baseline (non-Hybrid), separately by median, P95, size, latency.
    hybrid_rank_* ranks the variant among the baselines + itself (other Hybrid variants excluded)."""
    rows = []
    for k, d0 in df.groupby(keys):
        for hm in ("Hybrid", "Hybrid-Corrected"):
            h = d0[d0["model"] == hm]
            b = d0[~d0["model"].isin(HYBRIDS)].dropna(subset=[med])
            if h.empty or b.empty:
                continue
            d = pd.concat([b, h])
            h = h.iloc[0]
            r = {**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "hybrid_model": hm}
            _headline_row(r, d, b, h, med, p95, size, lat)
            rows.append(r)
    return pd.DataFrame(rows)


def _headline_row(r, d, b, h, med, p95, size, lat):
    if True:
        for name, col in (("median", med), ("p95", p95), ("size", size), ("latency", lat)):
            if col not in d or b[col].isna().all():
                continue
            best = b.loc[b[col].idxmin()]
            r.update({f"hybrid_{name}": h[col], f"best_baseline_{name}": best[col],
                      f"best_baseline_{name}_model": best["model"],
                      f"hybrid_rank_{name}": int((d[col] < h[col]).sum()) + 1,
                      f"hybrid_better_{name}": bool(h[col] < best[col])})


def _shares(cdf_json):
    if not isinstance(cdf_json, str):
        return None
    c = json.loads(cdf_json)
    tot = sum(c.values()) or 1
    return " ".join(f"{k}:{100 * v / tot:.0f}%" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def original_vs_corrected_static(st: pd.DataFrame) -> pd.DataFrame:
    keys = ["bin_strategy", "workload", "dataset"]
    vals = ["median_qerror", "p95_qerror", "mae_rows", "size_bytes", "infer_us_median", "build_time_s"]
    rows = []
    for k, d in st[st["model"].isin(HYBRIDS)].groupby(keys):
        r = dict(zip(keys, k))
        for _, x in d.iterrows():
            tag = {"Hybrid": "orig", "Hybrid-Corrected": "corr", "Hybrid-Corrected-PredictOnly": "predonly"}[x["model"]]
            for v in vals:
                r[f"{tag}_{v}"] = x[v]
            r[f"{tag}_cdf_shares"] = _shares(x["cdf_choice"])
        if "orig_median_qerror" in r and "corr_median_qerror" in r:
            r["delta_median_corr_minus_orig"] = r["corr_median_qerror"] - r["orig_median_qerror"]
            r["delta_p95_corr_minus_orig"] = r["corr_p95_qerror"] - r["orig_p95_qerror"]
        rows.append(r)
    return pd.DataFrame(rows)


def original_vs_corrected_drift(hy: pd.DataFrame, sc: str) -> pd.DataFrame:
    if hy.empty:
        return hy
    vals = ["final_median_mean", "final_p95_mean", "final_mae_rows_mean", "size_bytes_final", "final_infer_us_median",
            "rebuild_performed_steps", "checkpoint_triggered_steps", "rebuilt_buckets_total"]
    o = hy[hy["model"] == "Hybrid"].set_index(["bin_strategy", "dataset"])[vals].add_prefix("orig_")
    c = hy[hy["model"] == "Hybrid-Corrected"].set_index(["bin_strategy", "dataset"])[vals].add_prefix("corr_")
    return o.join(c, how="outer").reset_index().assign(scenario=sc)


def _xlsx(df_map: dict, path: Path):
    with pd.ExcelWriter(path) as xw:
        for name, df in df_map.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-cdf-workbooks", action="store_true")
    args = ap.parse_args()

    st = static_table()
    if not st.empty:
        cols = ID_COLUMNS + [c for c in st.columns if c not in ID_COLUMNS]
        st = st[cols]
        st.to_csv(RES / "static" / "static_results.csv", index=False)
        piv_med = st.pivot_table(index=["bin_strategy", "workload", "dataset"], columns="model", values="median_qerror")
        piv_p95 = st.pivot_table(index=["bin_strategy", "workload", "dataset"], columns="model", values="p95_qerror")
        piv_size = st.pivot_table(index=["bin_strategy", "dataset"], columns="model", values="size_bytes", aggfunc="first")
        piv_inf = st.pivot_table(index=["bin_strategy", "workload", "dataset"], columns="model", values="infer_us_median")
        _xlsx({"static_results": st, "median_qerror": piv_med.reset_index(), "p95_qerror": piv_p95.reset_index(),
               "size_bytes": piv_size.reset_index(), "infer_us_median": piv_inf.reset_index()},
              RES / "static" / "static_results.xlsx")
        static_cdf_workbooks()
        hl = headline(st, ["bin_strategy", "workload", "dataset"])
        hl.to_csv(RES / "static" / "static_headline.csv", index=False)
        original_vs_corrected_static(st).to_csv(RES / "static" / "original_vs_corrected_static.csv", index=False)
        print(f"static: {len(st)} rows")

    wb_report = []
    for sc, long in drift_long().items():
        cols = ID_COLUMNS + [c for c in long.columns if c not in ID_COLUMNS]
        long = long[cols]
        long.to_csv(RES / sc / "drift_long.csv", index=False)
        hy = hybrid_stage_table(long)
        fm = final_by_model(long)
        _xlsx({"drift_long": long, "hybrid_stages": hy, "final_by_model": fm,
               "median_by_stage": long.pivot_table(index=["bin_strategy", "dataset", "model", "drift_pct"],
                                                   columns="stage", values="median_qerror").reset_index(),
               "p95_by_stage": long.pivot_table(index=["bin_strategy", "dataset", "model", "drift_pct"],
                                                columns="stage", values="p95_qerror").reset_index()},
              RES / sc / f"drift_summary__{sc}.xlsx")
        hy.to_csv(RES / sc / "hybrid_stages.csv", index=False)
        original_vs_corrected_drift(hy, sc).to_csv(RES / sc / "original_vs_corrected_drift.csv", index=False)
        fm.to_csv(RES / sc / "final_by_model.csv", index=False)
        headline(fm, ["bin_strategy", "dataset"], med="final_median_mean", p95="final_p95_mean",
                 size="size_bytes_mean", lat="final_infer_us_median").to_csv(RES / sc / "drift_headline.csv", index=False)
        if not args.no_cdf_workbooks:
            wb_report += cdf_workbooks(sc, long)
        if not args.no_plots:
            plots(sc, long)
        print(f"{sc}: {len(long)} long rows, {long['dataset'].nunique()} columns, {long['model'].nunique()} models")
    if wb_report:
        pd.DataFrame(wb_report).to_csv(RES / "cdf_workbooks_check.csv", index=False)
    comp = completeness()
    print(comp.groupby(["kind", "bin", "status"]).size().to_string())


if __name__ == "__main__":
    main()

"""
DSB expansion, part 6: results/paper_experiments/dsb_expanded/README.md from the exported tables.

    poetry run python -m paper_experiments.dsb_readme      # after paper_experiments.dsb_export

Every number in the README is read from a CSV next to it; nothing is typed by hand.
"""
import pandas as pd

from paper_experiments.dsb_run import RES, SCENARIOS

COLS_SHORT = lambda c: c.split(".", 1)[1]  # noqa: E731


def _read(p):
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _md(df: pd.DataFrame, floatfmt=3) -> str:
    if df.empty:
        return "_no data (NOT_RUN)_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows]) + "\n"


def static_section():
    st = _read(RES / "static" / "static_results.csv")
    hl = _read(RES / "static" / "static_headline.csv")
    out = ["## Static results", ""]
    if st.empty:
        return out + ["_NOT_RUN_", ""]
    out += ["Source: `static/static_results.csv` (+ `.xlsx`), headline `static/static_headline.csv`. "
            "Inference latency is per scalar `predict()` call (µs) measured while up to 6 jobs ran in parallel "
            "on a shared 12-core host (load ≈ 20) — compare only within this table.", ""]
    for b in ("fd", "fixed100"):
        d = st[(st["bin_strategy"] == b) & (st["workload"] == "dd_narrow")]
        if d.empty:
            continue
        piv = d.pivot_table(index="column", columns="model", values="median_qerror")
        p95 = d.pivot_table(index="column", columns="model", values="p95_qerror")
        size = d.pivot_table(index="column", columns="model", values="size_bytes")
        lat = d.pivot_table(index="column", columns="model", values="infer_us_median")
        for title, t, fmt in (("median q-error", piv, 3), ("P95 q-error", p95, 3), ("model size, bytes", size, 0),
                              ("median inference, µs", lat, 2)):
            out += [f"### {b} buckets, workload dd_narrow (main) — {title}", "", _md(t.reset_index(), fmt)]
    if not hl.empty:
        out += ["### Where Hybrid / Hybrid-Corrected beats the best non-Hybrid baseline (all workloads)", ""]
        rows = []
        for (b, hm), d in hl.groupby(["bin_strategy", "hybrid_model"]):
            r = {"bin": b, "model": hm, "configs": len(d)}
            for k in ("median", "p95", "size", "latency"):
                col = f"hybrid_better_{k}"
                if col in d:
                    r[f"better_{k}"] = int(d[col].fillna(False).astype(bool).sum())
            rows.append(r)
        out += [_md(pd.DataFrame(rows)),
                "`better_x` counts the column × workload configurations where the Hybrid variant is strictly lower "
                "than every non-Hybrid model on metric x.", ""]
    ovc = _read(RES / "static" / "original_vs_corrected_static.csv")
    if not ovc.empty and "corr_median_qerror" in ovc:
        keep = ["bin_strategy", "workload", "dataset", "orig_median_qerror", "corr_median_qerror", "orig_p95_qerror",
                "corr_p95_qerror", "orig_mae_rows", "corr_mae_rows", "orig_size_bytes", "corr_size_bytes",
                "orig_infer_us_median", "corr_infer_us_median", "orig_cdf_shares", "corr_cdf_shares"]
        if "predonly_median_qerror" in ovc:
            keep += ["predonly_median_qerror", "predonly_p95_qerror"]
        d = ovc[[c for c in keep if c in ovc]]
        d = d[d["workload"] == "dd_narrow"].assign(dataset=lambda x: x["dataset"].map(COLS_SHORT))
        out += ["### Original vs Hybrid-Corrected (static, dd_narrow)", "",
                "Hybrid-Corrected is a labelled MODIFICATION (inclusive-interval fix of Tier-0 inference and of the "
                "waterfall validation count); Original = canonical Hybrid FD CDF. PredictOnly = inference fix only "
                "(diagnostic). Full table for every workload: `static/original_vs_corrected_static.csv`.", "",
                _md(d.drop(columns=["workload"]))]
    return out


def drift_section():
    out = ["## Drift results", ""]
    for sc, cfg in SCENARIOS.items():
        hy = _read(RES / sc / "hybrid_stages.csv")
        fm = _read(RES / sc / "final_by_model.csv")
        out += [f"### {sc} (0 → {cfg['total_pct']} %, {cfg['steps']} points)", ""]
        if fm.empty:
            out += ["_NOT_RUN_", ""]
            continue
        out += [f"Sources: `{sc}/drift_long.csv` (SHOCK / FT / RB rows; RB only where a rebuild was performed), "
                f"`{sc}/final_by_model.csv`, `{sc}/hybrid_stages.csv`, workbooks `{sc}/drift_summary__{sc}.xlsx`, "
                f"`{sc}/cdf_workbooks/`, figures `{sc}/figures/<bin>/<column>.png`.", ""]
        for b in ("fd", "fixed100"):
            d = fm[fm["bin_strategy"] == b]
            if d.empty:
                continue
            for title, v in (("mean final median q-error over steps 1..", "final_median_mean"),
                             ("mean final P95 q-error", "final_p95_mean")):
                t = d.pivot_table(index="dataset", columns="model", values=v).reset_index()
                t["dataset"] = t["dataset"].map(COLS_SHORT)
                out += [f"#### {b} — {title}", "", _md(t)]
        if not hy.empty:
            keep = ["model", "bin_strategy", "dataset", "steps", "checkpoint_triggered_steps", "rebuild_performed_steps",
                    "rebuilt_buckets_total", "shock_median_mean", "ft_median_mean", "final_median_mean",
                    "ft_median_on_rb_steps", "rb_median_on_rb_steps", "shock_p95_mean", "ft_p95_mean",
                    "ft_time_s_total", "rb_time_s_total", "size_bytes_final"]
            t = hy[[c for c in keep if c in hy]].assign(dataset=lambda x: x["dataset"].map(COLS_SHORT))
            out += ["#### Hybrid stages (Shock → FT → RB)", "",
                    "`ft_median_on_rb_steps` vs `rb_median_on_rb_steps` compare FT and RB only on steps where a rebuild "
                    "was actually performed; on all other steps the final state is FT and no change is attributed "
                    "to Rebuild.", "", _md(t)]
        ovc = _read(RES / sc / "original_vs_corrected_drift.csv")
        if not ovc.empty and "corr_final_median_mean" in ovc and ovc["corr_final_median_mean"].notna().any():
            out += ["#### Original vs Hybrid-Corrected (drift)", "",
                    _md(ovc.assign(dataset=lambda x: x["dataset"].map(COLS_SHORT)).drop(columns=["scenario"]))]
    return out


def completeness_section():
    comp = _read(RES / "completeness.csv")
    out = ["## Completeness", ""]
    if comp.empty:
        return out + ["_completeness.csv missing_", ""]
    g = comp.groupby(["kind", "bin", "status"]).size().unstack(fill_value=0).reset_index()
    out += [_md(g), "Full list with the exact command of every job: `completeness.csv`; per-attempt log: "
            "`jobs_status.csv`, `logs/`.", ""]
    nr = comp[comp["status"] != "DONE"]
    if not nr.empty:
        out += ["### Not DONE (NOT_RUN / FAILED) — commands", "", "```bash"]
        out += [f"# {r.status} {r.job_id}\n{r.command}" for r in nr.head(400).itertuples()]
        out += ["```", ""]
    return out


def main():
    dist = _read(RES / "drift_distribution_change.csv")
    wb = _read(RES / "cdf_workbooks_check.csv")
    lines = [
        "# DSB expanded benchmark (Agent 2)",
        "",
        "DSB SF = 1 (official toolkit, `DSB benchmark/setup_dsb.sh`), 8 columns: the 3 thesis columns + 5 from other "
        "tables — selection and profile in `DATASET_MANIFEST.md`, `dataset_manifest.csv`, "
        "`dataset_profile_candidates.csv`. Models: the 6 registered defaults (EquiWidth, EquiDepth, TKHist, "
        "EntropyHist-MEA, STHist, canonical Hybrid FD CDF) at `--buckets fd` (repo FD, n = N, cap 2000) and "
        "`--buckets 100`, plus the labelled modification Hybrid-Corrected. Seed 42, 100 000-row sample, "
        "1 000 000 queries per workload. Every configuration directory has a `manifest_<Model>.json` "
        "(git commit, work-tree hash, file hashes, model version, command).",
        "",
        "Code: `paper_experiments/dsb_columns.py` (profile/selection), `dsb_prepare.py` (meta.pkl + workloads), "
        "`dsb_drift_data.py` (drift snapshots), `dsb_run.py` (job queue), `dsb_export.py` (tables, workbooks, "
        "figures), `dsb_readme.py` (this file).",
        "",
        "## Workloads",
        "",
        "- `dd_narrow` (main): data-driven, anchor from the 100k sample, w ~ U{1..100}, low = anchor − U{0..w−1}, "
        "high = low + w, clipped to [mn, mx] (thesis §4.4.2). Stream `default_rng([42, 10])`.",
        "- `uniform_narrow`: low ~ U{mn..mx}, same width rule. `uniform_wide`: repo wide rule (10 %–100 % of the domain).",
        "- `official`: distinct `<col> BETWEEN a AND b` ranges of dsqgen on the unmodified official templates — only "
        "ss_net_profit (3 ranges), ss_sales_price (3), inv_quantity_on_hand (509); NOT_APPLICABLE for the other 5 "
        "columns (`data/dsb_expanded/official_status.json`). 3-query workloads are reported but carry no statistical weight.",
        "- Zero-truth queries are dropped (benchmark_utils), so query counts differ per column/workload.",
        "",
        "## Drift scenarios",
        "",
        "Target = \"Zipf popularity shift\" on the initial support (seeded rank permutation, P(r) ∝ 1/r, N_init rows). "
        "Displacement: f = round((1−α) f_init + α f_target), α = 0…1 (21 points, N constant). Additive: "
        "f = round(f_init + α f_target), α = 0…2 (41 points, final N = 3·N_init), `--drift-total-pct 200`. "
        "Per-step reservoir (Bernoulli 1.5S/N → 100k) and per-step data-driven workload of 1 000 000 queries "
        "(thesis §4.4.4); adapt/eval halves (seed 42). The distribution really changes — distance from step 0 "
        "(`drift_distribution_change.csv`):",
        "",
    ]
    if not dist.empty:
        last = dist.sort_values("step").groupby(["column", "scenario"]).tail(1)
        t = last[["column", "scenario", "drift_pct", "N", "tv_distance", "ks_statistic", "wasserstein1_over_domain",
                  "target_row_share"]].assign(column=lambda x: x["column"].map(COLS_SHORT))
        lines += [_md(t, 4)]
    lines += ["Official refresh-stream churn for the new columns: NOT_RUN (lowest priority; the existing "
              "`data/dsb/*/drift` churn of ws_wholesale_cost / ws_list_price is untouched).", ""]
    lines += static_section() + drift_section()
    if not wb.empty:
        lines += ["## CDF-selection workbooks check", "",
                  f"{int(wb['complete'].sum())} of {len(wb)} per-(column, scenario, bin, model) workbooks have one sheet "
                  "per drift point (`cdf_workbooks_check.csv`).", ""]
    lines += completeness_section()
    (RES / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RES / 'README.md'}")


if __name__ == "__main__":
    main()

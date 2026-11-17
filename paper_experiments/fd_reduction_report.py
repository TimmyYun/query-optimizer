#!/usr/bin/env python3
"""
Tables, figures and the three answers for the FD bucket reduction experiment
(paper_experiments/fd_reduction.py). Re-runnable on partial results.

Reads   results/paper_experiments/fd_reduction/raw/<dataset>/results.jsonl            (Original run)
        results/paper_experiments/fd_reduction/raw/<dataset>/results_corrected.jsonl  (Hybrid-Corrected ablation)
        results/paper_experiments/fd_reduction/latency_pass.csv   (optional, sequential latency pass)
Writes  results/paper_experiments/fd_reduction/
          fd_reduction_results.csv   same-K (fd_r*) and merge (fd_merge_r*) configurations
          memory_matched_results.csv baselines sized to each Hybrid configuration's n_values
          fd_reduction_all.csv       both, with latency_us (pass) and pareto flags
          pareto_frontier.csv        non-dominated configurations per dataset
          answers.csv / answers.md   per dataset and Hybrid variant (Hybrid, Hybrid-Corrected):
                                     (a) same K, (b) vs its own full-FD config, (c) same memory
          original_vs_corrected.csv  per dataset x r: Hybrid vs Hybrid-Corrected (and -Merge, PredictOnly)
          tier0_offbyone_effect.csv  EquiWidth vs EquiWidth-Merge (canonical Tier-0) at r = 0
          completeness.csv           expected vs present configurations
          fd_reduction_results.xlsx  all of the above + README sheet (columns and units)
          figures/<dataset>/fig1..fig5_*.png, figures/aggregate_*.png

    poetry run python -m paper_experiments.fd_reduction_report
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.fd_reduction import (  # noqa: E402
    MATCH_MODELS,
    MERGE_MODELS,
    OUT,
    REDUCTIONS,
    SAME_K_MODELS,
)
from paper_experiments.schema import ID_COLUMNS, fmt_bytes  # noqa: E402

COLORS = {
    "EquiWidth": "#1f77b4", "EquiDepth": "#ff7f0e", "TKHist": "#2ca02c", "EntropyHist-MEA": "#9467bd",
    "Hybrid": "#d62728", "Hybrid-Merge": "#8c564b", "EquiWidth-Merge": "#7f7f7f",
    "Hybrid-Corrected": "#e377c2", "Hybrid-Corrected-Merge": "#bcbd22", "EquiWidth-Merge-Corrected": "#17becf",
    "Hybrid-Corrected-PredictOnly": "#000000",
}
CORR_MODELS = ["Hybrid-Corrected", "Hybrid-Corrected-Merge", "EquiWidth-Merge-Corrected"]
DASHED = set(MERGE_MODELS) | {"Hybrid-Corrected-Merge", "EquiWidth-Merge-Corrected"}
DIAGNOSTIC = {"Hybrid-Corrected-PredictOnly"}  # attribution only, never a candidate (no Pareto, no answers)
MODEL_ORDER = SAME_K_MODELS + MERGE_MODELS + CORR_MODELS
VARIANTS = {"Hybrid": "Original (canonical)", "Hybrid-Corrected": "Corrected (MODIFICATION)"}
# answer tolerances (reported next to the strict test)
TOL_MEDIAN, TOL_P95 = 1.01, 1.05     # (b) "no worse" within 1 % median / 5 % P95
COMP_MEDIAN, COMP_P95 = 1.02, 1.10   # (c) "competitive" within 2 % median / 10 % P95 of the best baseline

README = [
    ("experiment_id … stage", "unified identifiers (paper_experiments/schema.py); stage = STATIC"),
    ("bin_strategy", "fd_r<rr>: K = round(K_FD*(1-rr/100)), each model by its own definition; "
                     "fd_merge_r<rr>: greedy Ward merge of the K_FD equal-width buckets to K (MODIFICATION); "
                     "mem_match_K<k>: baseline sized to the n_values of Hybrid at K=k"),
    ("K_FD", "Freedman–Diaconis bucket count of the dataset (meta.pkl k, N table rows, cap 2000)"),
    ("K_requested / size_param", "K, statistics_target (EquiDepth) or requested B (EntropyHist)"),
    ("n_buckets", "buckets the model actually has (EquiDepth: histogram buckets besides MCVs)"),
    ("n_values / size_bytes", "stored numbers (model.n_values()) / x 8 bytes — persistent-state accounting"),
    ("*_qerror", "q-error, dimensionless, >= 1, lower is better; truth and estimate clipped to >= 1 row"),
    ("mae_rows", "mean absolute error in rows on the same clipped values"),
    ("build_time_s / train_time_s", "seconds: whole build / Hybrid waterfall training only"),
    ("infer_us_*", "microseconds per predict() measured in the accuracy run (machine shared with other jobs)"),
    ("lat_us_*", "microseconds per predict() from the sequential latency pass (one process, 100k fixed queries)"),
    ("latency_us", "lat_us_median if the latency pass ran, else infer_us_median (latency_source says which)"),
    ("target_n_values / budget_gap_pct / budget_reachable",
     "memory matching: Hybrid's n_values, achieved gap in %, False = the model cannot grow to that size"),
    ("pareto_median / pareto_p95", "non-dominated in (median or P95 q-error, size_bytes, latency_us) within the dataset"),
]


def load():
    frames = []
    for f in sorted((OUT / "raw").glob("*/results*.jsonl")):
        d = pd.read_json(f, lines=True, dtype=False)
        d["slug"] = f.parent.name
        d["run"] = "corrected_ablation" if f.stem.endswith("corrected") else "original"
        frames.append(d)
    if not frames:
        raise SystemExit("no results yet")
    df = pd.concat(frames, ignore_index=True)
    lat = OUT / "latency_pass.csv"
    if lat.exists():
        L = pd.read_csv(lat).drop_duplicates(["dataset", "bin_strategy", "model"], keep="last")
        df = df.merge(L, on=["dataset", "bin_strategy", "model"], how="left")
    for c in ("lat_us_median", "lat_us_mean", "lat_us_p95", "lat_loadavg_1min"):
        if c not in df:
            df[c] = np.nan
    df["latency_source"] = np.where(df["lat_us_median"].notna(), "latency_pass", "accuracy_run")
    df["latency_us"] = df["lat_us_median"].fillna(df["infer_us_median"])
    if "matched_to_model" not in df:
        df["matched_to_model"] = np.nan
    df["family"] = np.select(
        [df["bin_strategy"].str.startswith("fd_merge_r"), df["bin_strategy"].str.startswith("fd_r")],
        ["merge", "same_K"], "memory_matched")
    return df


def pareto_flags(d, acc):
    """Non-dominated rows in (acc, size_bytes, latency_us), all minimized."""
    X = d[[acc, "size_bytes", "latency_us"]].to_numpy(dtype=float)
    flags = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        dom = np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
        flags[i] = not dom.any()
    return flags


def answers(df, variant="Hybrid"):
    out = []
    for ds, d in df.groupby("dataset", sort=True):
        sk = d[d["family"] == "same_K"]
        mm = d[(d["family"] == "memory_matched") & (d["matched_to_model"].fillna("Hybrid") == variant)]
        hyb = sk[sk["model"] == variant].set_index("reduction_pct").sort_index()
        if 0 not in hyb.index:
            continue
        h0 = hyb.loc[0]
        row = {"dataset": ds, "variant": variant, "K_FD": int(h0["K_FD"]),
               "hybrid_fd_median": h0["median_qerror"], "hybrid_fd_p95": h0["p95_qerror"],
               "hybrid_fd_bytes": int(h0["size_bytes"]), "hybrid_fd_latency_us": h0["latency_us"]}
        for m in ("EquiWidth", "EquiDepth"):
            r0 = sk[(sk["model"] == m) & (sk["reduction_pct"] == 0)]
            if len(r0):
                row[f"{m}_fd_median"] = r0["median_qerror"].iloc[0]
                row[f"{m}_fd_p95"] = r0["p95_qerror"].iloc[0]
                row[f"{m}_fd_bytes"] = int(r0["size_bytes"].iloc[0])
        a_ok, b_ok, b_tol, c_win, c_comp, c_detail = [], [], [], [], [], []
        for r, h in hyb.iterrows():
            base = sk[(sk["reduction_pct"] == r) & sk["model"].isin(["EquiWidth", "EquiDepth"])]
            if r > 0 and len(base) == 2 and (h["median_qerror"] < base["median_qerror"].min()) \
                    and (h["p95_qerror"] < base["p95_qerror"].min()):
                a_ok.append(int(r))
            if r > 0 and h["n_values"] < h0["n_values"]:
                if h["median_qerror"] <= h0["median_qerror"] and h["p95_qerror"] <= h0["p95_qerror"]:
                    b_ok.append(int(r))
                if h["median_qerror"] <= TOL_MEDIAN * h0["median_qerror"] and h["p95_qerror"] <= TOL_P95 * h0["p95_qerror"]:
                    b_tol.append(int(r))
            m = mm[mm["matched_to_K"] == h["K_requested"]]
            if len(m):
                best_med = m["median_qerror"].min()
                best_p95 = m["p95_qerror"].min()
                if h["median_qerror"] <= best_med and h["p95_qerror"] <= best_p95:
                    c_win.append(int(r))
                if h["median_qerror"] <= COMP_MEDIAN * best_med and h["p95_qerror"] <= COMP_P95 * best_p95:
                    c_comp.append(int(r))
                bm = m.loc[m["median_qerror"].idxmin()]
                c_detail.append(f"r{int(r)}: H {h['median_qerror']:.3f}/{h['p95_qerror']:.2f} vs "
                                f"{bm['model']} {bm['median_qerror']:.3f}/{bm['p95_qerror']:.2f}")
        best_b = hyb[hyb.index.isin(b_tol)]
        row.update({
            "a_same_K_beats_EW_and_ED_r": a_ok, "a_any": bool(a_ok),
            "b_no_worse_than_fd_less_memory_r": b_ok, "b_any": bool(b_ok),
            "b_within_tol_r": b_tol, "b_tol_any": bool(b_tol),
            "b_best_saving_pct": float(100 * (1 - best_b["n_values"].min() / h0["n_values"])) if len(best_b) else 0.0,
            "c_equal_memory_wins_r": c_win, "c_any": bool(c_win),
            "c_equal_memory_competitive_r": c_comp,
            "c_detail_median_p95": "; ".join(c_detail),
        })
        out.append(row)
    return pd.DataFrame(out)


def original_vs_corrected(df):
    """Per dataset x r: canonical Hybrid vs Hybrid-Corrected (same K, sample, workload, seed)."""
    def uniform_share(cc, nb):
        if not isinstance(cc, str) or not nb:
            return np.nan
        c = json.loads(cc)
        nonempty = nb - c.get("Empty", 0)
        return 100.0 * c.get("Uniform", 0) / nonempty if nonempty else np.nan

    rows = []
    pairs = [("Hybrid", "Hybrid-Corrected", "fd_r"), ("Hybrid-Merge", "Hybrid-Corrected-Merge", "fd_merge_r")]
    for ds, d in df.groupby("dataset"):
        for orig_m, corr_m, prefix in pairs:
            o = d[(d["model"] == orig_m) & d["bin_strategy"].str.startswith(prefix)].set_index("reduction_pct")
            c = d[(d["model"] == corr_m) & d["bin_strategy"].str.startswith(prefix)].set_index("reduction_pct")
            po = d[(d["model"] == "Hybrid-Corrected-PredictOnly")].set_index("reduction_pct")
            for r in sorted(set(o.index) & set(c.index)):
                a, b = o.loc[r], c.loc[r]
                row = {"dataset": ds, "pair": f"{orig_m} vs {corr_m}", "reduction_pct": int(r),
                       "K": int(a["K_requested"]), "n_buckets": int(a["n_buckets"])}
                for k in ("median_qerror", "p95_qerror", "avg_qerror", "mae_rows", "size_bytes", "latency_us",
                          "train_time_s"):
                    row[f"orig_{k}"] = a[k]
                    row[f"corr_{k}"] = b[k]
                    row[f"delta_pct_{k}"] = 100.0 * (b[k] - a[k]) / a[k] if a[k] else np.nan
                row["orig_uniform_share_pct"] = uniform_share(a["cdf_choice"], a["n_buckets"])
                row["corr_uniform_share_pct"] = uniform_share(b["cdf_choice"], b["n_buckets"])
                row["orig_cdf_choice"] = a["cdf_choice"]
                row["corr_cdf_choice"] = b["cdf_choice"]
                if prefix == "fd_r" and r in po.index:
                    p_ = po.loc[r]
                    row["predict_only_median_qerror"] = p_["median_qerror"]
                    row["predict_only_p95_qerror"] = p_["p95_qerror"]
                    row["predict_only_uniform_share_pct"] = uniform_share(p_["cdf_choice"], p_["n_buckets"])
                    row["predict_only_size_bytes"] = p_["size_bytes"]
                rows.append(row)
    return pd.DataFrame(rows)


def tier0_bug_effect(df):
    """
    EquiWidth vs EquiWidth-Merge at r = 0: identical FD grid and counts, both uniform inside a bucket.
    The only difference is HybridEstimator's Tier-0 uniform CDF (x_norm = (v - lo) / w against labels
    'fraction <= v'), one value short when a query starts at a bucket's lo and one value extra when it
    ends at its hi. So this table is the measured size of that off-by-one, not a partitioning effect.
    """
    rows = []
    cols = ["median_qerror", "p95_qerror", "avg_qerror", "mae_rows"]
    for ds, d in df.groupby("dataset"):
        ew = d[(d["model"] == "EquiWidth") & (d["bin_strategy"] == "fd_r00")]
        em = d[(d["model"] == "EquiWidth-Merge") & (d["bin_strategy"] == "fd_merge_r00")]
        ec = d[(d["model"] == "EquiWidth-Merge-Corrected") & (d["bin_strategy"] == "fd_merge_r00")]
        if ew.empty or em.empty:
            continue
        r = {"dataset": ds, "K_FD": int(ew["K_FD"].iloc[0])}
        for c in cols:
            a, b = float(ew[c].iloc[0]), float(em[c].iloc[0])
            r[f"EquiWidth_{c}"] = a
            r[f"Tier0Uniform_{c}"] = b
            r[f"delta_pct_{c}"] = 100.0 * (b - a) / a
            if not ec.empty:  # inclusive uniform on the same grid must reproduce EquiWidth
                r[f"Tier0Inclusive_{c}"] = float(ec[c].iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def completeness(df):
    rows = []
    for ds, d in df.groupby("dataset"):
        have = set(zip(d["bin_strategy"], d["model"]))
        exp = [(f"fd_r{r:02d}", m) for r in REDUCTIONS for m in SAME_K_MODELS]
        exp += [(f"fd_merge_r{r:02d}", m) for r in REDUCTIONS for m in MERGE_MODELS]
        hy = d[(d["model"] == "Hybrid") & (d["family"] == "same_K")]
        exp += [(f"mem_match_K{int(k)}", m) for k in hy["K_requested"].unique() for m in MATCH_MODELS]
        if (d["run"] == "corrected_ablation").any():
            exp += [(f"fd_r{r:02d}", "Hybrid-Corrected") for r in REDUCTIONS]
            exp += [(f"fd_merge_r{r:02d}", m) for r in REDUCTIONS for m in ("Hybrid-Corrected-Merge", "EquiWidth-Merge-Corrected")]
            exp += [(f"fd_r{r:02d}", "Hybrid-Corrected-PredictOnly") for r in (0, 50)]
            hc = d[(d["model"] == "Hybrid-Corrected") & (d["family"] == "same_K")]
            exp += [(f"mem_match_corrected_K{int(k)}", m) for k in hc["K_requested"].unique() for m in MATCH_MODELS]
        missing = [f"{b}:{m}" for b, m in exp if (b, m) not in have]
        rows.append({"dataset": ds, "expected": len(set(exp)), "present": len(have & set(exp)),
                     "missing": ";".join(missing) or ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})


def figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_root = OUT / "figures"
    for ds, d in df.groupby("dataset"):
        slug = d["slug"].iloc[0]
        fdir = fig_root / slug
        fdir.mkdir(parents=True, exist_ok=True)
        K_FD = int(d["K_FD"].iloc[0])
        curves = d[d["family"].isin(["same_K", "merge"])]

        def lines(ax, y, logy=False, x="n_buckets_plot"):
            for m in MODEL_ORDER:
                c = curves[curves["model"] == m].sort_values("K_requested")
                if c.empty:
                    continue
                xs = c["K_requested"] if x == "n_buckets_plot" else c[x]
                ax.plot(xs, c[y], marker="o", ms=3, lw=1.4, color=COLORS[m],
                        ls="--" if m in DASHED else "-", label=m)
            ax.axvline(K_FD, color="black", lw=0.8, ls=":") if x == "n_buckets_plot" else None
            if logy:
                ax.set_yscale("log")

        # 1. q-error vs buckets
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        lines(axes[0], "median_qerror")
        lines(axes[1], "p95_qerror", logy=True)
        axes[0].set_ylabel("median q-error")
        axes[1].set_ylabel("P95 q-error (log)")
        for ax in axes:
            ax.set_xlabel(f"buckets K (dotted: K_FD = {K_FD})")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{ds}: q-error vs number of buckets (fd_r* and merge)")
        fig.tight_layout()
        fig.savefig(fdir / "fig1_qerror_vs_buckets.png")
        plt.close(fig)

        # 2. memory vs buckets
        fig, ax = plt.subplots(figsize=(6.5, 4))
        lines(ax, "size_bytes", logy=True)
        ax.set_ylabel("model memory, bytes (log)")
        ax.set_xlabel("buckets K")
        ax.legend(fontsize=7)
        ax.set_title(f"{ds}: total model memory vs buckets")
        fig.tight_layout()
        fig.savefig(fdir / "fig2_memory_vs_buckets.png")
        plt.close(fig)

        # 3. q-error vs memory (incl. memory-matched points)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, y in zip(axes, ["median_qerror", "p95_qerror"]):
            for m in MODEL_ORDER:
                c = curves[curves["model"] == m].sort_values("size_bytes")
                if not c.empty:
                    ax.plot(c["size_bytes"], c[y], marker="o", ms=3, lw=1.2, color=COLORS[m],
                            ls="--" if m in DASHED else "-", label=f"{m} (K sweep)")
            mm = d[(d["family"] == "memory_matched") & (d["matched_to_model"].fillna("Hybrid") == "Hybrid")]
            for m in MATCH_MODELS:
                c = mm[mm["model"] == m].sort_values("size_bytes")
                if not c.empty:
                    ax.scatter(c["size_bytes"], c[y], marker="x", s=22, color=COLORS[m],
                               label=f"{m} (memory-matched)")
            ax.set_xscale("log")
            if y == "p95_qerror":
                ax.set_yscale("log")
            ax.set_xlabel("model memory, bytes (log)")
            ax.set_ylabel(y.replace("_", " "))
        axes[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{ds}: q-error vs total memory")
        fig.tight_layout()
        fig.savefig(fdir / "fig3_qerror_vs_memory.png")
        plt.close(fig)

        # 4. latency vs buckets
        fig, ax = plt.subplots(figsize=(6.5, 4))
        lines(ax, "latency_us")
        src = "sequential latency pass" if (curves["latency_source"] == "latency_pass").all() else "accuracy run (shared machine)"
        ax.set_ylabel(f"median inference, µs ({src})")
        ax.set_xlabel("buckets K")
        ax.legend(fontsize=7)
        ax.set_title(f"{ds}: inference latency vs buckets")
        fig.tight_layout()
        fig.savefig(fdir / "fig4_latency_vs_buckets.png")
        plt.close(fig)

        # 5. Pareto projections
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        pairs = [("size_bytes", "median_qerror"), ("latency_us", "median_qerror"), ("size_bytes", "latency_us")]
        for ax, (x, y) in zip(axes, pairs):
            for m in d["model"].unique():
                c = d[(d["model"] == m) & ~d["model"].isin(DIAGNOSTIC)]
                ax.scatter(c[x], c[y], s=12, color=COLORS.get(m, "black"), alpha=0.45, label=m)
            p = d[d["pareto_median"]]
            ax.scatter(p[x], p[y], s=40, facecolors="none", edgecolors="black", lw=1.0, label="Pareto (median)")
            if x == "size_bytes":
                ax.set_xscale("log")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{ds}: Pareto frontier over (median q-error, memory, latency) — circled = non-dominated")
        fig.tight_layout()
        fig.savefig(fdir / "fig5_pareto.png")
        plt.close(fig)

    # aggregate small multiples
    dsets = sorted(df["dataset"].unique())
    ncol = 5
    nrow = int(np.ceil(len(dsets) / ncol))
    for y, fname, xcol in [("median_qerror", "aggregate_median_vs_buckets.png", "K_requested"),
                           ("p95_qerror", "aggregate_p95_vs_buckets.png", "K_requested"),
                           ("median_qerror", "aggregate_median_vs_memory.png", "size_bytes")]:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
        for ax, ds in zip(axes.flat, dsets):
            d = df[(df["dataset"] == ds) & df["family"].isin(["same_K", "merge"])]
            for m in ["EquiWidth", "EquiDepth", "Hybrid", "Hybrid-Merge", "Hybrid-Corrected"]:
                c = d[d["model"] == m].sort_values(xcol)
                if not c.empty:
                    ax.plot(c[xcol], c[y], lw=1.2, marker=".", color=COLORS[m],
                            ls="--" if m in DASHED else "-", label=m)
            if xcol == "size_bytes":
                ax.set_xscale("log")
            if y == "p95_qerror":
                ax.set_yscale("log")
            ax.set_title(ds, fontsize=8)
        for ax in list(axes.flat)[len(dsets):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f"{y} vs {xcol} — EquiWidth / EquiDepth / Hybrid / Hybrid-Merge / Hybrid-Corrected")
        fig.tight_layout()
        fig.savefig(fig_root / fname)
        plt.close(fig)


def main():
    df = load()
    df["pareto_median"] = False
    df["pareto_p95"] = False
    cand = ~df["model"].isin(DIAGNOSTIC)
    for ds, idx in df[cand].groupby("dataset").groups.items():
        d = df.loc[idx]
        df.loc[idx, "pareto_median"] = pareto_flags(d, "median_qerror")
        df.loc[idx, "pareto_p95"] = pareto_flags(d, "p95_qerror")
    df["size_display"] = df["size_bytes"].map(fmt_bytes)
    lead = ID_COLUMNS + ["family", "reduction_pct", "K_FD", "K_requested", "size_param", "n_buckets",
                         "n_values", "size_bytes", "size_display", "median_qerror", "p25_qerror", "p75_qerror",
                         "p95_qerror", "avg_qerror", "mae_rows", "build_time_s", "train_time_s",
                         "infer_us_median", "infer_us_mean", "infer_us_p95", "lat_us_median", "lat_us_mean",
                         "lat_us_p95", "latency_us", "latency_source", "lat_loadavg_1min", "n_queries"]
    rest = [c for c in df.columns if c not in lead]
    df = df[[c for c in lead if c in df.columns] + rest]
    df = df.sort_values(["dataset", "family", "model", "K_requested"])
    df.to_csv(OUT / "fd_reduction_all.csv", index=False)
    df[df["family"] != "memory_matched"].to_csv(OUT / "fd_reduction_results.csv", index=False)
    mm = df[df["family"] == "memory_matched"]
    mm.to_csv(OUT / "memory_matched_results.csv", index=False)
    par = df[df["pareto_median"] | df["pareto_p95"]]
    par.to_csv(OUT / "pareto_frontier.csv", index=False)
    ans = pd.concat([answers(df, v) for v in VARIANTS], ignore_index=True)
    ans_x = ans.copy()
    for c in ans_x.columns:
        if ans_x[c].map(lambda v: isinstance(v, list)).any():
            ans_x[c] = ans_x[c].map(lambda v: ",".join(map(str, v)))
    ans_x.to_csv(OUT / "answers.csv", index=False)
    comp = completeness(df)
    comp.to_csv(OUT / "completeness.csv", index=False)
    bug = tier0_bug_effect(df)
    bug.to_csv(OUT / "tier0_offbyone_effect.csv", index=False)
    ovc = original_vs_corrected(df)
    ovc.to_csv(OUT / "original_vs_corrected.csv", index=False)
    with pd.ExcelWriter(OUT / "fd_reduction_results.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(README, columns=["column", "meaning / unit"]).to_excel(xw, sheet_name="README", index=False)
        for name, t in [("same_K", df[df["family"] == "same_K"]), ("merge", df[df["family"] == "merge"]),
                        ("memory_matched", mm), ("pareto", par), ("answers", ans_x),
                        ("original_vs_corrected", ovc), ("tier0_offbyone", bug), ("completeness", comp)]:
            t.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    md = ["| dataset | variant | K_FD | Hybrid FD median / P95 / bytes | (a) r where Hybrid beats EW & ED (med+P95) | "
          "(b) r no worse than FD with less memory (strict; tol) | (c) r winning / competitive at equal memory |",
          "|---|---|---|---|---|---|---|"]
    for _, a in ans_x.iterrows():
        md.append(f"| {a['dataset']} | {a['variant']} | {a['K_FD']} | {a['hybrid_fd_median']:.4f} / {a['hybrid_fd_p95']:.3f} / "
                  f"{fmt_bytes(a['hybrid_fd_bytes'])} | {a['a_same_K_beats_EW_and_ED_r'] or '—'} | "
                  f"{a['b_no_worse_than_fd_less_memory_r'] or '—'}; {a['b_within_tol_r'] or '—'} | "
                  f"{a['c_equal_memory_wins_r'] or '—'} / {a['c_equal_memory_competitive_r'] or '—'} |")
    (OUT / "answers.md").write_text("\n".join(md) + "\n")
    figures(df)
    print(json.dumps({"rows": len(df), "datasets": int(df["dataset"].nunique()),
                      "incomplete": comp[comp["missing"] != ""]["dataset"].tolist()}, indent=1))


if __name__ == "__main__":
    main()

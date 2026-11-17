#!/usr/bin/env python3
"""
Figures for the finished experiment categories, in the style of fd_reduction_report.py
(same model colours, grid, line+marker curves, small multiples per dataset).

  dsb_drift_<bin>/     x = drift %, y = q-error (final state of each step) per model; Hybrid stages
                       SHOCK / FT / RB (RB markers only where rebuild_performed); Hybrid memory vs drift
  dsb_static/          no natural x -> box plots (+ points) over the 8 columns: model x {fixed100, fd}
                       for median / P95 q-error, size, latency, one figure per workload
  reproduction/        the same box plots over the 5 synthetic distributions, per scenario
  ablation_corrected/  Corrected / Original ratio (median, P95, size) as box plots per category

Reads only finished outputs; a missing category is skipped and listed in figures/README.md.
Latency in static results was measured with other jobs running (marked in the titles).

    poetry run python -m paper_experiments.figures_categories [--drift-bins fd fixed100]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_experiments.fd_reduction_report import COLORS, _style
from paper_experiments.schema import drift_steps_to_long

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "paper_experiments"
OUT = RES / "figures"
MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid", "Hybrid-Corrected"]
COLORS = {**COLORS, "STHist": "#17becf"}  # STHist is not in the FD-reduction palette
BIN_HATCH = {"fixed100": "", "fd": "///"}
SCEN_LABEL = {"displacement": "displacement 0→100 %", "additive_200": "additive 0→200 %"}
METRICS = [("median_qerror", "median q-error", False), ("p95_qerror", "P95 q-error (log)", True),
           ("size_bytes", "model size, bytes (log)", True), ("infer_us", "median inference, µs (under load)", False)]
NOTES = []


def short(col):
    return col.split(".", 1)[1] if "." in col else col


# ----------------------------------------------------------------------------- loading

def static_summaries(root: Path, depth_labels):
    """summary_<Model>.json under root/<l1>/<l2>/... -> rows with the path parts named by depth_labels."""
    rows = []
    for p in root.rglob("summary_*.json"):
        rel = p.relative_to(root).parts[:-1]
        if len(rel) != len(depth_labels):
            continue
        s = json.loads(p.read_text())
        m = s["metrics"]
        rows.append({**dict(zip(depth_labels, rel)), "model": s["model"],
                     "median_qerror": m.get("median_q_error"), "p95_qerror": m.get("p95_q_error"),
                     "avg_qerror": m.get("avg_q_error"), "size_bytes": m.get("size_bytes"),
                     "infer_us": None if m.get("median_infer_time") is None else m["median_infer_time"] * 1e6})
    return pd.DataFrame(rows)


def drift_long(bin_):
    rows = []
    for sc in SCEN_LABEL:
        base = RES / "dsb_expanded" / sc / bin_
        for f in base.glob("*/gradual_*_to_zipfpop_*/drift_steps.csv"):
            col = f.parent.parent.name
            model = f.parent.name.split("_to_zipfpop_")[1]
            if model not in MODELS:
                continue
            d = pd.read_csv(f)
            L = drift_steps_to_long(d, {"model": model, "column": col}, staged=model.startswith("Hybrid"))
            L["scenario"], L["n_steps_csv"] = sc, len(d)
            rows.append(L)
    rows = [r for r in rows if not r.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def final_per_step(L):
    order = {"STATIC": 0, "SHOCK": 1, "FT": 2, "RB": 3}
    L = L.assign(_o=L["stage"].map(order)).sort_values("_o")
    return L.groupby(["scenario", "column", "model", "step"], as_index=False).tail(1).drop(columns="_o")


# ----------------------------------------------------------------------------- drift

def fig_drift(plt, bin_):
    L = drift_long(bin_)
    if L.empty:
        NOTES.append(f"dsb_drift_{bin_}: no finished runs — skipped")
        return
    expected = {"displacement": 21, "additive_200": 41}
    L = L[L.apply(lambda r: r["n_steps_csv"] == expected[r["scenario"]], axis=1)]
    out = OUT / f"dsb_drift_{bin_}"
    out.mkdir(parents=True, exist_ok=True)
    F = final_per_step(L)
    cols = sorted(F["column"].unique())
    missing = [(sc, c, m) for sc in SCEN_LABEL for c in cols for m in MODELS
               if F[(F.scenario == sc) & (F.column == c) & (F.model == m)].empty]
    if missing:
        NOTES.append(f"dsb_drift_{bin_}: {len(missing)} (scenario, column, model) curves not finished — not drawn")

    # aggregate small multiples: final-state q-error vs drift
    for sc in SCEN_LABEL:
        for y, logy, lab in [("median_qerror", False, "median"), ("p95_qerror", True, "p95")]:
            fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), squeeze=False)
            for ax, c in zip(axes.flat, cols):
                for m in MODELS:
                    d = F[(F.scenario == sc) & (F.column == c) & (F.model == m)].sort_values("drift_pct")
                    if not d.empty:
                        ax.plot(d["drift_pct"], d[y], marker=".", ms=3, lw=1.2, color=COLORS[m], label=m,
                                ls="--" if m == "Hybrid-Corrected" else "-")
                if logy:
                    ax.set_yscale("log")
                ax.set_title(short(c), fontsize=8)
                ax.set_xlabel("drift %")
            for ax in list(axes.flat)[len(cols):]:
                ax.axis("off")
            axes.flat[0].legend(fontsize=6)
            axes.flat[0].set_ylabel(f"{lab} q-error" + (" (log)" if logy else ""))
            fig.suptitle(f"DSB drift, {SCEN_LABEL[sc]}, {bin_} buckets — {lab} q-error after each step's action "
                         f"(Hybrid: RB if performed, else FT)")
            fig.tight_layout()
            fig.savefig(out / f"aggregate_{sc}_{lab}_vs_drift.png")
            plt.close(fig)

    # per column: all models + Hybrid stages
    for sc in SCEN_LABEL:
        for c in cols:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for ax, y, logy in [(axes[0, 0], "median_qerror", False), (axes[0, 1], "p95_qerror", True)]:
                for m in MODELS:
                    d = F[(F.scenario == sc) & (F.column == c) & (F.model == m)].sort_values("drift_pct")
                    if not d.empty:
                        ax.plot(d["drift_pct"], d[y], marker="o", ms=2.5, lw=1.3, color=COLORS[m], label=m,
                                ls="--" if m == "Hybrid-Corrected" else "-")
                ax.set_ylabel(("median" if y == "median_qerror" else "P95") + " q-error" + (" (log)" if logy else ""))
                if logy:
                    ax.set_yscale("log")
            axes[0, 0].legend(fontsize=7)
            for ax, m in [(axes[1, 0], "Hybrid"), (axes[1, 1], "Hybrid-Corrected")]:
                d = L[(L.scenario == sc) & (L.column == c) & (L.model == m)]
                for st, ls, mk in [("SHOCK", ":", "v"), ("FT", "-", "o")]:
                    s = d[d.stage.isin([st, "STATIC"])].sort_values("drift_pct")
                    ax.plot(s["drift_pct"], s["median_qerror"], ls=ls, marker=mk, ms=3, lw=1.2,
                            color=COLORS[m], alpha=0.6 if st == "SHOCK" else 1.0, label=st)
                rb = d[d.stage == "RB"]
                if not rb.empty:
                    ax.scatter(rb["drift_pct"], rb["median_qerror"], marker="*", s=70, color="black", zorder=5,
                               label=f"RB (rebuild performed, {len(rb)} steps)")
                ck = d[(d.stage == "FT") & d.checkpoint_triggered & ~d.rebuild_performed]
                if not ck.empty:
                    ax.scatter(ck["drift_pct"], ck["median_qerror"], marker="x", s=30, color="grey", zorder=4,
                               label="checkpoint, 0 marked → no RB")
                ax.set_title(f"{m}: Shock → FT → RB", fontsize=9)
                ax.set_ylabel("median q-error")
                ax.legend(fontsize=7)
            for ax in axes.flat:
                ax.set_xlabel("drift %")
            fig.suptitle(f"{short(c)} — DSB drift {SCEN_LABEL[sc]}, {bin_} buckets")
            fig.tight_layout()
            fig.savefig(out / f"{short(c)}__{sc}.png")
            plt.close(fig)

    # Hybrid memory vs drift
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        for sc, ls in [("displacement", "-"), ("additive_200", "--")]:
            for m in ["Hybrid", "Hybrid-Corrected"]:
                d = F[(F.scenario == sc) & (F.column == c) & (F.model == m)].sort_values("drift_pct")
                if not d.empty:
                    ax.plot(d["drift_pct"], d["size_bytes"], ls=ls, lw=1.2, marker=".", color=COLORS[m],
                            label=f"{m}, {sc}")
        ax.set_yscale("log")
        ax.set_title(short(c), fontsize=8)
        ax.set_xlabel("drift %")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    axes.flat[0].set_ylabel("model size, bytes (log)")
    fig.suptitle(f"DSB drift, {bin_} buckets — Hybrid memory after each step (solid: displacement, dashed: additive)")
    fig.tight_layout()
    fig.savefig(out / "aggregate_hybrid_memory_vs_drift.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- box plots

def grouped_boxes(ax, df, y, logy, models, groups=("fixed100", "fd"), group_col="bin"):
    pos, data, cols, hatches = [], [], [], []
    for i, m in enumerate(models):
        for j, g in enumerate(groups):
            v = df[(df.model == m) & (df[group_col] == g)][y].dropna().to_numpy()
            if len(v):
                pos.append(i * (len(groups) + 1) + j)
                data.append(v)
                cols.append(COLORS[m])
                hatches.append(BIN_HATCH.get(g, ""))
    if not data:
        return
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for patch, c, h in zip(bp["boxes"], cols, hatches):
        patch.set(facecolor=c, alpha=0.35, edgecolor=c, hatch=h)
    for med in bp["medians"]:
        med.set(color="black", lw=1)
    rng = np.random.default_rng([42, 99])
    for p, v, c in zip(pos, data, cols):
        ax.scatter(p + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=c, zorder=3, edgecolor="none")
    ax.set_xticks([i * (len(groups) + 1) + (len(groups) - 1) / 2 for i in range(len(models))])
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    if logy:
        ax.set_yscale("log")


def box_figure(plt, df, title, path, sample_label):
    models = [m for m in MODELS if m in set(df.model)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (y, lab, logy) in zip(axes, METRICS):
        grouped_boxes(ax, df, y, logy, models)
        ax.set_title(lab, fontsize=9)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=BIN_HATCH[b], label=b)
                            for b in ("fixed100", "fd")], fontsize=7, title="buckets", title_fontsize=7)
    fig.suptitle(f"{title} — box = {sample_label}, point = one {sample_label.rstrip('s')}; "
                 f"plain = fixed100, hatched = FD")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def by_dataset_figure(plt, df, key, title, path, label_fn=lambda x: x):
    """Small multiples, one panel per dataset (named): x = model, circle = fixed100, triangle = FD;
    one figure row per metric (median, P95, size)."""
    models = [m for m in MODELS if m in set(df.model)]
    keys = sorted(df[key].unique())
    rows = [("median_qerror", "median q-error", False), ("p95_qerror", "P95 q-error (log)", True),
            ("size_bytes", "size, bytes (log)", True)]
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(2.3 * len(keys) + 1, 2.6 * len(rows)), squeeze=False)
    xs = np.arange(len(models))
    for j, k in enumerate(keys):
        d = df[df[key] == k]
        for i, (y, lab, logy) in enumerate(rows):
            ax = axes[i, j]
            for b, mk, dx in [("fixed100", "o", -0.15), ("fd", "^", 0.15)]:
                for x, m in zip(xs, models):
                    v = d[(d.model == m) & (d.bin == b)][y]
                    if len(v):
                        ax.scatter(x + dx, v.iloc[0], marker=mk, s=22, color=COLORS[m],
                                   edgecolor="black" if b == "fd" else "none", linewidth=0.5, zorder=3)
            # best (lowest) value per bin marked with a thin horizontal line in the bin's style
            for b, ls in [("fixed100", ":"), ("fd", "--")]:
                v = d[d.bin == b][y].dropna()
                if len(v):
                    ax.axhline(v.min(), color="grey", lw=0.7, ls=ls)
            if logy:
                from matplotlib.ticker import FormatStrFormatter
                ax.set_yscale("log")
                if y == "p95_qerror":  # narrow ranges: plain numbers instead of 1.2x10^0
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
                    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
                    ax.tick_params(axis="y", which="minor", labelsize=6)
            ax.set_xticks(xs)
            ax.set_xticklabels(models if i == len(rows) - 1 else [], rotation=60, ha="right", fontsize=6)
            if i == 0:
                ax.set_title(label_fn(k), fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([], [], marker="o", ls="", color="grey", label="fixed100"),
                               Line2D([], [], marker="^", ls="", color="grey", markeredgecolor="black", label="FD"),
                               Line2D([], [], ls=":", color="grey", label="best, fixed100"),
                               Line2D([], [], ls="--", color="grey", label="best, FD")], fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


QUANTILES = [5, 25, 50, 75, 95]


def dsb_query_quantiles(df):
    """Per-query q-error quantiles of every DSB static run (bin x workload x column x model), cached:
    the per-query CSVs hold 1M rows each, the figure needs only 5 numbers per box."""
    cache = OUT / "dsb_static" / "qerror_quantiles.csv"
    old = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["bin", "workload", "column", "model"])
    have = set(map(tuple, old[["bin", "workload", "column", "model"]].astype(str).to_numpy()))
    rows = []
    for _, r in df.iterrows():
        key = (r["bin"], r["workload"], r["column"], r["model"])
        if key in have:
            continue
        f = RES / "dsb_expanded" / "static" / r["bin"] / r["workload"] / r["column"] / f"workload_{r['workload']}_{r['model']}.csv"
        if not f.exists():
            continue
        q = pd.read_csv(f, usecols=["Q_Error"])["Q_Error"].to_numpy()
        rows.append({**dict(zip(["bin", "workload", "column", "model"], key)), "n_queries": len(q),
                     "mean": float(q.mean()), **{f"p{k}": float(v) for k, v in zip(QUANTILES, np.percentile(q, QUANTILES))}})
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def dataset_box_figure(plt, Q, wl, path):
    """x = DSB column (named), one box per model: per-query q-error, box = P25-P75, line = median,
    whiskers = P5-P95, diamond = mean (log y); top row fixed100, bottom row FD."""
    models = [m for m in MODELS if m in set(Q.model)]
    cols = sorted(Q["column"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(2.4 * len(cols) + 2, 8.5), squeeze=False)
    width = 0.8 / len(models)
    for ax, b in zip(axes[:, 0], ["fixed100", "fd"]):
        d = Q[Q.bin == b]
        for k, m in enumerate(models):
            stats, pos = [], []
            for i, c in enumerate(cols):
                r = d[(d.model == m) & (d.column == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                stats.append({"med": r.p50, "q1": r.p25, "q3": r.p75, "whislo": r.p5, "whishi": r.p95,
                              "mean": r["mean"], "label": ""})
                pos.append(i - 0.4 + width * (k + 0.5))
            if not stats:
                continue
            bp = ax.bxp(stats, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=False,
                        showmeans=True, meanprops={"marker": "D", "markersize": 2.5, "markerfacecolor": "black",
                                                   "markeredgecolor": "black"})
            for patch in bp["boxes"]:
                patch.set(facecolor=COLORS[m], alpha=0.55, edgecolor=COLORS[m])
            for part in ("whiskers", "caps"):
                for ln in bp[part]:
                    ln.set(color=COLORS[m])
            for med in bp["medians"]:
                med.set(color="black", lw=1)
            ax.plot([], [], color=COLORS[m], lw=6, alpha=0.55, label=m)
        for i in range(1, len(cols)):
            ax.axvline(i - 0.5, color="grey", lw=0.5, alpha=0.5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([short(c) for c in cols], fontsize=9)
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylabel(f"q-error per query (log), {b}")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(fontsize=7, ncol=len(models), loc="upper left")
    n = int(Q["n_queries"].median()) if "n_queries" in Q else 0
    fig.suptitle(f"DSB static, workload {wl}: per-query q-error by column and model (~{n:,} queries per box); "
                 "box P25–P75, line median, whiskers P5–P95, ◆ mean")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_dsb_static(plt):
    df = static_summaries(RES / "dsb_expanded" / "static", ["bin", "workload", "column"])
    if df.empty:
        NOTES.append("dsb_static: no results — skipped")
        return df
    for wl, d in df.groupby("workload"):
        n = d["column"].nunique()
        box_figure(plt, d, f"DSB static, workload {wl} ({n} columns)", OUT / "dsb_static" / f"box_models_{wl}.png",
                   "columns")
        by_dataset_figure(plt, d, "column", f"DSB static, workload {wl} — per column (panel = DSB column)",
                          OUT / "dsb_static" / f"by_column_{wl}.png", short)
    Q = dsb_query_quantiles(df)
    for wl, q in Q.groupby("workload"):
        dataset_box_figure(plt, q, wl, OUT / "dsb_static" / f"box_{wl}.png")
    return df


def fig_reproduction(plt):
    a = static_summaries(RES / "methodology" / "reproduction" / "static", ["bin", "scenario", "dist"])
    b = static_summaries(RES / "methodology" / "reproduction" / "static_w100_thesis", ["bin", "scenario", "dist"])
    df = pd.concat([a, b], ignore_index=True)
    if df.empty:
        NOTES.append("reproduction: no results — skipped")
        return df
    for sc, d in df.groupby("scenario"):
        box_figure(plt, d, f"Thesis reproduction, {sc} (5 distributions)", OUT / "reproduction" / f"box_{sc}.png",
                   "distributions")
        by_dataset_figure(plt, d, "dist", f"Thesis reproduction, {sc} — per distribution",
                          OUT / "reproduction" / f"by_dist_{sc}.png")
    return df


def fig_ablation(plt, dsb, repro):
    parts = []
    for name, df, keys in [("DSB static", dsb, ["bin", "workload", "column"]),
                           ("reproduction", repro, ["bin", "scenario", "dist"])]:
        if df.empty:
            continue
        o = df[df.model == "Hybrid"].set_index(keys)
        c = df[df.model == "Hybrid-Corrected"].set_index(keys)
        j = o.join(c, lsuffix="_o", rsuffix="_c", how="inner")
        for y in ("median_qerror", "p95_qerror", "size_bytes"):
            parts.append(pd.DataFrame({"category": name, "bin": j.index.get_level_values("bin"),
                                       "metric": y, "ratio": j[f"{y}_c"] / j[f"{y}_o"]}))
    fd = RES / "fd_reduction" / "original_vs_corrected.csv"
    if fd.exists():
        r = pd.read_csv(fd)
        if "pair" in r:
            r = r[r["pair"].astype(str).str.startswith("Hybrid") & ~r["pair"].astype(str).str.contains("Merge")]
        for y in ("median_qerror", "p95_qerror", "size_bytes"):
            co, cc = f"orig_{y}", f"corr_{y}"
            if co in r and cc in r:
                parts.append(pd.DataFrame({"category": "FD reduction", "bin": "fd_r*", "metric": y,
                                           "ratio": r[cc] / r[co]}))
    if not parts:
        NOTES.append("ablation_corrected: no paired results — skipped")
        return
    R = pd.concat(parts, ignore_index=True).dropna()
    out = OUT / "ablation_corrected"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "corrected_over_original_ratios.csv", index=False)
    cats = [c for c in ["reproduction", "DSB static", "FD reduction"] if c in set(R.category)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    rng = np.random.default_rng([42, 98])
    for ax, (y, lab) in zip(axes, [("median_qerror", "median q-error"), ("p95_qerror", "P95 q-error"),
                                   ("size_bytes", "model size")]):
        data = [R[(R.category == c) & (R.metric == y)]["ratio"].to_numpy() for c in cats]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch in bp["boxes"]:
            patch.set(facecolor=COLORS["Hybrid-Corrected"], alpha=0.35, edgecolor=COLORS["Hybrid-Corrected"])
        for med in bp["medians"]:
            med.set(color="black")
        for i, v in enumerate(data, 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=COLORS["Hybrid-Corrected"], zorder=3)
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(cats, data)], fontsize=8)
        if y == "size_bytes":
            ax.set_yscale("log")
        ax.set_title(f"{lab}: Corrected / Original" + (" (log)" if y == "size_bytes" else ""), fontsize=9)
    fig.suptitle("Ablation Hybrid-Corrected vs Hybrid (Original), paired configurations; "
                 "ratio < 1 = Corrected better / smaller")
    fig.tight_layout()
    fig.savefig(out / "box_corrected_over_original.png")
    plt.close(fig)


DSB_DATA = ROOT / "data" / "dsb_expanded"
SCEN_POINTS = {"displacement": [(0, "0 %"), (10, "50 %"), (20, "100 %")],
               "additive_200": [(0, "0 %"), (20, "+100 %"), (40, "+200 %")]}


def _binned(freq, mn, lo, hi, n_bins):
    """Rows per value averaged over n_bins equal bins of [lo, hi] (freq is dense over [mn, mn+len-1])."""
    full = np.zeros(hi - lo + 1)
    a, b = max(lo, mn), min(hi, mn + len(freq) - 1)
    if a <= b:
        full[a - lo:b - lo + 1] = freq[a - mn:b - mn + 1]
    edges = np.linspace(0, len(full), n_bins + 1).astype(int)
    sums = np.add.reduceat(full, edges[:-1])
    widths = np.diff(edges)
    return lo + (edges[:-1] + widths / 2), sums / np.maximum(widths, 1)


def fig_dsb_distributions(plt):
    """The 8 DSB columns as used: data (freq of meta.pkl / step_*.pkl — ground truth, plotted only here)."""
    import pickle
    man = RES / "dsb_expanded" / "dataset_manifest.csv"
    if not DSB_DATA.exists() or not man.exists():
        NOTES.append("dsb_distributions: data/dsb_expanded or dataset_manifest.csv missing — skipped")
        return
    prof = pd.read_csv(man).set_index("key")
    cols = sorted(c.name for c in DSB_DATA.iterdir() if (c / "meta.pkl").exists())
    out = OUT / "dsb_distributions"
    out.mkdir(parents=True, exist_ok=True)
    blue = COLORS["EquiWidth"]

    # 1. initial data: whole domain (left) and a 400-value zoom around the median (right)
    fig, axes = plt.subplots(len(cols), 2, figsize=(13, 2.3 * len(cols)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1]})
    for i, c in enumerate(cols):
        mn, mx, N, freq, sample, k, skew, kurt = pickle.load(open(DSB_DATA / c / "meta.pkl", "rb"))
        x, y = _binned(freq, mn, mn, mx, min(400, mx - mn + 1))
        ax = axes[i, 0]
        ax.fill_between(x, y, step="mid", color=blue, alpha=0.5, lw=0)
        ax.set_yscale("log")
        ax.set_xlim(mn, mx)
        r = prof.loc[c] if c in prof.index else None
        info = (f"N={N:,}  NDV={int(r['ndv']):,}  domain={mx - mn + 1:,}  skew={skew:.2f}  "
                f"top-1={100 * r['top1_mass']:.1f} %  FD k={k}" if r is not None else f"N={N:,} FD k={k}")
        ax.set_title(f"{c}  —  {info}", fontsize=8, loc="left")
        ax.set_ylabel("rows / value (log)", fontsize=7)
        cdf = np.cumsum(freq) / freq.sum()
        med = mn + int(np.searchsorted(cdf, 0.5))
        lo, hi = max(mn, med - 200), min(mx, med + 199)
        ax2 = axes[i, 1]
        ax2.bar(np.arange(lo, hi + 1), freq[lo - mn:hi - mn + 1], width=1.0, color=blue, alpha=0.8, lw=0)
        bw = int(np.ceil((mx - mn + 1) / k))  # as EquiWidthHistogram: ceil(domain / k) -> ceil(domain / bw) buckets
        for e in np.arange(mn, mx + 1, bw):
            if lo <= e <= hi:
                ax2.axvline(e, color="black", lw=0.5, ls=":")
        n_b = int(np.ceil((mx - mn + 1) / bw))
        ax2.set_title(f"zoom: 400 values around the median (dotted = FD bucket edges: width {bw}, {n_b} buckets)",
                      fontsize=7)
        ax2.set_xlim(lo, hi)
        for a in (ax, ax2):
            a.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[-1, 0].set_xlabel("value (decimals as integer cents)")
    fig.suptitle("DSB SF=1 columns used in the experiments — initial data (step 0 / static)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "initial_distributions.png", bbox_inches="tight")
    plt.close(fig)

    # 2-3. drift: the distribution at the start, middle and end of each scenario (same x range per column)
    shades = ["#1f77b4", "#ff7f0e", "#d62728"]
    for sc, points in SCEN_POINTS.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), squeeze=False)
        for ax, c in zip(axes.flat, cols):
            steps = []
            for st, lab in points:
                f = DSB_DATA / c / sc / f"step_{st}.pkl"
                if f.exists():
                    mn, mx, N, freq, sample, k = pickle.load(open(f, "rb"))
                    steps.append((lab, mn, mx, N, freq))
            if not steps:
                continue
            lo, hi = min(s[1] for s in steps), max(s[2] for s in steps)
            for (lab, mn, mx, N, freq), col in zip(steps, shades):
                x, y = _binned(freq, mn, lo, hi, min(300, hi - lo + 1))
                ax.plot(x, y, color=col, lw=1.1, drawstyle="steps-mid", label=f"{lab}  N={N:,}")
            ax.set_yscale("log")
            ax.set_title(short(c), fontsize=8)
            ax.legend(fontsize=6)
        axes[0, 0].set_ylabel("rows / value (log)")
        axes[1, 0].set_ylabel("rows / value (log)")
        what = ("rows replaced by the Zipf-popularity target, N constant" if sc == "displacement"
                else "target rows added on top, N grows to 3x")
        fig.suptitle(f"DSB drift data, {SCEN_LABEL[sc]} — {what}")
        fig.tight_layout()
        fig.savefig(out / f"drift_{sc}.png")
        plt.close(fig)

    # 4. how much the distribution actually changes per step (TV distance to step 0)
    ch = RES / "dsb_expanded" / "drift_distribution_change.csv"
    if ch.exists():
        d = pd.read_csv(ch)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, sc in zip(axes, SCEN_LABEL):
            for c in cols:
                e = d[(d["column"] == c) & (d["scenario"] == sc)].sort_values("drift_pct")
                ax.plot(e["drift_pct"], e["tv_distance"], marker=".", lw=1.2, label=short(c))
            ax.set_title(SCEN_LABEL[sc], fontsize=9)
            ax.set_xlabel("drift %")
            ax.set_ylabel("total-variation distance to step 0")
        axes[0].legend(fontsize=7)
        fig.suptitle("Actual distribution change of the DSB drift data (0 = identical to the initial column)")
        fig.tight_layout()
        fig.savefig(out / "drift_tv_distance.png")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-bins", nargs="+", default=["fd"])
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    for b in a.drift_bins:
        fig_drift(plt, b)
    dsb = fig_dsb_static(plt)
    repro = fig_reproduction(plt)
    fig_ablation(plt, dsb, repro)
    fig_dsb_distributions(plt)
    files = sorted(str(f.relative_to(OUT)) for f in OUT.rglob("*.png"))
    (OUT / "README.md").write_text(
        "# Figures of the finished categories\n\nGenerated by `python -m paper_experiments.figures_categories`. "
        "FD-reduction figures: `results/paper_experiments/fd_reduction/figures/`.\n\n## Notes\n\n"
        + "".join(f"- {n}\n" for n in NOTES or ["none"]) + "\n## Files\n\n" + "".join(f"- `{f}`\n" for f in files))
    print(f"{len(files)} figures -> {OUT}")
    for n in NOTES:
        print("NOTE:", n)


if __name__ == "__main__":
    main()

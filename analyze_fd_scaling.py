"""
Статистика: при каком числе бакетов (FD x 1.0 / 1.2 / 1.4 / 2.0) Hybrid статистически точнее остальных.

Входы — готовые папки экспериментов (static/ + drift/), по одной на множитель:
    x1.0 -> results/exp2_freedman_diaconis   (FD; совпадает с fd*1 бит в бит)
    x1.2 -> results/exp3_fd_scaling/fd_x1.2  (run_experiments.py --buckets 'fd*1.2', ...)
    ...

Три уровня проверки (везде alpha = 0.05, односторонние альтернативы, поправка Холма
на 7 конкурентов внутри одной семьи):

1. Статика, внутри датасета (static_per_dataset.csv). Единица — запрос. Пары
   (q_Hybrid, q_X) на одних и тех же запросах; знаково-ранговый тест Уилкоксона на
   d = ln q_Hybrid - ln q_X. Hybrid «лучший» на датасете, если значимо лучше всех 7;
   «делит первое», если ни один конкурент не значимо лучше; иначе «побит».
2. Статика, по набору датасетов (static_aggregate.csv). Единица — датасет
   (Demsar 2006): Уилкоксон по ln медианы q-error Hybrid vs X; средний ранг; Фридман.
3. Дрифт (drift_stats.csv). Единица — шаг дрифта (step > 0, ошибка после действия на
   eval-половине, как в export_results.drift_summary). Внутри сценария шаги парные;
   шаги одного сценария не независимы (временной ряд), поэтому p — ориентир, не строгий.

    poetry run python analyze_fd_scaling.py            # пути по умолчанию
    poetry run python analyze_fd_scaling.py --runs x1.0=results/exp2_freedman_diaconis x2.0=...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

HYBRID = "Hybrid"
ALPHA = 0.05
DEFAULT_RUNS = {
    "x1.0": "results/exp2_freedman_diaconis",
    "x1.2": "results/exp3_fd_scaling/fd_x1.2",
    "x1.4": "results/exp3_fd_scaling/fd_x1.4",
    "x2.0": "results/exp3_fd_scaling/fd_x2.0",
}
SYNTH_NARROW = ["synth_hard_narrow", "synth_hard_narrow_data_driven", "synth_simple_narrow"]
SYNTH_WIDE = ["synth_hard_wide", "synth_simple_wide"]
REAL = ["dsb_static", "stats_ceb_official", "stats_ceb_driven"]
GROUPS = {
    "all": SYNTH_NARROW + SYNTH_WIDE + REAL,
    "synth_narrow": SYNTH_NARROW,
    "synth_wide": SYNTH_WIDE,
    "real": REAL,
    **{b: [b] for b in SYNTH_NARROW + SYNTH_WIDE + REAL},
}


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def one_sided(d):
    """(p that Hybrid is better, p that Hybrid is worse) from paired log-differences d."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if np.count_nonzero(d) == 0:
        return 1.0, 1.0
    return (float(wilcoxon(d, alternative="less").pvalue),
            float(wilcoxon(d, alternative="greater").pvalue))


# ---------------------------------------------------------------- static, per dataset
def load_qerrors(ds_dir: Path):
    """Per-query q-errors per model: <workload stem>_<Model>.csv next to summary_<Model>.json."""
    out = {}
    for js in sorted(ds_dir.glob("summary_*.json")):
        model = js.stem.removeprefix("summary_")
        files = sorted(ds_dir.glob(f"*_{model}.csv"))
        if len(files) != 1:
            print(f"  [skip] {ds_dir} {model}: per-query files {files}")
            continue
        out[model] = pd.read_csv(files[0], usecols=["Query_ID", "Q_Error"]).sort_values("Query_ID")
    return out


def static_per_dataset(label, root: Path):
    rows = []
    for bench_dir in sorted(p for p in (root / "static").iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            q = load_qerrors(ds_dir)
            if HYBRID not in q:
                print(f"  [skip] {ds_dir}: no Hybrid")
                continue
            h = q[HYBRID]
            lh = np.log(h["Q_Error"].to_numpy())
            for model, df in q.items():
                if model == HYBRID:
                    continue
                assert np.array_equal(df["Query_ID"].to_numpy(), h["Query_ID"].to_numpy()), (ds_dir, model)
                lx = np.log(df["Q_Error"].to_numpy())
                d = lh - lx
                p_better, p_worse = one_sided(d)
                nz = d[d != 0]
                rows.append({
                    "mult": label, "benchmark": bench_dir.name, "dataset": ds_dir.name, "competitor": model,
                    "n_queries": len(d),
                    "median_hybrid": float(np.exp(np.median(lh))), "median_comp": float(np.exp(np.median(lx))),
                    "p95_hybrid": float(np.exp(np.quantile(lh, 0.95))), "p95_comp": float(np.exp(np.quantile(lx, 0.95))),
                    # share of queries where Hybrid is strictly closer (ties excluded)
                    "hybrid_win_rate": float((nz < 0).mean()) if len(nz) else np.nan,
                    "p_hybrid_better": p_better, "p_hybrid_worse": p_worse,
                })
    df = pd.DataFrame(rows)
    key = ["mult", "benchmark", "dataset"]
    df["p_better_holm"] = df.groupby(key)["p_hybrid_better"].transform(holm)
    df["p_worse_holm"] = df.groupby(key)["p_hybrid_worse"].transform(holm)
    df["outcome"] = np.select(
        [df["p_better_holm"] < ALPHA, df["p_worse_holm"] < ALPHA], ["win", "loss"], "tie")
    return df


def dataset_verdicts(per):
    g = per.groupby(["mult", "benchmark", "dataset"])
    v = g.agg(
        n_queries=("n_queries", "first"),
        median_hybrid=("median_hybrid", "first"),
        best_competitor_median=("median_comp", "min"),
        wins=("outcome", lambda s: int((s == "win").sum())),
        ties=("outcome", lambda s: int((s == "tie").sum())),
        losses=("outcome", lambda s: int((s == "loss").sum())),
    ).reset_index()
    best = per.loc[per.groupby(["mult", "benchmark", "dataset"])["median_comp"].idxmin(),
                   ["mult", "benchmark", "dataset", "competitor"]].rename(columns={"competitor": "best_competitor"})
    v = v.merge(best, on=["mult", "benchmark", "dataset"])
    n_comp = v["wins"] + v["ties"] + v["losses"]
    v["verdict"] = np.select([v["wins"] == n_comp, v["losses"] == 0], ["best", "co-best"], "beaten")
    return v


# ---------------------------------------------------------------- static, across datasets
def static_aggregate(per):
    rows = []
    for (label, group), sub in [((m, g), per[(per["mult"] == m) & per["benchmark"].isin(b)])
                                for m in per["mult"].unique() for g, b in GROUPS.items()]:
        if sub.empty:
            continue
        # dataset x model matrix of median q-error (Hybrid + competitors)
        med = sub.pivot_table(index=["benchmark", "dataset"], columns="competitor", values="median_comp")
        med[HYBRID] = sub.groupby(["benchmark", "dataset"])["median_hybrid"].first()
        p95 = sub.pivot_table(index=["benchmark", "dataset"], columns="competitor", values="p95_comp")
        p95[HYBRID] = sub.groupby(["benchmark", "dataset"])["p95_hybrid"].first()
        ranks = med.rank(axis=1, method="average")
        try:
            friedman_p = friedmanchisquare(*[med[c] for c in med.columns]).pvalue if len(med) > 2 else np.nan
        except ValueError:
            friedman_p = np.nan
        comp_rows = []
        for c in med.columns.drop(HYBRID):
            dm = np.log(med[HYBRID]) - np.log(med[c])
            dp = np.log(p95[HYBRID]) - np.log(p95[c])
            pb, pw = one_sided(dm)
            pb95, pw95 = one_sided(dp)
            comp_rows.append({
                "mult": label, "group": group, "competitor": c, "n_datasets": len(med),
                "hybrid_better_datasets": int((dm < 0).sum()), "equal_datasets": int((dm == 0).sum()),
                "hybrid_worse_datasets": int((dm > 0).sum()),
                "gmean_median_hybrid": float(np.exp(np.log(med[HYBRID]).mean())),
                "gmean_median_comp": float(np.exp(np.log(med[c]).mean())),
                "p_better": pb, "p_worse": pw, "p_better_p95": pb95, "p_worse_p95": pw95,
                "hybrid_avg_rank": float(ranks[HYBRID].mean()), "comp_avg_rank": float(ranks[c].mean()),
                "friedman_p": friedman_p,
            })
        cr = pd.DataFrame(comp_rows)
        for col in ["p_better", "p_worse", "p_better_p95", "p_worse_p95"]:
            cr[col + "_holm"] = holm(cr[col])
        cr["outcome_median"] = np.select([cr["p_better_holm"] < ALPHA, cr["p_worse_holm"] < ALPHA], ["win", "loss"], "tie")
        cr["outcome_p95"] = np.select([cr["p_better_p95_holm"] < ALPHA, cr["p_worse_p95_holm"] < ALPHA], ["win", "loss"], "tie")
        rows.append(cr)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- drift
def drift_stats(label, root: Path):
    rows = []
    ddir = root / "drift"
    if not ddir.exists():
        return pd.DataFrame()
    for scen in sorted(p for p in ddir.iterdir() if p.is_dir()):
        curves = {}
        for mdir in sorted(p for p in scen.iterdir() if p.is_dir()):
            f = mdir / "drift_steps.csv"
            if f.exists():
                model = mdir.name.rsplit("_", 1)[-1]
                s = pd.read_csv(f)
                curves[model] = s[s["step"] > 0].set_index("step")
        if HYBRID not in curves:
            continue
        h = curves[HYBRID]
        crow = []
        for model, s in curves.items():
            if model == HYBRID:
                continue
            steps = h.index.intersection(s.index)
            d = np.log(h.loc[steps, "median_qerror"]) - np.log(s.loc[steps, "median_qerror"])
            d95 = np.log(h.loc[steps, "p95_qerror"]) - np.log(s.loc[steps, "p95_qerror"])
            pb, pw = one_sided(d)
            pb95, pw95 = one_sided(d95)
            crow.append({
                "mult": label, "scenario": scen.name, "competitor": model, "n_steps": len(steps),
                "buckets_hybrid": int(h["n_buckets"].iloc[0]),
                "mean_median_hybrid": float(h.loc[steps, "median_qerror"].mean()),
                "mean_median_comp": float(s.loc[steps, "median_qerror"].mean()),
                "hybrid_better_steps": int((d < 0).sum()), "hybrid_worse_steps": int((d > 0).sum()),
                "p_better": pb, "p_worse": pw, "p_better_p95": pb95, "p_worse_p95": pw95,
            })
        cr = pd.DataFrame(crow)
        for col in ["p_better", "p_worse", "p_better_p95", "p_worse_p95"]:
            cr[col + "_holm"] = holm(cr[col])
        cr["outcome_median"] = np.select([cr["p_better_holm"] < ALPHA, cr["p_worse_holm"] < ALPHA], ["win", "loss"], "tie")
        cr["outcome_p95"] = np.select([cr["p_better_p95_holm"] < ALPHA, cr["p_worse_p95_holm"] < ALPHA], ["win", "loss"], "tie")
        cr["hybrid_place"] = int((cr["mean_median_comp"] < cr["mean_median_hybrid"].iloc[0]).sum()) + 1
        rows.append(cr)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs", nargs="+", default=[f"{k}={v}" for k, v in DEFAULT_RUNS.items()],
                    help="label=experiment_dir pairs")
    ap.add_argument("--out-dir", default="results/exp3_fd_scaling")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per, drift = [], []
    for spec in args.runs:
        label, path = spec.split("=", 1)
        root = Path(path)
        if not root.exists():
            print(f"[skip] {label}: {root} missing")
            continue
        print(f"[{label}] {root}")
        per.append(static_per_dataset(label, root))
        drift.append(drift_stats(label, root))
    per = pd.concat(per, ignore_index=True)
    verdicts = dataset_verdicts(per)
    agg = static_aggregate(per)
    drift = pd.concat(drift, ignore_index=True)

    per.to_csv(out / "static_per_dataset.csv", index=False)
    verdicts.to_csv(out / "static_dataset_verdicts.csv", index=False)
    agg.to_csv(out / "static_aggregate.csv", index=False)
    drift.to_csv(out / "drift_stats.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nStatic, per dataset: Hybrid verdict counts")
    print(verdicts.pivot_table(index="mult", columns="verdict", values="dataset", aggfunc="count", fill_value=0))
    print("\nStatic, across all datasets (median q-error): outcome per competitor")
    print(agg[agg["group"] == "all"].pivot(index="mult", columns="competitor", values="outcome_median"))
    if not drift.empty:
        print("\nDrift: Hybrid wins/ties/losses over competitors per scenario (median)")
        print(drift.groupby(["mult", "scenario"])["outcome_median"].value_counts().unstack(fill_value=0))
    print(f"\n-> {out}/static_per_dataset.csv, static_dataset_verdicts.csv, static_aggregate.csv, drift_stats.csv")


if __name__ == "__main__":
    main()

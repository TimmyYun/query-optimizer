"""
DSB expansion, part 3: drift snapshots for the selected columns (results/paper_experiments/PLAN.md §3).

    poetry run python -m paper_experiments.dsb_drift_data [--only <table.column> ...] [--force]

Target distribution — "Zipf popularity shift" (f_target), defined on the initial column:
    support S   = values with f_init > 0 (only values that exist in the column: valid keys / dates / amounts)
    ranks       = default_rng([42, 3]).permutation(|S|): value S[perm[j]] gets rank j + 1
    P(rank r)   ∝ r^-1 (Zipf, s = 1)
    f_target    = default_rng([42, 4]).multinomial(N_init, P) placed on S[perm]
So the target keeps the column's value set but moves the popularity onto randomly chosen values.

Scenarios (thesis §4.3.2 formulas, rounded to whole rows, trimmed to the active range like
data/generate_shifts.py):
    displacement/step_i.pkl   f = round((1 - a) f_init + a f_target), a = i / 20, i = 0..20  (N constant)
    additive_200/step_i.pkl   f = round(f_init + a f_target),         a = i / 20, i = 0..40  (N -> 3 N_init)
Step i is drift_pct = 5 i % of the initial table (displacement: rows replaced; additive: rows added).

Reservoir per step (thesis §4.3.3): Bernoulli trial per row with p = min(1, 1.5 S / N), then a uniform
subset of S = 100 000 without replacement. Done per value (Binomial(f_v, p), then a multivariate
hypergeometric draw of S), which is the same distribution as the per-row procedure; stream
default_rng([42, 5, scenario, i]). k per step: data.datasets.freedman_diaconis_bins(sample, mn, mx, N).
Per-step workload step_<i>_workload_driven_1000000.csv: the dd_narrow rule of dsb_prepare on the step's
sample (thesis §4.4.4), stream default_rng([42, 6, scenario, i]).

drift_distribution_change.csv: per step, distance of the step's distribution from step 0 —
total variation, KS statistic (max CDF gap), Wasserstein-1 / domain width, the share of rows that come
from f_target, and TV(f_init, f_target). TV > 0 proves the drift is not a repetition of the initial data.
"""
import argparse
import pickle

import numpy as np
import pandas as pd

from data.datasets import freedman_diaconis_bins
from paper_experiments.dsb_columns import ROOT, SELECTED_ALL
from paper_experiments.dsb_prepare import N_QUERIES, dd_narrow, save_ranges
from paper_experiments.provenance import write_manifest

DATA = ROOT / "data" / "dsb_expanded"
RES = ROOT / "results" / "paper_experiments" / "dsb_expanded"
SAMPLE_SIZE = 100_000
ZIPF_S = 1.0
SCENARIOS = {"displacement": {"code": 1, "steps": 20, "max_alpha": 1.0},
             "additive_200": {"code": 2, "steps": 40, "max_alpha": 2.0}}


def zipf_target(f_init: np.ndarray, N: int) -> np.ndarray:
    support = np.nonzero(f_init)[0]
    perm = np.random.default_rng([42, 3]).permutation(len(support))
    p = 1.0 / np.arange(1, len(support) + 1) ** ZIPF_S
    p /= p.sum()
    counts = np.random.default_rng([42, 4]).multinomial(int(N), p)
    f = np.zeros_like(f_init, dtype=np.int64)
    f[support[perm]] = counts
    return f


def draw_sample(freq: np.ndarray, mn: int, N: int, rng) -> np.ndarray:
    p = min(1.0, SAMPLE_SIZE * 1.5 / max(N, 1))
    kept = rng.binomial(freq.astype(np.int64), p)
    if kept.sum() > SAMPLE_SIZE:
        kept = rng.multivariate_hypergeometric(kept, SAMPLE_SIZE)
    s = np.repeat(np.arange(len(freq), dtype=np.int64) + mn, kept)
    rng.shuffle(s)
    return s


def distances(f0: np.ndarray, f: np.ndarray) -> dict:
    p0, p = f0 / f0.sum(), f / f.sum()
    c0, c = np.cumsum(p0), np.cumsum(p)
    return {"tv_distance": 0.5 * float(np.abs(p - p0).sum()), "ks_statistic": float(np.abs(c - c0).max()),
            "wasserstein1_over_domain": float(np.abs(c - c0).sum() / len(f0))}


def build(key: str, force=False):
    mn, mx, N, freq, sample, k, *_ = pickle.load(open(DATA / key / "meta.pkl", "rb"))
    f_init = np.asarray(freq, dtype=np.int64)
    f_target = zipf_target(f_init, N)
    tv_target = distances(f_init, f_target)["tv_distance"]
    rows = []
    for sc, cfg in SCENARIOS.items():
        out = DATA / key / sc
        out.mkdir(parents=True, exist_ok=True)
        for i in range(cfg["steps"] + 1):
            a = cfg["max_alpha"] * i / cfg["steps"]
            mixed = f_init + a * f_target if sc == "additive_200" else (1 - a) * f_init + a * f_target
            mixed = np.round(mixed).astype(np.int64)
            step_path, wl_path = out / f"step_{i}.pkl", out / f"step_{i}_workload_driven_{N_QUERIES}.csv"
            active = np.nonzero(mixed)[0]
            lo_i, hi_i = int(active[0]), int(active[-1])
            s_mn, s_mx, s_N = mn + lo_i, mn + hi_i, int(mixed.sum())
            if force or not step_path.exists() or not wl_path.exists():
                trimmed = mixed[lo_i: hi_i + 1]
                s = draw_sample(trimmed, s_mn, s_N, np.random.default_rng([42, 5, cfg["code"], i]))
                s_k = freedman_diaconis_bins(s, s_mn, s_mx, s_N)
                with open(step_path, "wb") as f:
                    pickle.dump((s_mn, s_mx, s_N, trimmed, s, s_k), f)
                save_ranges(*dd_narrow(s, s_mn, s_mx, N_QUERIES, np.random.default_rng([42, 6, cfg["code"], i])),
                            wl_path)
            else:
                s_k = pickle.load(open(step_path, "rb"))[5]
            target_rows = a * f_target.sum()
            rows.append({"column": key, "scenario": sc, "step": i, "drift_pct": round(100 * a, 2), "alpha": a,
                         "N": s_N, "mn": s_mn, "mx": s_mx, "fd_k": int(s_k),
                         "target_row_share": float(target_rows / s_N),
                         **distances(f_init, mixed), "tv_init_vs_target": tv_target})
            print(f"[{key}] {sc} step {i:>2} a={a:.2f} N={s_N:,} k={s_k} TV={rows[-1]['tv_distance']:.4f}", flush=True)
        pd.DataFrame([r for r in rows if r["scenario"] == sc]).to_csv(out / "drift_distribution_change.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    keys = args.only or list(SELECTED_ALL)
    rows = []
    for key in keys:
        rows += build(key, force=args.force)
    RES.mkdir(parents=True, exist_ok=True)
    allp = RES / "drift_distribution_change.csv"
    df = pd.DataFrame(rows)
    if allp.exists() and args.only:
        old = pd.read_csv(allp)
        df = pd.concat([old[~old["column"].isin(keys)], df], ignore_index=True)
    df.to_csv(allp, index=False)
    write_manifest(DATA / "manifest_drift.json", part="dsb_expanded/drift-data", zipf_s=ZIPF_S,
                   sample_size=SAMPLE_SIZE, scenarios=SCENARIOS, streams={"rank_perm": [42, 3], "target": [42, 4],
                   "sample": "[42, 5, scenario, step]", "workload": "[42, 6, scenario, step]"}, columns=keys)


if __name__ == "__main__":
    main()

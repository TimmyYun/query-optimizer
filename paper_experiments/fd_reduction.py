#!/usr/bin/env python3
"""
FD bucket reduction: can Hybrid FD CDF compensate fewer Freedman–Diaconis buckets by modelling
the in-bucket distribution more accurately?

Per dataset (static protocol of run_*.py, in one process, no subprocess per point):
  K_FD = meta.pkl k  (FD with N table rows, capped at 2000)
  r in {0, 10, ..., 90} %,  K = max(1, round(K_FD * (1 - r)))
  same sample (meta.pkl reservoir), same workload (all queries after zero-truth filtering),
  seed 42 everywhere.

Configurations (bin_strategy):
  fd_r<rr>          EquiWidth(K), EquiDepth(statistics_target=K), TKHist(K), EntropyHist-MEA(K),
                    Hybrid (canonical, K equal-width buckets) — every model by its own definition
  fd_merge_r<rr>    Hybrid-Merge   [MODIFICATION of Hybrid FD CDF: non-equal-width buckets]
                    EquiWidth-Merge [control: the same merged boundaries, uniform inside a bucket]
                    Boundaries: the K_FD equal-width buckets are merged greedily, always the
                    adjacent pair with the smallest merge cost
                        cost(A, B) = w_A * w_B / (w_A + w_B) * (d_A - d_B)^2,  d = count / width,
                    i.e. the increase of the squared error of a piecewise-constant density
                    (Ward's criterion at bucket resolution; ties -> leftmost pair), until K buckets
                    remain. Counts are recomputed from the sample as in EquiWidth (int(n * N/|S|)).
                    Each bucket then stores its boundary, so n_values counts 2 numbers per bucket.
  mem_match_K<k>    EquiWidth / EquiDepth / TKHist / EntropyHist-MEA sized so that their n_values is
                    the closest achievable to the n_values M of Hybrid at K = k (same memory budget);
                    budget_reachable = False when the model cannot grow to M (e.g. limited by the
                    number of distinct sample values).

Memory = n_values() * 8 bytes, the repository's persistent-state accounting for every model.
Scores exactly as benchmark_utils.evaluate_and_save: truth and estimate clipped to >= 1 row,
q-error on selectivities, MAE in rows on the same clipped values; inference through
benchmark_utils.run_queries (scalar predict, time.perf_counter per call).

Built models are pickled to --models-dir for the latency pass (--part latency): a single process
re-times every configuration on the same fixed 100 000-query subset, interleaving models,
so latency is compared under the same load (loadavg recorded).

    poetry run python -m paper_experiments.fd_reduction --part accuracy --group synthetic dsb --workers 3 \
        --models-dir <scratch>/fd_models
    poetry run python -m paper_experiments.fd_reduction --part accuracy --group dsb_expanded --workers 3 ...
    poetry run python -m paper_experiments.fd_reduction --part latency --models-dir <scratch>/fd_models
    poetry run python -m paper_experiments.fd_reduction_report
"""
import argparse
import heapq
import json
import math
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_utils import load_and_filter_workload, run_queries  # noqa: E402
from models import (  # noqa: E402
    EntropyHistogram1D,
    EquiDepthHistogram,
    EquiWidthHistogram,
    HistogramConfig,
    HybridEstimator,
    TKHistogram,
    q_error_vec,
)
from models.common import Bucket  # noqa: E402
from paper_experiments.provenance import (  # noqa: E402
    MODEL_SOURCES,
    SEED,
    file_sha256,
    model_version,
    sources_sha,
    write_manifest,
)
from paper_experiments.schema import ID_COLUMNS  # noqa: E402

OUT = ROOT / "results/paper_experiments/fd_reduction"
REDUCTIONS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
POINTS = 200  # CDF points per bucket, as run_hybrid.py
LATENCY_QUERIES = 100_000
SAME_K_MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "Hybrid"]
MERGE_MODELS = ["Hybrid-Merge", "EquiWidth-Merge"]
MATCH_MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA"]
SYNTH = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
EXPERIMENT_ID = "fd_reduction"
MERGE_SOURCES = ["paper_experiments/fd_reduction.py"]


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

def dataset_specs(groups):
    specs = []
    if "synthetic" in groups:
        for kind in ("simple", "hard"):
            for d in SYNTH:
                ds = f"data/generated/60000000_{kind}/{d}"
                specs.append(dict(dir=ds, workload=f"{ds}/workload_driven_1000000.csv", dataset=f"{kind}/{d}",
                                  table=f"synthetic_{kind}", column=d, slug=f"{kind}__{d}"))
    if "dsb" in groups:
        for c in ("ws_wholesale_cost", "ws_list_price"):
            ds = f"data/dsb/{c}"
            specs.append(dict(dir=ds, workload=f"{ds}/workload_official.csv", dataset=f"dsb/{c}",
                              table="web_sales", column=c, slug=f"dsb__{c}"))
    if "dsb_expanded" in groups:
        for meta in sorted((ROOT / "data/dsb_expanded").glob("*/meta.pkl")):
            name = meta.parent.name  # <table>.<column>
            wl = meta.parent / "workload_dd_narrow.csv"
            if not wl.exists():
                continue
            table, _, column = name.partition(".")
            specs.append(dict(dir=str(meta.parent.relative_to(ROOT)), workload=str(wl.relative_to(ROOT)),
                              dataset=f"dsb_expanded/{name}", table=table, column=column, slug=f"dsbx__{name}"))
    return specs


class Ctx:
    """Everything one dataset's configurations share: metadata, sample, workload, truth."""

    def __init__(self, spec):
        self.spec = spec
        with open(ROOT / spec["dir"] / "meta.pkl", "rb") as f:
            self.mn, self.mx, self.N, freq, sample, self.k_fd, *_ = pickle.load(f)
        self.mn, self.mx, self.N, self.k_fd = int(self.mn), int(self.mx), int(self.N), int(self.k_fd)
        self.sample = np.asarray(sample, dtype=np.int64)
        self.sample_sorted = np.sort(self.sample)
        self.domain = self.mx - self.mn + 1
        self.scale = self.N / len(self.sample)
        # sample-only density, scaled to N (as run_hybrid.py); freq is used only for the truth below
        self.est_freq = np.bincount(self.sample - self.mn, minlength=self.domain) * self.scale
        self.queries, y_true_sel = load_and_filter_workload(str(ROOT / spec["workload"]), self.mn, self.mx, freq)
        self.true_counts = np.maximum(np.rint(y_true_sel * self.N), 1.0)
        self.n_distinct_sample = int(len(np.unique(self.sample)))
        del freq


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def sample_count(ctx, lo, hi):
    n = np.searchsorted(ctx.sample_sorted, hi, side="right") - np.searchsorted(ctx.sample_sorted, lo, side="left")
    return int(n * ctx.scale)  # same rounding as EquiWidthHistogram.build_from_sample


def merge_buckets(ctx, buckets, k_target):
    """Greedy adjacent merging by the density-SSE (Ward) cost; see the module docstring."""
    n = len(buckets)
    lo = [int(b.lo) for b in buckets]
    hi = [int(b.hi) for b in buckets]
    c = [float(b.count) for b in buckets]
    w = [float(b.hi - b.lo + 1) for b in buckets]
    if k_target >= n:
        return [Bucket(lo[i], hi[i], count=sample_count(ctx, lo[i], hi[i])) for i in range(n)]
    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(n - 1))
    alive = [True] * n
    ver = [0] * n

    def cost(i, j):
        di, dj = c[i] / w[i], c[j] / w[j]
        return w[i] * w[j] / (w[i] + w[j]) * (di - dj) ** 2

    heap = [(cost(i, i + 1), i, i + 1, 0, 0) for i in range(n - 1)]
    heapq.heapify(heap)
    remaining = n
    while remaining > k_target and heap:
        _, i, j, vi, vj = heapq.heappop(heap)
        if not (alive[i] and alive[j]) or ver[i] != vi or ver[j] != vj or nxt[i] != j:
            continue
        c[i] += c[j]
        w[i] += w[j]
        hi[i] = hi[j]
        alive[j] = False
        nxt[i] = nxt[j]
        if nxt[j] != -1:
            prv[nxt[j]] = i
        ver[i] += 1
        remaining -= 1
        if prv[i] != -1:
            p = prv[i]
            heapq.heappush(heap, (cost(p, i), p, i, ver[p], ver[i]))
        if nxt[i] != -1:
            q = nxt[i]
            heapq.heappush(heap, (cost(i, q), i, q, ver[i], ver[q]))
    return [Bucket(lo[i], hi[i], count=sample_count(ctx, lo[i], hi[i])) for i in range(n) if alive[i]]


def ew_buckets(ctx, K):
    return EquiWidthHistogram.build_from_sample(ctx.mn, ctx.mx, K, ctx.sample, ctx.N).buckets


def train_hybrid(ctx, buckets):
    h = HybridEstimator(buckets)
    train_s = h.train(ctx.est_freq, ctx.mn, POINTS, np.random.default_rng(SEED))
    return h, train_s


def build(ctx, model, size, merged=None):
    """-> (model object, info dict, build_s, train_s). size = K / statistics_target / B."""
    t0 = time.perf_counter()
    train_s = None
    if model == "EquiWidth":
        h = EquiWidthHistogram.build_from_sample(ctx.mn, ctx.mx, size, ctx.sample, ctx.N)
        info = {"n_buckets": len(h.buckets)}
    elif model == "EquiDepth":
        h = EquiDepthHistogram.build_from_sample(sample=ctx.sample, total_rows=ctx.N, statistics_target=size)
        info = {"n_buckets": h.n_buckets, "statistics_target": size, "num_mcv": h.num_mcv}
    elif model == "TKHist":
        h = TKHistogram.build_from_sample(ctx.mn, ctx.mx, size, ctx.sample, ctx.N)
        info = {"n_buckets": len(h.buckets), "n_topk_entries": h.n_topk_entries}
    elif model == "EntropyHist-MEA":
        h = EntropyHistogram1D.build_from_sample(ctx.sample, ctx.N, HistogramConfig(n_buckets=size))
        info = {"n_buckets": h.B_effective, "n_buckets_requested": size}
    elif model == "Hybrid":
        h, train_s = train_hybrid(ctx, ew_buckets(ctx, size))
        info = {"n_buckets": len(h.buckets)}
    elif model == "Hybrid-Merge":
        h, train_s = train_hybrid(ctx, [Bucket(b.lo, b.hi, b.count) for b in merged])
        info = {"n_buckets": len(h.buckets)}
    elif model == "EquiWidth-Merge":
        h = HybridEstimator([Bucket(b.lo, b.hi, b.count) for b in merged])  # no CDF trained: uniform
        h._bake_vectorized_data()
        info = {"n_buckets": len(h.buckets)}
    else:
        raise ValueError(model)
    return h, info, time.perf_counter() - t0, train_s


def score(ctx, h):
    pred, times, _ = run_queries(h.predict, ctx.queries, ctx.true_counts)
    y_true = ctx.true_counts / ctx.N
    pred_safe = np.maximum(pred, 1.0)
    qe = q_error_vec(y_true, pred_safe / ctx.N)
    return {
        "median_qerror": float(np.median(qe)), "p25_qerror": float(np.percentile(qe, 25)),
        "p75_qerror": float(np.percentile(qe, 75)), "p95_qerror": float(np.percentile(qe, 95)),
        "avg_qerror": float(np.mean(qe)), "mae_rows": float(np.mean(np.abs(pred_safe - ctx.true_counts))),
        "infer_us_median": float(np.median(times)) * 1e6, "infer_us_mean": float(np.mean(times)) * 1e6,
        "infer_us_p95": float(np.percentile(times, 95)) * 1e6, "infer_total_s": float(np.sum(times)),
        "n_queries": len(ctx.queries),
    }


def mversion(model):
    if model == "Hybrid-Merge":
        return model_version("Hybrid-Merge", extra_sources=MERGE_SOURCES)
    if model == "EquiWidth-Merge":  # uniform buckets on merged boundaries, evaluated by HybridEstimator
        return f"EquiWidth-Merge:control-uniform-on-merged-boundaries:{sources_sha(MODEL_SOURCES['Hybrid'] + MERGE_SOURCES)}"
    return model_version(model)


# ----------------------------------------------------------------------------
# Memory matching
# ----------------------------------------------------------------------------

def ew_n_values(ctx, K):
    bw = max(1, int(math.ceil(ctx.domain / K)))
    return 2 + int(math.ceil(ctx.domain / bw))


def match_size(ctx, model, M, cache):
    """Size whose n_values is closest to M (bisection on the monotone n_values(size)); -> (size, reachable)."""
    def f(s):
        if model == "EquiWidth":  # closed form, no build needed
            return ew_n_values(ctx, s)
        if (model, s) not in cache:
            h, *_ = build(ctx, model, s)
            cache[(model, s)] = h.n_values()
        return cache[(model, s)]

    if model == "EntropyHist-MEA":  # n_values = 4 * B_effective; B_effective <= B <= distinct sample values
        s = int(np.clip(round(M / 4), 1, max(1, ctx.n_distinct_sample)))
        return s, f(s) >= 0.9 * M
    # brackets from each model's n_values formula:
    #   EquiWidth 2 + K;  TKHist 2 + K + 2*topk (topk <= 5K);  EquiDepth bounds (<= T+1) + 2*MCV (<= 2T) + 1
    if model == "EquiWidth":
        lo, hi = 1, ctx.domain
    elif model == "TKHist":
        lo, hi = max(1, (M - 2) // 11), max(1, min(ctx.domain, M - 2))
    else:  # EquiDepth (statistics_target)
        lo, hi = max(1, (M - 2) // 3), 100_000
    if f(hi) < M:
        return hi, False
    while lo < hi:  # smallest s with f(s) >= M
        mid = (lo + hi) // 2
        if f(mid) >= M:
            hi = mid
        else:
            lo = mid + 1
    s = lo
    if s > 1 and abs(f(s - 1) - M) <= abs(f(s) - M):
        s -= 1
    return s, True


# ----------------------------------------------------------------------------
# One dataset
# ----------------------------------------------------------------------------

def ids(ctx, model, bin_strategy, n_buckets):
    s = ctx.spec
    row = {
        "experiment_id": EXPERIMENT_ID, "dataset": s["dataset"], "table": s["table"], "column": s["column"],
        "workload": Path(s["workload"]).stem, "drift_type": "none", "drift_pct": 0.0,
        "bin_strategy": bin_strategy, "bucket_count": n_buckets, "model": model,
        "model_version": mversion(model), "seed": SEED, "stage": "STATIC",
    }
    assert list(row) == ID_COLUMNS
    return row


def run_dataset(spec, models_dir):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _run_dataset(spec, Path(models_dir))


def _run_dataset(spec, models_dir):
    t_start = time.perf_counter()
    raw = OUT / "raw" / spec["slug"]
    cdf_dir = OUT / "cdf" / spec["slug"]
    mdir = models_dir / spec["slug"]
    for d in (raw, cdf_dir, mdir):
        d.mkdir(parents=True, exist_ok=True)
    res_path = raw / "results.csv"
    log_path = raw / "results.jsonl"  # one JSON row per finished configuration (resumable)
    done = set()
    if log_path.exists():
        prev = read_log(log_path)
        done = set(zip(prev["bin_strategy"], prev["model"]))
    ctx = Ctx(spec)
    print(f"[{spec['slug']}] K_FD={ctx.k_fd} N={ctx.N} domain={ctx.domain} queries={len(ctx.queries)}", flush=True)
    write_manifest(raw / "manifest.json", experiment_id=EXPERIMENT_ID, dataset=spec["dataset"],
                   dataset_dir=spec["dir"], meta_sha256=file_sha256(Path(spec["dir"]) / "meta.pkl"),
                   workload=spec["workload"], workload_sha256=file_sha256(spec["workload"]),
                   n_queries_after_filter=len(ctx.queries), K_FD=ctx.k_fd, reductions_pct=REDUCTIONS,
                   points_per_bucket=POINTS, models=SAME_K_MODELS + MERGE_MODELS,
                   merge_cost="w_A*w_B/(w_A+w_B)*(d_A-d_B)^2 (Ward, density SSE), greedy, leftmost on ties",
                   memory_accounting="n_values()*8 bytes", sample_size=len(ctx.sample), N=ctx.N)

    def emit(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def run_cfg(model, size, bin_strategy, extra, merged=None, save_table=False):
        if (bin_strategy, model) in done:
            return None
        h, info, build_s, train_s = build(ctx, model, size, merged)
        sc = score(ctx, h)
        fname = f"{model}__{bin_strategy}.pkl"
        with open(mdir / fname, "wb") as f:
            pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
        cdf = None
        if isinstance(h, HybridEstimator) and model != "EquiWidth-Merge":
            cdf = json.dumps(h.cdf_choice())
        if save_table and isinstance(h, HybridEstimator):
            pd.DataFrame(h.bucket_table()).to_csv(cdf_dir / f"{model}__{bin_strategy}.csv", index=False)
        nv = int(h.n_values())
        row = {**ids(ctx, model, bin_strategy, info["n_buckets"]), **extra,
               "K_FD": ctx.k_fd, "size_param": size, "n_buckets": info["n_buckets"],
               "n_values": nv, "size_bytes": 8 * nv, **sc, "build_time_s": build_s, "train_time_s": train_s,
               "cdf_choice": cdf, "model_info": json.dumps({k: v for k, v in info.items() if k != "n_buckets"}),
               "model_file": str(mdir / fname)}
        emit(row)
        done.add((bin_strategy, model))
        print(f"[{spec['slug']}] {bin_strategy:<16} {model:<16} K={info['n_buckets']:>6} nv={nv:>7} "
              f"med={sc['median_qerror']:.4f} p95={sc['p95_qerror']:.3f} {sc['infer_us_median']:.2f}us "
              f"build={build_s:.1f}s", flush=True)
        return row

    fd_buckets = ew_buckets(ctx, ctx.k_fd)
    for r in REDUCTIONS:
        K = max(1, int(round(ctx.k_fd * (1 - r / 100))))
        extra = {"reduction_pct": r, "K_requested": K}
        for model in SAME_K_MODELS:
            run_cfg(model, K, f"fd_r{r:02d}", extra, save_table=(model == "Hybrid"))
        merged = merge_buckets(ctx, fd_buckets, K)
        for model in MERGE_MODELS:
            run_cfg(model, K, f"fd_merge_r{r:02d}", extra, merged=merged, save_table=(model == "Hybrid-Merge"))

    # Same memory budget as each Hybrid configuration
    rows = read_log(log_path)
    hyb = rows[(rows["model"] == "Hybrid") & rows["bin_strategy"].str.startswith("fd_r")]
    cache = {}
    for _, hr in hyb.sort_values("reduction_pct").iterrows():
        M = int(hr["n_values"])
        Kh = int(hr["K_requested"])
        for model in MATCH_MODELS:
            bs = f"mem_match_K{Kh}"
            if (bs, model) in done:
                continue
            size, reachable = match_size(ctx, model, M, cache)
            run_cfg(model, size, bs, {"reduction_pct": int(hr["reduction_pct"]), "K_requested": size,
                                      "matched_to_model": "Hybrid", "matched_to_K": Kh,
                                      "target_n_values": M, "budget_reachable": reachable})
    rows = read_log(log_path)
    m = rows["target_n_values"].notna() if "target_n_values" in rows else pd.Series(False, index=rows.index)
    rows.loc[m, "budget_gap_pct"] = 100.0 * (rows.loc[m, "n_values"] - rows.loc[m, "target_n_values"]) / rows.loc[m, "target_n_values"]
    rows.to_csv(res_path, index=False)
    write_cdf_workbook(spec, cdf_dir)
    print(f"[{spec['slug']}] done in {time.perf_counter() - t_start:.0f}s", flush=True)
    return spec["slug"]


def read_log(path):
    return pd.read_json(path, lines=True, dtype=False)


def write_cdf_workbook(spec, cdf_dir):
    """One sheet per Hybrid / Hybrid-Merge configuration (bucket_table), plus an index."""
    files = sorted(cdf_dir.glob("*.csv"))
    if not files:
        return
    index = []
    path = cdf_dir / f"cdf_selection__{spec['slug']}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sheets = {}
        for f in files:
            model, bs = f.stem.split("__")
            short = ("H" if model == "Hybrid" else "HM") + "_" + bs.replace("fd_merge_", "").replace("fd_", "")
            df = pd.read_csv(f)
            sheets[short] = df
            ch = df["cdf_model"].value_counts().to_dict()
            index.append({"sheet": short, "model": model, "bin_strategy": bs, "n_buckets": len(df),
                          "total_bucket_bytes": int(df["bucket_bytes"].sum()),
                          **{f"n_{k}": v for k, v in ch.items()}})
        pd.DataFrame(index).fillna(0).to_excel(xw, sheet_name="index", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)


# ----------------------------------------------------------------------------
# Latency pass
# ----------------------------------------------------------------------------

def latency_pass(specs, models_dir):
    from threadpoolctl import threadpool_limits

    out = OUT / "latency_pass.csv"
    rows = []
    with threadpool_limits(limits=1):
        for spec in specs:
            res_path = OUT / "raw" / spec["slug"] / "results.csv"
            if not res_path.exists():
                continue
            res = pd.read_csv(res_path)
            ctx = Ctx(spec)
            rng = np.random.default_rng([SEED, 11])
            n = min(LATENCY_QUERIES, len(ctx.queries))
            sub = np.sort(rng.choice(len(ctx.queries), size=n, replace=False))
            qs = [ctx.queries[i] for i in sub]
            tc = ctx.true_counts[sub]
            # interleave: for every reduction level all models back to back
            res = res.assign(_o=res["reduction_pct"].fillna(0)).sort_values(["_o", "bin_strategy", "model"])
            for _, r in res.iterrows():
                with open(r["model_file"], "rb") as f:
                    h = pickle.load(f)
                la = os.getloadavg()[0]
                pred, times, _ = run_queries(h.predict, qs, tc)
                rows.append({"dataset": spec["dataset"], "bin_strategy": r["bin_strategy"], "model": r["model"],
                             "lat_us_median": float(np.median(times)) * 1e6, "lat_us_mean": float(np.mean(times)) * 1e6,
                             "lat_us_p95": float(np.percentile(times, 95)) * 1e6, "lat_n_queries": n,
                             "lat_loadavg_1min": la})
            print(f"[latency] {spec['slug']} {len(res)} configs, loadavg {os.getloadavg()[0]:.1f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--part", choices=["accuracy", "latency"], required=True)
    p.add_argument("--group", nargs="+", default=["synthetic", "dsb"], choices=["synthetic", "dsb", "dsb_expanded"])
    p.add_argument("--only", nargs="*", help="dataset slugs to run (default: all of --group)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--models-dir", default=str(OUT / "_models"))
    args = p.parse_args()
    specs = dataset_specs(args.group)
    if args.only:
        specs = [s for s in specs if s["slug"] in args.only]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "latency":
        print(latency_pass(specs, Path(args.models_dir)))
        return
    # largest K_FD first so the longest jobs start early
    specs.sort(key=lambda s: -pickle.load(open(ROOT / s["dir"] / "meta.pkl", "rb"))[5])
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_dataset, s, args.models_dir): s["slug"] for s in specs}
        for fut in as_completed(futs):
            try:
                print("DONE", fut.result(), flush=True)
            except Exception:
                failed.append(futs[fut])
                print("FAILED", futs[fut], traceback.format_exc(), flush=True)
    if failed:
        print("FAILED DATASETS:", failed, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

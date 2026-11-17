#!/usr/bin/env python3
"""
FD bucket reduction — ablation with Hybrid-Corrected (labelled MODIFICATION of Hybrid FD CDF,
models/hybrid_corrected.py: inclusive Tier-0 at inference + inclusive waterfall validation;
everything else identical). Runs after paper_experiments/fd_reduction.py, on the same datasets,
the same K grid, the same sample / workload / seed; Original rows are never touched.

Per dataset (rows -> raw/<dataset>/results_corrected.jsonl):
  fd_r<rr>            Hybrid-Corrected              every r, K equal-width buckets
  fd_merge_r<rr>      Hybrid-Corrected-Merge        every r, the same Ward-merged grids as Hybrid-Merge
                      EquiWidth-Merge-Corrected     control: merged grid, inclusive uniform inside a bucket
                                                    (the bug-free version of EquiWidth-Merge)
  fd_r00, fd_r50      Hybrid-Corrected-PredictOnly  DIAGNOSTIC: inference fix only, canonical validation
  mem_match_corrected_K<k>  EquiWidth / EquiDepth / TKHist / EntropyHist-MEA sized to the n_values of
                      Hybrid-Corrected at K = k. A baseline of the same size already scored in the
                      Original run is reused (same build, sample and workload -> same estimates;
                      `reused_from` names the source row); latency is re-measured in the pass below.
Bucket tables -> cdf_corrected/<dataset>/<model>__<bin_strategy>.csv + cdf_selection_corrected__<dataset>.xlsx.

--part latency: one sequential process re-times EVERY configuration of both runs (Original + Corrected)
on the same fixed 100 000-query subset (default_rng([42, 11])), models interleaved per reduction level,
loadavg recorded -> latency_pass.csv. This supersedes `fd_reduction.py --part latency` (Original only).

    poetry run python -m paper_experiments.fd_reduction_corrected --part corrected --workers 3 \
        --group synthetic dsb dsb_expanded --models-dir <scratch>/fd_models
    poetry run python -m paper_experiments.fd_reduction_corrected --part latency --group synthetic dsb dsb_expanded
"""
import argparse
import json
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

from benchmark_utils import run_queries  # noqa: E402
from models import HybridEstimator  # noqa: E402
from models.common import Bucket  # noqa: E402
from models.hybrid_corrected import HybridCorrectedEstimator, HybridCorrectedPredictOnlyEstimator  # noqa: E402
from paper_experiments.fd_reduction import (  # noqa: E402
    EXPERIMENT_ID,
    LATENCY_QUERIES,
    MATCH_MODELS,
    OUT,
    POINTS,
    REDUCTIONS,
    Ctx,
    build,
    dataset_specs,
    ew_buckets,
    merge_buckets,
    read_log,
    match_size,
    score,
)
from paper_experiments.provenance import MODEL_SOURCES, SEED, model_version, sources_sha, write_manifest  # noqa: E402
from paper_experiments.schema import ID_COLUMNS  # noqa: E402

CORR_SRC = "models/hybrid_corrected.py"
MERGE_SRC = "paper_experiments/fd_reduction.py"
PREDICT_ONLY_R = (0, 50)
VERSIONS = {
    "Hybrid-Corrected": lambda: model_version("Hybrid-Corrected", extra_sources=[CORR_SRC]),
    "Hybrid-Corrected-Merge": lambda: model_version("Hybrid-Corrected-Merge", extra_sources=[CORR_SRC, MERGE_SRC]),
    "Hybrid-Corrected-PredictOnly": lambda: model_version("Hybrid-Corrected-PredictOnly", extra_sources=[CORR_SRC]),
    "EquiWidth-Merge-Corrected": lambda: "EquiWidth-Merge-Corrected:control-inclusive-uniform-on-merged-boundaries:"
                                         + sources_sha(MODEL_SOURCES["Hybrid"] + [CORR_SRC, MERGE_SRC]),
}


def corrected_build(ctx, model, K, merged=None):
    t0 = time.perf_counter()
    train_s = None
    if model == "EquiWidth-Merge-Corrected":
        h = HybridCorrectedEstimator([Bucket(b.lo, b.hi, b.count) for b in merged])  # no CDF trained
        h._bake_vectorized_data()
    else:
        cls = HybridCorrectedPredictOnlyEstimator if model == "Hybrid-Corrected-PredictOnly" else HybridCorrectedEstimator
        buckets = [Bucket(b.lo, b.hi, b.count) for b in merged] if merged is not None else ew_buckets(ctx, K)
        h = cls(buckets)
        train_s = h.train(ctx.est_freq, ctx.mn, POINTS, np.random.default_rng(SEED))
    return h, {"n_buckets": len(h.buckets)}, time.perf_counter() - t0, train_s


def row_ids(ctx, model, bin_strategy, n_buckets, version):
    s = ctx.spec
    row = {
        "experiment_id": EXPERIMENT_ID, "dataset": s["dataset"], "table": s["table"], "column": s["column"],
        "workload": Path(s["workload"]).stem, "drift_type": "none", "drift_pct": 0.0,
        "bin_strategy": bin_strategy, "bucket_count": n_buckets, "model": model,
        "model_version": version, "seed": SEED, "stage": "STATIC",
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
    cdf_dir = OUT / "cdf_corrected" / spec["slug"]
    mdir = models_dir / spec["slug"]
    for d in (raw, cdf_dir, mdir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = raw / "results_corrected.jsonl"
    orig = read_log(raw / "results.jsonl")
    done = set()
    if log_path.exists():
        prev = read_log(log_path)
        done = set(zip(prev["bin_strategy"], prev["model"]))
    ctx = Ctx(spec)
    write_manifest(raw / "manifest_corrected.json", experiment_id=EXPERIMENT_ID, part="corrected-ablation",
                   dataset=spec["dataset"], workload=spec["workload"], K_FD=ctx.k_fd, reductions_pct=REDUCTIONS,
                   models=list(VERSIONS), model_versions_corrected={m: f() for m, f in VERSIONS.items()},
                   predict_only_reductions=list(PREDICT_ONLY_R), points_per_bucket=POINTS,
                   memory_matching="to Hybrid-Corrected n_values; baselines of an identical size reused from the Original run")
    versions = {m: f() for m, f in VERSIONS.items()}
    print(f"[{spec['slug']}] corrected part, K_FD={ctx.k_fd}", flush=True)

    def emit(row):
        with open(log_path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        done.add((row["bin_strategy"], row["model"]))

    def run_cfg(model, K, bin_strategy, extra, merged=None):
        if (bin_strategy, model) in done:
            return
        h, info, build_s, train_s = corrected_build(ctx, model, K, merged)
        sc = score(ctx, h)
        fname = f"{model}__{bin_strategy}.pkl"
        with open(mdir / fname, "wb") as f:
            pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
        trained = model != "EquiWidth-Merge-Corrected"
        if trained:
            pd.DataFrame(h.bucket_table()).to_csv(cdf_dir / f"{model}__{bin_strategy}.csv", index=False)
        nv = int(h.n_values())
        emit({**row_ids(ctx, model, bin_strategy, info["n_buckets"], versions[model]), **extra,
              "K_FD": ctx.k_fd, "size_param": K, "n_buckets": info["n_buckets"], "n_values": nv,
              "size_bytes": 8 * nv, **sc, "build_time_s": build_s, "train_time_s": train_s,
              "cdf_choice": json.dumps(h.cdf_choice()) if trained else None,
              "model_info": "{}", "model_file": str(mdir / fname)})
        print(f"[{spec['slug']}] {bin_strategy:<16} {model:<28} K={info['n_buckets']:>6} nv={nv:>7} "
              f"med={sc['median_qerror']:.4f} p95={sc['p95_qerror']:.3f} build={build_s:.1f}s", flush=True)

    fd = ew_buckets(ctx, ctx.k_fd)
    for r in REDUCTIONS:
        K = max(1, int(round(ctx.k_fd * (1 - r / 100))))
        extra = {"reduction_pct": r, "K_requested": K}
        run_cfg("Hybrid-Corrected", K, f"fd_r{r:02d}", extra)
        if r in PREDICT_ONLY_R:
            run_cfg("Hybrid-Corrected-PredictOnly", K, f"fd_r{r:02d}", extra)
        merged = merge_buckets(ctx, fd, K)
        run_cfg("Hybrid-Corrected-Merge", K, f"fd_merge_r{r:02d}", extra, merged=merged)
        run_cfg("EquiWidth-Merge-Corrected", K, f"fd_merge_r{r:02d}", extra, merged=merged)

    # memory matching to Hybrid-Corrected
    rows = read_log(log_path)
    hc = rows[(rows["model"] == "Hybrid-Corrected") & rows["bin_strategy"].str.startswith("fd_r")]
    om = orig[orig["bin_strategy"].str.startswith("mem_match_K")]
    cache = {}
    for _, hr in hc.sort_values("reduction_pct").iterrows():
        M, Kh = int(hr["n_values"]), int(hr["K_requested"])
        bs = f"mem_match_corrected_K{Kh}"
        for model in MATCH_MODELS:
            if (bs, model) in done:
                continue
            size, reachable = match_size(ctx, model, M, cache)
            extra = {"reduction_pct": int(hr["reduction_pct"]), "K_requested": size,
                     "matched_to_model": "Hybrid-Corrected", "matched_to_K": Kh,
                     "target_n_values": M, "budget_reachable": reachable,
                     "budget_gap_pct": None}
            same = om[(om["model"] == model) & (om["size_param"] == size)]
            if len(same):  # identical build already scored in the Original run
                src = same.iloc[0].to_dict()
                keep = {k: v for k, v in src.items() if k not in ID_COLUMNS and k not in extra
                        and k not in ("matched_to_model", "matched_to_K", "target_n_values", "budget_reachable",
                                      "budget_gap_pct")}
                row = {**row_ids(ctx, model, bs, src["bucket_count"], src["model_version"]), **extra, **keep,
                       "reused_from": f"{src['bin_strategy']}:{model}"}
            else:
                h, info, build_s, train_s = build(ctx, model, size)
                sc = score(ctx, h)
                fname = f"{model}__{bs}.pkl"
                with open(mdir / fname, "wb") as f:
                    pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
                nv = int(h.n_values())
                row = {**row_ids(ctx, model, bs, info["n_buckets"], model_version(model)), **extra,
                       "K_FD": ctx.k_fd, "size_param": size, "n_buckets": info["n_buckets"], "n_values": nv,
                       "size_bytes": 8 * nv, **sc, "build_time_s": build_s, "train_time_s": train_s,
                       "cdf_choice": None, "model_info": json.dumps({k: v for k, v in info.items() if k != "n_buckets"}),
                       "model_file": str(mdir / fname), "reused_from": None}
            row["budget_gap_pct"] = 100.0 * (row["n_values"] - M) / M
            emit(row)
            print(f"[{spec['slug']}] {bs:<24} {model:<16} nv={row['n_values']:>7} (target {M}) "
                  f"med={row['median_qerror']:.4f} reused={row['reused_from'] is not None}", flush=True)
    write_workbook(spec, cdf_dir)
    print(f"[{spec['slug']}] corrected done in {time.perf_counter() - t_start:.0f}s", flush=True)
    return spec["slug"]


SHORT = {"Hybrid-Corrected": "HC", "Hybrid-Corrected-Merge": "HCM", "Hybrid-Corrected-PredictOnly": "HCP"}


def write_workbook(spec, cdf_dir):
    files = sorted(cdf_dir.glob("*.csv"))
    if not files:
        return
    index, sheets = [], {}
    for f in files:
        model, bs = f.stem.split("__")
        name = f"{SHORT[model]}_" + bs.replace("fd_merge_", "").replace("fd_", "")
        df = pd.read_csv(f)
        sheets[name] = df
        index.append({"sheet": name, "model": model, "bin_strategy": bs, "n_buckets": len(df),
                      "total_bucket_bytes": int(df["bucket_bytes"].sum()),
                      **{f"n_{k}": v for k, v in df["cdf_model"].value_counts().to_dict().items()}})
    with pd.ExcelWriter(cdf_dir / f"cdf_selection_corrected__{spec['slug']}.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(index).fillna(0).to_excel(xw, sheet_name="index", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)


def latency_pass(specs):
    """Original + Corrected configurations, one process, same 100k subset, interleaved per r."""
    from threadpoolctl import threadpool_limits

    out = OUT / "latency_pass.csv"
    rows = []
    with threadpool_limits(limits=1):
        for spec in specs:
            raw = OUT / "raw" / spec["slug"]
            parts = [read_log(p) for p in (raw / "results.jsonl", raw / "results_corrected.jsonl") if p.exists()]
            if not parts:
                continue
            res = pd.concat(parts, ignore_index=True)
            res = res[res["model_file"].notna()]
            res = res.assign(_o=res["reduction_pct"].fillna(0)).sort_values(["_o", "bin_strategy", "model"])
            ctx = Ctx(spec)
            n = min(LATENCY_QUERIES, len(ctx.queries))
            sub = np.sort(np.random.default_rng([SEED, 11]).choice(len(ctx.queries), size=n, replace=False))
            qs = [ctx.queries[i] for i in sub]
            tc = ctx.true_counts[sub]
            timed = {}
            for _, r in res.iterrows():
                key = r["model_file"]
                if key not in timed:  # a reused row points to the same pickle: time it once
                    with open(key, "rb") as f:
                        h = pickle.load(f)
                    la = os.getloadavg()[0]
                    _, times, _ = run_queries(h.predict, qs, tc)
                    timed[key] = (float(np.median(times)) * 1e6, float(np.mean(times)) * 1e6,
                                  float(np.percentile(times, 95)) * 1e6, la)
                    del h
                med, mean, p95, la = timed[key]
                rows.append({"dataset": spec["dataset"], "bin_strategy": r["bin_strategy"], "model": r["model"],
                             "lat_us_median": med, "lat_us_mean": mean, "lat_us_p95": p95,
                             "lat_n_queries": n, "lat_loadavg_1min": la})
            print(f"[latency] {spec['slug']}: {len(timed)} models timed, loadavg {os.getloadavg()[0]:.1f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    p = argparse.ArgumentParser(description="FD reduction: Hybrid-Corrected ablation and the shared latency pass")
    p.add_argument("--part", choices=["corrected", "latency"], required=True)
    p.add_argument("--group", nargs="+", default=["synthetic", "dsb", "dsb_expanded"],
                   choices=["synthetic", "dsb", "dsb_expanded"])
    p.add_argument("--only", nargs="*")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--models-dir", default=str(OUT / "_models"))
    args = p.parse_args()
    specs = dataset_specs(args.group)
    if args.only:
        specs = [s for s in specs if s["slug"] in args.only]
    specs = [s for s in specs if (OUT / "raw" / s["slug"] / "results.jsonl").exists()]
    if args.part == "latency":
        print(latency_pass(specs))
        return
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
    _ = HybridEstimator  # imported for pickled models' class resolution in the latency pass
    main()

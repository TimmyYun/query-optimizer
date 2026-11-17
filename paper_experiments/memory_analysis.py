#!/usr/bin/env python3
"""
Memory analysis of every estimator (static builds) and of Hybrid FD CDF under drift.

    poetry run python -m paper_experiments.memory_analysis              # all datasets that exist
    poetry run python -m paper_experiments.memory_analysis --skip-drift # static only
    poetry run python -m paper_experiments.memory_analysis --report-only

Each (dataset, model, --buckets) configuration is measured in a FRESH subprocess
(`--worker`), so RSS numbers of one build are not polluted by another. One process at a time.

Byte metrics — they measure different things and are never mixed:
  logical_bytes        n_values() x 8 B: the numbers the model must persist (repo / thesis accounting;
                       every model's own n_values() — unchanged). Broken down into components
                       (grid, counts, boundaries, CDF parameters per model type / tier, type codes, MCV, ...)
                       and per bucket; shared numbers (grid min/width, Fourier frequencies) are counted
                       once — the 32 Fourier frequencies are fixed spec constants (logspace(0, 3, 32)),
                       identical for every MLP, so they are counted 0 times (like the code of the model).
  alt_*                alternative persistent accountings, NEXT to n_values: DecisionTree as the baked
                       lookup (thresholds + leaf values) instead of 3 x node_count.
  compact_raw_bytes    sum of nbytes of a minimal numpy encoding of the persisted numbers (int8 type codes,
                       int32 offsets, float64 values); compact_npz_bytes = the same written by np.savez
                       (adds zip/npy headers).
  pickle_bytes         len(pickle.dumps(model, protocol=5)) of the live Python object as the benchmark
                       holds it (includes sklearn fit state, caches, instrumentation).
  derived_logical_bytes / derived_python_bytes
                       runtime-only structures rebuilt from the persisted numbers (prefix sums, baked
                       arrays, lookup lists): 8 B per number / deep sys.getsizeof of those objects.
  object_graph_bytes   deep sys.getsizeof of the whole model object (allocated Python objects).
  tracemalloc_peak_bytes
                       peak of Python/numpy allocations traced while the model is built (second build in
                       the same process, tracemalloc on); includes transient training data.
  rss_peak_delta_bytes process VmHWM after the build minus VmRSS before it (Linux /proc, HWM reset via
                       /proc/self/clear_refs = 5); first build, tracemalloc off.
  reservoir_bytes      the 100k sample (sample.nbytes) — shared input, NOT part of any estimator.
  feedback_*           memory a feedback mechanism keeps: STHist processes feedback query by query
                       (persistent buffer 0, only an event Counter); Hybrid FT/RB are measured in the
                       drift check (transient arrays + the marked-bucket set + the adapt-query log).
"""
import argparse
import gc
import io
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import tracemalloc
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.provenance import model_version, worker_env, write_manifest  # noqa: E402
from paper_experiments.schema import ID_COLUMNS, bin_strategy_label  # noqa: E402

OUT = ROOT / "results" / "paper_experiments" / "memory_analysis"
RAW = OUT / "raw"
BYTES = 8
MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid",
          # labelled MODIFICATION (models/hybrid_corrected.py) and its diagnostic variant
          "Hybrid-Corrected", "Hybrid-Corrected-PredictOnly"]
HYBRID_CLASSES = {"Hybrid": "original", "Hybrid-Corrected": "corrected", "Hybrid-Corrected-PredictOnly": "corrected-predict-only"}
DRIFT_MODELS = ["Hybrid", "Hybrid-Corrected"]
EXTRA_MODELS = ["Hybrid-Merge", "EquiWidth-Merge"]  # only if Agent 3's builder exists
BUCKET_ARGS = ["100", "fd"]
STHIST_FEEDBACK_QUERIES = 100_000  # STHist static build = metadata + prequential feedback (run_sthist.py)
TK_TOPK, TK_NCAT = 5, 200  # run_tkhist.py defaults
HYBRID_POINTS = 200  # run_hybrid.py default
SEED = 42
SYNTH_DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
DSB_COLUMNS = ["ws_wholesale_cost", "ws_list_price"]
TIER_NAMES = {0: "Tier 0 Uniform", 1: "Tier 1 parametric", 2: "Tier 2 Isotonic/Tree", 3: "Tier 3 FourierMLP"}


# ----------------------------------------------------------------------------- datasets

def discover_datasets():
    """[{path, label, benchmark, table, column, workload}] for every dataset with meta.pkl that exists."""
    out = []
    for v in ("simple", "hard"):
        for d in SYNTH_DISTS:
            p = ROOT / f"data/generated/60000000_{v}/{d}"
            if (p / "meta.pkl").exists():
                out.append({"path": str(p.relative_to(ROOT)), "label": f"{v}/{d}", "benchmark": f"synth_{v}",
                            "table": None, "column": d, "workload": str((p / "workload_driven_1000000.csv").relative_to(ROOT))})
    for c in DSB_COLUMNS:
        p = ROOT / f"data/dsb/{c}"
        if (p / "meta.pkl").exists():
            out.append({"path": str(p.relative_to(ROOT)), "label": f"dsb/{c}", "benchmark": "dsb",
                        "table": "web_sales", "column": c, "workload": str((p / "workload_official.csv").relative_to(ROOT))})
    exp = ROOT / "data" / "dsb_expanded"
    if (exp / "READY_STATIC.json").exists():
        for p in sorted(exp.glob("*/meta.pkl")):
            d = p.parent
            wl = d / "workload_dd_narrow.csv"
            if not wl.exists():
                cands = sorted(d.glob("workload_*.csv"))
                wl = cands[0] if cands else wl
            table, _, column = d.name.partition(".")
            out.append({"path": str(d.relative_to(ROOT)), "label": f"dsb_expanded/{d.name}", "benchmark": "dsb_expanded",
                        "table": table or None, "column": column or d.name, "workload": str(wl.relative_to(ROOT))})
    return out


def load_meta(path):
    with open(ROOT / path / "meta.pkl", "rb") as f:
        mn, mx, N, freq, sample, k, *_ = pickle.load(f)
    return int(mn), int(mx), int(N), freq, np.asarray(sample, dtype=np.int64), int(k)


def resolve_k(bucket_arg, k):
    from benchmark_utils import resolve_buckets
    return resolve_buckets(bucket_arg, k)


# ----------------------------------------------------------------------------- builders (runner defaults)

def _runner_defaults(module, adder):
    p = argparse.ArgumentParser()
    getattr(module, adder)(p)
    return p.parse_args([])


def merge_builder():
    """
    Hybrid-Merge / EquiWidth-Merge builder of the FD-reduction experiment, if its module exposes
    build_for_memory(model, K, meta) -> estimator (meta = (mn, mx, N, freq, sample, k)). Otherwise None
    and the merged variants are skipped here (their n_values are in fd_reduction results).
    """
    try:
        import paper_experiments.fd_reduction as fr
    except Exception:
        return None
    fn = getattr(fr, "build_for_memory", None)
    return fn if callable(fn) else None


def sthist_feedback_queries(ds, meta):
    from benchmark_utils import load_and_filter_workload
    mn, mx, N, freq, _, _ = meta
    queries, sel = load_and_filter_workload(str(ROOT / ds["workload"]), mn, mx, freq)
    queries = queries[:STHIST_FEEDBACK_QUERIES]
    return queries, np.maximum(np.rint(sel[: len(queries)] * N), 1.0)


def make_builder(model, K, meta, ds):
    """Zero-arg build function reproducing the static runner's construction (defaults of each runner)."""
    from models import (EntropyHistogram1D, EquiDepthHistogram, EquiWidthHistogram, HybridEstimator,
                        STHistogram, TKHistogram)
    mn, mx, N, freq, sample, k = meta
    if model == "EquiWidth":
        return lambda: EquiWidthHistogram.build_from_sample(mn, mx, K, sample, N)
    if model == "EquiDepth":
        return lambda: EquiDepthHistogram.build_from_sample(sample=sample, total_rows=N, statistics_target=K)
    if model == "TKHist":
        return lambda: TKHistogram.build_from_sample(mn=mn, mx=mx, bins=K, sample=sample, total_rows=N,
                                                     top_k=TK_TOPK, n_categorical=TK_NCAT)
    if model == "EntropyHist-MEA":
        import run_entropyhist
        cfg = run_entropyhist.build_config(_runner_defaults(run_entropyhist, "add_entropyhist_args"), K)
        return lambda: EntropyHistogram1D.build_from_sample(sample, N, cfg)
    if model == "STHist":
        import run_sthist
        cfg_args = _runner_defaults(run_sthist, "add_sthist_args")
        queries, truth = sthist_feedback_queries(ds, meta)

        def build_st():
            st = STHistogram(run_sthist.build_config(cfg_args, K), total_rows=N, min_value=mn, max_value=mx)
            for q, t in zip(queries, truth):  # prequential, as evaluate_and_save(feedback=...)
                st.predict(q)
                st.observe_feedback(q.low, q.high, float(t))
            return st
        return build_st
    if model in HYBRID_CLASSES:
        from models.hybrid_corrected import VARIANTS
        cls = VARIANTS[HYBRID_CLASSES[model]]
        assert model != "Hybrid" or cls is HybridEstimator

        def build_hy():  # run_hybrid.run_logic (run_hybrid_corrected.py swaps only the estimator class)
            est = np.bincount(sample - mn, minlength=mx - mn + 1) * (N / len(sample))
            buckets = EquiWidthHistogram.build_from_sample(mn=mn, mx=mx, bins=K, sample=sample, total_rows=N).buckets
            h = cls(buckets)
            h.train(est, mn, HYBRID_POINTS, np.random.default_rng(SEED))
            return h
        return build_hy
    fn = merge_builder()
    if fn is None:
        raise RuntimeError(f"no builder for {model}")
    return lambda: fn(model, K, meta)


# ----------------------------------------------------------------------------- sizing helpers

def deep_sizeof(obj, seen=None, depth=0):
    """Deep sys.getsizeof (allocated Python objects; numpy arrays: header + owned data)."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 60:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        size = sys.getsizeof(obj)
        if obj.base is not None and isinstance(obj.base, np.ndarray):
            size += deep_sizeof(obj.base, seen, depth + 1)
        return size
    size = sys.getsizeof(obj)
    if isinstance(obj, (str, bytes, int, float, bool, type(None), np.generic)):
        return size
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_sizeof(k, seen, depth + 1) + deep_sizeof(v, seen, depth + 1)
        return size
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            size += deep_sizeof(v, seen, depth + 1)
        return size
    if callable(obj) and not hasattr(obj, "__dict__"):
        return size
    if hasattr(obj, "__dict__"):
        size += deep_sizeof(vars(obj), seen, depth + 1)
    for slot in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, slot):
            size += deep_sizeof(getattr(obj, slot), seen, depth + 1)
    return size


def n_numbers(x):
    """How many numbers a derived structure holds (8 B logical each)."""
    if x is None:
        return 0
    if isinstance(x, np.ndarray):
        return int(x.size)
    if isinstance(x, (list, tuple)):
        return int(sum(n_numbers(v) if isinstance(v, (list, tuple, np.ndarray)) else 1 for v in x))
    if isinstance(x, dict):
        return int(sum(n_numbers(v) for v in x.values()))
    return 1


def npz_bytes(arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return len(buf.getvalue()), int(sum(np.asarray(a).nbytes for a in arrays.values()))


def _mlp_params(mdl):
    m = mdl.model
    return np.concatenate([c.ravel() for c in m.coefs_] + [b.ravel() for b in m.intercepts_])


# ----------------------------------------------------------------------------- per-model breakdown

def breakdown(model, h):
    """
    -> dict(components={name: n_values}, per_bucket=np.array (values per bucket), empty=np.bool_ array,
            global_values=int, nonbucket={name: n_values}, derived={name: obj}, compact={name: array},
            extra={...}, bucket_table=list|None)
    Invariant (asserted): sum(components) == n_values == global + sum(per_bucket) + sum(nonbucket).
    """
    nv = int(h.n_values())
    extra, table = {}, None
    cls = type(h).__name__  # dispatch on the class: merged variants are HybridEstimator objects
    if cls == "EquiWidthHistogram":
        B = len(h.buckets)
        comps = {"grid (min, width)": 2, "bucket counts": B}
        if nv != 2 + B:  # merged boundaries (non-equal width) are stored per bucket
            comps["bucket boundaries"] = nv - 2 - B
        per_bucket = np.full(B, (nv - 2) / max(B, 1)) if B else np.zeros(0)
        empty = np.asarray(h.b_count) <= 0
        glob, nonb = 2, {}
        derived = {"b_lo": h.b_lo, "b_hi": h.b_hi, "b_count": h.b_count, "b_width": h.b_width,
                   "prefix_counts": h.prefix_counts}
        compact = {"grid": np.array([h.b_lo[0], h.b_width[0]]) if B else np.zeros(2), "counts": np.asarray(h.b_count)}
        if "bucket boundaries" in comps:
            compact["lo"] = np.asarray(h.b_lo, dtype=np.int64)
    elif cls == "TKHistogram":
        B = len(h.buckets)
        ent = np.array([len(t) for t in h.top_k], dtype=np.int64)
        E = int(ent.sum())
        comps = {"grid (min, width)": 2, "bucket counts": B, "top-k values": E, "top-k counts": E}
        per_bucket = 1 + 2 * ent
        empty = np.asarray(h.b_count) <= 0
        glob, nonb = 2, {}
        derived = {"b_lo": h.b_lo, "b_hi": h.b_hi, "b_count": h.b_count, "b_width": h.b_width,
                   "b_resid": h.b_resid, "prefix_counts": h.prefix_counts}
        vals = [v for t in h.top_k for v in t.keys()]
        cnts = [c for t in h.top_k for c in t.values()]
        compact = {"grid": np.array([h.b_lo[0], h.b_width[0]]), "counts": np.asarray(h.b_count),
                   "topk_values": np.asarray(vals, dtype=np.int64), "topk_counts": np.asarray(cnts, dtype=np.float64)}
        extra["topk_entries"] = E
    elif cls == "EquiDepthHistogram":
        L, m, nb = len(h.hist_bounds), h.num_mcv, h.n_buckets
        comps = {"histogram bounds": L, "MCV values": m, "MCV frequencies": m, "n_distinct": 1}
        per_bucket = np.ones(nb)  # lower bound of each bucket; the last bound is global
        empty = np.zeros(nb, dtype=bool)
        glob = 1 + (1 if L > 0 else 0)  # n_distinct + the last bound (L bounds = n_buckets + 1)
        nonb = {"MCV list (value, frequency)": 2 * m}
        derived = {"mcv_prefix": h.mcv_prefix}
        compact = {"bounds": np.asarray(h.hist_bounds, dtype=np.float64), "mcv_vals": np.asarray(h.mcv_vals, dtype=np.int64),
                   "mcv_freqs": np.asarray(h.mcv_freqs, dtype=np.float64), "n_distinct": np.array([h.n_distinct], dtype=np.float64)}
        extra.update(num_mcv=m, n_hist_bounds=L)
    elif cls == "EntropyHistogram1D":
        B = h.B_effective
        comps = {"bucket min": B, "bucket max": B, "bucket tuple count": B, "bucket distinct count": B}
        per_bucket = np.full(B, 4)
        empty = np.array([b.tuple_count <= 0 for b in h.buckets], dtype=bool)
        glob, nonb = 0, {}
        derived = {"_mins": h._mins, "_maxs": h._maxs, "_counts": h._counts, "_prefix": h._prefix, "_eq": h._eq,
                   # build statistics kept in every EntropyBucket object (not persisted by n_values)
                   "bucket_object_stats": [[b.start_idx, b.end_idx, b.entropy_frequency, b.entropy_area, b.bias_factor,
                                            b.sum_f2, b.sum_f3, b.sum_f_log2f] for b in h.buckets]}
        compact = {"buckets": np.array([[b.min_value, b.max_value, b.tuple_count, b.distinct_count] for b in h.buckets],
                                       dtype=np.float64).reshape(-1, 4)}
    elif cls == "STHistogram":
        B = len(h.freq)
        comps = {"bucket left bounds": B, "bucket frequencies": B, "domain right edge": 1}
        per_bucket = np.full(B, 2)
        empty = np.asarray(h.freq) <= 0
        glob, nonb = 1, {}
        derived = {"highs": h.highs, "widths": h.widths, "_lows_list": h._lows_list, "_prefix": h._prefix}
        compact = {"lows": np.asarray(h.lows, dtype=np.int64), "freq": np.asarray(h.freq, dtype=np.float64),
                   "domain_max": np.array([h.domain_max], dtype=np.int64)}
        extra["feedback_persistent_bytes_python"] = deep_sizeof(h.events)
    else:  # HybridEstimator: Hybrid, Hybrid-Merge, EquiWidth-Merge (no CDF trained)
        from models import HybridEstimator
        assert isinstance(h, HybridEstimator), cls
        table = h.bucket_table()
        B = len(table)
        per_bucket = np.array([r["bucket_values"] for r in table], dtype=np.int64)
        empty = np.array([r["count"] <= 0 for r in table], dtype=bool)
        glob, nonb = HybridEstimator.GLOBAL_VALUES, {}
        n_models = sum(1 for m in h.models.values() if m is not None)
        comps = {"grid (min, width)": glob, "bucket counts": B}
        if not h.uniform_grid:
            comps["bucket boundaries (non-equal width)"] = B
        comps["model-type codes"] = n_models
        tree_3n, tree_baked = 0, 0
        params, offsets, mb, mt = [], [0], [], []
        for r in table:
            if r["model_params"]:
                key = f"CDF params T{r['tier']} {r['cdf_model']}" if r["tier"] is not None else "CDF params (empty)"
                comps[key] = comps.get(key, 0) + int(r["model_params"])
        for i in sorted(h.models):
            mdl = h.models[i]
            if mdl is None:
                continue
            name = HybridEstimator.model_name(mdl)
            if isinstance(mdl, tuple):
                if name in ("Poly2", "Poly3"):  # legacy models only
                    p = np.concatenate([np.ravel(mdl[1]).astype(float), [float(mdl[2])]])
                else:
                    p = np.array([float(mdl[1]), float(mdl[2])])
            elif name == "Isotonic":
                p = np.concatenate([mdl.X_thresholds_, mdl.y_thresholds_]).astype(float)
            elif name == "DecisionTree":
                _, ts, vals = HybridEstimator._lookup(mdl)
                p = np.array(ts + vals, dtype=float)
                tree_3n += 3 * mdl.tree_.node_count
                tree_baked += len(ts) + len(vals)
            else:
                p = _mlp_params(mdl)
            params.append(p)
            offsets.append(offsets[-1] + len(p))
            mb.append(i)
            mt.append(int(h.mod_types[i]))
        compact = {"grid": np.array([h.b_lo[0], h.b_width[0]]) if B else np.zeros(2), "counts": np.asarray(h.b_count),
                   "model_bucket": np.asarray(mb, dtype=np.int32), "model_type": np.asarray(mt, dtype=np.int8),
                   "param_offsets": np.asarray(offsets, dtype=np.int32),
                   "params": np.concatenate(params) if params else np.zeros(0)}
        if not h.uniform_grid:
            compact["lo"] = np.asarray(h.b_lo, dtype=np.int64)
        lookups = [HybridEstimator._lookup(m) for m in h.models.values()
                   if m is not None and type(m).__name__ in ("IsotonicRegression", "DecisionTreeRegressor")]
        derived = {"b_lo": h.b_lo, "b_hi": h.b_hi, "b_count": h.b_count, "b_width": h.b_width, "_prefix": h._prefix,
                   "mod_types": h.mod_types, "lin_params": h.lin_params, "poly_params": h.poly_params,
                   "poly3_params": h.poly3_params, "log_params": h.log_params, "power_params": h.power_params,
                   "iso_tree_lookups": [[lk[1], lk[2]] for lk in lookups]}
        extra.update(n_models=n_models, uniform_grid=bool(h.uniform_grid),
                     alt_tree_3xnodes_values=tree_3n, alt_tree_baked_values=tree_baked,
                     alt_n_values_tree_baked=nv - tree_3n + tree_baked,
                     instrumentation_python_bytes=deep_sizeof(h.selection),
                     cdf_models_pickle_bytes=len(pickle.dumps(h.models, protocol=5)),
                     cdf_choice=json.dumps(h.cdf_choice()))
    total = sum(comps.values())
    assert total == nv, (model, comps, nv)
    assert abs(glob + float(np.sum(per_bucket)) + sum(nonb.values()) - nv) < 1e-6, (model, glob, per_bucket.sum(), nonb, nv)
    return {"components": comps, "per_bucket": np.asarray(per_bucket, dtype=np.float64), "empty": np.asarray(empty, bool),
            "global_values": glob, "nonbucket": nonb, "derived": derived, "compact": compact, "extra": extra,
            "bucket_table": table, "n_values": nv}


# ----------------------------------------------------------------------------- process memory

def _status_kb(field):
    try:
        for line in open("/proc/self/status"):
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def reset_hwm():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class RssProbe:
    """VmRSS before -> VmHWM after (HWM reset at start). Falls back to ru_maxrss (no reset possible)."""

    def __enter__(self):
        gc.collect()
        self.reset = reset_hwm()
        self.rss0 = _status_kb("VmRSS")
        if self.rss0 is None:
            import resource
            self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return self

    def __exit__(self, *exc):
        hwm = _status_kb("VmHWM")
        if hwm is None:
            import resource
            hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.hwm = hwm
        self.delta = max(0, hwm - self.rss0)
        self.method = "VmHWM-VmRSS (clear_refs reset)" if self.reset else "ru_maxrss (no reset; upper bound)"
        return False


def traced(fn, *a, **kw):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn(*a, **kw)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return res, peak, dt


# ----------------------------------------------------------------------------- worker: one static config

def _extra_sources(model):
    if model.startswith("Hybrid-Corrected"):
        return ["models/hybrid_corrected.py"]
    if "Merge" in model:
        return ["paper_experiments/fd_reduction.py"]
    return []


def ids_for(ds, model, bucket_arg, K, **kw):
    return {"experiment_id": "memory_analysis", "dataset": ds["label"], "table": ds["table"], "column": ds["column"],
            "workload": Path(ds["workload"]).stem if model == "STHist" else None, "drift_type": "none", "drift_pct": 0.0,
            "bin_strategy": bin_strategy_label(bucket_arg), "bucket_count": K, "model": model,
            "model_version": model_version(model, extra_sources=_extra_sources(model)),
            "seed": SEED, "stage": "STATIC", **kw}


def run_worker(ds, model, bucket_arg, out_json):
    meta = load_meta(ds["path"])
    mn, mx, N, freq, sample, k = meta
    K = resolve_k(bucket_arg, k)
    build = make_builder(model, K, meta, ds)
    with RssProbe() as rss:
        t0 = time.perf_counter()
        h = build()
        build_s = time.perf_counter() - t0
    _, tm_peak, tm_s = traced(build)
    b = breakdown(model, h)
    nv = b["n_values"]
    pk = len(pickle.dumps(h, protocol=5))
    npz, raw = npz_bytes(b["compact"])
    derived_logical = BYTES * sum(n_numbers(v) for v in b["derived"].values())
    derived_python = sum(deep_sizeof(v) for v in b["derived"].values())
    pb = b["per_bucket"] * BYTES
    empty = b["empty"]
    rec = {
        **ids_for(ds, model, bucket_arg, K),
        "n_rows": N, "domain_size": mx - mn + 1, "fd_k": k,
        "n_buckets": int(len(pb)), "n_empty_buckets": int(empty.sum()),
        "n_values": nv, "logical_bytes": nv * BYTES,
        "global_bytes": b["global_values"] * BYTES, "per_bucket_total_bytes": float(pb.sum()),
        "nonbucket_bytes": BYTES * sum(b["nonbucket"].values()),
        "empty_bucket_bytes": float(pb[empty].sum()) if len(pb) else 0.0,
        "empty_bucket_pct_of_logical": 100.0 * float(pb[empty].sum()) / (nv * BYTES) if nv else 0.0,
        "compact_raw_bytes": raw, "compact_npz_bytes": npz, "pickle_bytes": pk,
        "pickle_to_logical_ratio": pk / (nv * BYTES) if nv else None,
        "derived_logical_bytes": derived_logical, "derived_python_bytes": derived_python,
        "object_graph_bytes": deep_sizeof(h),
        "reservoir_bytes": int(sample.nbytes), "reservoir_rows": int(len(sample)),
        "build_s": build_s, "tracemalloc_build_s": tm_s,
        "tracemalloc_peak_bytes": tm_peak, "rss_peak_delta_bytes": rss.delta, "rss_baseline_bytes": rss.rss0,
        "rss_method": rss.method,
        "feedback_persistent_bytes": 0, "feedback_note": {
            "STHist": "online per-query feedback, no query buffer; only an event Counter (python bytes in extra)",
            "Hybrid": "static build has no feedback; FT/RB memory: drift check"}.get(model, "no feedback"),
        **{f"x_{k_}": v for k_, v in b["extra"].items()},
    }
    if model == "STHist":
        rec["feedback_persistent_bytes"] = b["extra"].get("feedback_persistent_bytes_python", 0)
        rec["sthist_feedback_queries"] = STHIST_FEEDBACK_QUERIES
    comps = [{"component": c, "n_values": int(v), "bytes": int(v) * BYTES, "pct_of_logical": 100.0 * v / nv if nv else 0.0}
             for c, v in b["components"].items()]
    payload = {"record": rec, "components": comps, "per_bucket_bytes": pb.tolist(), "empty": empty.tolist(),
               "bucket_table": b["bucket_table"]}
    if b["bucket_table"] is not None:
        tiers = [r["tier"] for r in b["bucket_table"]]
        payload["tiers"] = tiers
        payload["cdf_models"] = [r["cdf_model"] for r in b["bucket_table"]]
    Path(out_json).write_text(json.dumps(payload, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------- worker: drift FT / RB memory

DRIFT_CHECKS = [
    # (label, shift_dir, static workload or None for per-step, workload count)
    ("hard/shift_normal_to_zipf_5%", "data/generated/60000000_hard/shift_normal_to_zipf_5%",
     "data/generated/60000000_hard/normal/workload_driven_1000000.csv", "1000000"),
]
DSB_DRIFT_CHECK_COLUMNS = ["catalog_returns.cr_item_sk", "catalog_returns.cr_returned_time_sk", "web_sales.ws_item_sk"]


def discover_drift_checks():
    checks = list(DRIFT_CHECKS)
    for col in DSB_DRIFT_CHECK_COLUMNS:
        d = ROOT / "data" / "dsb_expanded" / col / "displacement"
        if (d / "step_1.pkl").exists() and (d / "step_1_workload_driven_1000000.csv").exists():
            checks.append((f"dsb_expanded/{col}/displacement", str(d.relative_to(ROOT)), None, "1000000"))
    return checks


def run_drift_worker(label, shift_dir, static_wl, wl_count, bucket_arg, mode, out_json, model_name="Hybrid"):
    """
    Runs drift_harness.run_drift on the first steps of a scenario up to (and incl.) the first 100k checkpoint,
    with HybridDrift.update / mark / rebuild wrapped to measure memory (mode 'tracemalloc' or 'rss').
    The truncated scenario is a folder of symlinks to the original step files (exact harness semantics).
    """
    import drift_harness
    import run_hybrid_drift
    from models.hybrid_corrected import VARIANTS
    from run_hybrid_drift import HybridDrift

    # run_hybrid_corrected_drift.py does exactly this: same HybridDrift, estimator class swapped
    run_hybrid_drift.HybridEstimator = VARIANTS[HYBRID_CLASSES[model_name]]

    src = ROOT / shift_dir
    paths = sorted(src.glob("step_*.pkl"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    with open(paths[0], "rb") as f:
        N0 = int(pickle.load(f)[2])
    n_steps = len(paths) - 1
    rows_per_step = N0 / n_steps
    k_star = max(1, math.ceil(drift_harness.CHECKPOINT_ROWS / rows_per_step))
    k_star = min(k_star, n_steps)
    work = RAW / "drift_work" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}__{model_name}__{bucket_arg}__{mode}"
    if work.exists():
        shutil.rmtree(work)
    tmp_shift = work / "shift"
    tmp_shift.mkdir(parents=True)
    for i in range(k_star + 1):
        (tmp_shift / f"step_{i}.pkl").symlink_to(paths[i].resolve())
        wl = src / f"step_{i}_workload_driven_{wl_count}.csv"
        if wl.exists():
            (tmp_shift / wl.name).symlink_to(wl.resolve())
    measurements = []

    class Measured(HybridDrift):
        def _measure(self, stage, fn, *a):
            if mode == "rss":
                with RssProbe() as rss:
                    t0 = time.perf_counter()
                    res = fn(*a)
                    dt = time.perf_counter() - t0
                peak, extra = None, {"rss_peak_delta_bytes": rss.delta, "rss_method": rss.method}
            else:
                res, peak, dt = traced(fn, *a)
                extra = {"tracemalloc_peak_bytes": peak}
            fb = a[-1] if a else None
            measurements.append({"stage": stage, "step": self.step_no, "time_s": dt, **extra,
                                 "adapt_queries": len(fb.queries) if fb is not None and hasattr(fb, "queries") else None,
                                 "marked_buckets": len(self.marked), "result": res if isinstance(res, (int, float)) else None,
                                 "n_values_after": self.h.n_values(),
                                 "marked_set_python_bytes": deep_sizeof(self.marked)})
            return res

        def update(self, feedback):
            return self._measure("FT", super().update, feedback)

        def mark(self, feedback):
            return self._measure("MARK", super().mark, feedback)

        def rebuild(self, step, feedback):
            return self._measure("RB", super().rebuild, step, feedback)

    # the drift_steps.csv is a by-product (scenario prefix only), kept under raw/ for traceability
    ns = Namespace(shift_dir=str(tmp_shift), workload=wl_count, static_workload_path=static_wl and str(ROOT / static_wl),
                   shift_workload=static_wl is None, drift_total_pct=100.0 * k_star / n_steps,
                   drift_type="memory_check", init_dist="mem", target_dist="check", out_dir=str(work / "out"),
                   experiment_name="run", buckets=bucket_arg)
    model = Measured(HYBRID_POINTS)
    drift_harness.run_drift(ns, model_name, model)
    Path(out_json).write_text(json.dumps({
        "label": label, "model": model_name, "bin_strategy": bin_strategy_label(bucket_arg), "mode": mode, "k_star": k_star,
        "n_steps_scenario": n_steps, "rows_per_step": rows_per_step, "N0": N0, "measurements": measurements,
        "n_buckets": len(model.h.buckets)}, default=_json_default))


# ----------------------------------------------------------------------------- FD-reduction models (read-only)

FD_RESULTS = ROOT / "results" / "paper_experiments" / "fd_reduction"


def analyse_fd_models(force=False):
    """
    Memory of every FD-reduction configuration from the model objects the FD-reduction experiment pickled
    (`model_file` column of fd_reduction_all.csv): same breakdown as the static builds, no rebuild, so no
    construction peaks. Writes raw/fd_models.json (list of payloads, cached by model file).
    """
    src = FD_RESULTS / "fd_reduction_all.csv"
    if not src.exists():
        return
    df = pd.read_csv(src, low_memory=False)
    if "model_file" not in df.columns:
        return
    cache_path = RAW / "fd_models.json"
    cache = {} if force or not cache_path.exists() else json.loads(cache_path.read_text())
    done = 0
    for _, r in df.iterrows():
        f = r.get("model_file")
        if not isinstance(f, str) or not Path(f).exists():
            continue
        key = f"{f}:{Path(f).stat().st_mtime_ns}"
        if key in cache:
            continue
        try:
            with open(f, "rb") as fh:
                h = pickle.load(fh)
            b = breakdown(str(r["model"]), h)
        except Exception as e:
            cache[key] = {"error": repr(e), "model_file": f}
            continue
        nv = b["n_values"]
        pb = b["per_bucket"] * BYTES
        em = b["empty"]
        npz, raw = npz_bytes(b["compact"])
        rec = {c: (None if pd.isna(r.get(c)) else r.get(c)) for c in ID_COLUMNS}
        rec.update({"n_buckets": int(len(pb)), "n_empty_buckets": int(em.sum()), "n_values": nv,
                    "logical_bytes": nv * BYTES, "n_values_reported": r.get("n_values"),
                    "empty_bucket_bytes": float(pb[em].sum()) if len(pb) else 0.0,
                    "empty_bucket_pct_of_logical": 100.0 * float(pb[em].sum()) / (nv * BYTES) if nv else 0.0,
                    "pickle_bytes": len(pickle.dumps(h, protocol=5)), "compact_raw_bytes": raw, "compact_npz_bytes": npz,
                    "derived_logical_bytes": BYTES * sum(n_numbers(v) for v in b["derived"].values()),
                    "global_bytes": b["global_values"] * BYTES, "model_file": f,
                    **{f"x_{k}": v for k, v in b["extra"].items()}})
        comps = [{"component": c, "n_values": int(v), "bytes": int(v) * BYTES, "pct_of_logical": 100.0 * v / nv if nv else 0.0}
                 for c, v in b["components"].items()]
        cache[key] = {"record": rec, "components": comps, "per_bucket_bytes": pb.tolist(), "empty": em.tolist(),
                      "bucket_table": b["bucket_table"]}
        done += 1
        if done % 100 == 0:
            cache_path.write_text(json.dumps(cache, default=_json_default))
            print(f"[memory fd_models] {done} analysed", flush=True)
    cache_path.write_text(json.dumps(cache, default=_json_default))
    print(f"[memory fd_models] {done} new, {len(cache)} total", flush=True)


# ----------------------------------------------------------------------------- master

def _cfg_name(ds_label, model, bucket_arg):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ds_label}__{model}__{bucket_arg}")


def run_all(args):
    RAW.mkdir(parents=True, exist_ok=True)
    datasets = discover_datasets()
    if args.datasets:
        datasets = [d for d in datasets if any(s in d["label"] for s in args.datasets)]
    models = list(args.models)
    if merge_builder() is not None and not args.models_given:
        models += EXTRA_MODELS
    status = []
    for ds in datasets:
        for model in models:
            for b in args.buckets:
                out = RAW / f"{_cfg_name(ds['label'], model, b)}.json"
                if out.exists() and not args.force:
                    status.append({"dataset": ds["label"], "model": model, "bin": b, "status": "OK (cached)"})
                    continue
                cmd = [sys.executable, "-m", "paper_experiments.memory_analysis", "--worker",
                       "--ds-json", json.dumps(ds), "--model", model, "--bucket-arg", b, "--out-json", str(out)]
                t0 = time.perf_counter()
                r = subprocess.run(cmd, cwd=ROOT, env=worker_env(1), capture_output=True, text=True)
                st = "OK" if r.returncode == 0 and out.exists() else f"FAILED rc={r.returncode}"
                if r.returncode != 0:
                    (RAW / f"{_cfg_name(ds['label'], model, b)}.err").write_text(r.stdout[-4000:] + "\n" + r.stderr[-8000:])
                status.append({"dataset": ds["label"], "model": model, "bin": b, "status": st,
                               "wall_s": round(time.perf_counter() - t0, 1)})
                print(f"[memory] {ds['label']:<40} {model:<16} {b:<4} {st} ({status[-1]['wall_s']} s)", flush=True)
    if not args.skip_drift:
        checks = [(c, dm) for c in discover_drift_checks() for dm in DRIFT_MODELS if dm in models]
        for (label, shift_dir, wl, cnt), dm in checks:
            for b in args.buckets:
                for mode in ("tracemalloc", "rss"):
                    out = RAW / "drift" / f"{_cfg_name(label, dm, b)}__{mode}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if out.exists() and not args.force:
                        continue
                    cmd = [sys.executable, "-m", "paper_experiments.memory_analysis", "--drift-worker",
                           "--ds-json", json.dumps([label, shift_dir, wl, cnt]), "--bucket-arg", b,
                           "--mode", mode, "--out-json", str(out), "--model", dm]
                    t0 = time.perf_counter()
                    r = subprocess.run(cmd, cwd=ROOT, env=worker_env(1), capture_output=True, text=True)
                    st = "OK" if r.returncode == 0 and out.exists() else f"FAILED rc={r.returncode}"
                    if r.returncode != 0:
                        (out.with_suffix(".err")).write_text(r.stdout[-4000:] + "\n" + r.stderr[-8000:])
                    status.append({"dataset": label, "model": f"{dm} (drift FT/RB)", "bin": b, "status": f"{mode}: {st}",
                                   "wall_s": round(time.perf_counter() - t0, 1)})
                    print(f"[memory drift] {label} {dm} {b} {mode}: {st}", flush=True)
    if not args.skip_fd_models:
        analyse_fd_models(force=args.force)
    prev = OUT / "run_status.csv"
    df = pd.DataFrame(status)
    if prev.exists():  # keep earlier runs' status (cached configs report "OK (cached)")
        old = pd.read_csv(prev)
        df = pd.concat([old, df]).drop_duplicates(["dataset", "model", "bin"], keep="last")
    df.to_csv(prev, index=False)


# ----------------------------------------------------------------------------- report

def _stats(values, prefix=""):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {f"{prefix}n_buckets": 0}
    return {f"{prefix}n_buckets": int(len(v)), f"{prefix}min_bytes": float(v.min()), f"{prefix}mean_bytes": float(v.mean()),
            f"{prefix}median_bytes": float(np.median(v)), f"{prefix}p95_bytes": float(np.percentile(v, 95)),
            f"{prefix}max_bytes": float(v.max()), f"{prefix}total_bytes": float(v.sum())}


def cdf_distribution(bt: pd.DataFrame, total_bytes: float, n_rows, ids: dict) -> list:
    """Per CDF model (+ per tier) table from a Hybrid bucket table (bucket_table() rows)."""
    bt = bt.copy()
    bt["tier_label"] = bt["tier"].map(lambda t: "Empty" if pd.isna(t) else TIER_NAMES.get(int(t), str(t)))
    tot_buckets = len(bt)
    tot_count = float(bt["count"].clip(lower=0).sum())
    rows = []
    for level, key in (("cdf_model", ["cdf_model", "tier_label"]), ("tier", ["tier_label"])):
        for k, g in bt.groupby(key, dropna=False):
            k = k if isinstance(k, tuple) else (k,)
            mem = g["bucket_bytes"].astype(float)
            cnt = float(g["count"].clip(lower=0).sum())
            rows.append({**ids, "level": level, "cdf_model": k[0] if level == "cdf_model" else "(all)",
                         "tier": k[-1], "n_buckets": int(len(g)), "pct_buckets": 100.0 * len(g) / tot_buckets,
                         "total_bytes": float(mem.sum()), "mean_bytes": float(mem.mean()), "median_bytes": float(mem.median()),
                         "pct_of_total_memory": 100.0 * float(mem.sum()) / total_bytes if total_bytes else None,
                         "rows_covered_est": cnt, "pct_cardinality_est": 100.0 * cnt / tot_count if tot_count else None,
                         "pct_of_N": 100.0 * cnt / n_rows if n_rows else None})
    rows.append({**ids, "level": "global", "cdf_model": "GLOBAL grid (min, width)", "tier": "-", "n_buckets": 0,
                 "pct_buckets": 0.0, "total_bytes": 16.0, "mean_bytes": None, "median_bytes": None,
                 "pct_of_total_memory": 100.0 * 16.0 / total_bytes if total_bytes else None,
                 "rows_covered_est": None, "pct_cardinality_est": None, "pct_of_N": None})
    return rows


def _drift_bucket_tables():
    """Hybrid per-drift-point bucket tables written by run_hybrid_drift.py anywhere under results/paper_experiments."""
    base = ROOT / "results" / "paper_experiments"
    for bt_dir in sorted(base.rglob("bucket_tables")):
        if "memory_analysis" in bt_dir.parts:
            continue
        run_dir = bt_dir.parent
        steps = run_dir / "drift_steps.csv"
        sdf = pd.read_csv(steps) if steps.exists() else None
        m = re.match(r"gradual_(.+)_to_(.+)_(Hybrid.*)$", run_dir.name)
        rel = run_dir.relative_to(base)
        bin_part = next((p for p in rel.parts if re.fullmatch(r"fd|fixed\d+|fd_x[\d.]+|fd_r\d+", p)), None)
        for f in sorted(bt_dir.glob("*.csv")):
            yield rel, f, m.group(1) if m else run_dir.name, bin_part, sdf, m.group(3) if m else "Hybrid"


def build_report(args):
    raws = sorted(f for f in RAW.glob("*.json") if f.name != "fd_models.json")
    recs, comps, pbs, cdf_rows = [], [], [], []
    for f in raws:
        p = json.loads(f.read_text())
        r = p["record"]
        recs.append(r)
        idc = {c: r.get(c) for c in ID_COLUMNS}
        for c in p["components"]:
            comps.append({**idc, **c})
        pb = np.asarray(p["per_bucket_bytes"], dtype=float)
        em = np.asarray(p["empty"], dtype=bool)
        if len(pb):
            pbs.append({**idc, "scope": "all buckets", **_stats(pb)})
            pbs.append({**idc, "scope": "non-empty buckets", **_stats(pb[~em])})
            if em.any():
                pbs.append({**idc, "scope": "empty buckets", **_stats(pb[em])})
        if p.get("bucket_table"):
            bt = pd.DataFrame(p["bucket_table"])
            for t, g in bt.groupby(bt["tier"].map(lambda t: "Empty" if pd.isna(t) else TIER_NAMES.get(int(t)))):
                pbs.append({**idc, "scope": t, **_stats(g["bucket_bytes"])})
            cdf_rows += cdf_distribution(bt, r["logical_bytes"], r["n_rows"], {**idc, "source": "static build"})
            bt_out = OUT / "bucket_tables"
            bt_out.mkdir(exist_ok=True)
            bt.to_csv(bt_out / f"{f.stem}.csv", index=False)
    # FD-reduction configurations (from pickled model files; no construction peaks)
    fd_recs = []
    fd_cache = RAW / "fd_models.json"
    if fd_cache.exists():
        for p in json.loads(fd_cache.read_text()).values():
            if "record" not in p:
                continue
            r = p["record"]
            fd_recs.append(r)
            idc = {c: r.get(c) for c in ID_COLUMNS}
            for c in p["components"]:
                comps.append({**idc, **c})
            pb = np.asarray(p["per_bucket_bytes"], dtype=float)
            em = np.asarray(p["empty"], dtype=bool)
            if len(pb):
                pbs.append({**idc, "scope": "all buckets", **_stats(pb)})
                if em.any():
                    pbs.append({**idc, "scope": "empty buckets", **_stats(pb[em])})
            if p.get("bucket_table"):
                bt = pd.DataFrame(p["bucket_table"])
                for t, g in bt.groupby(bt["tier"].map(lambda t: "Empty" if pd.isna(t) else TIER_NAMES.get(int(t)))):
                    pbs.append({**idc, "scope": t, **_stats(g["bucket_bytes"])})
                cdf_rows += cdf_distribution(bt, r["logical_bytes"], None, {**idc, "source": "fd_reduction model file"})
    if fd_recs:
        fdm = pd.DataFrame(fd_recs)
        fdm["n_values_matches_reported"] = fdm["n_values"] == fdm["n_values_reported"]
        fdm.to_csv(OUT / "fd_reduction_memory.csv", index=False)
    summary = pd.DataFrame(recs)
    if summary.empty:
        print("no raw measurements yet")
        return
    summary = summary.sort_values(["dataset", "bin_strategy", "model"])
    summary.insert(summary.columns.get_loc("logical_bytes") + 1, "logical_KiB", summary["logical_bytes"] / 1024)
    summary.to_csv(OUT / "memory_summary.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / "memory_breakdown.csv", index=False)
    pd.DataFrame(pbs).to_csv(OUT / "per_bucket_memory_stats.csv", index=False)
    pd.DataFrame(cdf_rows).to_csv(OUT / "hybrid_cdf_distribution.csv", index=False)

    peak_cols = ID_COLUMNS + ["n_buckets", "logical_bytes", "build_s", "tracemalloc_peak_bytes", "tracemalloc_build_s",
                              "rss_peak_delta_bytes", "rss_baseline_bytes", "rss_method", "reservoir_bytes"]
    peak = summary[[c for c in peak_cols if c in summary.columns]].copy()
    peak.insert(len(ID_COLUMNS), "phase", "static build")
    drift_rows = []
    for f in sorted((RAW / "drift").glob("*.json")) if (RAW / "drift").exists() else []:
        p = json.loads(f.read_text())
        for m in p["measurements"]:
            drift_rows.append({"experiment_id": "memory_analysis", "dataset": p["label"], "bin_strategy": p["bin_strategy"],
                               "model": p.get("model", "Hybrid"), "stage": m["stage"], "phase": f"drift {m['stage']}", "mode": p["mode"],
                               "step": m["step"], "checkpoint_step": p["k_star"], "n_buckets": p["n_buckets"],
                               "time_s": m["time_s"], "tracemalloc_peak_bytes": m.get("tracemalloc_peak_bytes"),
                               "rss_peak_delta_bytes": m.get("rss_peak_delta_bytes"), "adapt_queries": m.get("adapt_queries"),
                               "adapt_log_logical_bytes": None if m.get("adapt_queries") is None else 32 * m["adapt_queries"],
                               "marked_buckets": m["marked_buckets"], "marked_set_logical_bytes": 8 * m["marked_buckets"],
                               "marked_set_python_bytes": m["marked_set_python_bytes"], "result": m["result"],
                               "n_values_after": m["n_values_after"]})
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(OUT / "drift_stage_memory.csv", index=False)
    if not drift_df.empty:  # FT / MARK / RB peaks next to the static build peaks
        d = drift_df.rename(columns={"time_s": "build_s"})
        peak = pd.concat([peak, d[[c for c in d.columns if c in set(peak.columns) | {"step", "mode"}]]], ignore_index=True)
    peak.to_csv(OUT / "peak_memory.csv", index=False)

    # Hybrid CDF distribution per drift point (bucket tables of drift runs, read-only)
    drows = []
    for rel, f, init, bin_part, sdf, model in _drift_bucket_tables():
        bt = pd.read_csv(f)
        if bt.empty:
            continue
        step = int(bt["step"].iloc[0]) if "step" in bt else None
        srow = sdf[sdf["step"] == step].iloc[0] if sdf is not None and step is not None and (sdf["step"] == step).any() else None
        n_values = int(srow["n_values"]) if srow is not None else int(16 / 8 + bt["bucket_values"].sum())
        ids = {"experiment_id": rel.parts[0], "dataset": init, "table": None, "column": init,
               "workload": None, "drift_type": srow.get("drift_type") if srow is not None else None,
               "drift_pct": float(bt["drift_pct"].iloc[0]) if "drift_pct" in bt else None,
               "bin_strategy": bin_part, "bucket_count": len(bt), "model": model, "model_version": None,
               "seed": SEED, "stage": "STATIC" if step == 0 else "FINAL", "source": str(rel / "bucket_tables" / f.name),
               "step": step, "sum_check_ok": int(2 + bt["bucket_values"].sum()) == n_values}
        drows += cdf_distribution(bt, n_values * BYTES, srow["n_rows"] if srow is not None and "n_rows" in srow else None, ids)
    if drows:
        pd.DataFrame(drows).to_csv(OUT / "hybrid_cdf_distribution_drift.csv", index=False)
    # plots / markdown sections describe the static measurements; FD-reduction rows stay in the CSVs
    sc = pd.DataFrame(comps)
    sc = sc[sc["experiment_id"] == "memory_analysis"]
    cdf_df = pd.DataFrame(cdf_rows)
    make_plots(summary, sc)
    write_markdown(summary, sc, cdf_df[cdf_df["experiment_id"] == "memory_analysis"], drift_df, pd.DataFrame(pbs), bool(drows),
                   fd_mem=fdm if fd_recs else None)
    write_manifest(OUT / "manifest.json", part="memory_analysis", models=MODELS,
                   datasets=sorted(summary["dataset"].unique().tolist()), bucket_args=BUCKET_ARGS,
                   sthist_feedback_queries=STHIST_FEEDBACK_QUERIES, tkhist=(TK_TOPK, TK_NCAT), hybrid_points=HYBRID_POINTS,
                   raw_measurements=len(raws), drift_checks=[c[0] for c in discover_drift_checks()])
    try:
        from paper_experiments.excel_export import export_dir
        export_dir(OUT, OUT / "memory_analysis.xlsx", exclude=("raw", "bucket_tables"))
    except Exception as e:  # the CSVs are the primary output
        print(f"xlsx export failed: {e}")


# ----------------------------------------------------------------------------- plots

# fixed categorical order (validated reference palette, slots 1-8); merged variants are not in the static plots
COLORS = {"EquiWidth": "#2a78d6", "EquiDepth": "#eb6834", "TKHist": "#1baf7a", "EntropyHist-MEA": "#eda100",
          "STHist": "#e87ba4", "Hybrid": "#008300", "Hybrid-Corrected": "#6250d6", "Hybrid-Corrected-PredictOnly": "#e34948"}


def make_plots(summary, comps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 9, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb"})
    models = [m for m in COLORS if m in set(summary["model"])]
    for bs, g in summary.groupby("bin_strategy"):
        dsets = sorted(g["dataset"].unique())
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(dsets) + 3), 4.2))
        w = 0.8 / max(1, len(models))
        for j, m in enumerate(models):
            gm = g[g["model"] == m].set_index("dataset")
            xs = [i + (j - (len(models) - 1) / 2) * w for i in range(len(dsets))]
            ys = [gm.loc[d, "logical_bytes"] if d in gm.index else np.nan for d in dsets]
            ax.bar(xs, ys, width=w * 0.9, color=COLORS[m], label=m, edgecolor="#fcfcfb", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(dsets)))
        ax.set_xticklabels(dsets, rotation=35, ha="right")
        ax.set_ylabel("logical persistent memory, bytes (n_values x 8, log)")
        ax.set_title(f"Model size per dataset — bin strategy {bs}")
        ax.legend(ncol=min(6, len(models)), fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13))
        fig.tight_layout()
        fig.savefig(OUT / f"model_size_{bs}.png", dpi=150)
        plt.close(fig)

    hy = comps[comps["model"] == "Hybrid"].copy()
    if not hy.empty:
        def group(c):
            if c.startswith("CDF params T"):
                return f"CDF params Tier {c[12]}"
            return c
        hy["group"] = hy["component"].map(group)
        order = ["grid (min, width)", "bucket counts", "bucket boundaries (non-equal width)", "model-type codes",
                 "CDF params Tier 1", "CDF params Tier 2", "CDF params Tier 3"]
        palette = ["#52514e", "#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#e87ba4", "#6250d6"]
        for bs, g in hy.groupby("bin_strategy"):
            pv = g.pivot_table(index="dataset", columns="group", values="bytes", aggfunc="sum").fillna(0)
            cols = [c for c in order if c in pv.columns]
            fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pv) + 3), 4.2))
            bottom = np.zeros(len(pv))
            for c in cols:
                ax.bar(range(len(pv)), pv[c].values / 1024, bottom=bottom, color=palette[order.index(c)], label=c,
                       edgecolor="#fcfcfb", linewidth=1)
                bottom += pv[c].values / 1024
            ax.set_xticks(range(len(pv)))
            ax.set_xticklabels(pv.index, rotation=35, ha="right")
            ax.set_ylabel("KiB (logical, n_values x 8)")
            ax.set_title(f"Hybrid FD CDF memory breakdown — {bs}")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(OUT / f"hybrid_breakdown_{bs}.png", dpi=150)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        g = summary[summary["model"] == m]
        ax.scatter(g["logical_bytes"], g["pickle_bytes"], s=28, color=COLORS[m], label=m, edgecolor="#fcfcfb", linewidth=1)
    lim = [max(1, summary["logical_bytes"].min() / 2), summary["pickle_bytes"].max() * 2]
    ax.plot(lim, lim, color="#52514e", lw=1, ls="--", label="pickle = logical")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("logical persistent bytes (n_values x 8)")
    ax.set_ylabel("pickle bytes (live object)")
    ax.set_title("Serialized object vs logical persistent memory")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pickle_vs_logical.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown

def _kib(b):
    return "—" if b is None or (isinstance(b, float) and math.isnan(b)) else f"{b / 1024:,.1f}"


def key_findings(summary, comps):
    """Statements computed from the tables (re-generated on every report run)."""
    L = []
    for bs, g in summary.groupby("bin_strategy"):
        pv = g.pivot_table(index="dataset", columns="model", values="logical_bytes", aggfunc="first")
        if "Hybrid" not in pv:
            continue
        L.append(f"- **{bs}**: Hybrid хранит {_kib(pv['Hybrid'].median())} KiB (медиана по {pv['Hybrid'].notna().sum()} датасетам; "
                 f"от {_kib(pv['Hybrid'].min())} до {_kib(pv['Hybrid'].max())} KiB).")
        for base in ("EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist"):
            if base in pv:
                r = (pv["Hybrid"] / pv[base]).dropna()
                if len(r):
                    L.append(f"  - Hybrid / {base}: медиана ×{r.median():.1f} (min ×{r.min():.1f}, max ×{r.max():.1f}).")
        for mod in ("Hybrid-Corrected", "Hybrid-Corrected-PredictOnly"):
            if mod in pv:
                r = (pv[mod] / pv["Hybrid"]).dropna()
                if len(r):
                    L.append(f"  - {mod} / Hybrid (модификация против канонического): медиана ×{r.median():.2f} "
                             f"(min ×{r.min():.2f}, max ×{r.max():.2f}).")
    pv = summary.pivot_table(index=["dataset", "bin_strategy"], columns="model", values="logical_bytes", aggfunc="first")
    if {"Hybrid", "Hybrid-Corrected"} <= set(pv.columns):
        r = (pv["Hybrid-Corrected"] / pv["Hybrid"]).dropna().sort_values()
        ch = summary.set_index(["dataset", "bin_strategy", "model"])["x_cdf_choice"]
        L.append(f"- Original → Hybrid-Corrected: память меньше в {int((r < 0.95).sum())}, больше в {int((r > 1.05).sum())}, "
                 f"примерно равна в {int(((r >= 0.95) & (r <= 1.05)).sum())} из {len(r)} конфигураций. Крайние случаи:")
        for (d, bs), v in list(r.head(3).items()) + list(r.tail(2).items()):
            L.append(f"  - {d} / {bs}: ×{v:.3f} ({_kib(pv.loc[(d, bs), 'Hybrid'])} → {_kib(pv.loc[(d, bs), 'Hybrid-Corrected'])} KiB); "
                     f"CDF {ch.get((d, bs, 'Hybrid'))} → {ch.get((d, bs, 'Hybrid-Corrected'))}")
    c = comps[comps["model"] == "Hybrid"].copy()
    if not c.empty:
        c["part"] = np.where(c["component"].str.startswith("CDF params T"), "Tier " + c["component"].str[12], c["component"])
        dom = c.groupby(["dataset", "bin_strategy", "part"])["bytes"].sum().reset_index()
        dom = dom.loc[dom.groupby(["dataset", "bin_strategy"])["bytes"].idxmax()]
        v = dom.groupby(["bin_strategy", "part"]).size()
        for bs in sorted(dom["bin_strategy"].unique()):
            L.append(f"- Доминирующая часть памяти Hybrid ({bs}), число датасетов: "
                     + ", ".join(f"{k}: {n}" for k, n in v[bs].sort_values(ascending=False).items()) + ".")
    for m in ("Hybrid", "EquiWidth"):
        g = summary[(summary["model"] == m) & (summary["n_empty_buckets"] > 0)]
        if len(g):
            r = g.loc[g["empty_bucket_pct_of_logical"].idxmax()]
            L.append(f"- Пустые бакеты {m}: в {len(g)} конфигурациях; максимум {r['empty_bucket_pct_of_logical']:.1f} % логической "
                     f"памяти ({r['dataset']}, {r['bin_strategy']}: {int(r['n_empty_buckets'])} из {int(r['n_buckets'])}).")
    for m, g in summary.groupby("model"):
        L.append(f"- {m}: pickle / logical — медиана ×{np.median(g['pickle_bytes'] / g['logical_bytes']):.1f}; "
                 f"пик tracemalloc при построении — медиана {_kib(g['tracemalloc_peak_bytes'].median())} KiB.")
    return L


def write_markdown(summary, comps, cdf, drift_df, pbs, have_drift_tables, fd_mem=None):
    L = ["# Анализ памяти (MEMORY_ANALYSIS)", "",
         "Сгенерировано `python -m paper_experiments.memory_analysis` из измерений в `raw/`; все числа ниже — из "
         "`memory_summary.csv`, `memory_breakdown.csv`, `hybrid_cdf_distribution.csv`, `peak_memory.csv`, "
         "`drift_stage_memory.csv`. Единицы: байты (KiB = 1024 B).", "",
         "## Методика: какие байты мы считаем", "",
         "| Метрика | Что это | Как измерено |", "|---|---|---|",
         "| `logical_bytes` | постоянная (persistent) память модели — числа, которые надо хранить | `n_values() × 8 B`, определение каждой модели (как в тезисе и репо), не менялось |",
         "| разбивка (`memory_breakdown.csv`) | на что уходят `n_values` | компоненты по модели; assert: Σ компонент = n_values = глобальные + Σ по бакетам + не-бакетные (MCV) |",
         "| `per_bucket_memory_stats.csv` | память, приписанная бакету | Hybrid: `bucket_table().bucket_values × 8` (счётчик + граница при неравной ширине + параметры CDF + код типа); общие числа (min/ширина сетки) — один раз в глобальных |",
         "| `empty_bucket_bytes` | накладные расходы пустых бакетов | Σ памяти бакетов с count = 0 |",
         "| `alt_n_values_tree_baked` | альтернативный учёт DecisionTree | пороги + значения листьев запечённой таблицы вместо `3 × node_count` |",
         "| `compact_raw_bytes` / `compact_npz_bytes` | минимальная сериализация тех же чисел | numpy-массивы (int8 коды, int32 смещения, float64 значения); npz — с заголовками |",
         "| `pickle_bytes` | сериализованный живой объект | `len(pickle.dumps(model, protocol=5))` — включает состояние sklearn, кэши, инструментирование |",
         "| `derived_*_bytes` | runtime-структуры (префиксные суммы, запечённые массивы, lookup-списки) | 8 B на число / глубокий `sys.getsizeof`; НЕ входит в persistent |",
         "| `object_graph_bytes` | выделенные Python-объекты модели | глубокий `sys.getsizeof` всего объекта |",
         "| `tracemalloc_peak_bytes` | пик Python/numpy-аллокаций при построении | второе построение в том же процессе под tracemalloc (включает временные обучающие данные) |",
         "| `rss_peak_delta_bytes` | прирост RSS процесса при построении | свежий подпроцесс на конфигурацию; VmHWM после − VmRSS до (HWM сброшен через /proc/self/clear_refs) |",
         "| `reservoir_bytes` | 100k-выборка | `sample.nbytes`; общий вход, НЕ часть оценщика |",
         "| feedback | память обратной связи | STHist: онлайн по запросу, буфер 0 (только Counter событий); Hybrid FT/RB: см. drift check |",
         "",
         "Частоты Fourier-признаков (32 числа `logspace(0, 3, 32)`) — константы спецификации, одинаковые для всех MLP; "
         "они не входят ни в `n_values`, ни в per-bucket память (как код модели). Сетка (min, ширина) — 2 числа один раз.", ""]
    L += ["## Ключевые выводы (вычислены из таблиц)", ""] + key_findings(summary, comps) + [""]
    L += ["## Итог по моделям (logical bytes, KiB)", ""]
    pv = summary.pivot_table(index="dataset", columns=["bin_strategy", "model"], values="logical_bytes", aggfunc="first")
    for bs in sorted(summary["bin_strategy"].unique()):
        ms = [m for m in COLORS if (bs, m) in pv.columns]
        L += [f"### {bs}", "", "| dataset | " + " | ".join(ms) + " |", "|---" * (len(ms) + 1) + "|"]
        for d in pv.index:
            L.append(f"| {d} | " + " | ".join(_kib(pv.loc[d, (bs, m)]) for m in ms) + " |")
        L.append("")
    hy = summary[summary["model"] == "Hybrid"]
    if not hy.empty:
        L += ["## Hybrid: куда уходит память", ""]
        c = comps[comps["model"] == "Hybrid"].copy()
        c["tier"] = c["component"].str.extract(r"CDF params T(\d)")[0]
        for bs, g in c.groupby("bin_strategy"):
            tot = g.groupby("dataset")["bytes"].sum()
            by_tier = g.dropna(subset=["tier"]).groupby(["dataset", "tier"])["bytes"].sum().unstack(fill_value=0)
            counts = g[g["component"] == "bucket counts"].set_index("dataset")["bytes"]
            L += [f"### {bs}", "", "| dataset | total KiB | counts % | Tier1 % | Tier2 % | Tier3 % | dominant |", "|---|---|---|---|---|---|---|"]
            for d in tot.index:
                t = {k: (100 * by_tier.loc[d, k] / tot[d] if d in by_tier.index and k in by_tier.columns else 0.0) for k in "123"}
                cp = 100 * counts.get(d, 0) / tot[d]
                parts = {"counts": cp, "Tier1": t["1"], "Tier2": t["2"], "Tier3": t["3"]}
                L.append(f"| {d} | {_kib(tot[d])} | {cp:.1f} | {t['1']:.1f} | {t['2']:.1f} | {t['3']:.1f} | {max(parts, key=parts.get)} |")
            L.append("")
        L += ["Пустые бакеты (Hybrid): ", ""]
        for _, r in hy.iterrows():
            if r["n_empty_buckets"]:
                L.append(f"- {r['dataset']} / {r['bin_strategy']}: {int(r['n_empty_buckets'])} из {int(r['n_buckets'])} бакетов пусты, "
                         f"{r['empty_bucket_bytes']:.0f} B = {r['empty_bucket_pct_of_logical']:.2f} % логической памяти")
        L.append("")
        tree = hy[hy["x_alt_tree_3xnodes_values"] > 0] if "x_alt_tree_3xnodes_values" in hy else hy.iloc[0:0]
        if len(tree):
            L += ["DecisionTree: `3 × node_count` против запечённой таблицы (порог + значение листа):", ""]
            for _, r in tree.iterrows():
                L.append(f"- {r['dataset']} / {r['bin_strategy']}: {int(r['x_alt_tree_3xnodes_values'])} → "
                         f"{int(r['x_alt_tree_baked_values'])} чисел; n_values {int(r['n_values'])} → {int(r['x_alt_n_values_tree_baked'])}")
            L.append("")
    L += ["## Сериализация против логической памяти", "",
          "| model | bin | median pickle / logical | median compact_raw / logical | median object_graph / logical |", "|---|---|---|---|---|"]
    for (m, bs), g in summary.groupby(["model", "bin_strategy"]):
        L.append(f"| {m} | {bs} | {np.median(g['pickle_bytes'] / g['logical_bytes']):.2f} | "
                 f"{np.median(g['compact_raw_bytes'] / g['logical_bytes']):.2f} | {np.median(g['object_graph_bytes'] / g['logical_bytes']):.2f} |")
    L += ["", "## Пиковая память построения", "",
          "| model | bin | median tracemalloc peak, KiB | max tracemalloc peak, KiB | median RSS Δ, KiB | max RSS Δ, KiB |",
          "|---|---|---|---|---|---|"]
    for (m, bs), g in summary.groupby(["model", "bin_strategy"]):
        L.append(f"| {m} | {bs} | {_kib(g['tracemalloc_peak_bytes'].median())} | {_kib(g['tracemalloc_peak_bytes'].max())} | "
                 f"{_kib(g['rss_peak_delta_bytes'].median())} | {_kib(g['rss_peak_delta_bytes'].max())} |")
    res = summary["reservoir_bytes"].iloc[0]
    L += ["", f"Резервуар (общий вход всех моделей, не часть оценщика): {int(res):,} B = {res / 1024:.0f} KiB "
          f"({int(summary['reservoir_rows'].iloc[0]):,} строк × int64).", ""]
    if not drift_df.empty:
        L += ["## Hybrid под дрейфом: FT / разметка / RB (drift check)", "",
              "Первые шаги сценария до первого checkpoint (100 000 изменённых строк) прогнаны через `drift_harness.run_drift` "
              "с обёрнутыми `update` / `mark` / `rebuild` (два прогона: tracemalloc и RSS). "
              "Журнал adapt-запросов (low, high, truth, estimate) — 32 B на запрос, принадлежит харнессу/СУБД.", "",
              "| scenario | model | bin | stage | step | tracemalloc peak, KiB | RSS Δ, KiB | time, s | marked | adapt log, KiB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        key = ["dataset", "model", "bin_strategy", "stage", "step"]
        tm = drift_df[drift_df["mode"] == "tracemalloc"].set_index(key).sort_index()
        rs = drift_df[drift_df["mode"] == "rss"].set_index(key).sort_index()
        # FT / MARK run at every step up to the checkpoint: show the checkpoint step only (+ RB)
        tm = tm[tm.index.get_level_values("step") == tm["checkpoint_step"]]
        for idx, r in tm.iterrows():
            rss = rs["rss_peak_delta_bytes"].get(idx) if idx in rs.index else None
            rss = rss.iloc[0] if isinstance(rss, pd.Series) else rss
            L.append(f"| {idx[0]} | {idx[1]} | {idx[2]} | {idx[3]} | {idx[4]} | {_kib(r['tracemalloc_peak_bytes'])} | {_kib(rss)} | "
                     f"{r['time_s']:.2f} | {int(r['marked_buckets'])} | {_kib(r['adapt_log_logical_bytes'])} |")
        L += ["", "Время в таблице измерено под tracemalloc (замедляет обучение) — для латентности используйте результаты дрейфа.", ""]
    if fd_mem is not None and not fd_mem.empty:
        L += ["## FD-reduction: память по пиклам моделей (fd_reduction_memory.csv)", "",
              f"{len(fd_mem)} конфигураций проанализированы из сохранённых объектов (без перестроения, без пиков). "
              f"n_values совпадает с отчётом FD-reduction: {int(fd_mem['n_values_matches_reported'].sum())} из {len(fd_mem)}.", "",
              "| model | configs | median logical KiB | median pickle / logical | median empty-bucket % |", "|---|---|---|---|---|"]
        for m, g in fd_mem.groupby("model"):
            L.append(f"| {m} | {len(g)} | {_kib(g['logical_bytes'].median())} | {np.median(g['pickle_bytes'] / g['logical_bytes']):.1f} | "
                     f"{g['empty_bucket_pct_of_logical'].median():.2f} |")
        L.append("")
    L += ["## Распределение CDF Hybrid", "",
          "Полные таблицы: `hybrid_cdf_distribution.csv` (статические построения)" +
          (", `hybrid_cdf_distribution_drift.csv` (каждая точка дрейфа из bucket_tables)" if have_drift_tables else
           "; таблиц по точкам дрейфа пока нет (появятся после прогонов DSB, затем перезапуск `--report-only`)") + ".", ""]
    if not cdf.empty:
        t = cdf[(cdf["level"] == "tier")]
        L += ["| dataset | bin | tier | buckets % | memory % | cardinality % (est) |", "|---|---|---|---|---|---|"]
        for _, r in t.sort_values(["dataset", "bin_strategy", "tier"]).iterrows():
            L.append(f"| {r['dataset']} | {r['bin_strategy']} | {r['tier']} | {r['pct_buckets']:.1f} | "
                     f"{(r['pct_of_total_memory'] or 0):.1f} | {(r['pct_cardinality_est'] or 0):.1f} |")
        L.append("")
    L += ["## Ограничения", "",
          "- `logical_bytes` — модель хранения «8 байт на число» из тезиса/репо; реальная СУБД хранила бы float4/int4 и сжимала бы.",
          "- STHist в статике: метаданные + prequential-фидбэк по первым "
          f"{STHIST_FEEDBACK_QUERIES:,} запросам основного workload (n_values = 2B+1 не зависит от числа запросов).",
          "- RSS-приросты — грубая метрика (аллокатор не всегда возвращает память ОС; tracemalloc точнее для Python-аллокаций).",
          "- Пик построения Hybrid включает массив частот выборки по всему домену (8 B × размер домена).",
          "- Hybrid-Corrected-PredictOnly (диагностика) в статических замерах не строился: его валидация каноническая, "
          "поэтому выбранные модели и память совпадают с Hybrid по построению; его объекты из FD-reduction разобраны в "
          "`fd_reduction_memory.csv`.", ""]
    (OUT / "MEMORY_ANALYSIS.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--worker", action="store_true")
    p.add_argument("--drift-worker", action="store_true")
    p.add_argument("--ds-json")
    p.add_argument("--model")
    p.add_argument("--bucket-arg")
    p.add_argument("--mode", default="tracemalloc")
    p.add_argument("--out-json")
    p.add_argument("--datasets", nargs="*", help="substring filter on dataset labels")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--buckets", nargs="*", default=BUCKET_ARGS)
    p.add_argument("--skip-drift", action="store_true")
    p.add_argument("--skip-fd-models", action="store_true", help="skip the FD-reduction model-file pass")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--force", action="store_true", help="re-measure configurations already in raw/")
    a = p.parse_args()
    if a.worker:
        run_worker(json.loads(a.ds_json), a.model, a.bucket_arg, a.out_json)
        return
    if a.drift_worker:
        label, shift_dir, wl, cnt = json.loads(a.ds_json)
        run_drift_worker(label, shift_dir, wl, cnt, a.bucket_arg, a.mode, a.out_json, a.model or "Hybrid")
        return
    a.models_given = a.models is not None
    a.models = a.models or MODELS
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        run_all(a)
    build_report(a)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()

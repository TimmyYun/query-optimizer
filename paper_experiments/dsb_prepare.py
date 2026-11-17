"""
DSB expansion, part 2: static datasets and workloads for the selected columns.

    poetry run python -m paper_experiments.dsb_prepare            # meta.pkl + 3 workloads per column
    poetry run python -m paper_experiments.dsb_prepare --official # + official dsqgen ranges where applicable

data/dsb_expanded/<table>.<column>/
    meta.pkl                    (mn, mx, N, freq, sample, k, skew, kurt) — prepare_benchmark.build_meta, the
                                same pipeline as data/dsb and the synthetic data (sample 100 000, seed 42,
                                FD with n = N, cap 2000)
    workload_dd_narrow.csv      MAIN. Data-driven (thesis §4.4.2): anchor = uniform draw from the 100k sample,
                                w ~ U{1..100}, offset ~ U{0..w-1}, low = anchor - offset, high = low + w
                                (= generate_data_driven_workload(max_width=101) semantics), both ends clipped to
                                [mn, mx] (workload.py clips low at 0, which is wrong for negative domains such as
                                ss_net_profit). Stream default_rng([42, 10]).
    workload_uniform_narrow.csv low ~ U{mn..mx}, same w rule, high = min(low + w, mx). Stream [42, 11].
    workload_uniform_wide.csv   repo wide rule (workload.generate_workload(wide=True)): low ~ U{mn..mx-1},
                                w ~ U{floor(0.1 D)..D-1}, D = mx - mn, high = min(low + w, mx). Stream [42, 12].
    workload_official.csv       distinct "<col> BETWEEN a AND b" ranges of dsqgen on the unmodified official
                                templates (prepare_benchmark extraction), only for columns the templates filter.
All workloads have 1 000 000 queries (official: as many distinct ranges as the templates produce).
Zero-truth queries are dropped later by benchmark_utils.load_and_filter_workload, as for every benchmark.
Vectorized draws, so the streams differ from workload.py's per-query loop (documented, seeded).

Writes data/dsb_expanded/READY_STATIC.json when every column has meta.pkl + the three workloads.
"""
import argparse
import json
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from paper_experiments.dsb_columns import CANDIDATES, ROOT, SELECTED_ALL, load_column
from paper_experiments.provenance import file_sha256, write_manifest

DATA = ROOT / "data" / "dsb_expanded"
N_QUERIES = 1_000_000
SAMPLE_SIZE = 100_000
WIDTH_MAX = 100  # thesis §4.4.2: width in [1, 100]
STREAMS = {"dd_narrow": [42, 10], "uniform_narrow": [42, 11], "uniform_wide": [42, 12]}
TOOLS = ROOT / "DSB benchmark" / "dsb" / "code" / "tools"
TEMPLATES = ROOT / "DSB benchmark" / "dsb" / "query_templates_pg"


def scale_of(key: str) -> int:
    table, col = key.split(".")
    return CANDIDATES[table][col]


def dd_narrow(sample, mn, mx, n, rng):
    anchors = np.asarray(sample, dtype=np.int64)[rng.integers(0, len(sample), size=n)]
    w = rng.integers(1, WIDTH_MAX + 1, size=n)
    off = rng.integers(0, w)  # U{0..w-1}
    lo = np.clip(anchors - off, mn, mx)
    hi = np.clip(lo + w, mn, mx)
    return lo, hi


def uniform_narrow(mn, mx, n, rng):
    lo = rng.integers(mn, mx + 1, size=n)
    w = rng.integers(1, WIDTH_MAX + 1, size=n)
    return lo, np.minimum(lo + w, mx)


def uniform_wide(mn, mx, n, rng):
    D = mx - mn
    lo = rng.integers(mn, max(mn + 1, mx), size=n)
    w = rng.integers(max(1, int(D * 0.1)), max(2, D), size=n)
    return lo, np.minimum(lo + w, mx)


def save_ranges(lo, hi, path: Path):
    pd.DataFrame({"low": lo.astype(np.int64), "high": hi.astype(np.int64)}).to_csv(path, index=False)


def build_column(key: str, force=False):
    from prepare_benchmark import build_meta  # the repo pipeline, used read-only
    out = DATA / key
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.pkl"
    if force or not meta_path.exists():
        table, col = key.split(".")
        values, _ = load_column(table, col, scale_of(key))
        meta = build_meta(values, out, SAMPLE_SIZE)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
    mn, mx, N, freq, sample, k, skew, kurt = pickle.load(open(meta_path, "rb"))
    print(f"[{key}] N={N:,} domain=[{mn}, {mx}] k={k} sample={len(sample):,}", flush=True)
    for name, fn in (("dd_narrow", lambda r: dd_narrow(sample, mn, mx, N_QUERIES, r)),
                     ("uniform_narrow", lambda r: uniform_narrow(mn, mx, N_QUERIES, r)),
                     ("uniform_wide", lambda r: uniform_wide(mn, mx, N_QUERIES, r))):
        p = out / f"workload_{name}.csv"
        if force or not p.exists():
            save_ranges(*fn(np.random.default_rng(STREAMS[name])), p)
    return out


# ---------------- official queries (dsqgen) ----------------

def official(keys, instances=1000, rngseed=42):
    """dsqgen on the official templates that filter a selected column, then prepare_benchmark's range extraction."""
    from prepare_benchmark import dsb_extract_ranges, write_ranges
    cols = {k.split(".")[1]: k for k in keys}
    pattern = re.compile("|".join(rf"{c}\s+between" for c in cols), re.IGNORECASE)
    templates = sorted(t for t in TEMPLATES.rglob("*.tpl")
                       if t.name != "postgres.tpl" and pattern.search(t.read_text(errors="ignore")))
    qdir = DATA / "queries"
    staging = (qdir / "_templates").resolve()
    for src in sorted({t.parent for t in templates}):
        shutil.copytree(src, staging / src.name, dirs_exist_ok=True)
    for tpl in templates:
        tdir = qdir / f"{tpl.parent.name}__{tpl.stem}"
        if tdir.exists() and len(list(tdir.glob("query_*.sql"))) >= instances:
            continue
        tdir.mkdir(parents=True, exist_ok=True)
        cmd = [str((TOOLS / "dsqgen").resolve()), "-output_dir", str(tdir.resolve()), "-streams", str(instances),
               "-directory", str(staging / tpl.parent.name), "-template", tpl.name, "-dialect", "postgres",
               "-scale", "1", "-rngseed", str(rngseed)]
        print("  " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=TOOLS, check=True, stdout=subprocess.DEVNULL)
    sql = sorted(qdir.glob("*/query_*.sql"))
    status = {}
    for col, key in cols.items():
        ranges = [r for f in sql for r in dsb_extract_ranges(f.read_text(errors="ignore"), col, scale_of(key))]
        if ranges:
            uniq = write_ranges(ranges, DATA / key / "workload_official.csv")
            status[key] = {"official": "OK", "distinct_ranges": len(uniq), "predicates": len(ranges)}
        else:
            status[key] = {"official": "NOT_APPLICABLE", "reason": "no official template filters it with BETWEEN"}
    (DATA / "official_status.json").write_text(json.dumps(status, indent=1))
    return status


def write_ready():
    entries = {}
    for key in SELECTED_ALL:
        d = DATA / key
        wl = {n: d / f"workload_{n}.csv" for n in ("dd_narrow", "uniform_narrow", "uniform_wide", "official")}
        if not (d / "meta.pkl").exists() or not all(wl[n].exists() for n in ("dd_narrow", "uniform_narrow", "uniform_wide")):
            return None
        entries[key] = {
            "dataset_dir": str(d.relative_to(ROOT)),
            "meta_sha256": file_sha256((d / "meta.pkl").relative_to(ROOT)),
            "workloads": {n: {"path": str(p.relative_to(ROOT)), "sha256": file_sha256(p.relative_to(ROOT))}
                          for n, p in wl.items() if p.exists()},
            "main_workload": str(wl["dd_narrow"].relative_to(ROOT)),
        }
    rec = {"columns": entries, "n_queries": N_QUERIES, "sample_size": SAMPLE_SIZE, "seed": 42,
           "workload_streams": STREAMS, "width_rule": "w ~ U{1..100}"}
    (DATA / "READY_STATIC.json").write_text(json.dumps(rec, indent=1))
    write_manifest(DATA / "manifest_static.json", part="dsb_expanded/static-data", **rec)
    print(f"wrote {DATA / 'READY_STATIC.json'}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    keys = args.only or list(SELECTED_ALL)
    for key in keys:
        build_column(key, force=args.force)
    if args.official:
        print(json.dumps(official(keys), indent=1))
    write_ready()


if __name__ == "__main__":
    main()

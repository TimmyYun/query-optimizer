#!/usr/bin/env python3
"""
Compute true counts and selectivities for a single-column dataset (CSV with one integer per line)
against a workload SQL file containing equality and range queries.

Input:
  - data/*.csv (e.g., salary_uniform_500mb.csv, one int per line, no header)
  - workload/workload_100.sql with lines like:
      SELECT COUNT(*) FROM salary WHERE salary = 123456;
      SELECT COUNT(*) FROM salary WHERE salary BETWEEN 45000 AND 90000;

Output:
  - truth/truth_mapping.csv : one row per (dataset, query)
  - truth/<dataset_stem>_truth.csv : per-dataset breakdown
"""

# python compute_true_selectivity.py \
#   --data-glob "data/salary_*_500mb.csv" \
#   --workload "workload/100/workload_100.sql" \
#   --outdir truth

import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Workload parsing
# ---------------------------

EQ_RE = re.compile(
    r"select\s+count\(\*\)\s+from\s+([\w\.]+)\s+where\s+([\w\.]+)\s*=\s*(\d+)\s*;",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"select\s+count\(\*\)\s+from\s+([\w\.]+)\s+where\s+([\w\.]+)\s+between\s+(\d+)\s+and\s+(\d+)\s*;",
    re.IGNORECASE,
)

def parse_workload(sql_path: Path) -> List[Dict[str, Any]]:
    queries = []
    last_comment: Optional[str] = None

    for raw in sql_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            last_comment = line[2:].strip()
            continue

        m_eq = EQ_RE.match(line)
        if m_eq:
            table, column, val = m_eq.groups()
            queries.append({
                "kind": "equality",
                "table": table,
                "column": column,
                "value": int(val),
                "low": None,
                "high": None,
                "sql": line,
                "comment": last_comment,
            })
            last_comment = None
            continue

        m_rg = RANGE_RE.match(line)
        if m_rg:
            table, column, lo, hi = m_rg.groups()
            lo_i, hi_i = int(lo), int(hi)
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            queries.append({
                "kind": "range",
                "table": table,
                "column": column,
                "value": None,
                "low": lo_i,
                "high": hi_i,
                "sql": line,
                "comment": last_comment,
            })
            last_comment = None
            continue

        # Ignore other lines (e.g., psql meta), or raise if you want to be strict.
    return queries

# ---------------------------
# CSV streaming passes
# ---------------------------

def pass_min_max(csv_path: Path, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    mn, mx = None, None
    n = 0
    for chunk in pd.read_csv(
        csv_path, header=None, names=["v"], dtype="int64",
        chunksize=chunksize, engine="c"
    ):
        v = chunk["v"].to_numpy()
        n += v.size
        cmin, cmax = int(v.min()), int(v.max())
        mn = cmin if mn is None else min(mn, cmin)
        mx = cmax if mx is None else max(mx, cmax)
    if mn is None or mx is None:
        raise ValueError(f"Empty CSV: {csv_path}")
    return mn, mx, n

def pass_frequency(csv_path: Path, mn: int, mx: int, chunksize: int = 1_000_000) -> np.ndarray:
    width = mx - mn + 1
    freq = np.zeros(width, dtype=np.int64)
    for chunk in pd.read_csv(
        csv_path, header=None, names=["v"], dtype="int64",
        chunksize=chunksize, engine="c"
    ):
        vals = chunk["v"].to_numpy()
        # shift to [0, width-1]
        shifted = vals - mn
        # guard (if any value somehow falls outside)
        mask = (shifted >= 0) & (shifted < width)
        if not np.all(mask):
            shifted = shifted[mask]
        # bincount per chunk, aggregate
        if shifted.size:
            bc = np.bincount(shifted, minlength=width)
            freq += bc
    return freq

# ---------------------------
# Truth computation
# ---------------------------

def count_eq(freq: np.ndarray, mn: int, val: int) -> int:
    idx = val - mn
    if 0 <= idx < freq.size:
        return int(freq[idx])
    return 0

def count_range(freq_psum: np.ndarray, mn: int, lo: int, hi: int) -> int:
    if hi < lo:
        lo, hi = hi, lo
    lo_idx = max(0, lo - mn)
    hi_idx = min(freq_psum.size - 1, hi - mn)
    if hi_idx < 0 or lo_idx > freq_psum.size - 1:
        return 0
    if lo_idx == 0:
        return int(freq_psum[hi_idx])
    return int(freq_psum[hi_idx] - freq_psum[lo_idx - 1])

def compute_truth_for_dataset(csv_path: Path, queries: List[Dict[str, Any]]) -> pd.DataFrame:
    mn, mx, total = pass_min_max(csv_path)
    freq = pass_frequency(csv_path, mn, mx)
    assert total == int(freq.sum()), "Row count mismatch after frequency pass"
    psum = np.cumsum(freq)

    rows = []
    for i, q in enumerate(queries, start=1):
        if q["kind"] == "equality":
            cnt = count_eq(freq, mn, q["value"])
        else:
            cnt = count_range(psum, mn, q["low"], q["high"])
        sel = cnt / total if total > 0 else 0.0
        rows.append({
            "dataset": csv_path.name,
            "query_index": i,
            "kind": q["kind"],
            "value": q.get("value"),
            "low": q.get("low"),
            "high": q.get("high"),
            "true_count": cnt,
            "total_rows": total,
            "true_selectivity": sel,
            "sql": q["sql"],
            "comment": q.get("comment"),
        })
    return pd.DataFrame(rows)

# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Map workload queries to true selectivities for each dataset CSV.")
    ap.add_argument("--data-glob", default="data/salary_*_500mb.csv", help="Glob for input CSVs")
    ap.add_argument("--workload", default="workload/workload_100.sql", help="SQL workload file")
    ap.add_argument("--outdir", default="truth", help="Output folder")
    args = ap.parse_args()

    sql_path = Path(args.workload)
    queries = parse_workload(sql_path)
    if not queries:
        raise SystemExit(f"No queries parsed from {sql_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for csv_path in sorted(Path(".").glob(args.data_glob)):
        if csv_path.suffix.lower() != ".csv":
            continue
        print(f"[dataset] {csv_path.name}: scanning …")
        df = compute_truth_for_dataset(csv_path, queries)
        df.to_csv(outdir / f"{csv_path.stem}_truth.csv", index=False)
        all_rows.append(df)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(outdir / "truth_mapping.csv", index=False)
        print(f"[done] wrote: {outdir/'truth_mapping.csv'}")
    else:
        print(f"No CSV files matched pattern: {args.data_glob}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Adapter from public benchmarks to this repo's contract.

The data and the queries come from the original benchmark unchanged; this
script only converts them into
    data/<bench>/<column>/meta.pkl                 (mn, mx, N, freq, sample, k, skew, kurt)
    data/<bench>/<column>/workload_official.csv    1-D ranges taken from the benchmark's own queries
    data/<bench>/<column>/workload_driven_<n>.csv  (STATS-CEB only) our data-driven supplement, labelled as such
    data/<bench>/<column>/drift/step_<i>.pkl       (DSB only) one step per official refresh checkpoint

DSB (Ding et al., VLDB 2021), github.com/microsoft/dsb:
    columns  web_sales.ws_wholesale_cost, web_sales.ws_list_price — the web_sales
             columns that official templates filter with a plain BETWEEN and that the
             official refresh streams modify. DECIMAL(7,2) is stored as integer cents
             (x100, lossless), so queries and data are scaled the same way.
    queries  dsqgen on the unmodified official templates (query_templates_pg), the same
             call as the official scripts/generate_workload.py; every
             "<col> BETWEEN a AND b" is extracted, ratio predicates (x / col BETWEEN) are not.
    drift    checkpoints exported by "DSB benchmark/DSB_GENERATOR.ipynb" after replaying
             the official refresh streams (0, 5, ..., 100 % churn).

STATS-CEB (Han et al., VLDB 2022), github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark:
    data     datasets/stats_simplified/{posts,users}.csv at a pinned commit; NULLs are
             dropped, since no range predicate selects a NULL.
    queries  workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql:
             each sub-query's predicates on one column are intersected into one range.
             Truth of the pure 1-D sub-queries is checked against the cardinality in the file.

Usage:
    python prepare_benchmark.py --bench stats-ceb               # downloads ~6 MB, then converts
    python prepare_benchmark.py --bench dsb --generate-queries  # needs the built DSB toolkit
    python prepare_benchmark.py --bench dsb                     # only convert what exists
"""
import argparse
import collections
import pickle
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from data.datasets import (
    scan_min_max_count,
    calculate_skew_kurt,
    build_frequency_and_sample,
    freedman_diaconis_bins,
)
from workload.workload import generate_data_driven_workload, save_workload_csv
from models.common import RangeQuery
from benchmark_utils import exact_counts

SEED = 42
# Sample the models are built from. DSB keeps the synthetic benchmark's 100k.
# STATS tables have only 40k-92k rows, so 100k would hand the models the whole
# column; 30k is what PostgreSQL ANALYZE samples (300 x default_statistics_target=100).
SAMPLE_SIZE = {"dsb": 100_000, "stats-ceb": 30_000}

# ==========================================
# Shared: values -> contract files
# ==========================================


def build_meta(values: np.ndarray, work_dir: Path, sample_size: int):
    """Same pipeline as data/datasets.py, so real and synthetic datasets are built identically."""
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "data.csv"
    pd.Series(values.astype(np.int64)).to_csv(data_path, index=False, header=False)
    mn, mx, N = scan_min_max_count(data_path)
    freq, sample = build_frequency_and_sample(data_path, mn, mx, N, sample_size, SEED)
    k = freedman_diaconis_bins(sample, mn, mx, N)
    skew, kurt = calculate_skew_kurt(data_path)
    data_path.unlink()
    return mn, mx, N, freq, sample, k, skew, kurt


def write_meta(values, out_dir: Path, sample_size: int):
    meta = build_meta(values, out_dir, sample_size)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    mn, mx, N, _, sample, k, _, _ = meta
    print(f"  meta.pkl: N={N:,} domain=[{mn}, {mx}] k={k} sample={len(sample):,}")
    return meta


def write_step(values, step_path: Path, sample_size: int):
    mn, mx, N, freq, sample, k, _, _ = build_meta(values, step_path.parent, sample_size)
    with open(step_path, "wb") as f:
        pickle.dump((mn, mx, N, freq, sample, k), f)
    print(f"  {step_path.name}: N={N:,} domain=[{mn}, {mx}]")


def write_ranges(ranges, path: Path):
    """Deduplicated, sorted; each distinct predicate is one query."""
    uniq = sorted(set(ranges))
    save_workload_csv([RangeQuery(lo, hi) for lo, hi in uniq], path)
    print(f"  {path.name}: {len(uniq)} distinct ranges (from {len(ranges)} predicates)")
    return uniq


# ==========================================
# DSB
# ==========================================

DSB_COLUMNS = {
    # column -> (table, scale to integer)
    "ws_wholesale_cost": ("web_sales", 100),
    "ws_list_price": ("web_sales", 100),
}
DSB_CHURN = ["initial"] + [f"churn_{p}pct" for p in range(5, 105, 5)]
NUM = r"(-?\d+(?:\.\d+)?)"


def dsb_generate_queries(tools_dir: Path, templates_root: Path, out_dir: Path, instances: int, rngseed: int, scale: int):
    """dsqgen per official template, mirroring scripts/generate_workload.py (param_dist default)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile("|".join(rf"{c}\s+between" for c in DSB_COLUMNS), re.IGNORECASE)
    templates = sorted(
        t for t in templates_root.rglob("*.tpl")
        if t.name != "postgres.tpl" and pattern.search(t.read_text(errors="ignore"))
    )
    print(f"  {len(templates)} official templates filter a target column")

    # dsqgen splits -directory at spaces ("DSB benchmark/..."), so run it on a
    # byte-for-byte copy of the official template folders under out_dir.
    staging = (out_dir / "_templates").resolve()
    if " " in str(staging):
        raise ValueError(f"dsqgen cannot handle spaces in paths: {staging}; pass --out-root without spaces")
    for src_dir in sorted({t.parent for t in templates}):
        shutil.copytree(src_dir, staging / src_dir.name, dirs_exist_ok=True)

    for tpl in templates:
        tpl_out = out_dir / tpl.stem
        if tpl_out.exists() and len(list(tpl_out.glob("query_*.sql"))) >= instances:
            continue
        tpl_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str((tools_dir / "dsqgen").resolve()),  # cwd below is tools_dir, so the path must be absolute
            "-output_dir", str(tpl_out.resolve()),
            "-streams", str(instances),
            "-directory", str(staging / tpl.parent.name),
            "-template", tpl.name,
            "-dialect", "postgres",
            "-scale", str(scale),
            "-rngseed", str(rngseed),
        ]
        print("  " + " ".join(cmd))
        # dsqgen reads tpcds.idx from its working directory
        subprocess.run(cmd, cwd=tools_dir, check=True, stdout=subprocess.DEVNULL)


def dsb_extract_ranges(sql: str, column: str, scale: int):
    """Plain '<alias.>col BETWEEN a AND b' only; 'x / col BETWEEN', 'a * 0.01' bounds are skipped."""
    out = []
    rx = re.compile(rf"(?:\w+\.)?{column}\s+between\s+{NUM}\s+and\s+{NUM}(?!\s*[*/+-])", re.IGNORECASE)
    for m in rx.finditer(sql):
        if sql[: m.start()].rstrip().endswith(("/", "*")):
            continue
        lo, hi = (int(round(float(v) * scale)) for v in m.groups())
        out.append((lo, hi))
    return out


def dsb_load_checkpoint(path: Path, scale: int):
    s = pd.read_csv(path).iloc[:, 0].dropna()
    return np.rint(s.to_numpy(dtype=np.float64) * scale).astype(np.int64)


def prepare_dsb(args):
    root = Path(args.out_root or "data/dsb")
    dsb_dir = Path(args.source_dir or "DSB benchmark/dsb")
    query_dir = root / "queries"

    if args.generate_queries:
        tools = dsb_dir / "code" / "tools"
        if not (tools / "dsqgen").exists():
            raise FileNotFoundError(f"{tools}/dsqgen not found. Run 'DSB benchmark/setup_dsb.sh' first.")
        dsb_generate_queries(tools, dsb_dir / "query_templates_pg", query_dir, args.instances, args.rngseed, args.scale)

    for col, (table, scale) in DSB_COLUMNS.items():
        print(f"\n[DSB] {table}.{col} (x{scale})")
        out = root / col
        out.mkdir(parents=True, exist_ok=True)

        sql_files = sorted(query_dir.glob("query*/query_*.sql")) if query_dir.exists() else []
        if sql_files:
            ranges = [r for f in sql_files for r in dsb_extract_ranges(f.read_text(errors="ignore"), col, scale)]
            write_ranges(ranges, out / "workload_official.csv")
        elif not (out / "workload_official.csv").exists():
            print(f"  no generated queries in {query_dir}; run with --generate-queries")

        ckpt = {s: root / "drift" / f"{table}__{col}__{s}.csv" for s in DSB_CHURN}
        if not ckpt["initial"].exists():
            print(f"  {ckpt['initial']} not found; run './setup_env.sh --dsb' to export checkpoints")
            continue
        write_meta(dsb_load_checkpoint(ckpt["initial"], scale), out, args.sample_size)
        for i, step in enumerate(DSB_CHURN):
            if ckpt[step].exists():
                write_step(dsb_load_checkpoint(ckpt[step], scale), out / "drift" / f"step_{i}.pkl", args.sample_size)
            else:
                print(f"  missing checkpoint {ckpt[step].name}: drift stops at step {i - 1}")
                break


# ==========================================
# STATS-CEB
# ==========================================

STATS_COMMIT = "670cb8d4bf4cbfa32f94fdf17f33973d3fd67d1b"  # master, 2023-03-14
STATS_URL = f"https://raw.githubusercontent.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/{STATS_COMMIT}"
STATS_FILES = {
    "posts.csv": "datasets/stats_simplified/posts.csv",
    "users.csv": "datasets/stats_simplified/users.csv",
    "stats_CEB_single_table_sub_query.sql": "workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql",
}
# Integer columns with >= 8 distinct 1-D ranges in the official sub-queries
# (timestamps are excluded: second resolution does not fit a dense freq array).
STATS_COLUMNS = [
    "posts.Score", "posts.ViewCount", "posts.CommentCount", "posts.AnswerCount", "posts.FavoriteCount",
    "users.Reputation", "users.Views", "users.UpVotes", "users.DownVotes",
]
PRED = re.compile(r"(\w+)\.(\w+)\s*(>=|<=|=|>|<)\s*(-?\d+)(?![\d.:])")
# Any filter on a column, including '...'::timestamp ones (used only to decide what is pure 1-D)
ANY_PRED = re.compile(r"(\w+)\.(\w+)\s*(?:>=|<=|=|>|<)\s*(?:-?\d|')")
FROM = re.compile(r"FROM\s+(\w+)\s+as\s+(\w+)", re.IGNORECASE)


def stats_download(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    for name, remote in STATS_FILES.items():
        dst = src / name
        if not dst.exists():
            print(f"  downloading {remote} @ {STATS_COMMIT[:7]}")
            urllib.request.urlretrieve(f"{STATS_URL}/{remote}", dst)


def stats_parse_subqueries(sql_path: Path):
    """
    -> {table.Column: [(ops, true_card_or_None)]}, ops = list of (op, value) on that column.
    true_card is kept only when the sub-query filters that single column (pure 1-D).
    """
    per_col = collections.defaultdict(list)
    for line in sql_path.read_text().splitlines():
        if not line.strip():
            continue
        sql, _, card = line.split("||")
        tables = {alias: tbl for tbl, alias in FROM.findall(sql)}
        where = sql.split("WHERE", 1)[1] if "WHERE" in sql else ""
        cols = collections.defaultdict(list)
        for alias, col, op, val in PRED.findall(where):
            cols[f"{tables[alias]}.{col}"].append((op, int(val)))
        n_filtered_cols = len({(a, c) for a, c in ANY_PRED.findall(where)})
        for key, ops in cols.items():
            per_col[key].append((ops, int(card) if n_filtered_cols == 1 else None))
    return per_col


def ops_to_range(ops, mn, mx):
    """Intersect the predicates on one integer column into [lo, hi]; open ends -> column min/max."""
    lo, hi = mn, mx
    for op, v in ops:
        if op == "=":
            lo, hi = max(lo, v), min(hi, v)
        elif op == ">=":
            lo = max(lo, v)
        elif op == ">":
            lo = max(lo, v + 1)
        elif op == "<=":
            hi = min(hi, v)
        elif op == "<":
            hi = min(hi, v - 1)
    return lo, hi


def prepare_stats(args):
    root = Path(args.out_root or "data/stats-ceb")
    src = Path(args.source_dir or root / "source")
    stats_download(src)
    per_col = stats_parse_subqueries(src / "stats_CEB_single_table_sub_query.sql")
    tables = {t: pd.read_csv(src / f"{t}.csv") for t in {c.split(".")[0] for c in STATS_COLUMNS}}

    for key in STATS_COLUMNS:
        table, col = key.split(".")
        print(f"\n[STATS-CEB] {key}")
        out = root / key
        values = tables[table][col].dropna().to_numpy(dtype=np.int64)
        mn, mx, N, freq, sample, k, _, _ = write_meta(values, out, args.sample_size)

        preds = per_col.get(key, [])
        ranges = [ops_to_range(ops, mn, mx) for ops, _ in preds]
        write_ranges([r for r in ranges if r[0] <= r[1]], out / "workload_official.csv")

        # Our truth must reproduce the benchmark's own cardinality for pure 1-D sub-queries
        checked = [(RangeQuery(*r), card) for r, (_, card) in zip(ranges, preds) if card is not None and r[0] <= r[1]]
        if checked:
            ours = exact_counts([q for q, _ in checked], freq, mn)
            bad = [(q, c, o) for (q, c), o in zip(checked, ours) if int(o) != c]
            print(f"  truth check vs official cardinalities: {len(checked) - len(bad)}/{len(checked)} match")
            for q, c, o in bad[:5]:
                print(f"    MISMATCH [{q.low}, {q.high}]: official {c}, ours {int(o)}")

        # Supplement for statistical power: our generator on the official data, never mixed with the above
        width = max(2, (mx - mn) // 20)
        wl = generate_data_driven_workload(args.driven_count, sample, seed=SEED, max_width=width)
        save_workload_csv(wl, out / f"workload_driven_{args.driven_count}.csv")
        print(f"  workload_driven_{args.driven_count}.csv: data-driven supplement, max width {width}")


def main():
    parser = argparse.ArgumentParser(description="Convert DSB / STATS-CEB into meta.pkl + workload csv")
    parser.add_argument("--bench", choices=["dsb", "stats-ceb"], required=True)
    parser.add_argument("--source-dir", default=None,
                        help="dsb: DSB toolkit clone (default 'DSB benchmark/dsb'); "
                             "stats-ceb: where the official files are downloaded (default data/stats-ceb/source)")
    parser.add_argument("--out-root", default=None, help="Output root (default data/dsb or data/stats-ceb)")
    parser.add_argument("--generate-queries", action="store_true",
                        help="dsb: run dsqgen on the official templates before extracting ranges")
    parser.add_argument("--instances", type=int, default=1000,
                        help="dsb: query instances per template (dsqgen -streams)")
    parser.add_argument("--rngseed", type=int, default=SEED, help="dsb: dsqgen -rngseed")
    parser.add_argument("--scale", type=int, default=1, help="dsb: scale factor the data was generated with")
    parser.add_argument("--driven-count", type=int, default=100_000,
                        help="stats-ceb: size of the data-driven supplement workload")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="rows sampled for the models (default: dsb 100000, stats-ceb 30000)")
    args = parser.parse_args()
    if args.sample_size is None:
        args.sample_size = SAMPLE_SIZE[args.bench]

    if args.bench == "dsb":
        prepare_dsb(args)
    else:
        prepare_stats(args)


if __name__ == "__main__":
    main()

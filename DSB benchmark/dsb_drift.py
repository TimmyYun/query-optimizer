"""
DSB: выбор колонок и генерация дрейфа по потокам обновлений dsdgen.

    python dsb_drift.py rank  --tools T --base B --out rank.csv
    python dsb_drift.py drift --tools T --base B --updates U --streams 300 --columns store_sales.ss_item_sk --out OUT
"""
import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

FACTS = ["store_sales", "store_returns", "catalog_sales", "catalog_returns", "web_sales", "web_returns", "inventory"]

SALES = {
    "store_sales": ("ss_sold_date_sk", "ss_ticket_number"),
    "catalog_sales": ("cs_sold_date_sk", "cs_order_number"),
    "web_sales": ("ws_sold_date_sk", "ws_order_number"),
}
RETURNS = {
    "store_returns": ("store_sales", "sr_returned_date_sk", "sr_ticket_number"),
    "catalog_returns": ("catalog_sales", "cr_returned_date_sk", "cr_order_number"),
    "web_returns": ("web_sales", "wr_returned_date_sk", "wr_order_number"),
}
INSERTS = {
    "store_sales": {
        "head": "s_purchase", "line": "s_purchase_lineitem", "on": ("purc_purchase_id", "plin_purchase_id"),
        "cols": {
            "ss_sold_date_sk": ("date", "purc_purchase_date"),
            "ss_sold_time_sk": ("time", "purc_purchase_time"),
            "ss_item_sk": ("item", "plin_item_id"),
            "ss_ticket_number": ("int", "purc_purchase_id"),
        },
    },
    "catalog_sales": {
        "head": "s_catalog_order", "line": "s_catalog_order_lineitem", "on": ("cord_order_id", "clin_order_id"),
        "cols": {
            "cs_sold_date_sk": ("date", "cord_order_date"),
            "cs_sold_time_sk": ("time", "cord_order_time"),
            "cs_ship_date_sk": ("date", "clin_ship_date"),
            "cs_item_sk": ("item", "clin_item_id"),
            "cs_order_number": ("int", "cord_order_id"),
        },
    },
    "web_sales": {
        "head": "s_web_order", "line": "s_web_order_lineitem", "on": ("word_order_id", "wlin_order_id"),
        "cols": {
            "ws_sold_date_sk": ("date", "word_order_date"),
            "ws_sold_time_sk": ("time", "word_order_time"),
            "ws_ship_date_sk": ("date", "wlin_ship_date"),
            "ws_item_sk": ("item", "wlin_item_id"),
            "ws_order_number": ("int", "word_order_id"),
        },
    },
    "store_returns": {
        "head": "s_store_returns", "line": None,
        "cols": {
            "sr_returned_date_sk": ("date", "sret_return_date"),
            "sr_return_time_sk": ("time", "sret_return_time"),
            "sr_item_sk": ("item", "sret_item_id"),
            "sr_ticket_number": ("int", "sret_ticket_number"),
        },
    },
    "catalog_returns": {
        "head": "s_catalog_returns", "line": None,
        "cols": {
            "cr_returned_date_sk": ("date", "cret_return_date"),
            "cr_returned_time_sk": ("time", "cret_return_time"),
            "cr_item_sk": ("item", "cret_item_id"),
            "cr_order_number": ("int", "cret_order_id"),
        },
    },
    "web_returns": {
        "head": "s_web_returns", "line": None,
        "cols": {
            "wr_returned_date_sk": ("date", "wret_return_date"),
            "wr_returned_time_sk": ("time", "wret_return_time"),
            "wr_item_sk": ("item", "wret_item_id"),
            "wr_order_number": ("int", "wret_order_id"),
        },
    },
}
NULL = -1


def parse_schema(sql_path):
    out = {}
    for table, body in re.findall(r"create table (\w+)\s*\((.*?)\n\);", Path(sql_path).read_text(), re.S):
        cols = []
        for line in body.splitlines():
            m = re.match(r"\s*(\w+)\s+(\w+)", line)
            if m and m.group(1) != "primary":
                cols.append((m.group(1), m.group(2)))
        out[table] = cols
    return out


def read_dat(path, columns, wanted, dtype):
    names = [c for c, _ in columns] + ["_trailing"]
    return pd.read_csv(path, sep="|", header=None, names=names, usecols=wanted, dtype=dtype,
                       keep_default_na=False, na_values=[""], engine="c")


def load_int_columns(path, columns, wanted):
    df = read_dat(path, columns, wanted, "Int64")
    return {c: df[c].to_numpy("int64", na_value=NULL) for c in wanted}


def fd_bins(sample, mn, mx, n_rows, cap=2000):
    q25, q75 = np.quantile(sample, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0:
        return 10
    return max(1, min(int((mx - mn) / (2 * iqr / n_rows ** (1 / 3))), cap))


def sample_rows(freq, n, rng, replace=False):
    total = int(freq.sum())
    rows = rng.choice(total, size=min(n, total) if not replace else n, replace=replace)
    return np.searchsorted(np.cumsum(freq), rows, side="right").astype(np.int64)


def grid(mn, mx, k):
    bw = int(np.ceil((mx - mn + 1) / k))
    return bw, int(np.ceil((mx - mn + 1) / bw))


def uniform_bucket_qerror(freq, mn, mx, k, lows, highs):
    bw, nb = grid(mn, mx, k)
    seg = np.zeros(nb * bw)
    seg[: mx - mn + 1] = freq[mn : mx + 1]
    cnt = seg.reshape(nb, bw).sum(1)
    width = np.full(nb, bw, dtype=float)
    width[-1] = (mx - mn + 1) - bw * (nb - 1)
    pfx = np.concatenate([[0.0], np.cumsum(cnt)])
    lo = np.clip(lows, mn, mx) - mn
    hi = np.clip(highs, mn, mx) - mn
    bl, bh = lo // bw, hi // bw
    left_hi = np.where(bl == bh, hi, (bl + 1) * bw - 1)
    est = (left_hi - lo + 1) / width[bl] * cnt[bl]
    est = est + np.where(bl == bh, 0.0, (hi - bh * bw + 1) / width[bh] * cnt[bh] + pfx[bh] - pfx[np.minimum(bl + 1, bh)])
    return est


def q_errors(truth, est):
    t, p = np.maximum(truth, 1.0), np.maximum(est, 1.0)
    return np.maximum(t / p, p / t)


def true_counts(freq, lows, highs):
    ps = np.concatenate([[0], np.cumsum(freq)])
    lo = np.clip(lows, 0, len(freq))
    hi = np.clip(highs + 1, 0, len(freq))
    return np.maximum(ps[hi] - ps[np.minimum(lo, hi)], 0)


def narrow_workload(freq, n, max_width, rng, max_stalls=50):
    last = len(freq) - 1
    lows = np.empty(0, dtype=np.int64)
    highs = np.empty(0, dtype=np.int64)
    stalls = 0
    while len(lows) < n:
        batch = max(2 * (n - len(lows)), 1024)
        centers = sample_rows(freq, batch, rng, replace=True)
        widths = rng.integers(1, max_width + 1, batch)
        lo = np.clip(centers - rng.integers(0, widths), 0, last)
        hi = np.clip(lo + widths - 1, 0, last)
        cand_lo = np.concatenate([lows, lo])
        cand_hi = np.concatenate([highs, hi])
        _, first = np.unique(cand_lo * (last + 1) + cand_hi, return_index=True)
        first.sort()
        stalls = stalls + 1 if len(first) == len(lows) else 0
        if stalls >= max_stalls:
            raise SystemExit(f"уникальных узких запросов не больше {len(lows)}, запрошено {n}")
        lows, highs = cand_lo[first], cand_hi[first]
    return lows[:n], highs[:n]


def column_profile(values, seed, n_queries=50_000):
    mn, mx = int(values.min()), int(values.max())
    freq = np.bincount(values)
    N = int(freq.sum())
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, min(100_000, N), replace=False)
    k = fd_bins(sample, mn, mx, N)
    bw, nb = grid(mn, mx, k)
    seg = np.zeros(nb * bw)
    seg[: mx - mn + 1] = freq[mn : mx + 1]
    buckets = seg.reshape(nb, bw)
    cnt = buckets.sum(1)
    width = np.full(nb, bw, dtype=float)
    width[-1] = (mx - mn + 1) - bw * (nb - 1)
    mean = cnt / width
    var = np.maximum((buckets ** 2).sum(1) / width - mean ** 2, 0.0)
    cv = np.sqrt(var) / np.where(mean > 0, mean, 1.0)
    lows, highs = narrow_workload(freq, n_queries, bw, rng)
    q = q_errors(true_counts(freq, lows, highs), uniform_bucket_qerror(freq, mn, mx, k, lows, highs))
    return {
        "N": N, "mn": mn, "mx": mx, "ndv": int((freq > 0).sum()), "k": k, "bucket_width": bw,
        "intra_bucket_cv": float((cv * cnt).sum() / N),
        "uniform_qerr_median": float(np.median(q)), "uniform_qerr_p95": float(np.percentile(q, 95)),
        "rows_per_sample_point": N / len(sample),
    }


def cmd_rank(args):
    schema = parse_schema(Path(args.tools) / "tpcds.sql")
    rows = []
    for table in args.tables:
        ints = [c for c, t in schema[table] if t in ("integer", "int", "bigint")]
        data = load_int_columns(Path(args.base) / f"{table}.dat", schema[table], ints)
        for col in ints:
            v = data[col]
            nulls = int((v == NULL).sum())
            v = v[v != NULL]
            if len(v) == 0 or len(np.unique(v)) < args.min_ndv:
                continue
            rows.append({"table": table, "column": col, "nulls": nulls, **column_profile(v, args.seed)})
        print(f"{table}: {len(ints)} integer columns", file=sys.stderr)
    res = pd.DataFrame(rows).sort_values("uniform_qerr_p95", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


class Warehouse:
    def __init__(self, tools, base, updates, columns, returns_delete, item_history):
        self.schema = parse_schema(Path(tools) / "tpcds.sql")
        self.staging = parse_schema(Path(tools) / "tpcds_source.sql")
        self.staging["delete"] = [("del_date_start", "date"), ("del_date_end", "date")]
        self.base, self.updates = Path(base), Path(updates)
        self.returns_delete = returns_delete
        self.item_history = item_history
        self.targets = columns

        need = {}
        for table, col in columns:
            if table not in INSERTS or col not in INSERTS[table]["cols"]:
                raise SystemExit(f"неподдерживаемая колонка {table}.{col}")
            need.setdefault(table, set()).add(col)
            if table in SALES:
                need[table].add(SALES[table][0])
            else:
                parent, date_col, order_col = RETURNS[table]
                need[table].update({date_col, order_col})
                if returns_delete == "order":
                    need.setdefault(parent, set()).update(SALES[parent])
        for table in list(need):
            if table in SALES and any(RETURNS[r][0] == table for r in need if r in RETURNS):
                need[table].add(SALES[table][1])
        self.tables = {t: load_int_columns(self.base / f"{t}.dat", self.schema[t], sorted(cols)) for t, cols in need.items()}
        self.baseline = {t: len(next(iter(cols.values()))) for t, cols in self.tables.items()}

        dd = read_dat(self.base / "date_dim.dat", self.schema["date_dim"], ["d_date_sk", "d_date"], str)
        self.date_sk = dict(zip(dd["d_date"], dd["d_date_sk"].astype(np.int64)))
        td = read_dat(self.base / "time_dim.dat", self.schema["time_dim"], ["t_time_sk", "t_time"], str)
        self.time_sk = dict(zip(td["t_time"].astype(np.int64), td["t_time_sk"].astype(np.int64)))
        it = read_dat(self.base / "item.dat", self.schema["item"], ["i_item_sk", "i_item_id", "i_rec_end_date"], str)
        current = it[it["i_rec_end_date"].isna()]
        self.item_sk = dict(zip(current["i_item_id"].str.strip(), current["i_item_sk"].astype(np.int64)))
        self.next_item_sk = int(it["i_item_sk"].astype(np.int64).max()) + 1

    def staging_frame(self, name, stream):
        path = self.updates / f"{name}_{stream}.dat"
        if not path.exists():
            return None
        return read_dat(path, self.staging[name], [c for c, _ in self.staging[name]], str)

    def apply_item_history(self, stream):
        upd = self.staging_frame("s_item", stream)
        if upd is None:
            return 0
        for item_id in upd["item_item_id"].str.strip():
            self.item_sk[item_id] = self.next_item_sk
            self.next_item_sk += 1
        return len(upd)

    def delete_ranges(self, stream):
        df = self.staging_frame("delete", stream)
        if df is None:
            return []
        return [(self.date_sk[a], self.date_sk[b]) for a, b in zip(df["del_date_start"], df["del_date_end"])]

    @staticmethod
    def in_ranges(values, ranges):
        mask = np.zeros(len(values), dtype=bool)
        for a, b in ranges:
            mask |= (values >= a) & (values <= b)
        return mask

    def drop(self, table, mask):
        self.tables[table] = {c: v[~mask] for c, v in self.tables[table].items()}
        return int(mask.sum())

    def apply_deletes(self, stream):
        ranges = self.delete_ranges(stream)
        deleted = {}
        removed_orders = {}
        for table in [t for t in self.tables if t in SALES]:
            date_col, order_col = SALES[table]
            mask = self.in_ranges(self.tables[table][date_col], ranges)
            if order_col in self.tables[table]:
                removed_orders[table] = np.unique(self.tables[table][order_col][mask])
            deleted[table] = self.drop(table, mask)
        for table in [t for t in self.tables if t in RETURNS]:
            parent, date_col, order_col = RETURNS[table]
            if self.returns_delete == "order":
                mask = np.isin(self.tables[table][order_col], removed_orders[parent])
            else:
                mask = self.in_ranges(self.tables[table][date_col], ranges)
            deleted[table] = self.drop(table, mask)
        return deleted

    def convert(self, kind, series):
        s = series.fillna("").str.strip()
        if kind == "date":
            return s.map(self.date_sk)
        if kind == "item":
            return s.map(self.item_sk)
        if kind == "time":
            hms = s.str.split(":", expand=True)
            if hms.shape[1] == 3:
                secs = pd.to_numeric(hms[0], errors="coerce") * 3600 + pd.to_numeric(hms[1], errors="coerce") * 60 + pd.to_numeric(hms[2], errors="coerce")
            else:
                secs = pd.to_numeric(s, errors="coerce")
            return secs.map(self.time_sk)
        return pd.to_numeric(s, errors="coerce")

    def apply_inserts(self, stream):
        inserted = {}
        for table, cols in self.tables.items():
            spec = INSERTS[table]
            head = self.staging_frame(spec["head"], stream)
            if head is None:
                inserted[table] = 0
                continue
            if spec["line"]:
                line = self.staging_frame(spec["line"], stream)
                if line is None:
                    inserted[table] = 0
                    continue
                head_key, line_key = spec["on"]
                frame = line.merge(head, left_on=line_key, right_on=head_key, how="inner")
            else:
                frame = head
            for col in cols:
                kind, src = spec["cols"][col]
                v = self.convert(kind, frame[src]).to_numpy(dtype="float64", na_value=np.nan)
                cols[col] = np.concatenate([cols[col], np.where(np.isnan(v), NULL, v).astype(np.int64)])
            inserted[table] = len(frame)
        return inserted

    def column(self, table, col):
        v = self.tables[table][col]
        return v[v != NULL]


def ks_exact(freq_a, freq_b):
    n = max(len(freq_a), len(freq_b))
    a = np.zeros(n)
    b = np.zeros(n)
    a[: len(freq_a)] = freq_a
    b[: len(freq_b)] = freq_b
    return float(np.max(np.abs(np.cumsum(a) / a.sum() - np.cumsum(b) / b.sum())))


def tv_on_grid(freq_a, freq_b, mn, bw):
    def bucketed(f):
        idx = np.maximum(np.arange(len(f)) - mn, 0) // bw
        return np.bincount(idx, weights=f)

    a, b = bucketed(freq_a), bucketed(freq_b)
    n = max(len(a), len(b))
    pa = np.zeros(n)
    pb = np.zeros(n)
    pa[: len(a)] = a / a.sum()
    pb[: len(b)] = b / b.sum()
    return float(0.5 * np.abs(pa - pb).sum())


def parse_queries(path):
    lows, highs = [], []
    pattern = re.compile(r"between\s+(\d+)\s+and\s+(\d+)\s*(?:\+\s*(\d+))?", re.I)
    seen = set()
    for line in Path(path).read_text().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) + (int(m.group(3)) if m.group(3) else 0)
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        lows.append(lo)
        highs.append(hi)
    return np.array(lows, dtype=np.int64), np.array(highs, dtype=np.int64)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_drift(args):
    columns = [tuple(c.split(".", 1)) for c in args.columns]
    wh = Warehouse(args.tools, args.base, args.updates, columns, args.returns_delete, args.item_history)
    thresholds = list(range(args.churn_step, args.max_churn + 1, args.churn_step))
    churn = {t: 0 for t in wh.tables}
    snaps = {c: [(0, 0.0, np.bincount(wh.column(*c)))] for c in columns}
    next_idx = {c: 0 for c in columns}
    log = []

    for stream in range(1, args.streams + 1):
        if all(next_idx[c] >= len(thresholds) for c in columns):
            break
        n_items = wh.apply_item_history(stream) if args.item_history else 0
        deleted = wh.apply_deletes(stream)
        inserted = wh.apply_inserts(stream)
        for t in wh.tables:
            churn[t] += deleted.get(t, 0) + inserted.get(t, 0)
        entry = {"stream": stream, "items_versioned": n_items}
        for t in wh.tables:
            entry[f"{t}_rows"] = len(next(iter(wh.tables[t].values())))
            entry[f"{t}_churn_pct"] = 100 * churn[t] / wh.baseline[t]
        log.append(entry)
        for c in columns:
            pct = 100 * churn[c[0]] / wh.baseline[c[0]]
            while next_idx[c] < len(thresholds) and pct >= thresholds[next_idx[c]]:
                snaps[c].append((stream, pct, np.bincount(wh.column(*c))))
                next_idx[c] += 1
        if stream % 25 == 0:
            done = ", ".join(f"{t}.{col}:{next_idx[(t, col)]}/{len(thresholds)}" for t, col in columns)
            print(f"stream {stream}: {done}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / "streams.csv", index=False)
    queries_dir = Path(args.queries) if args.queries else None
    files = {}
    summary = []

    for table, col in columns:
        name = f"{table}.{col}"
        steps = snaps[(table, col)]
        if next_idx[(table, col)] < len(thresholds):
            print(f"! {name}: потоков хватило только на {next_idx[(table, col)]}/{len(thresholds)} порогов", file=sys.stderr)
        length = max(len(f) for _, _, f in steps)
        shutil.rmtree(out / name, ignore_errors=True)

        def padded(f):
            full = np.zeros(length, dtype=np.int64)
            full[: len(f)] = f
            return full

        def step_rngs(i):
            ss = np.random.SeedSequence([args.seed, zlib.crc32(name.encode()), i])
            return [np.random.default_rng(s) for s in ss.spawn(2)]

        freq0 = padded(steps[0][2])
        nz0 = np.nonzero(freq0)[0]
        mn0, mx0 = int(nz0[0]), int(nz0[-1])
        k0 = fd_bins(sample_rows(freq0, args.sample_size, step_rngs(0)[0]), mn0, mx0, int(freq0.sum()))
        bw0, _ = grid(mn0, mx0, k0)
        dsb_queries = None
        if queries_dir and (queries_dir / f"{name}.sql").exists():
            dsb_queries = parse_queries(queries_dir / f"{name}.sql")

        for i, (stream, pct, f) in enumerate(steps):
            freq = padded(f)
            rng_sample, rng_wl = step_rngs(i)
            sample = sample_rows(freq, args.sample_size, rng_sample)
            nz = np.nonzero(freq)[0]
            mn, mx, N = int(nz[0]), int(nz[-1]), int(freq.sum())
            lows, highs = narrow_workload(freq, args.queries_per_step, bw0, rng_wl)
            truth = true_counts(freq, lows, highs)
            stale = freq0 * (N / freq0.sum())
            q_stale = q_errors(truth, uniform_bucket_qerror(stale, mn0, mx0, k0, np.maximum(lows, mn0), np.minimum(highs, mx0)))
            q_stale = np.where((highs < mn0) | (lows > mx0), np.maximum(truth, 1.0), q_stale)

            step_dir = out / name / f"step_{i:02d}"
            (step_dir / "workloads").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(step_dir / "data.npz", freq=freq, sample=sample)
            np.savez_compressed(step_dir / "workloads" / "narrow.npz", lows=lows.astype(np.int32), highs=highs.astype(np.int32), truth=truth)
            if dsb_queries is not None:
                ql, qh = dsb_queries
                np.savez_compressed(step_dir / "workloads" / "dsb.npz", lows=ql.astype(np.int32), highs=qh.astype(np.int32), truth=true_counts(freq, ql, qh))
            meta = {
                "table": table, "column": col, "step": i, "stream": stream, "churn_pct": pct,
                "threshold_pct": 0 if i == 0 else thresholds[i - 1],
                "N": N, "mn": mn, "mx": mx, "ndv": int((freq > 0).sum()), "domain_len": length,
                "k": fd_bins(sample, mn, mx, N), "sample_size": int(len(sample)), "sample_fraction": len(sample) / N,
                "ks_vs_step0": ks_exact(freq0, freq), "tv_vs_step0": tv_on_grid(freq0, freq, mn0, bw0),
                "stale_equiwidth_qerr_median": float(np.median(q_stale)), "stale_equiwidth_qerr_p95": float(np.percentile(q_stale, 95)),
                "narrow_max_width": bw0, "queries_per_step": int(len(lows)),
                "dsb_queries": None if dsb_queries is None else int(len(dsb_queries[0])),
                "seed": args.seed, "returns_delete": args.returns_delete, "item_history": args.item_history,
                "freq_sha256": hashlib.sha256(freq.tobytes()).hexdigest(),
            }
            (step_dir / "meta.json").write_text(json.dumps(meta, indent=1))
            summary.append(meta)
            for p in sorted(step_dir.rglob("*")):
                if p.is_file():
                    files[str(p.relative_to(out))] = sha256(p)
        print(f"{name}: {len(steps)} шагов", file=sys.stderr)

    pd.DataFrame(summary).drop(columns=["freq_sha256"]).to_csv(out / "summary.csv", index=False)
    manifest = {
        "args": vars(args),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank")
    r.add_argument("--tools", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tables", nargs="+", default=FACTS)
    r.add_argument("--min-ndv", type=int, default=200)
    r.add_argument("--seed", type=int, default=42)

    d = sub.add_parser("drift")
    d.add_argument("--tools", required=True)
    d.add_argument("--base", required=True)
    d.add_argument("--updates", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--columns", nargs="+", required=True)
    d.add_argument("--streams", type=int, required=True)
    d.add_argument("--churn-step", type=int, default=5)
    d.add_argument("--max-churn", type=int, default=100)
    d.add_argument("--returns-delete", choices=["order", "returned_date"], default="order")
    d.add_argument("--item-history", action="store_true")
    d.add_argument("--sample-size", type=int, default=100_000)
    d.add_argument("--queries-per-step", type=int, default=200_000)
    d.add_argument("--queries", default=None)
    d.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.cmd == "rank":
        cmd_rank(args)
    else:
        cmd_drift(args)


if __name__ == "__main__":
    main()

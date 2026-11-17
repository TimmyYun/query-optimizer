#!/usr/bin/env python3
"""
Generic Excel exporter for results/paper_experiments/** (works on future folders without code changes).

    poetry run python -m paper_experiments.excel_export --dir results/paper_experiments/fd_reduction \
        --out results/paper_experiments/fd_reduction/fd_reduction_all.xlsx

* every CSV under --dir becomes one sheet (unique, <= 31 chars, derived from its relative path);
* skipped: per-query result CSVs (Query_ID, Model, Prediction, Truth, Q_Error), workload CSVs (low, high),
  files over --max-mb; tables over --max-rows are truncated (README says so);
* headers carry units from the schema naming convention (paper_experiments/schema.py):
  *_bytes [bytes] (+ *_KiB / *_MiB display columns), *_s [s], *_us* [µs], *qerror* [q-error],
  *_rows [rows], pct* [%];
* number formats, bold header, frozen header row, autofilter, column widths;
* first sheet README: sheet -> source path, rows, truncation, and a per-column unit/description list.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

PER_QUERY_HEADER = ["Query_ID", "Model", "Prediction", "Truth", "Q_Error"]
INVALID_SHEET = re.compile(r"[\[\]:*?/\\]")
EXCEL_MAX_ROWS = 1_048_575

# (regex on the column name, unit, Excel number format) — first match wins
UNIT_RULES = [
    (r"_KiB$", "KiB", "#,##0.00"),
    (r"_MiB$", "MiB", "#,##0.000"),
    (r"(^|_)bytes($|_)|_bytes$", "bytes", "#,##0"),
    (r"qerror|q_error|Q_Error", "q-error, >=1, lower is better", "0.0000"),
    (r"(^|_)us($|_)|_us_|_us$", "µs", "0.00"),
    (r"(^|_)s$|_s_|_time_s$|_time$|^time_s$", "s", "0.000000"),
    (r"(^|_)rows($|_)", "rows", "#,##0.00"),
    (r"(^|_)pct($|_)|_pct$|^pct_", "%", "0.00"),
    (r"^n_|_count$|^bucket_count$|_buckets$", "count", "#,##0"),
    (r"ratio", "ratio", "0.000"),
]

COLUMN_DOCS = {
    "experiment_id": "experiment family (folder of results/paper_experiments)",
    "dataset": "dataset label (synthetic <simple|hard>/<dist>, dsb/<column>, dsb_expanded/<table.column>)",
    "table": "source table (DSB) or empty", "column": "source column / distribution",
    "workload": "workload name (file stem)", "drift_type": "none / displacement / additive / churn_official",
    "drift_pct": "drift point, % of the step-0 table changed (additive: rows added)",
    "bin_strategy": "fixed<B> / fd / fd_r<pct> (FD reduced) / fd_merge_r<pct> (merging, modification) / mem_match_*",
    "bucket_count": "number of buckets (requested or actual, see source)", "model": "estimator",
    "model_version": "<Model>:<spec>:<sha10 of its source files>", "seed": "random seed",
    "stage": "STATIC / SHOCK / FT / RB (RB only where a rebuild was performed)",
    "checkpoint_triggered": "Hybrid: >= 100 000 rows changed since the last rebuild at this step boundary",
    "rebuild_performed": "checkpoint triggered and >= 1 marked bucket retrained (baselines: full rebuild)",
    "rebuilt_buckets": "buckets retrained at this step", "ft_buckets": "buckets whose count FT changed",
    "marked_buckets": "buckets marked for rebuild (accumulated)", "median_qerror": "median q-error",
    "p95_qerror": "95th percentile q-error", "avg_qerror": "mean q-error", "mae_rows": "mean absolute error, rows",
    "n_values": "stored numbers (8 B each) — logical persistent model size", "size_bytes": "n_values x 8",
    "logical_bytes": "n_values x 8", "pickle_bytes": "len(pickle.dumps(model)) of the live object",
    "infer_us_median": "median per-query inference latency (scalar predict)",
}


def unit_of(col: str):
    for rx, unit, fmt in UNIT_RULES:
        if re.search(rx, col):
            return unit, fmt
    return None, None


def _is_skippable(path: Path, max_mb: float):
    if path.stat().st_size > max_mb * (1 << 20):
        return f"larger than {max_mb} MB"
    try:
        head = pd.read_csv(path, nrows=2)
    except Exception as e:
        return f"unreadable: {e}"
    cols = list(head.columns)
    if cols[:5] == PER_QUERY_HEADER:
        return "per-query results"
    if cols == ["low", "high"]:
        return "workload"
    return None


def sheet_name(rel: Path, used: set) -> str:
    parts = [INVALID_SHEET.sub("_", p) for p in rel.with_suffix("").parts]
    name = parts[-1]
    for p in reversed(parts[:-1]):  # prepend parents while they fit
        cand = f"{p}_{name}"
        if len(cand) > 31:
            break
        name = cand
    name = name[:31] or "sheet"
    base, i = name, 2
    while name.lower() in used or name.lower() == "readme":
        suffix = f"~{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(name.lower())
    return name


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """KiB / MiB next to every *_bytes column (display only; bytes stay the stored unit)."""
    out = df.copy()
    for c in list(df.columns):
        if re.search(r"_bytes$|^bytes$", str(c)) and pd.api.types.is_numeric_dtype(df[c]):
            pos = out.columns.get_loc(c) + 1
            out.insert(pos, f"{c}_KiB", df[c] / 1024.0)
            out.insert(pos + 1, f"{c}_MiB", df[c] / (1 << 20))
    return out


def write_workbook(entries, out_xlsx, skipped=(), format_rows_limit=50_000):
    """
    entries: [(sheet_base, DataFrame, source, note)] -> one formatted sheet each + README.
    sheet_base is a path-like name used to derive a unique sheet name.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    used = set()
    index, coldocs = [], []
    bold = Font(bold=True)
    for base, df, source, note in entries:
        df = add_display_columns(df)
        name = sheet_name(Path(base), used)
        ws = wb.create_sheet(name)
        headers = []
        for c in df.columns:
            unit, _ = unit_of(str(c))
            headers.append(f"{c} [{unit}]" if unit else str(c))
            coldocs.append((name, str(c), unit or "", COLUMN_DOCS.get(str(c), "")))
        ws.append(headers)
        for row in df.itertuples(index=False, name=None):
            ws.append([None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else
                       (str(v) if isinstance(v, (dict, list, tuple)) else v)) for v in row])
        for j, c in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=j)
            cell.font = bold
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            _, fmt = unit_of(str(c))
            letter = get_column_letter(j)
            sample = df[c].head(200).astype(str)
            width = max([len(headers[j - 1]) * 0.9] + [len(v) for v in sample]) if len(df) else len(headers[j - 1])
            ws.column_dimensions[letter].width = min(max(8, width + 2), 50)
            if fmt and pd.api.types.is_numeric_dtype(df[c]) and len(df) <= format_rows_limit:
                for (cell,) in ws.iter_rows(min_row=2, max_row=len(df) + 1, min_col=j, max_col=j):
                    cell.number_format = fmt
        ws.freeze_panes = "A2"
        if len(df):
            ws.auto_filter.ref = f"A1:{get_column_letter(max(1, len(df.columns)))}{len(df) + 1}"
        index.append((name, str(source), len(df), len(df.columns), note or ""))

    readme.append(["Workbook generated by paper_experiments/excel_export.py"])
    readme.append(["Units are part of the header: [bytes] (KiB/MiB columns are display copies), [s], [µs], "
                   "[q-error, >=1, lower is better], [rows], [%]."])
    readme.append([])
    readme.append(["sheet", "source", "rows", "columns", "note"])
    for r in index:
        readme.append(list(r))
    if skipped:
        readme.append([])
        readme.append(["skipped file", "reason"])
        for path, why in skipped:
            readme.append([str(path), why])
    readme.append([])
    readme.append(["sheet", "column", "unit", "description"])
    for r in coldocs:
        readme.append(list(r))
    for row in readme.iter_rows(min_row=1, max_row=readme.max_row):
        if row[0].value in ("sheet", "skipped file"):
            for cell in row:
                cell.font = bold
    readme.column_dimensions["A"].width = 34
    readme.column_dimensions["B"].width = 70
    wb.save(out_xlsx)
    return {"sheets": len(index), "skipped": len(skipped), "path": str(out_xlsx)}


def export_dir(results_dir, out_xlsx, exclude=(), max_mb=25.0, max_rows=200_000, pattern="*.csv"):
    """Every CSV under results_dir (recursively, except excluded path parts) -> one sheet of out_xlsx."""
    root = Path(results_dir)
    entries, skipped = [], []
    for path in sorted(root.rglob(pattern)):
        rel = path.relative_to(root)
        if any(part in exclude for part in rel.parts):
            continue
        why = _is_skippable(path, max_mb)
        if why:
            skipped.append((rel, why))
            continue
        df = pd.read_csv(path, low_memory=False)
        note = ""
        lim = min(max_rows, EXCEL_MAX_ROWS)
        if len(df) > lim:
            note = f"TRUNCATED: first {lim:,} of {len(df):,} rows (full data in the CSV)"
            df = df.head(lim)
        entries.append((str(rel), df, rel, note))
    if not entries:
        raise FileNotFoundError(f"no exportable CSV under {root}")
    return write_workbook(entries, out_xlsx, skipped)


def main():
    p = argparse.ArgumentParser(description="CSV tables of a results folder -> one formatted xlsx")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="path parts to skip (e.g. raw logs)")
    p.add_argument("--max-mb", type=float, default=25.0)
    p.add_argument("--max-rows", type=int, default=200_000)
    a = p.parse_args()
    print(export_dir(a.dir, a.out, exclude=tuple(a.exclude), max_mb=a.max_mb, max_rows=a.max_rows))


if __name__ == "__main__":
    main()

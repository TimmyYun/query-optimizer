#!/usr/bin/env python3
"""
Collect every result into one Excel workbook: one sheet per logical benchmark.

    poetry run python export_results.py                      # -> results/all_results.xlsx
    poetry run python export_results.py --out paper.xlsx --all

A logical benchmark is one experiment folder results/<experiment>/:
    static  (registry.STATIC_SCENARIOS)  rows = summary_<Model>.json of every dataset
    drift   (registry.DRIFT_SCENARIOS)   rows = every step x model x stage of the drift CSVs
The first sheet, "index", says what each sheet is: data, workload, number of rows.
Each data sheet holds the full long table and, to its right, a pivot of the median q-error.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from merge_drift import collect_drift_rows
from registry import DRIFT_SCENARIOS, STATIC_SCENARIOS

STATIC_COLUMNS = {
    "n_queries": "Queries",
    "median_q_error": "Median Q-Error",
    "p25_q_error": "P25 Q-Error",
    "p75_q_error": "P75 Q-Error",
    "p95_q_error": "P95 Q-Error",
    "avg_q_error": "Avg Q-Error",
    "build_time": "Build (s)",
    "update_time_total": "Feedback Updates (s)",
    "train_time": "Train = Build + Updates (s)",
    "infer_time": "Inference, all queries (s)",
    "avg_infer_time": "Inference / query, mean (s)",
    "median_infer_time": "Inference / query, median (s)",
    "p95_infer_time": "Inference / query, p95 (s)",
    "n_buckets": "Buckets",
    "n_floats": "Floats stored (Hybrid)",
}
SHEET_NAME_MAX = 31  # Excel limit


def static_table(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(exp_dir.rglob("summary_*.json")):
        s = json.loads(path.read_text())
        m = s["metrics"]
        row = {"Dataset": str(path.parent.relative_to(exp_dir)), "Workload": s["wl_name"], "Model": s["model"]}
        row.update({label: m.get(key) for key, label in STATIC_COLUMNS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(axis=1, how="all").sort_values(["Dataset", "Model"])
    return df


def drift_table(exp_dir: Path) -> pd.DataFrame:
    rows = [{"Run": prefix, **r} for prefix, rs in collect_drift_rows(exp_dir).items() for r in rs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={
        "drift": "Drift %", "step": "Step", "model": "Model", "stage": "Stage", "label": "Line",
        "Median": "Median Q-Error", "P95": "P95 Q-Error", "Avg": "Avg Q-Error",
        "Infer_Med": "Inference / query, median (s)", "Infer_Avg": "Inference / query, mean (s)",
    })
    return df.dropna(axis=1, how="all").sort_values(["Run", "Line", "Drift %"])


def describe(name: str, kind: str) -> str:
    if kind == "static":
        cfg = STATIC_SCENARIOS[name]
        return f"{len(cfg['datasets'])} datasets, workload {cfg['workload']}"
    if kind == "drift":
        cfg = DRIFT_SCENARIOS[name]
        wl = cfg["static_workload"] or "per-step workload (drifts with the data)"
        return f"shift {cfg['shift_dir']}, workload {wl}; scored on the eval half of each step"
    return "not in registry.py"


def style_sheet(ws, n_cols):
    header = PatternFill("solid", fgColor="DDE4EE")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    for col in range(1, n_cols + 1):
        letter = get_column_letter(col)
        width = max((len(str(c.value)) for c in ws[letter][:200] if c.value is not None), default=8)
        ws.column_dimensions[letter].width = min(max(10, width + 2), 45)
        for c in ws[letter][1:]:
            if isinstance(c.value, float):
                c.number_format = "0.000000000" if "(s)" in str(ws[letter][0].value) else "0.0000"


def write_sheet(writer, sheet, df, pivot):
    df.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]
    if pivot is not None and not pivot.empty:
        start = len(df.columns) + 2
        ws.cell(row=1, column=start, value="Median Q-Error").font = Font(bold=True)
        pivot.to_excel(writer, sheet_name=sheet, startrow=1, startcol=start - 1)
    style_sheet(ws, ws.max_column)


def main():
    parser = argparse.ArgumentParser(description="All results -> one xlsx, one sheet per benchmark")
    parser.add_argument("--results-dir", default="results", help="Root of the results tree")
    parser.add_argument("--out", default="results/all_results.xlsx", help="Output workbook")
    parser.add_argument("--all", action="store_true",
                        help="Also include experiment folders that are not scenarios in registry.py (e.g. smoke)")
    args = parser.parse_args()

    root = Path(args.results_dir)
    present = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    plan = [(n, "static") for n in STATIC_SCENARIOS if STATIC_SCENARIOS[n]["experiment"] in present]
    plan += [(n, "drift") for n in DRIFT_SCENARIOS if n in present]
    if args.all:
        known = {STATIC_SCENARIOS[n]["experiment"] for n in STATIC_SCENARIOS} | set(DRIFT_SCENARIOS)
        plan += [(n, "other") for n in sorted(present - known)]
    if not plan:
        print(f"No experiment folders found in {root}")
        return

    index = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="index")  # placeholder, keeps it first
        for name, kind in plan:
            folder = STATIC_SCENARIOS[name]["experiment"] if kind == "static" else name
            exp_dir = root / folder
            is_drift = kind == "drift" or (kind == "other" and not any(exp_dir.rglob("summary_*.json")))
            df = drift_table(exp_dir) if is_drift else static_table(exp_dir)
            if df.empty:
                print(f"skip {folder}: no results")
                continue
            if is_drift:
                pivot = df.pivot_table(index="Drift %", columns="Line", values="Median Q-Error")
            else:
                pivot = df.pivot_table(index="Dataset", columns="Model", values="Median Q-Error")
            sheet = folder[:SHEET_NAME_MAX]
            write_sheet(writer, sheet, df, pivot)
            index.append({
                "Sheet": sheet,
                "Kind": "drift" if is_drift else "static",
                "Results folder": str(exp_dir),
                "What": describe(name, kind),
                "Rows": len(df),
                "Models": ", ".join(sorted(df["Model"].unique())),
            })
            print(f"{sheet:32s} {len(df):5d} rows")

        idx = pd.DataFrame(index)
        idx.to_excel(writer, sheet_name="index", index=False)
        style_sheet(writer.sheets["index"], len(idx.columns))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# python make_histograms.py --glob "data/salary_*_500mb.csv" --bins 100 --outdir plots

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pass_min_max(csv_path, chunksize=1_000_000):
    mn, mx = None, None
    for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64",
                             chunksize=chunksize, engine="c"):
        cmin, cmax = int(chunk["v"].min()), int(chunk["v"].max())
        mn = cmin if mn is None else min(mn, cmin)
        mx = cmax if mx is None else max(mx, cmax)
    return mn, mx

def pass_hist(csv_path, edges, chunksize=1_000_000):
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64",
                             chunksize=chunksize, engine="c"):
        vals = chunk["v"].to_numpy()
        h, _ = np.histogram(vals, bins=edges)
        counts += h
    return counts

def plot_hist(edges, counts, out_png, title, density=False):
    mids = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]
    y = counts.astype(float)
    if density:
        total = y.sum()
        if total > 0:
            y /= total
    plt.figure()  # one chart per figure (no subplots)
    plt.bar(mids, y, width=width)
    plt.title(title)
    plt.xlabel("salary")
    plt.ylabel("density" if density else "count")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Make 100-bin histograms for large one-column CSVs (streaming).")
    ap.add_argument("--glob", default="data/salary_*_500mb.csv", help="Glob for input CSVs")
    ap.add_argument("--bins", type=int, default=100, help="Number of bins")
    ap.add_argument("--density", action="store_true", help="Normalize histogram")
    ap.add_argument("--outdir", default="plots", help="Output folder for PNGs")
    args = ap.parse_args()

    files = sorted([p for p in Path(".").glob(args.glob)])
    if not files:
        print(f"No files matched {args.glob}")
        return

    for f in files:
        if f.suffix.lower() != ".csv":
            continue
        print(f"[{f.name}] first pass: min/max …")
        mn, mx = pass_min_max(f)
        if mn is None or mx is None or mn == mx:
            print(f"  skipped (empty or constant): {f}")
            continue
        edges = np.linspace(mn, mx, args.bins + 1)
        print(f"  range: [{mn}, {mx}]  bins: {args.bins}")

        print(f"[{f.name}] second pass: histogram …")
        counts = pass_hist(f, edges)

        out_png = Path(args.outdir) / (f.stem + "_hist.png")
        print(f"[{f.name}] plotting → {out_png}")
        plot_hist(edges, counts, out_png, title=f.stem, density=args.density)

    print("Done.")

if __name__ == "__main__":
    main()

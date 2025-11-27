#!/usr/bin/env python3
import argparse, csv, os, random
from pathlib import Path

import matplotlib.pyplot as plt

def reservoir_scatter_sample(csv_path: Path, sample_size: int, delimiter=","):
    """
    Stream the CSV (assumes single numeric column per row), perform reservoir sampling
    of (row_index, value) pairs to get ~sample_size points for scatter plotting.
    Returns two lists: xs (indices), ys (values).
    """
    xs, ys = [], []
    n = 0  # total seen (0-based index)
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            s = row[0].strip()
            # try to parse integer; skip non-numeric header or bad rows
            try:
                v = int(s)
            except ValueError:
                # try float (just in case); cast to int for plotting if desired
                try:
                    v = float(s)
                except ValueError:
                    continue  # header or malformed
            if len(xs) < sample_size:
                xs.append(n)
                ys.append(v)
            else:
                j = random.randint(0, n)
                if j < sample_size:
                    xs[j] = n
                    ys[j] = v
            n += 1
    return xs, ys, n

def plot_scatter(xs, ys, title: str, out_path: Path):
    plt.figure(figsize=(12, 4))
    plt.scatter(xs, ys, s=1)  # small markers; no styles/colors specified
    plt.xlabel("Row index")
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Scatter-plot sampled points (index vs value) from large 1-column CSVs.")
    ap.add_argument("--data-dir", default="data", help="Directory with CSV files")
    ap.add_argument("--pattern", default="salary_*.csv", help="Glob pattern for files")
    ap.add_argument("--sample", type=int, default=100_000, help="Reservoir sample size per file")
    ap.add_argument("--out", default="plots", help="Output directory for PNGs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob(args.pattern))
    if not files:
        print(f"No files found matching {args.pattern} in {data_dir}")
        return

    for fp in files:
        print(f"[+] Sampling {fp.name} ...")
        xs, ys, total = reservoir_scatter_sample(fp, args.sample)
        print(f"    sampled {len(xs)} of ~{total:,} rows")
        title = f"{fp.name} (sampled {len(xs)} of ~{total:,} rows)"
        out_png = out_dir / (fp.stem + "_scatter.png")
        plot_scatter(xs, ys, title, out_png)
        print(f"    saved -> {out_png}")

if __name__ == "__main__":
    main()


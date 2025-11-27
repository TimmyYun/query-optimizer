import argparse, os, math, csv, sys
from pathlib import Path
import numpy as np

# ----------------------------
# Helpers
# ----------------------------
def clamp(x, lo, hi):
    return np.clip(x, lo, hi)

def to_int_salaries(x, lo, hi):
    return clamp(np.round(x).astype(np.int64), lo, hi)

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def approx_rows_for_size(sample_values, target_bytes, newline_bytes=1):
    # estimate avg bytes/row as len(str(value)) + newline
    as_str = [str(int(v)) for v in sample_values[: min(200000, len(sample_values))]]
    avg_len = sum(len(s) for s in as_str) / len(as_str) + newline_bytes
    return int(target_bytes // avg_len)

def write_rows_stream(values_iter, total_rows, out_path):
    ensure_dir(out_path)
    n_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for arr in values_iter:
            need = total_rows - n_written
            if need <= 0:
                break
            chunk = arr[:need]
            # csv.writer is fast enough for ints; avoid Python loops where possible
            writer.writerows((int(x),) for x in chunk)
            n_written += len(chunk)
    return n_written

# ----------------------------
# Distributions
# ----------------------------
def gen_uniform(rng, n, lo, hi):
    return to_int_salaries(rng.integers(lo, hi + 1, size=n), lo, hi)

def gen_normal(rng, n, mean, std, lo, hi):
    x = rng.normal(loc=mean, scale=std, size=n)
    return to_int_salaries(x, lo, hi)

def gen_laplace(rng, n, loc, scale, lo, hi):
    x = rng.laplace(loc=loc, scale=scale, size=n)  # leptokurtic (high kurtosis)
    return to_int_salaries(x, lo, hi)

def gen_student_t(rng, n, df, loc, scale, lo, hi):
    x = loc + scale * rng.standard_t(df=df, size=n)  # very heavy tails for small df
    return to_int_salaries(x, lo, hi)

def gen_lognormal(rng, n, mean, sigma, lo, hi):
    # mean here is *median-ish* center after exponentiation; tune sigma for skew
    x = rng.lognormal(mean=math.log(max(mean, 1)), sigma=sigma, size=n)
    return to_int_salaries(x, lo, hi)

def gen_pareto(rng, n, xm, alpha, lo, hi):
    # Pareto Type I: xm * (1 + Y), Y ~ Pareto(alpha)
    y = rng.pareto(alpha, size=n) + 1.0
    x = xm * y
    return to_int_salaries(x, lo, hi)

def gen_zipf(rng, n, a, base, mult, lo, hi):
    # Zipf returns positive integers with heavy skew
    z = rng.zipf(a, size=n)
    x = base + mult * z
    return to_int_salaries(x, lo, hi)

# ----------------------------
# Chunked generator
# ----------------------------
def stream_values(kind, rng, total, chunk, params, lo, hi):
    rem = total
    while rem > 0:
        m = min(chunk, rem)
        if kind == "uniform":
            arr = gen_uniform(rng, m, lo, hi)
        elif kind == "normal":
            arr = gen_normal(rng, m, params["mean"], params["std"], lo, hi)
        elif kind == "laplace":
            arr = gen_laplace(rng, m, params["loc"], params["scale"], lo, hi)
        elif kind == "student_t":
            arr = gen_student_t(rng, m, params["df"], params["loc"], params["scale"], lo, hi)
        elif kind == "lognormal":
            arr = gen_lognormal(rng, m, params["median_like"], params["sigma"], lo, hi)
        elif kind == "pareto":
            arr = gen_pareto(rng, m, params["xm"], params["alpha"], lo, hi)
        elif kind == "zipf":
            arr = gen_zipf(rng, m, params["a"], params["base"], params["mult"], lo, hi)
        elif kind == "skewed":
            # default skewed = lognormal unless overridden
            arr = gen_lognormal(rng, m, params["median_like"], params["sigma"], lo, hi)
        else:
            raise ValueError(f"Unknown dist: {kind}")
        yield arr
        rem -= m

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Generate a one-column salary dataset targeting a given file size.")
    p.add_argument("--dist", required=True,
                   choices=["uniform","normal","laplace","student_t","lognormal","pareto","zipf","skewed"],
                   help="Distribution shape.")
    p.add_argument("--target-mb", type=float, default=500.0, help="Approx CSV size to generate (megabytes).")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=1_000_000, help="Rows per write chunk.")
    # Salary range
    p.add_argument("--min-salary", type=int, default=30000)
    p.add_argument("--max-salary", type=int, default=300000)
    # Normal params
    p.add_argument("--mean", type=float, default=120000.0)
    p.add_argument("--std", type=float, default=30000.0)
    # Laplace / Student-t params
    p.add_argument("--loc", type=float, default=120000.0)
    p.add_argument("--scale", type=float, default=25000.0)
    p.add_argument("--df", type=float, default=3.0)
    # Lognormal (skewed) params
    p.add_argument("--median-like", type=float, default=80000.0, dest="median_like")
    p.add_argument("--sigma", type=float, default=0.8)
    # Pareto / Zipf
    p.add_argument("--xm", type=float, default=30000.0, help="Pareto minimum scale.")
    p.add_argument("--alpha", type=float, default=2.0, help="Pareto tail parameter.")
    p.add_argument("--zipf-a", type=float, default=2.0, dest="a")
    p.add_argument("--zipf-base", type=float, default=25000.0, dest="base")
    p.add_argument("--zipf-mult", type=float, default=5000.0, dest="mult")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    target_bytes = int(args.target_mb * (1024**2))

    # Calibration pass: sample 200k to estimate avg bytes/row for this distribution & range
    sample = next(stream_values(
        args.dist, rng, total=200_000, chunk=200_000,
        params={
            "mean": args.mean, "std": args.std,
            "loc": args.loc, "scale": args.scale, "df": args.df,
            "median_like": args.median_like, "sigma": args.sigma,
            "xm": args.xm, "alpha": args.alpha,
            "a": args.a, "base": args.base, "mult": args.mult
        },
        lo=args.min_salary, hi=args.max_salary
    ))
    rows = approx_rows_for_size(sample, target_bytes)
    print(f"[calibrate] avg bytes/row ≈ {target_bytes/rows:.2f} → rows ≈ {rows:,}")

    # Re-seed so calibration does not consume stream
    rng = np.random.default_rng(args.seed)

    # Stream to CSV
    n_written = write_rows_stream(
        values_iter=stream_values(
            args.dist, rng, rows, args.chunk,
            params={
                "mean": args.mean, "std": args.std,
                "loc": args.loc, "scale": args.scale, "df": args.df,
                "median_like": args.median_like, "sigma": args.sigma,
                "xm": args.xm, "alpha": args.alpha,
                "a": args.a, "base": args.base, "mult": args.mult
            },
            lo=args.min_salary, hi=args.max_salary
        ),
        total_rows=rows,
        out_path=args.out
    )

    final_bytes = os.path.getsize(args.out)
    print(f"[done] wrote {n_written:,} rows → {final_bytes/1024/1024:.2f} MB to {args.out}")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass


# # Uniform in [30k, 300k]
# python make_salary_dataset.py \
#   --dist uniform --target-mb 500 --out data/salary_uniform_500mb.csv
#
# # Normal centered at 120k with std 30k (clamped into [30k, 300k])
# python make_salary_dataset.py \
#   --dist normal --mean 120000 --std 30000 \
#   --target-mb 500 --out data/salary_normal_500mb.csv
#
# # High-kurtosis (Laplace, sharper peak & heavier tails than normal)
# python make_salary_dataset.py \
#   --dist laplace --loc 120000 --scale 25000 \
#   --target-mb 500 --out data/salary_laplace_500mb.csv
#
# # Very heavy tails (Student-t with low df)
# python make_salary_dataset.py \
#   --dist student_t --df 3 --loc 120000 --scale 20000 \
#   --target-mb 500 --out data/salary_t3_500mb.csv
#
# # Skewed right (Lognormal)
# python make_salary_dataset.py \
#   --dist lognormal --median-like 80000 --sigma 0.8 \
#   --target-mb 500 --out data/salary_lognormal_500mb.csv
#
# # Skewed right (Pareto)
# python make_salary_dataset.py \
#   --dist pareto --xm 30000 --alpha 2.0 \
#   --target-mb 500 --out data/salary_pareto_500mb.csv
#
# # Skewed right (Zipf)
# python make_salary_dataset.py \
#   --dist zipf --zipf-a 2.0 --zipf-base 25000 --zipf-mult 5000 \
#   --target-mb 500 --out data/salary_zipf_500mb.csv

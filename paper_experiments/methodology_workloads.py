"""
Thesis-width workloads for the static reproduction (thesis §4.4.1 eq. 4.1-4.2, §4.4.2).

The repo's registered workloads use widths 1..999 (workload/workload.py max_width=1000); the thesis
uses W ~ U(1, 100). This writes thesis-width copies under
results/paper_experiments/methodology/workloads/ with the repo's own generators and seed 42:
  narrow100_1000000.csv                 L ~ U(D_min, D_max - 1), W ~ U(1, 100), D = [0, 1 000 000]
  hard/<dist>/driven100_1000000.csv     anchor ~ sample (100k reservoir), W ~ U(1, 100), anchor inside

    poetry run python -m paper_experiments.methodology_workloads
"""
import pickle
from pathlib import Path

from workload.workload import generate_data_driven_workload, generate_workload, save_workload_csv

OUT = Path("results/paper_experiments/methodology/workloads")
COUNT = 1_000_000
SEED = 42
MAX_WIDTH = 101  # rng.integers(1, 101) -> W in [1, 100] (upper bound exclusive)
HARD = Path("data/generated/60000000_hard")
DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"narrow100_{COUNT}.csv"
    if not p.exists():
        save_workload_csv(generate_workload(COUNT, 0, 1_000_000, seed=SEED, max_width=MAX_WIDTH), p)
        print("wrote", p)
    for d in DISTS:
        p = OUT / "hard" / d / f"driven100_{COUNT}.csv"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(HARD / d / "meta.pkl", "rb") as f:
            sample = pickle.load(f)[4]
        save_workload_csv(generate_data_driven_workload(COUNT, sample, seed=SEED, max_width=MAX_WIDTH), p)
        print("wrote", p)


if __name__ == "__main__":
    main()

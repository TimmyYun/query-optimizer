#!/usr/bin/env python3
"""
Дополнительные эксперименты вокруг Hybrid (см. EXPERIMENT_RESULTS.md, раздел 9):

  minpts    — модель в бакете обучается, только если в нём >= P точек выборки
  partition — Hybrid поверх equi-depth бакетов вместо equi-width
  width     — ширина запроса как доля ширины бакета (все 6 моделей)
  memory    — все модели при одинаковом бюджете памяти (число float-параметров Hybrid)
  variants  — Hybrid: только параметрические модели, equi-depth, equi-depth без моделей
  memory_ed — базовые модели при бюджете памяти Hybrid-ED-param (после variants)

Каждая часть пишет в отдельный results/<exp>/, базовые результаты не трогаются.
"""
import argparse
import json
import math
import pickle
import subprocess
from pathlib import Path

import numpy as np

from workload.workload import RangeQuery, save_workload_csv

SYNTH = ["normal", "zipf", "uniform", "anti_zipf", "sparse_cluster"]
DSB = ["ws_item_sk", "cr_item_sk", "cr_returned_time_sk"]
MIN_POINTS = [10, 30, 100]
WIDTH_FACTORS = [0.01, 0.1, 0.25, 0.5, 1, 2, 4, 16]
WIDTH_DISTS = ["normal", "sparse_cluster", "zipf", "uniform"]
WIDTH_QUERIES = 100_000

ALL_MODELS = {
    "EquiWidth": "run_equiwidth.py",
    "Hybrid": "run_hybrid.py --no-debug-plot",
    "TKHist": "run_tkhist.py",
    "EquiDepth": "run_equidepth.py",
    "STHist": "run_sthist.py",
    "EntropyHist-MEA": "run_entropyhist.py",
}


def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def static_targets():
    """(сценарий, путь к датасету, путь к ворклоаду) — те же входы, что у run_experiments.py."""
    out = []
    for dist in SYNTH:
        ds = f"data/generated/60000000_hard/{dist}"
        out.append(("narrow", ds, "workload/1000000/narrow/workload.csv"))
        out.append(("data_driven", ds, f"{ds}/workload_driven_1000000.csv"))
    for col in DSB:
        ds = f"data/dsb/{col}"
        out.append(("dsb", ds, str(next(Path(ds).glob("workload_driven_*.csv")))))
    return out


def part_minpts():
    for sc, ds, wl in static_targets():
        for p in MIN_POINTS:
            run_cmd(
                f"poetry run python run_hybrid.py --no-debug-plot --min-sample-points {p} "
                f"--experiment-name abl_minpts_{sc} --dataset {ds} --workload {wl}"
            )


def part_partition():
    for sc, ds, wl in static_targets():
        for extra in ["", "--min-sample-points 30"]:
            run_cmd(
                f"poetry run python run_hybrid.py --no-debug-plot --partition equidepth {extra} "
                f"--experiment-name abl_partition_{sc} --dataset {ds} --workload {wl}"
            )


def make_width_workload(dist, factor):
    """Data-driven запросы фиксированной ширины factor * ширина_бакета (seed 42)."""
    ds = Path(f"data/generated/60000000_hard/{dist}")
    mn, mx, N, freq, sample, k, *_ = pickle.load(open(ds / "meta.pkl", "rb"))
    bw = max(1, int(math.ceil((mx - mn + 1) / k)))
    w = max(1, int(round(factor * bw)))
    rng = np.random.default_rng(42)
    centers = rng.choice(sample, size=WIDTH_QUERIES, replace=True)
    offsets = rng.integers(0, w, size=WIDTH_QUERIES)
    lows = np.clip(centers - offsets, mn, mx)
    highs = np.clip(lows + w - 1, mn, mx)
    path = Path(f"data/workloads/width_sweep/{dist}/w{factor}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_workload_csv([RangeQuery(int(lo), int(hi)) for lo, hi in zip(lows, highs)], path)
    return path


def part_width():
    for dist in WIDTH_DISTS:
        for f in WIDTH_FACTORS:
            wl = make_width_workload(dist, f)
            for script in ALL_MODELS.values():
                run_cmd(
                    f"poetry run python {script} --experiment-name width_sweep_x{f} "
                    f"--dataset data/generated/60000000_hard/{dist} --workload {wl}"
                )


def count_model_floats(model):
    """Число float-параметров одной локальной модели Hybrid (None -> 0)."""
    if model is None:
        return 0
    if isinstance(model, tuple):
        return 4 if model[0] == "poly3" else 2
    name = type(model).__name__
    if name == "IsotonicRegression":
        return 2 * len(model.X_thresholds_)
    if name == "DecisionTreeRegressor":
        return 3 * model.tree_.node_count  # порог, значение, ссылка на ребёнка
    mlp = model.model  # FourierModelWrapper
    return int(sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))


def hybrid_float_budget(dist):
    """
    Сколько float хранит Hybrid на датасете: 1 счётчик на бакет (сетка equi-width
    задаётся mn и шириной) + параметры моделей. Hybrid обучается так же, как в
    run_hybrid.py (seed 42, 200 точек), поэтому модели те же, что в базовом прогоне.
    """
    from models import EquiWidthHistogram, HybridEstimator

    mn, mx, N, freq, sample, k, *_ = pickle.load(
        open(f"data/generated/60000000_hard/{dist}/meta.pkl", "rb")
    )
    est_freq = np.bincount(sample - mn, minlength=len(freq)) * (N / len(sample))
    hist = EquiWidthHistogram.build_from_sample(mn, mx, k, sample, N)
    hyb = HybridEstimator(hist.buckets)
    hyb.train(est_freq, mn, 200, np.random.default_rng(42))
    model_floats = sum(count_model_floats(m) for m in hyb.models.values())
    return {"n_buckets": len(hyb.buckets), "model_floats": model_floats,
            "total_floats": len(hyb.buckets) + model_floats}


def part_memory():
    out = Path("results/abl_memory")
    out.mkdir(parents=True, exist_ok=True)
    budgets = {dist: hybrid_float_budget(dist) for dist in SYNTH}
    json.dump(budgets, open(out / "budgets.json", "w"), indent=2)
    print(json.dumps(budgets, indent=2))

    for sc, ds, wl in static_targets():
        if sc == "dsb":
            continue
        b = budgets[Path(ds).name]["total_floats"]
        # Бакетов столько, чтобы занять тот же бюджет float:
        # EquiWidth — 1 счётчик/бакет; EntropyHist — min, max, count;
        # TKHist — счётчик + 5 пар (значение, счётчик); EquiDepth — граница + до 1 MCV-пары
        runs = {
            "run_equiwidth.py": b,
            "run_entropyhist.py": b // 3,
            "run_tkhist.py": b // 11,
            "run_equidepth.py": b // 3,
        }
        for script, buckets in runs.items():
            run_cmd(
                f"poetry run python {script} --buckets {max(1, buckets)} "
                f"--experiment-name abl_memory_{sc} --dataset {ds} --workload {wl}"
            )


# Варианты Hybrid с подсчётом памяти (n_floats в summary):
# флаги run_hybrid.py; NO_MODELS — те же бакеты без локальных моделей (контроль нарезки)
NO_MODELS = "--min-sample-points 1000000000"
VARIANTS = [
    "",
    "--parametric-only",
    "--partition equidepth",
    "--partition equidepth --parametric-only",
    f"--partition equidepth {NO_MODELS}",
]


def part_variants():
    for sc, ds, wl in static_targets():
        for flags in VARIANTS:
            run_cmd(
                f"poetry run python run_hybrid.py --no-debug-plot {flags} "
                f"--experiment-name abl_variants_{sc} --dataset {ds} --workload {wl}"
            )


def part_memory_ed():
    """Базовые модели при бюджете памяти Hybrid-ED-param (берётся из abl_variants)."""
    for sc, ds, wl in static_targets():
        name = Path(ds).name
        summ = Path(f"results/abl_variants_{sc}/{name}/summary_Hybrid-ED-param.json")
        b = json.load(open(summ))["metrics"]["n_floats"]
        runs = {
            "run_equiwidth.py": b,
            "run_entropyhist.py": b // 3,
            "run_tkhist.py": b // 11,
            "run_equidepth.py": b // 3,
        }
        for script, buckets in runs.items():
            run_cmd(
                f"poetry run python {script} --buckets {max(1, buckets)} "
                f"--experiment-name abl_memory_ed_{sc} --dataset {ds} --workload {wl}"
            )


PARTS = {
    "minpts": part_minpts,
    "partition": part_partition,
    "width": part_width,
    "memory": part_memory,
    "variants": part_variants,
    "memory_ed": part_memory_ed,
}


def main():
    parser = argparse.ArgumentParser(description="Hybrid ablations")
    parser.add_argument("--part", choices=[*PARTS, "all"], required=True)
    args = parser.parse_args()
    for name, fn in PARTS.items():
        if args.part in (name, "all"):
            fn()


if __name__ == "__main__":
    main()

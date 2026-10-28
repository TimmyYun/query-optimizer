"""
Общий стенд для проверки уроков: загрузка датасетов, точная истина,
прогон модели по контракту из lessons/README.md.

Стенд намеренно не зависит от твоего кода (estimators/) и от benchmark_utils:
если ты ошибся в уроке 0, проверки всё равно покажут правду.
"""
import importlib
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHECK_DIR = Path(__file__).resolve().parent
ROOT = CHECK_DIR.parents[1]
for p in (str(ROOT), str(CHECK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

SYNTH = ROOT / "data" / "generated" / "60000000_hard"
DISTS = ["uniform", "anti_zipf", "normal", "sparse_cluster", "zipf"]
DSB = ["cr_item_sk", "cr_returned_time_sk", "ws_item_sk"]
RANDOM_NARROW = ROOT / "workload" / "1000000" / "narrow" / "workload.csv"
# Два сценария дрейфа normal -> zipf за 20 шагов по 5% (см. EXPERIMENT_RESULTS.md, раздел 7):
#   replace — zipf ЗАМЕНЯЕТ normal (N постоянно), жёсткий;
#   add     — zipf ДОБАВЛЯЕТСЯ к normal (N растёт 60M -> 120M), мягкий.
DRIFT_DIRS = {
    "replace": SYNTH / "shift_normal_to_zipf_5%",
    "add": SYNTH / "shift_normal_to_zipf_5%_dataset",
}
SEED = 42


@dataclass(frozen=True)
class Query:
    low: int
    high: int


def all_datasets():
    """name -> (meta.pkl, workload.csv). '@random' — узкие запросы в случайных местах домена."""
    out = {}
    for d in DISTS:
        out[d] = (SYNTH / d / "meta.pkl", SYNTH / d / "workload_driven_1000000.csv")
    for d in DISTS:
        out[f"{d}@random"] = (SYNTH / d / "meta.pkl", RANDOM_NARROW)
    for c in DSB:
        d = ROOT / "data" / "dsb" / c
        wl = sorted(d.glob("workload_driven_*.csv")) or [d / "workload_driven_*.csv"]
        out[f"dsb/{c}"] = (d / "meta.pkl", wl[0])
    return out


def load_meta(path):
    """meta.pkl — 8-tuple, step_N.pkl — 6-tuple; первые 6 полей совпадают."""
    with open(path, "rb") as f:
        m = pickle.load(f)
    mn, mx, N, freq, sample, k = m[:6]
    return int(mn), int(mx), int(N), np.asarray(freq), np.asarray(sample), int(k)


def load_queries(path, n=None):
    """Первые/случайные n запросов (seed 42, исходный порядок сохраняется)."""
    df = pd.read_csv(path)
    lows = df["low"].to_numpy(np.int64)
    highs = df["high"].to_numpy(np.int64)
    if n is not None and n < len(lows):
        idx = np.sort(np.random.default_rng(SEED).choice(len(lows), n, replace=False))
        lows, highs = lows[idx], highs[idx]
    return lows, highs


def true_counts(freq, mn, lows, highs):
    """Точное число строк в [low, high] по префиксным суммам freq. Части вне домена = 0."""
    ps = np.concatenate([[0.0], np.cumsum(freq, dtype=np.float64)])
    lo = np.clip(lows - mn, 0, len(freq))
    hi = np.clip(highs - mn + 1, 0, len(freq))
    return np.maximum(ps[hi] - ps[np.minimum(lo, hi)], 0.0)


def q_errors(truth, pred):
    t = np.maximum(np.asarray(truth, dtype=np.float64), 1.0)
    p = np.maximum(np.asarray(pred, dtype=np.float64), 1.0)
    return np.maximum(t / p, p / t)


def summary(qe):
    return {
        "median": float(np.median(qe)),
        "p95": float(np.percentile(qe, 95)),
        "p99": float(np.percentile(qe, 99)),
        "max": float(np.max(qe)),
        "avg": float(np.mean(qe)),
    }


def import_model(package, model):
    try:
        return importlib.import_module(f"{package}.{model}")
    except ModuleNotFoundError as e:
        if e.name in (package, f"{package}.{model}"):
            sys.exit(f"нет модуля {package}/{model}.py — сначала напиши его (см. lessons/README.md)")
        raise


def build_estimator(mod, sample, mn, mx, N, k, kw):
    # Копия: модель не должна портить выборку. freq модели не передаётся — никогда.
    t0 = time.perf_counter()
    est = mod.build(sample.copy(), mn, mx, N, k, **kw)
    return est, time.perf_counter() - t0


def predict_all(est, lows, highs, truth=None):
    """Оценки по одному запросу (как в харнессе). Если у модели есть observe и
    передана truth — прогон prequential: predict, затем observe с точным ответом."""
    online = truth is not None and hasattr(est, "observe")
    preds = np.empty(len(lows))
    times = np.empty(len(lows))
    t_obs = 0.0
    for i in range(len(lows)):
        q = Query(int(lows[i]), int(highs[i]))
        t0 = time.perf_counter()
        preds[i] = est.predict(q)
        times[i] = time.perf_counter() - t0
        if online:
            t1 = time.perf_counter()
            est.observe(q, float(truth[i]))
            t_obs += time.perf_counter() - t1
    return preds, times, t_obs


def n_floats(est):
    return int(est.n_floats()) if hasattr(est, "n_floats") else None


def run_static(package, model, name, meta_path, wl_path, n_queries, kw):
    mn, mx, N, freq, sample, k = load_meta(meta_path)
    lows, highs = load_queries(wl_path, n_queries)
    truth = true_counts(freq, mn, lows, highs)
    keep = truth > 0  # как load_and_filter_workload: пустые запросы выбрасываются
    lows, highs, truth = lows[keep], highs[keep], truth[keep]

    mod = import_model(package, model)
    est, build_s = build_estimator(mod, sample, mn, mx, N, k, kw)
    preds, times, t_obs = predict_all(est, lows, highs, truth)
    row = summary(q_errors(truth, preds))
    row.update(
        dataset=name,
        n_queries=int(len(lows)),
        build_s=build_s,
        infer_us=float(np.median(times) * 1e6),
        observe_s=t_obs,
        n_floats=n_floats(est),
        n_buckets=k,
    )
    return row


def run_drift(package, model, n_queries, kw, steps=None, scenario="replace"):
    """
    Протокол дрейфа (одинаковый для всех моделей):
      шаг 0: build по выборке шага 0;
      шаг s > 0: refresh(sample_s) если есть (ANALYZE), иначе модель стареет;
      оценка "shock" — на НЕЧЁТНЫХ запросах шага;
      фидбэк — только с ЧЁТНЫХ запросов: observe(q, actual) по одному
      или adapt(queries, actual, ...) пачкой;
      оценка "adapted" — снова на нечётных (честный holdout, без утечки).
    """
    mod = import_model(package, model)
    DRIFT_DIR = DRIFT_DIRS[scenario]
    n_steps = len(list(DRIFT_DIR.glob("step_*.pkl")))
    steps = list(range(n_steps)) if steps is None else steps
    mn0, mx0, N0, _, sample0, k0 = load_meta(DRIFT_DIR / "step_0.pkl")
    est, build_s = build_estimator(mod, sample0, mn0, mx0, N0, k0, kw)

    rows = []
    for s in range(n_steps):
        mn, mx, N, freq, sample, _ = load_meta(DRIFT_DIR / f"step_{s}.pkl")
        refresh_s = 0.0
        if s > 0 and hasattr(est, "refresh"):
            t0 = time.perf_counter()
            est.refresh(sample.copy(), mn, mx, N)
            refresh_s = time.perf_counter() - t0
        if s not in steps:
            continue  # шаг пропускаем в отчёте, но ANALYZE всё равно произошёл

        lows, highs = load_queries(DRIFT_DIR / f"step_{s}_workload_driven_1000000.csv", n_queries)
        truth = true_counts(freq, mn, lows, highs)
        ev, fb = np.arange(1, len(lows), 2), np.arange(0, len(lows), 2)

        p_shock, t_shock, _ = predict_all(est, lows[ev], highs[ev])
        m_shock = summary(q_errors(truth[ev], p_shock))

        feedback_s = 0.0
        info = {}
        if hasattr(est, "observe"):
            _, _, feedback_s = predict_all(est, lows[fb], highs[fb], truth[fb])
        elif s > 0 and hasattr(est, "adapt"):
            qs = [Query(int(a), int(b)) for a, b in zip(lows[fb], highs[fb])]
            t0 = time.perf_counter()
            info = est.adapt(qs, truth[fb].copy(), sample.copy(), mn, mx, N) or {}
            feedback_s = time.perf_counter() - t0

        if feedback_s > 0 or info:
            p_ad, _, _ = predict_all(est, lows[ev], highs[ev])
            m_ad = summary(q_errors(truth[ev], p_ad))
        else:
            m_ad = m_shock

        rows.append({
            "step": s,
            "shift": f"{s * 100 // max(1, n_steps - 1)}%",
            "shock_median": m_shock["median"],
            "shock_p95": m_shock["p95"],
            "adapted_median": m_ad["median"],
            "adapted_p95": m_ad["p95"],
            "build_s": build_s if s == 0 else 0.0,
            "refresh_s": refresh_s,
            "feedback_s": feedback_s,
            "infer_us": float(np.median(t_shock) * 1e6),
            "n_floats": n_floats(est),
            **{f"info_{k}": v for k, v in info.items()},
        })
    return rows

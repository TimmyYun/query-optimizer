#!/usr/bin/env python3
"""
Переборы бюджета и масштабов: где Hybrid выигрывает и почему.

  memory  — ошибка от памяти: каждая модель на сетке бюджетов, ось X — число float
  width   — ширина запроса x ширина бакета: EquiWidth и Hybrid на одних и тех же бакетах
  sample  — размер сэмпла n: сколько точек сэмпла на бакет нужно, чтобы выучить форму
  period  — синтетика с периодом микроструктуры P относительно ширины бакета
  plots   — графики по готовым results/sweep_<part>/results.csv

Каждая часть пишет results/sweep_<part>/results.csv (строка = модель x точка перебора).
Всё считается в одном процессе на датасет, без subprocess на каждую точку, и на
QUERIES запросах, а не на 1 млн: медиана по 100k запросов устойчива до 3-го знака.

Модели строятся только из сэмпла. freq харнесс использует для истины, для центров
запросов (по данным, а не по сэмплу) и для новых сэмплов в частях sample и period.
"""
import argparse
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmark_utils import exact_counts, load_and_filter_workload, run_queries
from data.datasets import DOMAIN_MAX, freedman_diaconis_bins
from models import (
    EntropyHistogram1D,
    EquiDepthHistogram,
    EquiWidthHistogram,
    HistogramConfig,
    HybridEstimator,
    RangeQuery,
    TKHistogram,
    q_error_vec,
)

HARD = "data/generated/60000000_hard"
SIMPLE = "data/generated/60000000_simple"
NARROW_WL = "workload/1000000/narrow/workload.csv"
QUERIES = 100_000  # memory, sample: первые 100k запросов основного workload
SWEEP_QUERIES = 30_000  # width, period: сгенерированные запросы на точку
POINTS = 200  # точек CDF на бакет, как в run_hybrid.py
WORKERS = 6
RESULTS = "results"  # корень результатов (--out-dir); старые прогоны *_v1 всегда читаются из results/

HYBRID_VARIANTS = {
    "Hybrid": {},
    "Hybrid-param": {"parametric_only": True},
}
HIST_MODELS = ["EquiWidth", "TKHist", "EntropyHist-MEA"]

# memory
MEMORY_DATASETS = [f"{HARD}/normal", f"{HARD}/sparse_cluster", f"{HARD}/zipf", f"{HARD}/uniform", f"{SIMPLE}/normal"]
HIST_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
# statistics_target; ANALYZE берёт 300 * target строк, при target > 333 это весь 100k сэмпл
ED_GRID = [10, 20, 50, 100, 200, 333, 1000, 3000, 10_000]
HYBRID_GRID = [10, 25, 50, 100, 200, 400, 800, 1600]

# width
WIDTH_DATASETS = [f"{HARD}/normal", f"{HARD}/sparse_cluster", f"{HARD}/zipf", f"{SIMPLE}/normal"]
WIDTH_BUCKETS = [25, 100, 400, 1600]
WIDTHS = [10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]

# sample
SAMPLE_DATASETS = WIDTH_DATASETS
SAMPLE_SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000]
SAMPLE_MODELS = ["EquiWidth", "EntropyHist-MEA", "Hybrid"]

# period
PERIODS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16_000, 32_000, 64_000]
PERIOD_BUCKETS = [100, 873]  # ширина бакета ~10k (FD по сэмплу) и ~1146 (FD по N)
PERIOD_N = 60_000_000
PERIOD_SAMPLE = 100_000


def label(ds):
    """data/generated/60000000_hard/normal -> hard/normal"""
    return f"{Path(ds).parent.name.split('_')[-1]}/{Path(ds).name}"


def load_meta(ds):
    mn, mx, N, freq, sample, k, *_ = pickle.load(open(Path(ds) / "meta.pkl", "rb"))
    return mn, mx, N, freq, sample, k


def first_queries(path, mn, mx, freq, n):
    """Первые n запросов workload после отброса нулевой истины; истина в строках (>= 1)."""
    queries, sel = load_and_filter_workload(path, mn, mx, freq)
    return queries[:n], np.maximum(np.rint(sel[:n] * freq.sum()), 1.0)


def data_driven_queries(freq, mn, mx, width, n, seed):
    """
    Центры запросов — строки таблицы (по freq), ширина — число или (lo, hi) включительно.
    Центр внутри запроса, поэтому истина всегда > 0.
    """
    rng = np.random.default_rng(seed)
    centers = mn + rng.choice(len(freq), size=n, p=freq / freq.sum())
    if isinstance(width, tuple):
        w = rng.integers(width[0], width[1] + 1, size=n)
    else:
        w = np.full(n, width)
    offsets = (rng.random(n) * w).astype(np.int64)
    lows = np.clip(centers - offsets, mn, mx)
    highs = np.clip(lows + w - 1, mn, mx)
    queries = [RangeQuery(int(lo), int(hi)) for lo, hi in zip(lows, highs)]
    return queries, np.maximum(exact_counts(queries, freq, mn), 1.0)


def draw_sample(freq, mn, n, seed):
    """Сэмпл n строк без возвращения из таблицы, заданной freq (как reservoir sampling)."""
    counts = np.random.default_rng(seed).multivariate_hypergeometric(freq.astype(np.int64), n)
    return mn + np.repeat(np.arange(len(freq)), counts)


def build(model, B, mn, mx, N, sample):
    """
    Строит модель при бюджете B (бакеты; у EquiDepth — statistics_target).
    Возвращает (predict, n_floats, build_s). n_floats — сколько чисел хранит модель:
      EquiWidth — счётчик на бакет (сетку задают mn и ширина);
      TKHist — счётчик + пары (значение, счётчик) top-k;
      EquiDepth — границы бакетов + пары MCV;
      EntropyHist — min, max, count на бакет;
      Hybrid — HybridEstimator.param_count() (счётчики + параметры моделей).
    """
    t0 = time.perf_counter()
    if model == "EquiWidth":
        h = EquiWidthHistogram.build_from_sample(mn, mx, B, sample, N)
        floats = len(h.buckets)
    elif model == "TKHist":
        h = TKHistogram.build_from_sample(mn, mx, B, sample, N)
        floats = len(h.buckets) + 2 * h.n_topk_entries
    elif model == "EquiDepth":
        h = EquiDepthHistogram.build_from_sample(sample=sample, total_rows=N, statistics_target=B)
        floats = (h.n_buckets + 1) + 2 * h.num_mcv
    elif model == "EntropyHist-MEA":
        h = EntropyHistogram1D.build_from_sample(sample, N, HistogramConfig(n_buckets=B))
        floats = 3 * h.B_effective
    else:
        buckets = EquiWidthHistogram.build_from_sample(mn, mx, B, sample, N).buckets
        h = HybridEstimator(buckets, **HYBRID_VARIANTS[model])
        est_freq = np.bincount(sample - mn, minlength=mx - mn + 1) * (N / len(sample))
        h.train(est_freq, mn, POINTS, np.random.default_rng(42))
        floats = h.param_count()
    return h.predict, floats, time.perf_counter() - t0


def score(predict, queries, true_counts):
    """q-error как в evaluate_and_save: истина и оценка обрезаны снизу до 1 строки."""
    pred, times, _ = run_queries(predict, queries, true_counts)
    qe = q_error_vec(true_counts, np.maximum(pred, 1.0))
    return {
        "median": float(np.median(qe)),
        "p25": float(np.percentile(qe, 25)),
        "p75": float(np.percentile(qe, 75)),
        "p95": float(np.percentile(qe, 95)),
        "mean": float(np.mean(qe)),
        "infer_us": float(np.median(times) * 1e6),
        "n_queries": len(queries),
    }


def bucket_width(mn, mx, B):
    return max(1, int(math.ceil((mx - mn + 1) / B)))


# ----------------------------------------------------------------------
# Части
# ----------------------------------------------------------------------


def memory_task(ds):
    mn, mx, N, freq, sample, k = load_meta(ds)
    k_n = freedman_diaconis_bins(sample, mn, mx, len(sample))
    workloads = {
        "narrow": first_queries(NARROW_WL, mn, mx, freq, QUERIES),
        "data_driven": first_queries(f"{ds}/workload_driven_1000000.csv", mn, mx, freq, QUERIES),
    }

    def fd_tag(B):
        return "FD(N)" if B == k else "FD(n)" if B == k_n else ""

    plan = [(m, B) for m in HIST_MODELS for B in sorted(set(HIST_GRID + [k, k_n]))]
    plan += [("EquiDepth", t) for t in ED_GRID]
    plan += [(v, B) for v in HYBRID_VARIANTS for B in sorted(set(HYBRID_GRID + [k, k_n]))]

    rows = []
    for model, B in plan:
        predict, floats, build_s = build(model, B, mn, mx, N, sample)
        for wl, (queries, true) in workloads.items():
            rows.append({
                "dataset": label(ds), "workload": wl, "model": model, "B": B,
                "fd": "" if model == "EquiDepth" else fd_tag(B),
                "n_floats": floats, "build_s": build_s, **score(predict, queries, true),
            })
        print(f"[memory] {label(ds)} {model} B={B} floats={floats} "
              f"med={rows[-2]['median']:.3f}/{rows[-1]['median']:.3f}", flush=True)
    return rows


def width_task(ds):
    mn, mx, N, freq, sample, _ = load_meta(ds)
    workloads = {w: data_driven_queries(freq, mn, mx, w, SWEEP_QUERIES, [42, w]) for w in WIDTHS}
    rows = []
    for B in WIDTH_BUCKETS:
        bw = bucket_width(mn, mx, B)
        for model in ["EquiWidth", "Hybrid"]:
            predict, floats, build_s = build(model, B, mn, mx, N, sample)
            for w, (queries, true) in workloads.items():
                rows.append({
                    "dataset": label(ds), "model": model, "B": B, "bucket_width": bw,
                    "query_width": w, "r": w / bw, "n_floats": floats, "build_s": build_s,
                    **score(predict, queries, true),
                })
            print(f"[width] {label(ds)} {model} B={B} done", flush=True)
    return rows


def sample_task(ds):
    mn, mx, N, freq, _, k = load_meta(ds)
    queries, true = first_queries(NARROW_WL, mn, mx, freq, QUERIES)
    rows = []
    for n in SAMPLE_SIZES:
        sample = draw_sample(freq, mn, n, [42, n])
        for B in [100, k]:
            for model in SAMPLE_MODELS:
                predict, floats, build_s = build(model, B, mn, mx, N, sample)
                rows.append({
                    "dataset": label(ds), "model": model, "n_sample": n, "B": B,
                    "fd": "FD(N)" if B == k else "", "pts_per_bucket": n / B,
                    "n_floats": floats, "build_s": build_s, **score(predict, queries, true),
                })
            print(f"[sample] {label(ds)} n={n} B={B} done", flush=True)
    return rows


def make_period_dataset(P, chunk=5_000_000):
    """
    Как gen_values(dist="normal") в data/datasets.py, но с ячейкой ширины P вместо 1000:
    глобальный колокол по ячейкам + мини-колокол (sd = P/6) в середине каждой ячейки.
    Строится сразу freq (60 млн строк кусками), сэмпл — 100k строк без возвращения.
    """
    rng = np.random.default_rng([42, P])
    n_chunks = DOMAIN_MAX // P
    counts = np.zeros(DOMAIN_MAX + 1, dtype=np.int64)
    left = PERIOD_N
    while left > 0:
        n = min(chunk, left)
        macro = np.clip(np.round(rng.normal(n_chunks / 2.0, n_chunks / 6.0, n)), 0, n_chunks - 1)
        micro = np.clip(np.round(rng.normal(P / 2.0, P / 6.0, n)), 0, P - 1)
        counts += np.bincount((macro * P + micro).astype(np.int64), minlength=DOMAIN_MAX + 1)
        left -= n
    nz = np.flatnonzero(counts)
    mn, mx = int(nz[0]), int(nz[-1])
    freq = counts[mn : mx + 1]
    # Отдельный поток для сэмпла: тот же поток, что у генератора, повторил бы его случайности
    sample = draw_sample(freq, mn, PERIOD_SAMPLE, [42, P, 1])
    return mn, mx, int(freq.sum()), freq, sample


def period_task(P):
    mn, mx, N, freq, sample = make_period_dataset(P)
    queries, true = data_driven_queries(freq, mn, mx, (2, 1000), SWEEP_QUERIES, [42, P, 2])
    rows = []

    def add(model, B, predict, floats, build_s, base_B):
        rows.append({
            "period": P, "model": model, "B": B, "base_B": base_B,
            "bucket_width": bucket_width(mn, mx, base_B), "n_floats": floats,
            "build_s": build_s, **score(predict, queries, true),
        })

    for B in PERIOD_BUCKETS:
        floats_of = {}
        for model in ["EquiWidth", *HYBRID_VARIANTS]:
            predict, floats, build_s = build(model, B, mn, mx, N, sample)
            floats_of[model] = floats
            add(model, B, predict, floats, build_s, B)
        # EquiWidth при той же памяти, что у каждого варианта Hybrid
        for ref in HYBRID_VARIANTS:
            B_eq = min(floats_of[ref], mx - mn + 1)
            predict, floats, build_s = build("EquiWidth", B_eq, mn, mx, N, sample)
            add(f"EquiWidth@{ref}", B_eq, predict, floats, build_s, B)
        print(f"[period] P={P} B={B} done", flush=True)
    return rows


PARTS = {
    "memory": (memory_task, MEMORY_DATASETS),
    "width": (width_task, WIDTH_DATASETS),
    "sample": (sample_task, SAMPLE_DATASETS),
    "period": (period_task, PERIODS),
}


def limit_blas_threads():
    # Процессов несколько, у каждого свой пул потоков Hybrid: BLAS-потоки ограничиваем
    threadpool_limits(limits=2)


def run_part(name):
    task, items = PARTS[name]
    out = Path(f"{RESULTS}/sweep_{name}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(items)), initializer=limit_blas_threads) as ex:
        futures = {ex.submit(task, item): item for item in items}
        for fut in as_completed(futures):
            rows += fut.result()
            # Пишем после каждого датасета: упавший прогон не теряет готовые точки
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            print(f"=== [{name}] {futures[fut]} finished", flush=True)
    print(f"Saved {out / 'results.csv'} ({len(rows)} rows)")


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

# Цвет закреплён за моделью во всех графиках (категориальная палитра, фиксированный порядок)
COLORS = {
    "Hybrid": "#2a78d6",
    "EquiWidth": "#eb6834",
    "Hybrid-param": "#eda100",
    "TKHist": "#e87ba4",
    "EquiDepth": "#008300",
    "EntropyHist-MEA": "#6250d6",
    "EquiWidth@Hybrid": "#eb6834",
    "EquiWidth@Hybrid-param": "#eb6834",
    "Hybrid v1": "#898781",  # старый выбор модели, для сравнения
}
STYLES = {"EquiWidth@Hybrid": "--", "EquiWidth@Hybrid-param": "-.", "Hybrid v1": "--"}
MARKERS = {"Hybrid": "o", "Hybrid v1": "o", "Hybrid-param": "^", "EquiWidth": "D",
           "TKHist": "v", "EquiDepth": "P", "EntropyHist-MEA": "X",
           "EquiWidth@Hybrid": "D", "EquiWidth@Hybrid-param": "D"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def setup_style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 5,
    })


def line(ax, d, x, y, model):
    ax.plot(d[x], d[y], color=COLORS[model], ls=STYLES.get(model, "-"),
            marker=MARKERS[model], label=model)


def plain_log_y(ax):
    """Лог-ось Y с обычными числами (1.05, 2, 100) вместо 1.05×10^0."""
    from matplotlib.ticker import LogFormatter

    ax.yaxis.set_major_formatter(LogFormatter())
    ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))


def finish(plt, fig, axes, title, path):
    """Одна легенда над всеми панелями (а не поверх линий), заголовок, сохранение."""
    handles = {}
    for ax in np.ravel(axes):
        for h, name in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(name, h)
    names = [m for m in COLORS if m in handles]
    fig.legend([handles[m] for m in names], names, loc="upper center", ncol=len(names),
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_results(part):
    """
    results.csv части + строки Hybrid из results/sweep_<part>_v1 (прогон до нового
    выбора модели) под именем "Hybrid v1" — на графиках видно «до» и «после».
    """
    df = pd.read_csv(f"{RESULTS}/sweep_{part}/results.csv")
    old = Path(f"results/sweep_{part}_v1/results.csv")
    if old.exists():
        v1 = pd.read_csv(old)
        df = pd.concat([df, v1[v1.model == "Hybrid"].assign(model="Hybrid v1")], ignore_index=True)
    return df


def plot_memory(plt):
    df = load_results("memory")
    datasets = [label(d) for d in MEMORY_DATASETS if label(d) in set(df["dataset"])]
    models = [m for m in COLORS if m in set(df["model"])]
    for wl in ["narrow", "data_driven"]:
        fig, axes = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 7), squeeze=False)
        for j, ds in enumerate(datasets):
            for i, metric in enumerate(["median", "p95"]):
                ax = axes[i][j]
                for m in models:
                    d = df[(df.dataset == ds) & (df.workload == wl) & (df.model == m)].sort_values("n_floats")
                    if d.empty:
                        continue
                    line(ax, d, "n_floats", metric, m)
                    # Текущая настройка бенчмарка (FD по N) и FD по сэмплу — кольцом
                    for tag, mk in [("FD(N)", "o"), ("FD(n)", "s")]:
                        p = d[d.fd == tag]
                        if m == "Hybrid" and not p.empty:
                            ax.scatter(p.n_floats, p[metric], s=110, facecolors="none",
                                       edgecolors=INK, linewidths=1.2, marker=mk, zorder=5)
                            ax.annotate(tag, (p.n_floats.iloc[0], p[metric].iloc[0]),
                                        xytext=(6, 6), textcoords="offset points", color=INK2, fontsize=8)
                ax.set_xscale("log")
                ax.set_yscale("log")
                plain_log_y(ax)
                ax.set_title(f"{ds} — {metric} q-error")
                ax.set_xlabel("memory, floats")
        finish(plt, fig, axes,
               f"Error vs memory, workload: {wl} (ring = Hybrid at FD(N), square = Hybrid at FD(n))",
               f"{RESULTS}/sweep_memory/memory_{wl}.png")


def plot_width(plt):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    df = load_results("width")
    datasets = [label(d) for d in WIDTH_DATASETS if label(d) in set(df["dataset"])]
    # Расходящаяся шкала: синий — Hybrid лучше EquiWidth, красный — хуже, серый — паритет
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.2 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        base = df[(df.dataset == ds) & (df.model == "EquiWidth")].set_index(["B", "query_width"])["median"]
        for j, model in enumerate(["Hybrid", "Hybrid v1"]):
            ax = axes[i][j]
            d = df[(df.dataset == ds) & (df.model == model)].set_index(["B", "query_width"])["median"]
            ratio = (base / d).unstack("query_width").sort_index(ascending=False)
            logr = np.log2(ratio.values)
            lim = max(0.05, np.nanmax(np.abs(logr)))
            ax.imshow(logr, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
            ax.grid(False)
            bws = df[df.dataset == ds].groupby("B")["bucket_width"].first()
            ax.set_yticks(range(len(ratio.index)), [f"B={b}\n(bw={bws[b]})" for b in ratio.index])
            ax.set_xticks(range(len(ratio.columns)), [f"{w:g}" for w in ratio.columns], rotation=45)
            ax.set_xlabel("query width")
            for (r, c), v in np.ndenumerate(ratio.values):
                dark = abs(logr[r, c]) > 0.6 * lim
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK)
            ax.set_title(f"{ds}: EquiWidth / {model}")
    fig.suptitle("Median q-error of EquiWidth divided by Hybrid on the same B buckets "
                 "(blue > 1: Hybrid better, red < 1: worse; bw = bucket width)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"{RESULTS}/sweep_width/width_heatmap.png", dpi=130)
    plt.close(fig)


def plot_sample(plt):
    df = load_results("sample")
    datasets = [label(d) for d in SAMPLE_DATASETS if label(d) in set(df["dataset"])]
    fig, axes = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 7), squeeze=False)
    for j, ds in enumerate(datasets):
        for i, fd in enumerate(["", "FD(N)"]):
            ax = axes[i][j]
            sub = df[(df.dataset == ds) & (df.fd.fillna("") == fd)]
            for m in [m for m in COLORS if m in set(sub.model)]:
                d = sub[sub.model == m].sort_values("pts_per_bucket")
                if not d.empty:
                    line(ax, d, "pts_per_bucket", "median", m)
            ax.set_xscale("log")
            ax.set_yscale("log")
            plain_log_y(ax)
            B = int(sub.B.iloc[0]) if not sub.empty else 0
            ax.set_title(f"{ds}, B={B}" + (" (FD(N))" if fd else ""))
            ax.set_xlabel("sample points per bucket (n / B)")
            ax.set_ylabel("median q-error")
    finish(plt, fig, axes, "Sample size sweep (n = 10k … 1M), narrow workload",
           f"{RESULTS}/sweep_sample/sample.png")


def plot_period(plt):
    df = load_results("period")
    df["P_over_bw"] = df["period"] / df["bucket_width"]
    fig, axes = plt.subplots(2, len(PERIOD_BUCKETS), figsize=(6 * len(PERIOD_BUCKETS), 7), squeeze=False)
    for j, B in enumerate(PERIOD_BUCKETS):
        sub = df[df.base_B == B]
        for i, metric in enumerate(["median", "n_floats"]):
            ax = axes[i][j]
            for m in COLORS:
                d = sub[sub.model == m].sort_values("P_over_bw")
                if not d.empty:
                    line(ax, d, "P_over_bw", metric, m)
            ax.axvline(1.0, color=AXIS, lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if metric == "median":
                plain_log_y(ax)
            bw = int(sub.bucket_width.iloc[0]) if not sub.empty else 0
            ax.set_title(f"B={B} (bucket width {bw}) — {'median q-error' if metric == 'median' else 'memory, floats'}")
            ax.set_xlabel("micro-structure period P / bucket width")
    finish(plt, fig, axes, "Controlled intra-bucket structure: hard-normal generator with period P",
           f"{RESULTS}/sweep_period/period.png")


def make_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style(plt)
    for name, fn in [("memory", plot_memory), ("width", plot_width), ("sample", plot_sample), ("period", plot_period)]:
        if Path(f"{RESULTS}/sweep_{name}/results.csv").exists():
            fn(plt)
            print(f"Plotted sweep_{name}")


def main():
    parser = argparse.ArgumentParser(description="Budget / scale sweeps around Hybrid")
    parser.add_argument("--part", choices=[*PARTS, "plots", "all"], required=True)
    parser.add_argument("--out-dir", default="results", help="results root (default: results)")
    args = parser.parse_args()
    global RESULTS
    RESULTS = args.out_dir  # воркеры наследуют через fork
    for name in PARTS:
        if args.part in (name, "all"):
            run_part(name)
    if args.part in ("plots", "all"):
        make_plots()


if __name__ == "__main__":
    main()

"""
Тесты Entropy-based гистограмм по разделу 10 спецификации
(entropy_histograms_1d_cleanroom_spec.md). Пункты 8–9 (StreamingWrapper,
shift-тесты) не покрыты: расширение не реализовано.

    poetry run python -m pytest tests/test_entropy_histogram.py -q
"""
import itertools
import json
import math

import numpy as np
import pytest

from models.entropy_histogram import (
    EQ_ESTIMATORS,
    ME,
    MB,
    MSE,
    TIE_DECIMALS,
    WEIGHTED_ABLATION,
    EntropyHistogram1D,
    HistogramConfig,
    PrefixMoments,
    aggregate_distribution,
    area_weights,
    entropy,
)

# Figure 1 статьи
FIG1_VALUES = [1, 2, 5, 6, 7, 8]
FIG1_FREQ = [5, 5, 7, 2, 3, 4]

ALL_CONFIGS = [
    dict(method=ME, me_input="area"),
    dict(method=ME, me_input="frequency"),
    dict(method=ME, me_input="area", me_objective=WEIGHTED_ABLATION),
    dict(method=ME, me_input="frequency", me_objective=WEIGHTED_ABLATION),
    dict(method=MSE),
    dict(method=MB),
]


def build(values, freq, B, **kw):
    return EntropyHistogram1D.build(values, freq, HistogramConfig(n_buckets=B, **kw))


# ----------------------------------------------------------------------
# Прямые O(length) формулы — независимая от префиксов реализация (оракул)
# ----------------------------------------------------------------------


def direct_mse_cost(f):
    f = np.asarray(f, dtype=np.float64)
    p = f / f.sum()
    s = 2.0 ** -entropy(f)
    return f.sum() * float(np.sum(p * (p - s) ** 2))


def direct_bf(f):
    return len(f) * 2.0 ** -entropy(f) - 1.0


def direct_mb_cost(f):
    return float(np.sum(f)) * max(0.0, direct_bf(f))


def direct_score(w_seg, f_seg, k, cfg, W_total):
    """Скор разреза k (относительно начала бакета) по прямым формулам."""
    if cfg.method == ME:
        L, R = w_seg[:k], w_seg[k:]
        if cfg.me_objective == WEIGHTED_ABLATION:
            score = sum(w_seg) * entropy(w_seg) - (
                sum(L) * entropy(L) + sum(R) * entropy(R)
            )
        else:
            score = sum(w_seg) * (entropy(w_seg) - (entropy(L) + entropy(R)))
    else:
        cost = direct_mse_cost if cfg.method == MSE else direct_mb_cost
        score = -(cost(f_seg) - cost(f_seg[:k]) - cost(f_seg[k:]))
    return round(score / W_total, TIE_DECIMALS)


def brute_force_splits(values, freq, cfg):
    """
    Медленный жадный оракул: на каждом шаге перебирает ВСЕ разрезы ВСЕХ живых
    бакетов с тем же ключом ничьих (скор, -вес родителя, разрез, start).
    """
    v = np.asarray(values)
    f = np.asarray(freq, dtype=np.float64)
    use_area = cfg.method == ME and cfg.me_input == "area"
    w = area_weights(v, f) if use_area else f
    W_total = (w if cfg.method == ME else f).sum()
    live = [(0, len(v))]
    splits = []
    while len(live) < min(cfg.n_buckets, len(v)):
        best = None
        for start, end in live:
            parent_w = (w if cfg.method == ME else f)[start:end].sum()
            for k in range(1, end - start):
                key = (direct_score(w[start:end], f[start:end], k, cfg, W_total), -parent_w, start + k, start)
                if best is None or key < best[0]:
                    best = (key, start, end)
        (_, _, k, _), start, end = best
        live.remove((start, end))
        live += [(start, k), (k, end)]
        splits.append(k)
    return sorted(splits)


# ----------------------------------------------------------------------
# 1. Математика
# ----------------------------------------------------------------------


def test_entropy_reference_values():
    assert entropy([5]) == 0.0
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert entropy([8, 2]) == pytest.approx(0.7219280949, abs=1e-10)


def test_bias_factor_and_mse_reference_values():
    assert direct_bf([5, 5]) == pytest.approx(0.0, abs=1e-12)
    assert direct_bf([8, 2]) > 0
    # p = (0.8, 0.2), s = 2^-H; E = sum p (p - s)^2
    s = 2 ** -0.7219280948873623
    expected = 10 * (0.8 * (0.8 - s) ** 2 + 0.2 * (0.2 - s) ** 2)
    assert direct_mse_cost([8, 2]) == pytest.approx(expected)
    assert direct_mse_cost([7]) == pytest.approx(0.0, abs=1e-12)


def test_entropy_decomposition_eq3():
    """H(X) = p_L H(X_L) + p_R H(X_R) + h(p_L) для непересекающихся носителей."""
    L, R = [3, 1, 4], [1, 5, 9, 2]
    pL = sum(L) / (sum(L) + sum(R))
    h = -(pL * math.log2(pL) + (1 - pL) * math.log2(1 - pL))
    assert entropy(L + R) == pytest.approx(
        pL * entropy(L) + (1 - pL) * entropy(R) + h
    )


def test_me_area_uses_areas_not_counts():
    h = build(FIG1_VALUES, FIG1_FREQ, 2, method=ME, me_input="area")
    first = h.buckets[0]
    assert first.entropy_area == pytest.approx(entropy([5, 15, 7, 2, 3, 4][: first.end_idx]))
    assert first.entropy_frequency == pytest.approx(entropy(FIG1_FREQ[: first.end_idx]))


# ----------------------------------------------------------------------
# 2. Figure 1 статьи
# ----------------------------------------------------------------------


def test_fig1_spreads_and_areas():
    v, f = aggregate_distribution(FIG1_VALUES, FIG1_FREQ)
    assert len(v) == 6 and f.sum() == 26
    assert area_weights(v, f).tolist() == [5, 15, 7, 2, 3, 4]


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_fig1_one_bucket_per_value_is_exact(cfg):
    h = build(FIG1_VALUES, FIG1_FREQ, 6, **cfg)
    assert h.B_effective == 6 and h.total_tuples == 26
    for v, f in zip(FIG1_VALUES, FIG1_FREQ):
        for name in EQ_ESTIMATORS:
            assert h.estimate_eq(v, name) == pytest.approx(f)
    assert h.estimate_range(0, 100) == 26
    assert h.estimate_range(1, 8) == 26


# ----------------------------------------------------------------------
# 3. Префиксные моменты == прямые формулы
# ----------------------------------------------------------------------


def test_prefix_moments_match_direct():
    rng = np.random.default_rng(42)
    for _ in range(50):
        f = rng.integers(1, 1000, size=rng.integers(1, 40)).astype(np.float64)
        m = PrefixMoments(f, with_powers=True)
        for start, end in itertools.combinations(range(len(f) + 1), 2):
            seg = f[start:end]
            assert float(m.entropy(start, end)) == pytest.approx(entropy(seg), abs=1e-10)
            assert float(m.mse_cost(start, end)) == pytest.approx(direct_mse_cost(seg), rel=1e-8, abs=1e-8)
            assert float(m.mb_cost(start, end)) == pytest.approx(direct_mb_cost(seg), rel=1e-8, abs=1e-8)


# ----------------------------------------------------------------------
# 4. Куча == медленный оракул
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_heap_matches_brute_force_oracle(cfg):
    rng = np.random.default_rng(42)
    for _ in range(40):
        N = int(rng.integers(2, 16))
        values = np.sort(rng.choice(200, size=N, replace=False))
        # Малые частоты -> много ничьих, проверяем и правило ничьих
        freq = rng.integers(1, 6, size=N)
        for B in range(1, N + 1):
            config = HistogramConfig(n_buckets=B, **cfg)
            h = EntropyHistogram1D.build(values, freq, config)
            assert h.split_indices == brute_force_splits(values, freq, config), (values, freq, B)


# ----------------------------------------------------------------------
# 5. Инварианты, детерминизм
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_bucket_invariants_and_shuffle_invariance(cfg):
    rng = np.random.default_rng(42)
    raw = rng.zipf(1.5, size=5000) % 3000  # сырые строки с дубликатами
    h = EntropyHistogram1D.build(raw, None, HistogramConfig(n_buckets=40, **cfg))
    h2 = EntropyHistogram1D.build(rng.permutation(raw), None, HistogramConfig(n_buckets=40, **cfg))

    assert h.split_indices == h2.split_indices
    assert h.build_input_signature == h2.build_input_signature
    assert h.B_effective == 40
    assert sum(b.tuple_count for b in h.buckets) == len(raw)
    assert sum(b.distinct_count for b in h.buckets) == len(np.unique(raw))
    covered = [i for b in h.buckets for i in range(b.start_idx, b.end_idx)]
    assert covered == list(range(h.distinct_count))


def test_build_is_deterministic():
    rng = np.random.default_rng(7)
    values, freq = rng.choice(10_000, 500, replace=False), rng.integers(1, 100, 500)
    a = build(values, freq, 30)
    b = build(values, freq, 30)
    assert a.to_dict() == b.to_dict()


def test_serialization_roundtrip():
    rng = np.random.default_rng(1)
    h = EntropyHistogram1D.build_from_sample(
        rng.integers(0, 10_000, 3000), 600_000, HistogramConfig(method=MB, n_buckets=25)
    )
    h2 = EntropyHistogram1D.from_dict(json.loads(json.dumps(h.to_dict())))
    for lo, hi in rng.integers(-100, 10_100, size=(200, 2)):
        assert h2.estimate_range(lo, hi) == h.estimate_range(lo, hi)


# ----------------------------------------------------------------------
# 6. Граничные случаи
# ----------------------------------------------------------------------


def test_empty_distribution():
    h = build([], [], 5)
    assert h.B_effective == 0
    assert h.estimate_range(0, 10) == 0.0 and h.estimate_eq(3) == 0.0


def test_single_value():
    h = build([42], [7], 10)
    assert h.B_effective == 1
    assert h.estimate_eq(42) == 7 and h.estimate_eq(41) == 0
    assert h.estimate_range(0, 100) == 7 and h.estimate_range(43, 100) == 0


def test_one_bucket_and_budget_above_N():
    assert build(FIG1_VALUES, FIG1_FREQ, 1).B_effective == 1
    assert build(FIG1_VALUES, FIG1_FREQ, 100).B_effective == 6


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_identical_frequencies_and_hotspot(cfg):
    assert build(range(20), [3] * 20, 5, **cfg).B_effective == 5
    f = [1] * 50
    f[17] = 10_000  # 99%-hotspot
    h = build(range(50), f, 3, **cfg)
    assert h.B_effective == 3 and h.total_tuples == sum(f)


def test_huge_counts_and_uneven_float_spacing():
    values = [0.0, 1e-3, 0.5, 0.5000001, 10.0, 1e6]
    freq = [10**12, 3, 10**15, 1, 7, 10**9]
    for cfg in ALL_CONFIGS:
        h = build(values, freq, 3, **cfg)
        assert h.total_tuples == sum(freq)
        assert h.estimate_range(-1, 2e6) == pytest.approx(sum(freq))


def test_point_inside_gap_and_out_of_domain():
    # MSE, B = 3: бакеты [1,5] (17 строк, 3 значения), [6,7], [8]
    h = build(FIG1_VALUES, FIG1_FREQ, 3, method=MSE)
    b0 = h.buckets[0]
    assert (b0.min_value, b0.max_value) == (1, 5)
    # 3 — внутри интервала бакета, но не в выборке: эвристика даёт > 0
    assert h.estimate_eq(3) == pytest.approx(17 / 3)
    assert h.estimate_eq(0) == 0 and h.estimate_eq(9) == 0
    assert h.estimate_range(100, 200) == 0 and h.estimate_range(-5, 0) == 0


def test_point_and_reversed_ranges():
    h = build(FIG1_VALUES, FIG1_FREQ, 3, method=MSE)
    assert h.estimate_range(5, 5) == h.estimate_eq(5) > 0
    assert h.estimate_range(6, 1) == 0


def test_range_interpolation_paper_linear():
    # Один бакет [1, 8], T = 26: [1, 4] -> (4 - 1) / (8 - 1)
    h = build(FIG1_VALUES, FIG1_FREQ, 1)
    assert h.estimate_range(1, 4) == pytest.approx(26 * 3 / 7)
    assert h.estimate_range(4, 100) == pytest.approx(26 * 4 / 7)
    assert h.estimate_range(1, 8) == 26  # касание обеих границ = полное покрытие
    assert h.estimate_range(0, 1) == 0  # нулевая ширина пересечения


def test_range_interpolation_discrete_extension():
    h = build(FIG1_VALUES, FIG1_FREQ, 1, range_estimator="discrete_linear")
    assert h.estimate_range(1, 4) == pytest.approx(26 * 4 / 8)
    assert h.estimate_range(0, 1) == pytest.approx(26 * 1 / 8)


def test_range_uses_tuple_counts_and_row_scale():
    sample = np.repeat(FIG1_VALUES, FIG1_FREQ)
    h = EntropyHistogram1D.build_from_sample(sample, 2600, HistogramConfig(n_buckets=3))
    assert h.row_scale == 100
    assert h.estimate_range(0, 100) == pytest.approx(2600)
    assert h.estimate_range(-1e9, 1e9) <= 2600


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        build([1, float("nan")], [1, 1], 2)
    with pytest.raises(ValueError):
        build([1, float("inf")], [1, 1], 2)
    with pytest.raises(ValueError):
        build([1, 2], [1, 0], 2)
    with pytest.raises(ValueError):
        build([1, 2], [1, -3], 2)
    with pytest.raises(ValueError):
        HistogramConfig(n_buckets=0)
    with pytest.raises(ValueError):
        HistogramConfig(method=MSE, me_input="frequency")


# ----------------------------------------------------------------------
# 7. Оценки равенства
# ----------------------------------------------------------------------


def test_entropy_and_bias_factor_estimators_identical():
    rng = np.random.default_rng(3)
    h = build(rng.choice(5000, 400, replace=False), rng.integers(1, 50, 400), 20)
    for b in h.buckets:
        v = b.min_value
        assert h.estimate_eq(v, "paper_entropy") == pytest.approx(
            h.estimate_eq(v, "paper_bias_factor"), rel=1e-12
        )


def test_weighted_frequency_mean():
    h = build([1, 2], [8, 2], 1)
    assert h.estimate_eq(1, "weighted_frequency_mean") == pytest.approx(6.8)
    assert h.estimate_eq(1, "paper_uniform") == pytest.approx(5.0)
    assert h.estimate_eq(1, "paper_entropy") == pytest.approx(10 * 2 ** -entropy([8, 2]))

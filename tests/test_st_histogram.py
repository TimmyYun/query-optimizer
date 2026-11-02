"""
Тесты ST-гистограммы по разделу 9 спецификации (ST_Histogram_1D_Claude_Code_Spec.md).
Все ожидаемые значения посчитаны вручную из формул Fig. 2 / Fig. 3.

    poetry run python -m pytest tests/ -q
"""
import json
import math

import numpy as np
import pytest

from models.common import Bucket
from models.st_histogram import (
    ROBUST_EXTENDED,
    STHistogram,
    STHistogramConfig,
    allocate_split_slots,
    greedy_merge_runs,
)


def two_bucket_hist(alpha=0.5, **kw):
    """[1,5]:50, [6,10]:50 — пример из §3.1."""
    cfg = STHistogramConfig(bucket_count=2, alpha=alpha, **kw)
    return STHistogram(cfg, total_rows=100, min_value=1, max_value=10)


def hist_from(freqs, widths, T, **kw):
    buckets, lo = [], 0
    for f, w in zip(freqs, widths):
        buckets.append(Bucket(lo, lo + w - 1, count=f))
        lo += w
    cfg = STHistogramConfig(bucket_count=len(freqs), **kw)
    return STHistogram.from_buckets(cfg, T, buckets)


def as_tuples(hist):
    return [(b.lo, b.hi, b.count) for b in hist.buckets]


# ----------------------------------------------------------------------
# A. Инициализация
# ----------------------------------------------------------------------


def test_init_two_buckets():
    assert as_tuples(two_bucket_hist()) == [(1, 5, 50.0), (6, 10, 50.0)]


def test_init_covers_domain_exactly():
    h = STHistogram(STHistogramConfig(bucket_count=100), 100_000, 1, 1000)
    assert len(h.buckets) == 100
    covered = np.concatenate([np.arange(b.lo, b.hi + 1) for b in h.buckets])
    assert np.array_equal(covered, np.arange(1, 1001))
    assert all(b.count == 1000.0 for b in h.buckets)


def test_init_domain_smaller_than_budget():
    h = STHistogram(STHistogramConfig(bucket_count=10), 30, 0, 2)
    assert as_tuples(h) == [(0, 0, 10.0), (1, 1, 10.0), (2, 2, 10.0)]
    assert h.events["effective_bucket_count_reduced"] == 1


def test_init_zero_rows():
    h = STHistogram(STHistogramConfig(bucket_count=4), 0, 0, 99)
    assert h.total_mass == 0.0


def test_init_rejects_invalid():
    with pytest.raises(ValueError):
        STHistogram(STHistogramConfig(), 100, 10, 1)
    with pytest.raises(ValueError):
        STHistogram(STHistogramConfig(), -1, 0, 10)
    with pytest.raises(ValueError):
        STHistogramConfig(alpha=0.0)
    with pytest.raises(ValueError):
        STHistogramConfig(renormalize_mass=True)  # extension без robust_extended


# ----------------------------------------------------------------------
# B. Оценка
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "low, high, expected",
    [
        (3, 8, 60.0),
        (1, 10, 100.0),
        (6, 10, 50.0),
        (11, 20, 0.0),
        (5, 5, 10.0),
        (5, 6, 20.0),  # граница бакетов — без двойного счёта
        (-5, 2, 20.0),  # обрезка по домену
    ],
)
def test_estimate(low, high, expected):
    assert two_bucket_hist().estimate_range(low, high) == pytest.approx(expected)


def test_estimate_matches_direct_sum():
    """Префиксные суммы в estimate_range == прямая сумма w * f."""
    rng = np.random.default_rng(42)
    h = hist_from(rng.uniform(0, 100, 50), rng.integers(1, 30, 50), T=2500)
    for _ in range(500):
        lo, hi = sorted(rng.integers(-10, h.domain_max + 10, 2))
        direct = sum(
            max(0, min(hi, b.hi) - max(lo, b.lo) + 1) / (b.hi - b.lo + 1) * b.count
            for b in h.buckets
        )
        assert h.estimate_range(lo, hi) == pytest.approx(direct)


# ----------------------------------------------------------------------
# C. Обновление частот (Fig. 2)
# ----------------------------------------------------------------------


def test_update_full_bucket_no_normalization():
    h = two_bucket_hist()
    res = h.observe_feedback(1, 5, 30)
    assert res.estimate == 50.0 and res.error == -20.0
    assert as_tuples(h) == [(1, 5, 40.0), (6, 10, 50.0)]
    assert h.total_mass == 90.0  # НЕ перенормировано к 100


def test_update_partial_buckets():
    h = two_bucket_hist()
    res = h.observe_feedback(3, 8, 48)
    assert res.estimate == pytest.approx(60.0)
    assert [b.count for b in h.buckets] == pytest.approx([47.0, 47.0])


def test_update_shares_proportional_to_contribution():
    # [0,9]:100, [10,19]:20; запрос [5,19]: вклады 50 и 20, est = 70
    h = hist_from([100.0, 20.0, 7.0], [10, 10, 10], T=127)
    h.observe_feedback(5, 19, 140)  # err = +70
    assert [b.count for b in h.buckets] == pytest.approx(
        [100 + 0.5 * 70 * 50 / 70, 20 + 0.5 * 70 * 20 / 70, 7.0]
    )


def test_update_clamps_at_zero():
    h = two_bucket_hist(alpha=1.0)
    h.observe_feedback(1, 10, 0)
    assert h.total_mass == 0.0 and all(b.count >= 0 for b in h.buckets)


def test_zero_estimate_paper_mode():
    h = hist_from([0.0, 50.0], [5, 5], T=50)
    before = h.get_state()["freq"]

    assert h.observe_feedback(0, 4, 0).event == "zero_estimate_correct"
    res = h.observe_feedback(0, 4, 30)
    assert res.event == "unlearnable_zero_estimate"
    assert h.get_state()["freq"] == before
    assert h.processed_queries == 2  # no-op фидбэк всё равно считается

    assert h.observe_feedback(100, 200, 5).event == "out_of_domain"


def test_zero_bucket_stays_zero_with_positive_neighbour():
    # Мультипликативное обновление не оживляет f = 0 (слабость Fig. 2, не баг)
    h = hist_from([0.0, 50.0], [5, 5], T=50)
    h.observe_feedback(0, 9, 500)
    assert h.buckets[0].count == 0.0
    assert h.buckets[1].count > 50.0


def test_zero_estimate_robust_revival():
    h = hist_from([0.0, 0.0, 50.0], [5, 5, 5], T=50, mode=ROBUST_EXTENDED)
    res = h.observe_feedback(2, 9, 40)  # 3 значения в b0, 5 в b1
    assert res.event == "zero_estimate_revived"
    assert [b.count for b in h.buckets] == pytest.approx([20 * 3 / 8, 20 * 5 / 8, 50.0])


def test_robust_renormalize_mass():
    cfg = STHistogramConfig(bucket_count=2, mode=ROBUST_EXTENDED, renormalize_mass=True)
    h = STHistogram(cfg, 100, 1, 10)
    h.observe_feedback(1, 5, 30)
    assert h.total_mass == pytest.approx(100.0)


@pytest.mark.parametrize(
    "low, high, actual", [(5, 3, 1), (1, 2, -1), (1, 2, math.nan), (1, 2, math.inf)]
)
def test_update_rejects_invalid_feedback(low, high, actual):
    h = two_bucket_hist()
    with pytest.raises(ValueError):
        h.observe_feedback(low, high, actual)
    assert h.processed_queries == 0


# ----------------------------------------------------------------------
# D. Слияние (Fig. 3)
# ----------------------------------------------------------------------


def naive_merge_runs(freq, threshold):
    """Прямая реализация Fig. 3 за O(B^2) — эталон для кучи."""
    runs = [[i] for i in range(len(freq))]
    while len(runs) > 1:
        d = [
            max(max(freq[j] for j in a) - min(freq[j] for j in b),
                max(freq[j] for j in b) - min(freq[j] for j in a))
            for a, b in zip(runs, runs[1:])
        ]
        pos = int(np.argmin(d))  # argmin -> самая левая при равенстве
        if d[pos] > threshold:
            break
        runs[pos : pos + 2] = [runs[pos] + runs[pos + 1]]
    return [(r[0], r[-1]) for r in runs]


def test_merge_compares_constituents_not_sums():
    # [1,2] сливаются (d=1); затем d([1,2],[4]) = max(4-1, 2-4) = 3 > 2 — стоп.
    # Сравнение суммы 1+2=3 с 4 ошибочно слило бы всё.
    assert greedy_merge_runs([1, 2, 4], 2) == [(0, 1), (2, 2)]


def test_merge_threshold_inclusive():
    assert greedy_merge_runs([10, 12], 2) == [(0, 1)]
    assert greedy_merge_runs([10, 12.5], 2) == [(0, 0), (1, 1)]


def test_merge_ties_leftmost():
    # d = 1 у обеих пар; слева сначала -> [0,1], затем d([5,6],[7]) = 2 > 1
    assert greedy_merge_runs([5, 6, 7], 1) == [(0, 1), (2, 2)]


def test_merge_heap_equals_naive():
    rng = np.random.default_rng(42)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        freq = rng.integers(0, 20, n).astype(float)  # много равенств
        thr = float(rng.integers(0, 8))
        assert greedy_merge_runs(freq, thr) == naive_merge_runs(freq, thr)


def test_merge_threshold_is_fraction_of_T():
    cfg = STHistogramConfig(bucket_count=3, merge_threshold=0.00025, split_fraction=0.0)
    assert cfg.merge_threshold * 100_000 == 25.0


# ----------------------------------------------------------------------
# E. Разбиение / реструктуризация
# ----------------------------------------------------------------------


def test_allocation_proportional():
    assert allocate_split_slots(3, [40, 20], [10, 10], [0, 1]) == [2, 1]


def test_allocation_respects_capacity():
    # Кандидат 0 вмещает только 1 слот -> остальное уходит кандидату 1
    assert allocate_split_slots(4, [90, 10], [1, 10], [0, 1]) == [1, 3]


def test_allocation_failures():
    assert allocate_split_slots(3, [], [], []) is None
    assert allocate_split_slots(3, [0, 0], [5, 5], [0, 1]) is None
    assert allocate_split_slots(3, [0, 0], [5, 5], [0, 1], uniform_if_zero=True) == [2, 1]
    assert allocate_split_slots(5, [1, 1], [1, 1], [0, 1]) is None


def test_restructure_merge_and_split():
    # m * T = 2: [1,2] сливаются; F = 1 слот достаётся единственному кандидату [4]
    h = hist_from([1.0, 2.0, 4.0], [4, 4, 4], T=8000, split_fraction=0.10)
    res = h.restructure()
    assert res.applied and res.freed == 1 and res.n_split == 1
    assert as_tuples(h) == [(0, 7, 3.0), (8, 9, 2.0), (10, 11, 2.0)]


def test_restructure_split_three_children():
    # [1..6]:60 на три -> [1,2]:20, [3,4]:20, [5,6]:20; два одинаковых соседа сливаются
    buckets = [Bucket(1, 6, 60.0), Bucket(7, 7, 1.0), Bucket(8, 8, 1.0), Bucket(9, 9, 1.0)]
    cfg = STHistogramConfig(bucket_count=4, merge_threshold=0.0, split_fraction=0.25)
    h = STHistogram.from_buckets(cfg, 63, buckets)
    h.restructure()
    assert as_tuples(h) == [(1, 2, 20.0), (3, 4, 20.0), (5, 6, 20.0), (7, 9, 3.0)]


def test_restructure_preserves_invariants_randomized():
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(5, 60))
        h = hist_from(
            rng.integers(0, 50, n).astype(float),
            rng.integers(1, 20, n),
            T=1000,
            merge_threshold=0.005,
        )
        mass = h.total_mass
        h.restructure()
        assert len(h.buckets) == n
        assert h.total_mass == pytest.approx(mass)
        assert h.buckets[0].lo == h.domain_min and h.buckets[-1].hi == h.domain_max


def test_merged_and_single_value_buckets_not_split():
    # [0,0]:100 одиночный (нельзя делить); [1..10]:5 и [11..20]:5 сливаются
    buckets = [Bucket(0, 0, 100.0), Bucket(1, 10, 5.0), Bucket(11, 20, 5.0)]
    cfg = STHistogramConfig(bucket_count=3, merge_threshold=0.0, split_fraction=1.0)
    h = STHistogram.from_buckets(cfg, 110, buckets)
    res = h.restructure()
    assert not res.applied and res.reason == "split_allocation_failed"
    assert as_tuples(h) == [(0, 0, 100.0), (1, 10, 5.0), (11, 20, 5.0)]  # атомарно


def test_restructure_nothing_to_merge():
    h = hist_from([1.0, 100.0], [5, 5], T=101)
    assert h.restructure().reason == "nothing_to_merge"


def test_restructure_only_every_R_queries():
    cfg = STHistogramConfig(bucket_count=10, restructure_every=5)
    h = STHistogram(cfg, 1000, 0, 99)
    fired = []
    for i in range(1, 16):
        res = h.observe_feedback(0, 99, 5000)  # огромная ошибка — триггер всё равно по счётчику
        if res.restructure is not None:
            fired.append(i)
    assert fired == [5, 10, 15]

    h = STHistogram(STHistogramConfig(bucket_count=10, restructure_every=5,
                                      enable_restructuring=False), 1000, 0, 99)
    assert all(h.observe_feedback(0, 99, 5000).restructure is None for _ in range(10))


# ----------------------------------------------------------------------
# F. Дрейф: модель узнаёт об изменениях только через фидбэк
# ----------------------------------------------------------------------


def test_no_feedback_no_change():
    h = STHistogram(STHistogramConfig(), 100_000, 1, 1000)
    state = json.dumps(h.get_state())
    for lo in range(1, 1000, 37):
        h.estimate_range(lo, lo + 50)
    assert json.dumps(h.get_state()) == state


def test_learns_shifted_distribution_from_feedback():
    # Вся масса в [0,99]; гистограмма стартует равномерной на [0,999]
    freq = np.zeros(1000)
    freq[:100] = 100
    ps = np.concatenate([[0], np.cumsum(freq)])
    rng = np.random.default_rng(42)
    h = STHistogram(STHistogramConfig(bucket_count=20, restructure_every=50), 10_000, 0, 999)
    assert h.estimate_range(0, 99) == pytest.approx(1000)
    for _ in range(2000):
        lo, hi = sorted(rng.integers(0, 1000, 2))
        h.observe_feedback(lo, hi, ps[hi + 1] - ps[lo])
    assert h.estimate_range(0, 99) == pytest.approx(10_000, rel=0.05)
    assert h.estimate_range(500, 999) < 100


# ----------------------------------------------------------------------
# G. Воспроизводимость / устойчивость
# ----------------------------------------------------------------------


def random_stream(seed, n, dmax=999):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        lo, hi = sorted(rng.integers(0, dmax + 1, 2))
        yield int(lo), int(hi), int(rng.integers(0, 5000))


def test_serialization_resume_is_exact():
    cfg = STHistogramConfig(bucket_count=30, restructure_every=40)
    stream = list(random_stream(42, 1000))

    a = STHistogram(cfg, 100_000, 0, 999)
    for q in stream:
        a.observe_feedback(*q)

    b = STHistogram(cfg, 100_000, 0, 999)
    for q in stream[:437]:
        b.observe_feedback(*q)
    b = STHistogram.from_state(json.loads(json.dumps(b.get_state())))
    for q in stream[437:]:
        b.observe_feedback(*q)

    assert a.get_state() == b.get_state()


@pytest.mark.parametrize("mode", ["paper_faithful", ROBUST_EXTENDED])
def test_stays_finite_after_many_observations(mode):
    h = STHistogram(STHistogramConfig(bucket_count=50, mode=mode), 100_000, 0, 999)
    for q in random_stream(7, 10_000):
        h.observe_feedback(*q)
    assert np.all(np.isfinite(h.freq)) and np.all(h.freq >= 0)
    assert len(h.buckets) <= 50

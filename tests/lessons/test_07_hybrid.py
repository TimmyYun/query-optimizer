"""Урок 7: estimators/hybrid.py — equi-width + локальная CDF-модель в бакете."""
import numpy as np
import pytest

from conftest import Q, maybe, need, random_queries


def ramp_sample(n=100_000, width=1000, seed=7):
    # плотность растёт линейно: P(x <= t) = (t/width)^2
    u = np.random.default_rng(seed).random(n)
    return np.minimum((np.sqrt(u) * width).astype(np.int64), width - 1)


def test_uniform_data_trains_no_models():
    hy = need("hybrid")
    rng = np.random.default_rng(8)
    s = rng.integers(0, 100_000, 100_000)
    h = hy.build(s, 0, 99_999, 1_000_000, 20)
    assert sum(m is not None for m in h.models.values()) == 0
    ew = maybe("equi_width")
    if ew is not None:
        e = ew.build(s, 0, 99_999, 1_000_000, 20)
        for q in random_queries(rng, 0, 99_999, 200, 3000):
            assert h.predict(q) == pytest.approx(e.predict(q), rel=1e-4, abs=1e-3)


def test_learns_shape_inside_bucket():
    """
    Эталон из feature/new_baseline этот тест НЕ проходит: на широких валидационных
    запросах равномерная модель даёт медиану q-error 1.46 < 1.5, и каскад
    останавливается сразу, хотя узкие запросы у левого края ошибаются в 10 раз.
    См. lessons/07_hybrid.md, ловушка 3.
    """
    hy = need("hybrid")
    h = hy.build(ramp_sample(), 0, 999, 1_000_000, 1)  # один бакет на весь домен
    # правда ~1% = 10 000 строк; равномерное предположение дало бы 100 000
    assert 10_000 / 1.5 <= h.predict(Q(0, 99)) <= 10_000 * 1.5
    assert h.predict(Q(0, 499)) + h.predict(Q(500, 999)) == pytest.approx(1_000_000, rel=0.03)


def test_inner_buckets_are_plain_counts():
    hy = need("hybrid")
    s = ramp_sample()
    h = hy.build(s, 0, 999, 1_000_000, 10)  # бакеты по 100 значений
    exact = ((s >= 100) & (s <= 899)).sum() * 10  # бакеты 1..8 целиком
    assert h.predict(Q(100, 899)) == pytest.approx(exact, rel=1e-6)


def test_deterministic():
    hy = need("hybrid")
    s = ramp_sample()
    a = hy.build(s, 0, 999, 1_000_000, 10)
    b = hy.build(s, 0, 999, 1_000_000, 10)
    for q in random_queries(np.random.default_rng(9), 0, 999, 200, 150):
        assert a.predict(q) == b.predict(q)


def test_memory_is_reported():
    hy = need("hybrid")
    h = hy.build(ramp_sample(), 0, 999, 1_000_000, 10)
    assert h.n_floats() > 10  # счётчики + параметры моделей

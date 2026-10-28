"""Урок 2: estimators/tkhist.py — equi-width + top-k частых значений в бакете."""
import numpy as np
import pytest

from conftest import Q, maybe, need, random_queries


def heavy_sample():
    # 1000 разных значений по одному + ещё 50 раз значение 3; N = 10 * 1050
    return np.concatenate([np.arange(1000), np.full(50, 3)])


def test_frequent_value_is_exact():
    tk = need("tkhist")
    h = tk.build(heavy_sample(), 0, 999, 10_500, 4, top_k=1)
    assert h.predict(Q(3, 3)) == pytest.approx(510, rel=0.02)  # 51 точка * 10
    # соседнее значение — из равномерного остатка бакета, а не 1/250 всего бакета с пиком
    assert h.predict(Q(4, 4)) == pytest.approx(10, rel=0.05)


def test_full_bucket_count_unchanged():
    tk = need("tkhist")
    h = tk.build(heavy_sample(), 0, 999, 10_500, 4, top_k=1)
    assert h.predict(Q(0, 249)) == pytest.approx(3000)


def test_point_queries_add_up_to_total():
    tk = need("tkhist")
    h = tk.build(heavy_sample(), 0, 999, 10_500, 4, top_k=3)
    assert sum(h.predict(Q(v, v)) for v in range(1000)) == pytest.approx(10_500, rel=1e-6)


def test_topk_zero_is_equi_width():
    tk = need("tkhist")
    ew = maybe("equi_width")
    if ew is None:
        pytest.skip("нужен estimators/equi_width.py из урока 1")
    rng = np.random.default_rng(2)
    s = rng.zipf(1.5, 20_000) % 5000
    a = tk.build(s, 0, 4999, 100_000, 20, top_k=0)
    b = ew.build(s, 0, 4999, 100_000, 20)
    for q in random_queries(rng, 0, 4999, 300, 600):
        assert a.predict(q) == pytest.approx(b.predict(q), rel=1e-6, abs=1e-6)

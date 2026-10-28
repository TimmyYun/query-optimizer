"""Урок 1: estimators/equi_width.py."""
import numpy as np
import pytest

from conftest import Q, need


def toy():
    # 10 точек в 0 и 10 в 9, N = 200 -> каждая точка выборки = 10 строк
    ew = need("equi_width")
    return ew.build(np.array([0] * 10 + [9] * 10), 0, 9, 200, 2)


def test_full_buckets_are_exact():
    h = toy()
    assert h.predict(Q(0, 4)) == pytest.approx(100)
    assert h.predict(Q(5, 9)) == pytest.approx(100)
    assert h.predict(Q(0, 9)) == pytest.approx(200)


def test_partial_bucket_is_uniform():
    h = toy()
    assert h.predict(Q(0, 1)) == pytest.approx(40)  # 2/5 ширины * 100
    assert h.predict(Q(3, 6)) == pytest.approx(80)  # 2/5*100 + 2/5*100


def test_outside_domain():
    h = toy()
    assert h.predict(Q(20, 30)) == pytest.approx(0)
    assert h.predict(Q(-5, 100)) == pytest.approx(200)


def test_last_bucket_can_be_narrower():
    # домен из 10 значений, 3 бакета -> ширина ceil(10/3)=4: [0,3] [4,7] [8,9]
    ew = need("equi_width")
    h = ew.build(np.arange(10), 0, 9, 10, 3)
    assert [tuple(map(int, r)) for r in h.bucket_ranges] == [(0, 3), (4, 7), (8, 9)]
    # в последнем бакете 2 строки на 2 значения -> 1 строка на значение (а не 2/4)
    assert h.predict(Q(8, 8)) == pytest.approx(1.0)


def test_point_queries_add_up_to_total():
    ew = need("equi_width")
    rng = np.random.default_rng(1)
    s = rng.integers(0, 1000, 5000)
    h = ew.build(s, 0, 999, 50_000, 7)
    assert sum(h.predict(Q(v, v)) for v in range(1000)) == pytest.approx(50_000, rel=1e-3)

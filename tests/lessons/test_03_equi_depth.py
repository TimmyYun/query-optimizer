"""Урок 3: estimators/equi_depth.py — гистограмма PostgreSQL (MCV + equi-depth bounds)."""
import numpy as np
import pytest

from conftest import Q, need


def test_n_distinct_haas_stokes():
    ed = need("equi_depth")
    # все значения уникальны -> считаем колонку уникальной
    assert ed.estimate_n_distinct(500, 0, 1000, 1_000_000) == 1_000_000
    # все значения встретились >= 2 раз -> видели всё
    assert ed.estimate_n_distinct(500, 500, 1000, 1_000_000) == 500
    # f1 = 400: 1000*500 / (600 + 400*1000/1e6) = 832.78 -> 833
    assert ed.estimate_n_distinct(500, 100, 1000, 1_000_000) == 833


def test_bounds_split_sample_into_equal_parts():
    ed = need("equi_depth")
    rng = np.random.default_rng(3)
    s = rng.choice(1_000_000, 30_000, replace=False)  # без повторов -> MCV пуст
    h = ed.build(s, 0, 999_999, 1_000_000, 100, statistics_target=100)
    b = np.asarray(h.hist_bounds, dtype=float)
    assert len(b) == 101 and np.all(np.diff(b) > 0)
    per_bucket = np.histogram(s, bins=b)[0]
    assert per_bucket.min() >= 297 and per_bucket.max() <= 303


def test_mcv_catches_heavy_value():
    ed = need("equi_depth")
    rng = np.random.default_rng(4)
    s = np.concatenate([np.full(9000, 777), rng.choice(1_000_000, 21_000, replace=False)])
    h = ed.build(s, 0, 999_999, 1_000_000, 100)
    assert 777 in [int(v) for v in h.mcv_vals]
    assert h.predict(Q(777, 777)) == pytest.approx(300_000, rel=0.01)


def test_range_on_uniform():
    ed = need("equi_depth")
    rng = np.random.default_rng(5)
    s = rng.integers(0, 1_000_000, 30_000)
    h = ed.build(s, 0, 999_999, 1_000_000, 100)
    assert h.predict(Q(200_000, 299_999)) == pytest.approx(100_000, rel=0.05)

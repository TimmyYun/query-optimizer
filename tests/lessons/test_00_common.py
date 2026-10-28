"""Урок 0: estimators/common.py — q-error, точная истина, сводка."""
import numpy as np

from conftest import need


def test_q_error_symmetric():
    c = need("common")
    assert np.allclose(c.q_error([100.0], [50.0]), [2.0])
    assert np.allclose(c.q_error([50.0], [100.0]), [2.0])
    assert np.allclose(c.q_error([77.0], [77.0]), [1.0])


def test_q_error_floors_at_one_row():
    # 0 строк и 0.3 строки — одно и то же "почти ничего": обе стороны прижимаются к 1
    c = need("common")
    assert np.allclose(c.q_error([0.0], [0.3]), [1.0])
    assert np.allclose(c.q_error([10.0], [0.0]), [10.0])


def test_true_counts_matches_bruteforce():
    c = need("common")
    rng = np.random.default_rng(0)
    mn, freq = 10, rng.integers(0, 20, 50)  # домен [10, 59]
    mx = mn + len(freq) - 1
    lows = rng.integers(-5, 80, 300)
    highs = lows + rng.integers(0, 30, 300)
    got = c.true_counts(freq, mn, lows, highs)
    for lo, hi, g in zip(lows, highs, got):
        exp = sum(freq[v - mn] for v in range(max(lo, mn), min(hi, mx) + 1))
        assert g == exp, (lo, hi)


def test_summarize_has_median_and_tail():
    c = need("common")
    s = c.summarize(np.array([1.0, 1.0, 2.0, 4.0, 100.0]))
    assert s["median"] == 2.0
    assert s["max"] == 100.0
    assert {"median", "p95", "avg", "max"} <= set(s)

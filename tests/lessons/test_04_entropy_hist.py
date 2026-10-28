"""Урок 4: estimators/entropy_hist.py — границы бакетов выбираются по энтропии."""
import numpy as np
import pytest

from conftest import Q, need


def test_entropy_bits():
    eh = need("entropy_hist")
    assert eh.entropy([1, 1]) == pytest.approx(1.0)
    assert eh.entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert eh.entropy([7]) == pytest.approx(0.0)
    assert eh.entropy([1, 0, 1]) == pytest.approx(1.0)  # нули не считаются


def plateaus():
    # значения 0..9 по 1 разу, 10..19 по 100 раз: два "плато"
    return np.concatenate([np.arange(10), np.repeat(np.arange(10, 20), 100)])


def test_cut_between_plateaus():
    eh = need("entropy_hist")
    s = plateaus()
    h = eh.build(s, 0, 19, len(s), 2)
    assert [tuple(map(int, r)) for r in h.bucket_ranges] == [(0, 9), (10, 19)]


def test_whole_domain_and_bucket_budget():
    eh = need("entropy_hist")
    s = plateaus()
    h = eh.build(s, 0, 19, 10 * len(s), 5)
    assert len(h.bucket_ranges) == 5
    assert h.predict(Q(0, 19)) == pytest.approx(10 * len(s))
    # больше бакетов, чем различных значений, не бывает
    assert len(eh.build(s, 0, 19, len(s), 50).bucket_ranges) == 20

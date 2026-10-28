"""Урок 5: estimators/equi_hist.py — equi-width, который учится по ответам запросов."""
import numpy as np
import pytest

from conftest import Q, need


def toy():
    eq = need("equi_hist")
    return eq.build(np.array([0] * 10 + [9] * 10), 0, 9, 200, 2, learning_rate=0.5)


def test_full_overlap_update():
    h = toy()
    h.observe(Q(0, 4), 400.0)  # ratio = 4, весь бакет -> count * 4**0.5
    assert h.predict(Q(0, 4)) == pytest.approx(200)
    assert h.predict(Q(5, 9)) == pytest.approx(100)  # неперекрытый бакет не трогаем


def test_partial_overlap_and_ratio_clip():
    h = toy()
    h.observe(Q(5, 6), 4000.0)  # pred = 40, ratio = 100 -> обрезается до 10; доля 2/5
    assert h.predict(Q(5, 9)) == pytest.approx(100 * 10 ** (0.5 * 0.4), rel=1e-6)
    assert h.predict(Q(0, 4)) == pytest.approx(100)

"""Урок 6: estimators/st_hist.py — ST-гистограмма (Aboulnaga & Chaudhuri 1999)."""
import numpy as np
import pytest

from conftest import Q, need


def two_buckets(**kw):
    st = need("st_hist")
    return st.from_buckets([0, 5], [4, 9], [50.0, 50.0], 100, alpha=0.5, **kw)


def test_does_not_look_at_data():
    st = need("st_hist")
    rng = np.random.default_rng(6)
    a = st.build(rng.integers(0, 100, 1000), 0, 99, 1000, 10)
    b = st.build(np.full(1000, 7), 0, 99, 1000, 10)
    for lo in range(0, 100, 7):
        assert a.predict(Q(lo, lo + 5)) == pytest.approx(b.predict(Q(lo, lo + 5)))
    assert a.predict(Q(0, 9)) == pytest.approx(100)  # T / B


def test_fig2_full_bucket_update():
    h = two_buckets()
    h.observe(Q(0, 4), 80.0)  # est 50, err 30 -> 50 + 0.5*30
    assert h.bucket_freqs == pytest.approx([65.0, 50.0])


def test_fig2_error_split_by_contribution():
    h = two_buckets()
    h.observe(Q(3, 6), 10.0)  # вклады 20 и 20, est 40, err -30 -> каждому -7.5
    assert h.bucket_freqs == pytest.approx([42.5, 42.5])


def test_restructure_merges_similar_and_splits_heavy():
    st = need("st_hist")
    # m*T = 1: три бакета по 10 сливаются в один, 2 слота уходят на тяжёлый
    h = st.from_buckets([0, 4, 8, 12], [3, 7, 11, 15], [10.0, 10.0, 10.0, 1000.0], 1030,
                        merge_threshold=1 / 1030, split_fraction=0.25)
    h.restructure()
    r = [tuple(map(int, x)) for x in h.bucket_ranges]
    f = h.bucket_freqs
    assert len(r) == 4  # бюджет бакетов не меняется
    assert r[0] == (0, 11) and f[0] == pytest.approx(30)
    assert r[1][0] == 12 and r[-1][1] == 15
    assert f[1:] == pytest.approx([1000 / 3] * 3)
    assert h.total_mass() == pytest.approx(1030)

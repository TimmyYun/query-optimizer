"""Урок 8: refresh() — ANALYZE при дрейфе для гистограмм, которые строятся по выборке."""
import numpy as np
import pytest

from conftest import Q, maybe, need


def samples():
    rng = np.random.default_rng(10)
    return rng.integers(0, 10_000, 20_000), rng.integers(0, 1000, 20_000)


@pytest.mark.parametrize("name", ["equi_width", "tkhist", "equi_depth", "entropy_hist"])
def test_refresh_follows_new_sample(name):
    mod = maybe(name)
    if mod is None:
        pytest.skip(f"{name}.py ещё не написан")
    old, new = samples()
    h = mod.build(old, 0, 9999, 1_000_000, 50)
    assert h.predict(Q(5000, 9999)) == pytest.approx(500_000, rel=0.05)
    h.refresh(new, 0, 9999, 2_000_000)  # данных стало вдвое больше и все в [0, 999]
    # если predict читает закэшированные массивы — тут останется старый ответ
    assert h.predict(Q(5000, 9999)) < 0.01 * 2_000_000
    assert h.predict(Q(0, 999)) == pytest.approx(2_000_000, rel=0.03)


def test_equi_width_keeps_grid_on_refresh():
    ew = need("equi_width")
    old, new = samples()
    h = ew.build(old, 0, 9999, 1_000_000, 50)
    before = [tuple(r) for r in h.bucket_ranges]
    h.refresh(new, 0, 9999, 1_000_000)
    assert [tuple(r) for r in h.bucket_ranges] == before

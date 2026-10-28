"""Урок 9: Hybrid под дрейфом — refresh (ANALYZE) + adapt (FT по фидбэку, RB плохих бакетов)."""
import numpy as np
import pytest

from conftest import Q, need


def ramp_freq(width=10_000, total=1_000_000):
    # плотность ~ x: "новые" данные после дрейфа
    w = np.arange(1, width + 1, dtype=float)
    f = np.floor(w / w.sum() * total)
    return f


def sample_from(freq, n, seed):
    p = freq / freq.sum()
    return np.random.default_rng(seed).choice(len(freq), n, p=p)


def setup():
    hy = need("hybrid")
    from bench import true_counts

    rng = np.random.default_rng(11)
    h = hy.build(rng.integers(0, 10_000, 100_000), 0, 9999, 1_000_000, 10)
    freq = ramp_freq()
    new_sample = sample_from(freq, 100_000, 12)
    lo = rng.integers(0, 9900, 4000)
    qs = [Q(int(a), int(a + rng.integers(0, 100))) for a in lo]
    truth = true_counts(freq, 0, np.array([q.low for q in qs]), np.array([q.high for q in qs]))
    return h, freq, new_sample, qs, truth


def qerr(h, qs, truth):
    p = np.maximum([h.predict(q) for q in qs], 1.0)
    t = np.maximum(truth, 1.0)
    return float(np.median(np.maximum(p / t, t / p)))


def test_refresh_then_adapt_improves_holdout():
    h, freq, s, qs, truth = setup()
    h.refresh(s, 0, 9999, int(freq.sum()))
    train, test = qs[::2], qs[1::2]
    before = qerr(h, test, truth[1::2])
    info = h.adapt(train, truth[::2].copy(), s, 0, 9999, int(freq.sum()))
    assert {"finetuned", "rebuilt"} <= set(info)
    assert qerr(h, test, truth[1::2]) <= before * 1.01  # оцениваем на запросах, которых не было в фидбэке


def test_refresh_recounts_buckets():
    # ANALYZE: масса каждого бакета = точки новой выборки * N/n, и predict это видит
    h, freq, s, qs, truth = setup()
    N = int(freq.sum())
    h.refresh(s, 0, 9999, N)
    for lo, hi in h.bucket_ranges:
        exp = ((s >= lo) & (s <= hi)).sum() * N / len(s)
        assert h.predict(Q(int(lo), int(hi))) == pytest.approx(exp, rel=1e-6)

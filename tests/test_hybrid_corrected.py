"""Hybrid-Corrected: inclusive-interval fix on the discrete domain; everything else as canonical."""
import numpy as np
import pytest

from models import EquiWidthHistogram, HybridEstimator
from models.common import Bucket, RangeQuery
from models.hybrid_corrected import HybridCorrectedEstimator, HybridCorrectedPredictOnlyEstimator

N = 60_000_000


def _uniform(cls, buckets):
    return cls([Bucket(lo, hi, count=float(c)) for lo, hi, c in buckets])


@pytest.mark.parametrize("q, true", [((0, 9), 10), ((90, 99), 10), ((10, 19), 10), ((0, 99), 100),
                                     ((0, 0), 1), ((99, 99), 1), ((57, 57), 1), ((-5, 3), 4), ((95, 300), 5)])
def test_uniform_bucket_is_exact_on_inclusive_ranges(q, true):
    h = _uniform(HybridCorrectedEstimator, [(0, 99, 100)])  # one row per value
    assert h.predict(RangeQuery(*q)) == pytest.approx(true)


def test_canonical_bug_is_pinned():
    """The canonical class keeps its behaviour (reproducibility): edges are off by one value."""
    h = _uniform(HybridEstimator, [(0, 99, 100)])
    assert h.predict(RangeQuery(0, 9)) == pytest.approx(9)
    assert h.predict(RangeQuery(90, 99)) == pytest.approx(11)
    assert h.predict(RangeQuery(0, 0)) == pytest.approx(0.05)


def test_multi_bucket_ranges_match_brute_force():
    """Uniform-within-bucket estimate of any inclusive range, edges and points included."""
    spec = [(0, 9, 30), (10, 19, 0), (20, 49, 300), (50, 52, 7)]
    h = _uniform(HybridCorrectedEstimator, spec)
    per_value = {v: c / (hi - lo + 1) for lo, hi, c in spec for v in range(lo, hi + 1)}
    rng = np.random.default_rng([42, 11])
    for _ in range(500):
        a = int(rng.integers(-3, 56))
        b = int(rng.integers(a, 56))
        assert h.predict(RangeQuery(a, b)) == pytest.approx(sum(per_value.get(v, 0.0) for v in range(a, b + 1)))


def test_validation_scores_exact_cdfs_as_perfect():
    """Validation = predict's in-bucket estimate: an exact CDF and Uniform on uniform data score 1.0."""
    w = 100
    b = Bucket(0, w - 1, count=1000.0 * w)
    b_ps = np.cumsum(np.full(w, 1000.0))
    rng = np.random.default_rng([42, 12])
    lo = rng.integers(0, w, 500)
    hi = np.array([rng.integers(a, w) for a in lo])
    exact = ("linear", 1.0, 1.0 / w)  # F(x_norm) = x_norm + 1/w = (v - lo + 1) / w
    for cls, perfect in ((HybridCorrectedEstimator, True), (HybridEstimator, False)):
        h = cls([b])
        for mdl in (None, exact):
            med, p95 = h._eval_model_q_error_vec(mdl, lo, hi, b, b_ps, 0, w)
            assert (med == pytest.approx(1.0) and p95 == pytest.approx(1.0)) == perfect


def _train(cls, values, bins, mn, mx):
    est = np.bincount(values - mn, minlength=mx - mn + 1) * (N / len(values))
    h = cls(EquiWidthHistogram.build_from_sample(mn, mx, bins, values, N).buckets)
    h.train(est, mn, 200, np.random.default_rng(42))
    return h


def test_predict_only_variant_keeps_canonical_selection():
    rng = np.random.default_rng([42, 13])
    values = np.concatenate([rng.integers(0, 5000, 40_000), 5000 + np.minimum(rng.zipf(1.4, 60_000), 4999)])
    a = _train(HybridEstimator, values, 12, 0, 9999)
    b = _train(HybridCorrectedPredictOnlyEstimator, values, 12, 0, 9999)
    names = lambda h: [HybridEstimator.model_name(h.models.get(i)) for i in range(len(h.buckets))]  # noqa: E731
    assert names(a) == names(b)
    # buckets with a trained model predict exactly as the canonical code
    for i, bk in enumerate(a.buckets):
        if a.models.get(i) is not None:
            q = RangeQuery(bk.lo + 3, bk.hi - 2)
            assert b.predict(q) == pytest.approx(a.predict(q))

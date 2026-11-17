"""
Documents a known off-by-one in the canonical Hybrid FD CDF (BUG-IN-CANONICAL-CODE, not fixed in this round;
see results/paper_experiments/methodology/IMPLEMENTATION_AUDIT.md §C).

Training labels are "fraction of rows <= v" at x_norm = (v - lo) / w, so the exact uniform CDF is
F(x_norm) = x_norm + 1/w. Two places use a different convention:
  1. Tier 0 (model None) predicts F(x_norm) = x_norm in _get_bucket_overlap_count: a range starting at
     the bucket start loses one value, a range ending at the bucket end gains one; interior ranges cancel.
  2. _eval_model_q_error_vec (the waterfall's 500-query validation) scores every candidate with
     F(hi_norm) - F(lo_norm), i.e. on (lo, hi], against truth on [lo, hi].
These tests pin the CURRENT behaviour with explicit values; if the code is fixed (as a labelled
modification), they must be updated together with the audit.
"""
import numpy as np
import pytest

from models import EquiWidthHistogram, HybridEstimator
from models.common import Bucket, RangeQuery


def _one_uniform_bucket():
    return HybridEstimator([Bucket(0, 99, count=100.0)])  # 1 row per value, Tier 0 (no model)


@pytest.mark.parametrize("q, current, exact", [
    ((0, 9), 9.0, 10.0),      # starts at bucket start: one value short
    ((90, 99), 11.0, 10.0),   # ends at bucket end: one value extra
    ((10, 19), 10.0, 10.0),   # interior: offsets cancel
    ((0, 0), 0.05, 1.0),      # point at bucket start: only the 5 % uniform floor
    ((99, 99), 2.0, 1.0),     # point at bucket end
    ((0, 99), 100.0, 100.0),  # whole bucket
])
def test_tier0_uniform_offset_current_behaviour(q, current, exact):
    h = _one_uniform_bucket()
    assert h.predict(RangeQuery(*q)) == pytest.approx(current)
    ew = EquiWidthHistogram.build_from_sample(0, 99, 1, np.arange(100), 100)
    assert ew.predict(RangeQuery(*q)) == pytest.approx(exact)  # EquiWidth is inclusive: (hi - lo + 1) / w


@pytest.mark.parametrize("w, median", [(10, 2.0), (100, 1.0556), (1000, 1.0054)])
def test_validation_scores_open_interval(w, median):
    """Perfectly uniform bucket: an exact validation would give median q-error 1.0 for Uniform."""
    b = Bucket(0, w - 1, count=1000.0 * w)
    h = HybridEstimator([b])
    b_ps = np.cumsum(np.full(w, 1000.0))
    rng = np.random.default_rng(1)
    lo = rng.integers(0, w, 500)
    hi = np.array([rng.integers(v, w) for v in lo])
    q_m, _ = h._eval_model_q_error_vec(None, lo, hi, b, b_ps, 0, w)
    assert q_m == pytest.approx(median, abs=5e-4)
    # a width-10 perfectly uniform bucket fails the Tier-0 threshold (1.3) in the current validation
    if w == 10:
        assert q_m > HybridEstimator.UNIFORM_THRESHOLD

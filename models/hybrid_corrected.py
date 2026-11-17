"""
Hybrid-Corrected — a labelled MODIFICATION of the canonical Hybrid FD CDF (models/hybrid.py).

Only one thing changes: inclusive-interval arithmetic on the discrete integer domain. Everything
else — FD grid, training rows, density weights, waterfall tiers, models, thresholds (1.3 / 1.05),
500 validation queries, FT, rebuild, the 5 % uniform floor in predict — is inherited unchanged.

Definition used everywhere: F(v) = fraction of the bucket's rows with value <= v (this is what the
training labels already are, y = (Rank(v) - Rank(lo - 1)) / C), and an inclusive range [a, b]
inside a bucket holds (F(b) - F(a - 1)) * C rows, with F(lo - 1) = 0 and F(hi) = 1.

The canonical code violates it in two places (found 2026-11-10, see
results/paper_experiments/methodology/IMPLEMENTATION_AUDIT.md):
  1. inference, Tier 0: a bucket without a model uses F(x_norm) = x_norm = (v - lo) / w, i.e. the
     fraction of values < v. Queries starting at the bucket's lo lose one value, queries ending at
     its hi gain one; a point query at lo gets 0 rows (only the 5 % floor). Fixed: (b - a + 1) / w.
  2. waterfall validation (_eval_model_q_error_vec), every candidate: the estimate is
     F(b) - F(a), the fraction in (a, b], while the truth counts [a, b]. Fixed: F(b) - F(a - 1)
     with the same endpoint conventions predict() uses (F(lo - 1) = 0, F(hi) = 1); Tier 0 is scored
     as (b - a + 1) / w. Validation therefore scores exactly the in-bucket estimate predict() makes
     (the 5 % floor of predict is not added: the canonical validation does not use it either).

fix_validation = False gives the diagnostic variant Hybrid-Corrected-PredictOnly (fix 1 only),
used to attribute the effect of each fix; it is not a candidate method.
"""
import numpy as np

from .hybrid import HybridEstimator


class HybridCorrectedEstimator(HybridEstimator):
    SPEC_VERSION = "hybrid-fd-cdf-corrected-inclusive (MODIFICATION of hybrid-fd-cdf-canonical)"
    fix_validation = True

    def _get_bucket_overlap_count(self, b_idx: int, q_lo: int, q_hi: int) -> float:
        if self.models.get(b_idx) is not None:
            return super()._get_bucket_overlap_count(b_idx, q_lo, q_hi)
        b = self.buckets[b_idx]
        lo, hi = max(q_lo, b.lo), min(q_hi, b.hi)
        if lo > hi:
            return 0.0
        # Tier 0: F(v) = (v - b.lo + 1) / w, so [lo, hi] holds (hi - lo + 1) / w of the bucket
        return (hi - lo + 1) / (b.hi - b.lo + 1) * b.count

    def _eval_model_q_error_vec(self, model, queries_lo, queries_hi, bucket, b_ps, b_lo, width):
        if not self.fix_validation:
            return super()._eval_model_q_error_vec(model, queries_lo, queries_hi, bucket, b_ps, b_lo, width)
        idx_hi = queries_hi - b_lo
        idx_lo = queries_lo - b_lo
        act_counts = b_ps[idx_hi] - np.where(idx_lo > 0, b_ps[idx_lo - 1], 0)
        act_counts = np.maximum(1.0, act_counts)

        if model is None:
            frac = (queries_hi - queries_lo + 1) / width
        else:
            prev = queries_lo - 1
            f_hi = np.where(queries_hi >= bucket.hi, 1.0,
                            self._predict_local_cdf_vec(model, (queries_hi - bucket.lo) / width))
            f_lo = np.where(prev < bucket.lo, 0.0,
                            self._predict_local_cdf_vec(model, (prev - bucket.lo) / width))
            frac = np.maximum(0.0, f_hi - f_lo)
        pred_counts = np.maximum(1.0, frac * bucket.count)

        q_errs = np.maximum(act_counts / pred_counts, pred_counts / act_counts)
        return np.median(q_errs), np.percentile(q_errs, 95)


class HybridCorrectedPredictOnlyEstimator(HybridCorrectedEstimator):
    """Diagnostic: inference fix only, canonical validation (to attribute the two fixes)."""

    SPEC_VERSION = "hybrid-fd-cdf-corrected-predict-only (DIAGNOSTIC, MODIFICATION of hybrid-fd-cdf-canonical)"
    fix_validation = False


VARIANTS = {
    "original": HybridEstimator,
    "corrected": HybridCorrectedEstimator,
    "corrected-predict-only": HybridCorrectedPredictOnlyEstimator,
}

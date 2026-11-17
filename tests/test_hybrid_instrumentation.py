"""Instrumentation of Hybrid FD CDF (selection info, bucket table) and the paper-experiment schema."""
import numpy as np
import pandas as pd

from models import EquiWidthHistogram, HybridEstimator
from paper_experiments.schema import drift_steps_to_long

N = 60_000_000


def _build(values, bins, mn, mx):
    est = np.bincount(values - mn, minlength=mx - mn + 1) * (N / len(values))
    h = HybridEstimator(EquiWidthHistogram.build_from_sample(mn, mx, bins, values, N).buckets)
    h.train(est, mn, 200, np.random.default_rng(42))
    return h


def _mixed_sample():
    rng = np.random.default_rng([42, 7])
    span = 20_000
    a = rng.integers(0, span // 2, size=50_000)                          # uniform half
    b = span // 2 + np.minimum(rng.zipf(1.3, size=50_000), span // 2 - 1)  # skewed half
    return np.concatenate([a, b]).astype(np.int64), span


def test_bucket_table_memory_sums_to_n_values():
    values, span = _mixed_sample()
    h = _build(values, 20, 0, span - 1)
    rows = h.bucket_table()
    assert len(rows) == len(h.buckets)
    assert HybridEstimator.GLOBAL_VALUES + sum(r["bucket_values"] for r in rows) == h.n_values()


def test_selection_matches_chosen_models():
    values, span = _mixed_sample()
    h = _build(values, 20, 0, span - 1)
    for r in h.bucket_table():
        if r["cdf_model"] == "Empty":
            assert r["tier"] is None
            continue
        assert r["cdf_model"] == HybridEstimator.model_name(h.models.get(r["bucket"]))
        assert r["tier"] == HybridEstimator.TIER[r["cdf_model"]]
        if r["passed_threshold"]:
            assert r["val_median_qerror"] <= r["threshold"] + 1e-12
        if r["selection_reason"] == "waterfall" and r["tier"] == 0:
            assert r["threshold"] == HybridEstimator.UNIFORM_THRESHOLD


def test_feedback_update_records_touched_buckets():
    values, span = _mixed_sample()
    h = _build(values, 20, 0, span - 1)
    from models.common import RangeQuery
    qs = [RangeQuery(100, 900), RangeQuery(15_000, 15_500)]
    pred = np.array([h.predict(q) for q in qs])
    n = h.feedback_update(qs, pred * 3.0, np.maximum(pred, 1.0))
    assert n == len(h.last_ft_idx) > 0


def test_long_format_never_emits_rb_without_rebuild():
    wide = pd.DataFrame([
        {"step": 0, "drift_pct": 0, "action": "initial build", "action_time_s": 1.0, "median_qerror": 1.1,
         "p95_qerror": 2, "avg_qerror": 1.2, "infer_us_median": 3, "n_values": 10, "n_buckets": 4},
        {"step": 1, "drift_pct": 5, "action": "FT + rebuild", "action_time_s": 1.0, "median_qerror": 1.3,
         "p95_qerror": 2, "avg_qerror": 1.2, "infer_us_median": 3, "n_values": 10, "n_buckets": 4,
         "before_median_qerror": 1.5, "before_p95_qerror": 3, "before_avg_qerror": 2,
         "ft_median_qerror": 1.3, "ft_p95_qerror": 2, "ft_avg_qerror": 1.2,
         "rb_median_qerror": 1.3, "rb_p95_qerror": 2, "rb_avg_qerror": 1.2,
         "checkpoint_triggered": True, "rebuild_performed": False, "rebuilt_buckets": 0},
        {"step": 2, "drift_pct": 10, "action": "FT + rebuild", "action_time_s": 1.0, "median_qerror": 1.1,
         "p95_qerror": 2, "avg_qerror": 1.2, "infer_us_median": 3, "n_values": 10, "n_buckets": 4,
         "before_median_qerror": 1.5, "before_p95_qerror": 3, "before_avg_qerror": 2,
         "ft_median_qerror": 1.3, "ft_p95_qerror": 2, "ft_avg_qerror": 1.2,
         "rb_median_qerror": 1.1, "rb_p95_qerror": 2, "rb_avg_qerror": 1.2,
         "checkpoint_triggered": True, "rebuild_performed": True, "rebuilt_buckets": 2},
    ])
    long = drift_steps_to_long(wide, {"model": "Hybrid"}, staged=True)
    stages = long.groupby("step")["stage"].apply(list).to_dict()
    assert stages == {0: ["STATIC"], 1: ["SHOCK", "FT"], 2: ["SHOCK", "FT", "RB"]}

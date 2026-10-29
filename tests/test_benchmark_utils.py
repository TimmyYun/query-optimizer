"""Shared harness: exact truth and the evaluation loop every method goes through."""
import numpy as np

from benchmark_utils import exact_counts, run_queries
from models.common import RangeQuery


def brute_force(queries, freq, mn):
    out = []
    for q in queries:
        out.append(sum(int(freq[v - mn]) for v in range(q.low, q.high + 1) if mn <= v < mn + len(freq)))
    return np.array(out, dtype=np.float64)


def test_exact_counts_matches_brute_force_including_edges():
    rng = np.random.default_rng(42)
    mn = 10
    freq = rng.integers(0, 5, size=30)
    queries = [
        RangeQuery(0, 5),      # fully below the domain -> 0
        RangeQuery(50, 60),    # fully above -> 0
        RangeQuery(0, 12),     # overlaps the left edge
        RangeQuery(35, 100),   # overlaps the right edge
        RangeQuery(10, 39),    # exactly the domain
        RangeQuery(17, 17),    # point query
    ] + [RangeQuery(int(a), int(a + w)) for a, w in zip(rng.integers(0, 45, 50), rng.integers(0, 10, 50))]

    np.testing.assert_array_equal(exact_counts(queries, freq, mn), brute_force(queries, freq, mn))


def test_run_queries_scores_before_feedback():
    """Prequential: the estimate of query i must not see the feedback of query i."""
    state = {"value": 1.0}
    seen = []

    def predict(q):
        return state["value"]

    def feedback(q, truth, pred):
        seen.append((truth, pred))
        state["value"] = truth

    queries = [RangeQuery(0, 1), RangeQuery(0, 1), RangeQuery(0, 1)]
    preds, times, update_time = run_queries(predict, queries, np.array([5.0, 7.0, 9.0]), feedback)

    np.testing.assert_array_equal(preds, [1.0, 5.0, 7.0])
    assert seen == [(5.0, 1.0), (7.0, 5.0), (9.0, 7.0)]
    assert len(times) == 3 and update_time >= 0.0


def test_run_queries_feedback_gets_clipped_estimate():
    seen = []
    preds, _, _ = run_queries(
        lambda q: 0.0, [RangeQuery(0, 1)], np.array([3.0]), lambda q, t, p: seen.append(p)
    )
    assert preds[0] == 0.0       # raw estimate is returned unclipped (EntropyHist RMSE needs it)
    assert seen == [1.0]         # the model learns from the same clipped value that is scored

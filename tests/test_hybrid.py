"""Hybrid FD CDF: waterfall-выбор локальной CDF, FT, predict, частичная перестройка под дрейфом."""
import numpy as np

from benchmark_utils import exact_counts
from models import EquiWidthHistogram, HybridEstimator
from models.common import RangeQuery

N = 60_000_000
SAMPLE = 100_000


def build(values, bins, mn, mx, seed=42):
    """Hybrid по сэмплу values (как run_hybrid.py): бакеты equi-width, частоты x N/n."""
    est = np.bincount(values - mn, minlength=mx - mn + 1) * (N / len(values))
    buckets = EquiWidthHistogram.build_from_sample(mn, mx, bins, values, N).buckets
    h = HybridEstimator(buckets)
    h.train(est, mn, 200, np.random.default_rng(seed))
    return h


def periodic_values(rng, n, span, period):
    """Колокола шириной period подряд — структура мельче бакета (как hard-генератор)."""
    cell = rng.integers(0, span // period, size=n)
    micro = np.clip(np.round(rng.normal(period / 2, period / 6, size=n)), 0, period - 1)
    return (cell * period + micro).astype(np.int64)


def median_q_error(h, queries, freq, mn):
    truth = np.maximum(exact_counts(queries, freq * (N / freq.sum()), mn), 1.0)
    pred = np.maximum([h.predict(q) for q in queries], 1.0)
    return float(np.median(np.maximum(pred / truth, truth / pred)))


def test_uniform_data_keeps_buckets_uniform():
    """Равномерные данные проходят Tier 0 (медиана <= 1.3): ни одной модели."""
    rng = np.random.default_rng([42, 1])
    values = rng.integers(0, 100_000, size=SAMPLE)
    h = build(values, 10, 0, 99_999)
    assert all(m is None for m in h.models.values())
    assert h.param_count() == 10


CANONICAL_CDFS = {"Uniform", "Linear", "Power", "Log-Linear", "Isotonic", "DecisionTree", "FourierMLP"}


def test_only_canonical_cdf_models_are_chosen():
    """Канонический Hybrid FD CDF: Tier 0-3 и ничего больше (ни poly3, ни мини-гистограмм)."""
    rng = np.random.default_rng([42, 2])
    span = 20_000
    values = rng.choice(periodic_values(rng, 1_000_000, span, 1000), size=SAMPLE, replace=False)
    skewed = np.minimum(rng.zipf(1.5, size=SAMPLE), span - 1)
    for data in (values, skewed):
        h = build(data, 8, 0, span - 1)
        names = {HybridEstimator.model_name(m) for m in h.models.values()}
        assert names <= CANONICAL_CDFS, names


def _waterfall(qerr, parametric_only=False):
    """
    Выбор модели в одном бакете, где медианная q-error каждой модели задана таблицей qerr
    (по имени модели). Возвращает (выбранное имя, имена проверенных по порядку).
    """
    from models.common import Bucket
    rng = np.random.default_rng([42, 11])
    values = rng.integers(0, 1000, size=SAMPLE)
    est = np.bincount(values, minlength=1000) * (N / SAMPLE)
    h = HybridEstimator([Bucket(0, 999, float(N))], parametric_only=parametric_only)
    checked = []

    def fake_exam(mdl, *args):
        name = HybridEstimator.model_name(mdl)
        checked.append(name)
        return qerr[name], qerr[name]

    h._eval_model_q_error_vec = fake_exam
    h.train(est, 0, 50, np.random.default_rng(7))
    return HybridEstimator.model_name(h.models[0]), checked


def test_waterfall_takes_first_model_that_passes_not_the_best():
    chosen, checked = _waterfall({"Uniform": 1.5, "Linear": 1.2, "Power": 1.04, "Log-Linear": 1.0,
                                  "Isotonic": 1.0, "DecisionTree": 1.0, "FourierMLP": 1.0})
    assert chosen == "Power"
    assert checked == ["Uniform", "Linear", "Power"]


def test_tier0_threshold_is_1_3_and_stops_the_waterfall():
    chosen, checked = _waterfall(dict.fromkeys(CANONICAL_CDFS, 1.0) | {"Uniform": 1.3})
    assert chosen == "Uniform" and checked == ["Uniform"]
    chosen, _ = _waterfall(dict.fromkeys(CANONICAL_CDFS, 1.0) | {"Uniform": 1.31})
    assert chosen == "Linear"


def test_tiers_2_and_3_run_in_order_only_when_earlier_tiers_fail():
    base = {"Uniform": 2.0, "Linear": 1.9, "Power": 1.8, "Log-Linear": 1.7}
    chosen, checked = _waterfall(base | {"Isotonic": 1.05, "DecisionTree": 1.0, "FourierMLP": 1.0})
    assert chosen == "Isotonic"
    assert checked == ["Uniform", "Linear", "Power", "Log-Linear", "Isotonic"]
    chosen, checked = _waterfall(base | {"Isotonic": 1.2, "DecisionTree": 1.1, "FourierMLP": 1.05})
    assert chosen == "FourierMLP" and checked[-3:] == ["Isotonic", "DecisionTree", "FourierMLP"]


def test_no_model_passes_keeps_the_most_accurate():
    chosen, _ = _waterfall({"Uniform": 2.0, "Linear": 1.9, "Power": 1.8, "Log-Linear": 1.7,
                            "Isotonic": 1.3, "DecisionTree": 1.6, "FourierMLP": 1.5})
    assert chosen == "Isotonic"
    chosen, checked = _waterfall({"Uniform": 2.0, "Linear": 1.9, "Power": 1.8, "Log-Linear": 1.7,
                                  "Isotonic": 1.0, "DecisionTree": 1.0, "FourierMLP": 1.0}, parametric_only=True)
    assert chosen == "Log-Linear" and "Isotonic" not in checked


def test_retrain_touches_only_requested_buckets():
    rng = np.random.default_rng([42, 4])
    span = 20_000
    values = rng.choice(periodic_values(rng, 1_000_000, span, 1000), size=SAMPLE, replace=False)
    h = build(values, 4, 0, span - 1)
    before = dict(h.models)
    est = np.bincount(values, minlength=span) * (N / SAMPLE)
    h.models[2] = "sentinel"
    h.train(est, 0, 200, np.random.default_rng(7), bucket_indices=[1])
    assert h.models[2] == "sentinel"
    assert h.models[0] is before[0] and h.models[3] is before[3]


def test_training_is_deterministic():
    rng = np.random.default_rng([42, 5])
    values = rng.choice(periodic_values(rng, 1_000_000, 20_000, 1000), size=SAMPLE, replace=False)
    a, b = build(values, 4, 0, 19_999), build(values, 4, 0, 19_999)
    qs = [RangeQuery(int(s), int(s + 150)) for s in range(0, 19_000, 977)]
    assert [a.predict(q) for q in qs] == [b.predict(q) for q in qs]


def test_isotonic_and_tree_fast_path_matches_sklearn():
    """predict() идёт по спискам чисел вместо sklearn: значения должны совпасть с model.predict."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng([42, 7])
    X = np.sort(rng.random(300))
    y = np.clip(np.cumsum(rng.random(300)) / 150 + rng.normal(0, 0.02, 300), 0, 1)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X, y)
    tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X.reshape(-1, 1), y)
    xs = np.concatenate([rng.random(2000), X, tree.tree_.threshold[tree.tree_.children_left != -1], [0.0, 1.0]])
    h = HybridEstimator([])
    for model in (iso, tree):
        expected = np.clip(model.predict(xs.reshape(-1, 1) if model is tree else xs), 0, 1)
        fast = np.array([h._predict_local_cdf(model, float(x)) for x in xs])
        np.testing.assert_allclose(fast, expected, rtol=0, atol=1e-12)


def test_cdf_choice_and_size():
    rng = np.random.default_rng([42, 8])
    span = 20_000
    values = rng.choice(periodic_values(rng, 1_000_000, span, 1000), size=SAMPLE, replace=False)
    h = build(values, 4, 0, span - 1)
    choice = h.cdf_choice()
    assert sum(choice.values()) == 4
    assert h.n_values() == 2 + h.param_count() + sum(m is not None for m in h.models.values())


def test_feedback_revives_empty_bucket_at_its_density_not_a_sum():
    """Пустой в сэмпле бакет: масса = alpha x плотность по запросам x ширина, а не сумма по запросам."""
    from models.common import Bucket
    h = HybridEstimator([Bucket(0, 99, 0.0), Bucket(100, 199, 1000.0)])
    h._bake_vectorized_data()
    # 200 одинаковых запросов внутри пустого бакета: в каждом 50 строк на 10 значений
    queries = [RangeQuery(10, 19)] * 200
    h.feedback_update(queries, np.full(200, 50.0), np.full(200, 1.0), None, alpha=0.15)
    assert abs(h.buckets[0].count - 0.15 * 5.0 * 100) < 1e-9


def test_predict_prefix_sum_matches_bucket_loop():
    """predict складывает целые бакеты префиксной суммой: ответ тот же, что у цикла по бакетам."""
    rng = np.random.default_rng([42, 9])
    span = 20_000
    values = rng.choice(periodic_values(rng, 1_000_000, span, 1000), size=SAMPLE, replace=False)
    h = build(values, 16, 0, span - 1)
    h.buckets[3].count = 0.0
    h._bake_vectorized_data()

    def loop_predict(q):
        total = 0.0
        for i, b in enumerate(h.buckets):
            if b.hi < q.low or b.lo > q.high or b.count == 0:
                continue
            if q.low <= b.lo and q.high >= b.hi:
                total += b.count
            else:
                total += h._get_bucket_overlap_count(i, q.low, q.high)
        return total

    lo = rng.integers(-500, span + 500, size=3000)
    queries = [RangeQuery(int(a), int(a + w)) for a, w in zip(lo, rng.integers(0, span, size=3000))]
    for q in queries:
        assert abs(h.predict(q) - loop_predict(q)) <= 1e-6 * max(1.0, loop_predict(q))


def _reference_feedback_update(h, queries, y_true_counts, y_pred_counts, error_threshold=1.2, alpha=0.15):
    """Прежняя реализация FT циклом по запросам и бакетам — эталон для векторизованной."""
    q_errs = np.maximum(y_true_counts / (y_pred_counts + 1e-9), y_pred_counts / (y_true_counts + 1e-9))
    b_lo = np.array([b.lo for b in h.buckets])
    b_hi = np.array([b.hi for b in h.buckets])
    true_m, pred_m, cov = {}, {}, {}
    for idx in np.where(q_errs > error_threshold)[0]:
        q = queries[idx]
        overlaps = []
        for i in range(np.searchsorted(b_hi, q.low), np.searchsorted(b_lo, q.high, side="right")):
            w = min(q.high, h.buckets[i].hi) - max(q.low, h.buckets[i].lo) + 1
            if w > 0:
                overlaps.append((i, w))
        total = sum(w for _, w in overlaps)
        for i, w in overlaps:
            true_m[i] = true_m.get(i, 0.0) + y_true_counts[idx] * w / total
            pred_m[i] = pred_m.get(i, 0.0) + y_pred_counts[idx] * w / total
            cov[i] = cov.get(i, 0.0) + w
    for i in true_m:
        b = h.buckets[i]
        ratio = np.clip(true_m[i] / (1.0 if pred_m[i] < 1e-5 else pred_m[i]), 0.2, 5.0)
        if b.count < 1.0 and true_m[i] > 0:
            b.count = alpha * true_m[i] / cov[i] * (b.hi - b.lo + 1)
        else:
            b.count = max(0.0, b.count * (1.0 - alpha) + b.count * ratio * alpha)
    return len(true_m)


def test_vectorized_feedback_update_matches_reference_loop():
    from copy import deepcopy

    rng = np.random.default_rng([42, 10])
    span = 20_000
    values = rng.choice(periodic_values(rng, 1_000_000, span, 1000), size=SAMPLE, replace=False)
    h = build(values, 37, 0, span - 1)  # последний бакет уже остальных
    h.buckets[5].count = 0.0
    h.buckets[6].count = 0.4
    h._bake_vectorized_data()
    ref = deepcopy(h)

    lo = rng.integers(-300, span + 300, size=4000)
    queries = [RangeQuery(int(a), int(a + w)) for a, w in zip(lo, rng.integers(0, 3000, size=4000))]
    y_true = rng.integers(0, 50_000, size=4000).astype(float)
    y_pred = np.maximum(rng.integers(0, 50_000, size=4000).astype(float), 1.0)

    n_vec = h.feedback_update(queries, y_true, y_pred)
    n_ref = _reference_feedback_update(ref, queries, y_true, y_pred)
    assert n_vec == n_ref
    np.testing.assert_allclose([b.count for b in h.buckets], [b.count for b in ref.buckets], rtol=1e-9, atol=1e-9)


def test_checkpoint_rule_counts_rows_not_percent():
    from drift_harness import CHECKPOINT_ROWS, RebuildRule
    rule = RebuildRule(60_000_000, CHECKPOINT_ROWS)
    rule.add(99_999)
    assert not rule.due()
    rule.add(1)
    assert rule.due()
    rule.reset(60_000_000)
    assert not rule.due()


def test_drift_rebuild_retrains_only_marked_buckets(tmp_path):
    """Checkpoint: помеченные бакеты получают счётчик и CDF по выборке шага, остальные не трогаются."""
    from drift_harness import Feedback, Step
    from run_hybrid_drift import HybridDrift

    rng = np.random.default_rng([42, 12])
    span = 20_000
    sample0 = rng.choice(periodic_values(rng, 1_000_000, span, 1000), size=SAMPLE, replace=False)
    model = HybridDrift(points=200)
    model.B, model.out_dir = 8, tmp_path
    model.build(Step(0, span - 1, N, sample0, 8))
    counts0 = [b.count for b in model.h.buckets]
    models0 = dict(model.h.models)

    sample1 = rng.integers(0, span, size=SAMPLE)  # данные сменились на равномерные
    lo = rng.integers(0, 2400, size=200)  # запросы только в бакете 0 (ширина 2500)
    queries = [RangeQuery(int(a), int(a) + 50) for a in lo]
    fb = Feedback(queries, np.full(200, 1e6), np.ones(200))  # оценки в 1e6 раз меньше правды
    assert model.mark(fb) == 1 and model.marked == {0}
    assert model.rebuild(Step(0, span - 1, N, sample1, 8), fb) == 1
    assert model.marked == set()
    est1 = np.bincount(sample1, minlength=span) * (N / SAMPLE)
    assert abs(model.h.buckets[0].count - est1[:2500].sum()) < 1e-6
    for i in range(1, 8):
        assert model.h.buckets[i].count == counts0[i]
        assert model.h.models[i] is models0[i]

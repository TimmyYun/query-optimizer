"""
Drift protocol of Hybrid FD CDF: Shock -> FT -> RB at every step (drift_harness._three_stages),
and the 10 % rule of the sample-based baselines.

Every test builds a tiny drift folder (step_<i>.pkl = (mn, mx, N, freq, sample, k)) in tmp_path,
runs drift_harness.run_drift on it, and spies on the model to check what each stage saw and did:
  (a) SHOCK is the score of the state the previous step left, counts not refreshed;
  (b) FT changes bucket counts only (boundaries, CDF models and waterfall records stay);
  (c) marking uses post-FT estimates of the adapt queries;
  (d) RB only at a checkpoint (>= CHECKPOINT_ROWS changed rows since the last rebuild), only
      marked buckets are retrained, the rest is untouched; otherwise RB == FT, rebuild_performed False;
  (e) no eval query ever reaches update / mark / rebuild, and rebuild gets no ground truth;
  (f) --drift-total-pct 200 over 40 steps changes 5 % of N0 per step, drift_pct 0..200;
  (g) baselines rebuild by the 10 % rule and report it in checkpoint_triggered / rebuild_performed.
"""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import drift_harness
from benchmark_utils import exact_counts, load_and_filter_workload, split_adapt_eval
from drift_harness import CHECKPOINT_ROWS, SampleHistogram, drift_parser, run_drift
from models import EquiWidthHistogram, RangeQuery
from run_hybrid_drift import HybridDrift
from workload.workload import save_workload_csv

D = 4_000  # domain [0, D - 1]


def _target_probs():
    """Skewed target: Zipf(1.2) over a seeded permutation of the domain (the drift destination)."""
    p = 1.0 / np.arange(1, D + 1) ** 1.2
    p = p[np.random.default_rng([42, 11]).permutation(D)]
    return p / p.sum()


def make_drift(tmp_path, n0, n_steps, total=1.0, mode="replace", sample_size=4_000, n_queries=1_500):
    """step_0..step_<n_steps>.pkl drifting uniform -> skewed, plus a unique static workload."""
    rng = np.random.default_rng([42, 10])
    f_init = rng.multinomial(n0, np.full(D, 1.0 / D)).astype(np.float64)
    f_target = rng.multinomial(n0, _target_probs()).astype(np.float64)
    shift = tmp_path / "shift"
    shift.mkdir()
    for k in range(n_steps + 1):
        a = total * k / n_steps
        mixed = f_init + a * f_target if mode == "add" else (1.0 - a) * f_init + a * f_target
        freq = np.round(mixed).astype(np.int64)
        srng = np.random.default_rng([42, 12, k])
        sample = srng.choice(D, size=sample_size, p=freq / freq.sum()).astype(np.int64)
        with open(shift / f"step_{k}.pkl", "wb") as f:
            pickle.dump((0, D - 1, int(freq.sum()), freq, sample, 10), f)
    qrng = np.random.default_rng([42, 13])
    lows = qrng.integers(0, D - 1, size=4 * n_queries)
    widths = qrng.integers(1, 150, size=4 * n_queries)
    uniq = sorted({(int(lo), int(min(D - 1, lo + w))) for lo, w in zip(lows, widths)})
    pick = np.random.default_rng([42, 14]).permutation(len(uniq))[:n_queries]
    wl = tmp_path / "workload.csv"
    save_workload_csv([RangeQuery(*uniq[i]) for i in sorted(pick)], wl)
    return shift, wl


def drift_args(tmp_path, shift, wl, buckets="8", total_pct=100.0):
    return drift_parser("test").parse_args([
        "--shift-dir", str(shift), "--workload", "test", "--static-workload-path", str(wl),
        "--out-dir", str(tmp_path / "out"), "--experiment-name", "t", "--buckets", buckets,
        "--drift-total-pct", str(total_pct),
    ])


def step_data(shift, k):
    with open(shift / f"step_{k}.pkl", "rb") as f:
        mn, mx, N, freq, sample, _ = pickle.load(f)
    return mn, N, freq


def _key(q):
    return (int(q.low), int(q.high))


def _model_sig(m):
    """Identity of a local CDF: tuples by value, sklearn objects by object identity."""
    return m if (m is None or isinstance(m, tuple)) else id(m)


class SpyHybrid(HybridDrift):
    """HybridDrift that records, per step, what each stage received and changed."""

    def __init__(self, points, queries, adapt_keys, eval_keys):
        super().__init__(points)
        self.queries, self.adapt_keys, self.eval_keys = queries, adapt_keys, eval_keys
        self.log = {}
        self.end_state = {}

    # --- helpers ---
    def _state(self):
        h = self.h
        return {
            "counts": np.array([b.count for b in h.buckets]),
            "bounds": [(b.lo, b.hi) for b in h.buckets],
            "models": {i: _model_sig(m) for i, m in h.models.items()},
            "selection": copy.deepcopy(h.selection),
        }

    def _check_adapt_only(self, feedback, stage):
        keys = [_key(q) for q in feedback.queries]
        assert set(keys) == self.adapt_keys, f"{stage}: feedback is not exactly the adapt half"
        assert not set(keys) & self.eval_keys, f"{stage}: eval query leaked"

    # --- stages ---
    def update(self, feedback):
        k = self.step_no + 1
        state = self._state()
        prev = self.end_state[k - 1]
        # (a) the Shock model is the previous step's final state: no count refresh in between
        np.testing.assert_array_equal(state["counts"], prev["counts"])
        assert state["models"] == prev["models"]
        self._check_adapt_only(feedback, "FT")
        n = super().update(feedback)
        after = self._state()
        # (b) FT: counts only
        assert after["bounds"] == state["bounds"]
        assert after["models"] == state["models"]
        assert after["selection"] == state["selection"]
        changed = set(np.nonzero(after["counts"] != state["counts"])[0].tolist())
        assert changed <= set(self.h.last_ft_idx.tolist())
        self.log[k] = {"ft_n": n, "ft_changed": changed, "post_ft": after}
        return n

    def mark(self, feedback):
        k = self.step_no
        self._check_adapt_only(feedback, "mark")
        # (c) estimates passed to marking are the model's own post-FT estimates
        post = np.maximum([self.h.predict(q) for q in feedback.queries], 1.0)
        np.testing.assert_allclose(feedback.pred_counts, post, rtol=0, atol=1e-9)
        n = super().mark(feedback)
        self.log[k]["marked"] = set(self.marked)
        return n

    def rebuild(self, step, feedback):
        k = self.step_no
        self._check_adapt_only(feedback, "rebuild")
        assert not hasattr(step, "freq"), "rebuild must not see the ground truth"
        before = self._state()
        marked = set(self.marked)
        n = super().rebuild(step, feedback)
        after = self._state()
        rebuilt = set(self._rb_idx)
        assert rebuilt == marked and n == len(marked)  # (d) only marked buckets, all of them
        for i in range(len(self.h.buckets)):
            if i not in rebuilt:
                assert after["counts"][i] == before["counts"][i]
                assert after["models"].get(i) == before["models"].get(i)
                assert after["selection"].get(i) == before["selection"].get(i)
        self.log[k]["rebuilt"] = rebuilt
        return n

    def end_step(self, k, drift_pct):
        super().end_step(k, drift_pct)
        self.end_state[k] = self._state()
        self.end_state[k]["h"] = copy.deepcopy(self.h)


@pytest.fixture(scope="module")
def hybrid_run(tmp_path_factory):
    """
    N0 = 400 000 rows, 10 steps of 10 %: 40 000 rows per step, so the 100 000-row checkpoint
    is reached at steps 3, 6 and 9 only (120 000 rows each time).
    """
    tmp_path = tmp_path_factory.mktemp("hybrid")
    shift, wl = make_drift(tmp_path, n0=400_000, n_steps=10)
    args = drift_args(tmp_path, shift, wl)
    mn, N0, freq0 = step_data(shift, 0)
    queries, _ = load_and_filter_workload(str(wl), mn, D - 1, freq0)
    adapt_idx, eval_idx = split_adapt_eval(len(queries))
    model = SpyHybrid(points=40, queries=queries,
                      adapt_keys={_key(queries[i]) for i in adapt_idx},
                      eval_keys={_key(queries[i]) for i in eval_idx})
    run_drift(args, "Hybrid", model)
    out = tmp_path / "out" / "t" / "gradual_normal_to_zipf_Hybrid"
    df = pd.read_csv(out / "drift_steps.csv")
    return {"df": df, "model": model, "shift": shift, "queries": queries,
            "eval_idx": eval_idx, "out": out, "N0": N0}


def test_shock_is_previous_state_scored_on_eval(hybrid_run):
    """(a) before_* equals the previous step's final model scored on this step's eval half."""
    df, model, shift = hybrid_run["df"], hybrid_run["model"], hybrid_run["shift"]
    queries, eval_idx = hybrid_run["queries"], hybrid_run["eval_idx"]
    for k in range(1, len(df)):
        mn, N, freq = step_data(shift, k)
        truth = np.maximum(exact_counts(queries, freq, mn)[eval_idx], 1.0)
        prev = model.end_state[k - 1]["h"]
        pred = np.array([prev.predict(queries[i]) for i in eval_idx])
        med, p95, avg, mae = drift_harness._scores(truth, pred, N)
        row = df.iloc[k]
        assert row["before_median_qerror"] == pytest.approx(med, rel=0, abs=1e-12)
        assert row["before_p95_qerror"] == pytest.approx(p95, rel=0, abs=1e-12)
        assert row["before_mae_rows"] == pytest.approx(mae, rel=1e-12)


def test_ft_counts_and_marking_logged_every_step(hybrid_run):
    """(b)/(c) ran inside the spy at every drift step; FT bucket count is reported."""
    df, model = hybrid_run["df"], hybrid_run["model"]
    assert sorted(model.log) == list(range(1, len(df)))
    for k in range(1, len(df)):
        assert df.iloc[k]["ft_buckets"] == model.log[k]["ft_n"]
        assert df.iloc[k]["marked_buckets"] == len(model.log[k]["marked"])


def test_rebuild_only_at_checkpoint_and_flags(hybrid_run):
    """(d) checkpoint every 100 000 changed rows; RB == FT and no RB credit without a rebuild."""
    df, model = hybrid_run["df"], hybrid_run["model"]
    rows_per_step = hybrid_run["N0"] / 10
    assert rows_per_step * 3 >= CHECKPOINT_ROWS > rows_per_step * 2
    triggered = df.index[df["checkpoint_triggered"].astype(bool)].tolist()
    assert triggered == [3, 6, 9]
    assert not df.loc[0, "checkpoint_triggered"] and not df.loc[0, "rebuild_performed"]
    for k in range(1, len(df)):
        r = df.iloc[k]
        did_rebuild = "rebuilt" in model.log[k]
        assert did_rebuild == bool(r["checkpoint_triggered"])
        assert bool(r["rebuild_performed"]) == (bool(r["checkpoint_triggered"]) and r["rebuilt_buckets"] > 0)
        if not r["checkpoint_triggered"]:
            assert r["rebuilt_buckets"] == 0 and r["rb_time_s"] == 0.0
            for m in ("median", "p95", "avg"):
                assert r[f"rb_{m}_qerror"] == r[f"ft_{m}_qerror"]
            assert r["action"] == "FT"
        else:
            assert r["action"] == "FT + rebuild"
            assert r["rebuilt_buckets"] == len(model.log[k]["rebuilt"])
        assert r["median_qerror"] == r["rb_median_qerror"]  # "after" is the RB score
    # the uniform -> skewed drift must actually exercise a performed rebuild
    assert df["rebuild_performed"].astype(bool).any()


def test_marks_accumulate_until_rebuild(hybrid_run):
    """Marks persist between checkpoints and are cleared by the rebuild."""
    model = hybrid_run["model"]
    for k in (1, 2):
        assert model.log[k]["marked"] <= model.log[k + 1]["marked"]
    for k in (3, 6, 9):
        assert model.log[k]["rebuilt"] == model.log[k]["marked"]


def test_bucket_tables_match_stage_logs(hybrid_run):
    """Per-drift-point CDF tables exist for every step and agree with what the stages did."""
    model, out = hybrid_run["model"], hybrid_run["out"]
    for k in range(0, 11):
        t = pd.read_csv(out / "bucket_tables" / f"p{10 * k:03d}.csv")
        assert len(t) == len(model.h.buckets)
        if k == 0:
            assert (t["step_status"] == "INITIAL").all()
            continue
        assert set(t.index[t["rebuilt"].astype(bool)]) == model.log[k].get("rebuilt", set())
        assert set(t.index[t["marked_after_ft"].astype(bool)]) == model.log[k]["marked"]
        assert model.log[k]["ft_changed"] <= set(t.index[t["ft_updated"].astype(bool)])


def test_additive_200_percent_rows_per_step(tmp_path):
    """(f) + (g): 40 steps to +200 %: 5 % of N0 per step; EquiWidth rebuilds by the 10 % rule."""
    n0 = 100_000
    shift, wl = make_drift(tmp_path, n0=n0, n_steps=40, total=2.0, mode="add", sample_size=1_000, n_queries=300)
    args = drift_args(tmp_path, shift, wl, buckets="10", total_pct=200.0)
    model = SampleHistogram(
        lambda s, B: EquiWidthHistogram.build_from_sample(s.mn, s.mx, B, s.sample, s.N), lambda h: len(h.buckets))
    run_drift(args, "EquiWidth", model)
    df = pd.read_csv(tmp_path / "out" / "t" / "gradual_normal_to_zipf_EquiWidth" / "drift_steps.csv")
    assert len(df) == 41
    np.testing.assert_allclose(df["drift_pct"], np.arange(0, 205, 5))
    assert df.loc[1, "changed_since_rebuild_rows"] == int(round(0.05 * n0))
    assert df["n_rows"].iloc[-1] == pytest.approx(3 * n0, rel=0.01)
    # 10 % rule: rows changed since the last rebuild, as a share of the table at that rebuild
    base, changed = float(df.loc[0, "n_rows"]), 0.0
    for k in range(1, 41):
        changed += 0.05 * n0
        due = round(100.0 * changed / base, 1) > 10.0
        r = df.iloc[k]
        assert bool(r["checkpoint_triggered"]) == due == bool(r["rebuild_performed"]) == (r["action"] == "rebuild")
        if due:
            base, changed = float(r["n_rows"]), 0.0
        else:
            assert r["median_qerror"] == r["before_median_qerror"]  # action "none": nothing changed


def test_feedback_only_baseline_never_checkpoints(tmp_path):
    """(g) STHist (rebuild_rule 'never') only takes feedback from the adapt half."""
    from run_sthist import add_sthist_args
    from run_sthist_drift import STHistDrift

    shift, wl = make_drift(tmp_path, n0=200_000, n_steps=4, sample_size=1_000, n_queries=300)
    p = drift_parser("t")
    add_sthist_args(p)
    args = p.parse_args(["--shift-dir", str(shift), "--workload", "t", "--static-workload-path", str(wl),
                         "--out-dir", str(tmp_path / "out"), "--experiment-name", "t", "--buckets", "10"])
    mn, _, freq0 = step_data(shift, 0)
    queries, _ = load_and_filter_workload(str(wl), mn, D - 1, freq0)
    adapt_idx, eval_idx = split_adapt_eval(len(queries))
    adapt_keys = {_key(queries[i]) for i in adapt_idx}
    seen = []

    class Spy(STHistDrift):
        def update(self, feedback):
            seen.append({_key(q) for q in feedback.queries})
            return super().update(feedback)

    run_drift(args, "STHist", Spy(args))
    df = pd.read_csv(next((tmp_path / "out" / "t").glob("gradual_*")) / "drift_steps.csv")
    assert all(s == adapt_keys for s in seen) and len(seen) == 4
    assert not df["checkpoint_triggered"].astype(bool).any()
    assert not df["rebuild_performed"].astype(bool).any()
    assert (df["action"].iloc[1:] == "feedback update").all()

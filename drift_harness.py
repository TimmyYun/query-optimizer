"""
One drift loop for every model: same steps, same queries, same rebuild rule, same output.

A drift scenario is a folder of snapshots step_<i>.pkl = (mn, mx, N, freq, sample, k).
Every scenario changes an equal share of the initial table per step, 0 % -> --drift-total-pct
(default 100 %) over the steps (synthetic: 5 % or 10 % of the rows per step; DSB: official refresh
streams, ~5 % churn per step; additive growth to +200 %: --drift-total-pct 200). So rows changed
per step = N(step 0) * drift_total_pct / 100 / (number of steps - 1).

At every step:
  1. the step's queries are split once into adapt / eval halves (split_adapt_eval);
  2. "before": the model as it is, scored on the eval half;
  3. the model takes at most one action, and its wall time is measured:
       rebuild          rows changed since the model's last (re)build exceed 10 % of the
                        table at that time (PostgreSQL autovacuum: autovacuum_analyze_scale_factor
                        = 0.1). Only models with rebuild_rule == "10%";
       feedback update  a feedback model learns from the adapt half (STHist; Hybrid between rebuilds);
       none             nothing happens;
  4. "after": scored again on the eval half. No model ever sees an eval query's truth.

Hybrid FD CDF (staged = True) instead runs its three stages one after another at every step:
  Shock    the model as it is (no count refresh), = "before";
  FT       feedback update from the adapt half, scored -> ft_*;
           then buckets whose post-FT error on the adapt half is still high are marked;
  RB       only at a checkpoint: >= 100 000 rows changed since the last rebuild, checked at the
           step boundary (one rebuild per step at most). Only the marked buckets are retrained on
           the step's sample. Without a checkpoint RB = FT and rebuilt_buckets = 0.
  "after" (median_qerror, ...) is the RB score.
  checkpoint_triggered: the checkpoint was reached at this step's boundary;
  rebuild_performed:    checkpoint_triggered and at least one marked bucket was retrained.
  Without rebuild_performed the RB columns repeat FT and no improvement belongs to Rebuild.

Models never see freq (the ground truth): they get Step (catalog metadata + the sample)
and Feedback (adapt queries with their true counts and the model's own estimates).

Output: <out-dir>/<experiment>/gradual_<init>_to_<target>_<Model>/drift_steps.csv, one row per step.
"""
import argparse
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import (
    BYTES_PER_VALUE,
    add_buckets_arg,
    exact_counts,
    load_and_filter_workload,
    resolve_buckets,
    run_queries,
    setup_out_dir,
    split_adapt_eval,
)
from models import summarize

REBUILD_FRACTION = 0.10  # rebuild once more than 10 % of the rows changed since the last one
CHECKPOINT_ROWS = 100_000  # Hybrid FD CDF: rebuild checkpoint after 100 000 changed rows


@dataclass
class Step:
    """What a model may see of a snapshot: catalog metadata and the sample, never freq."""

    mn: int
    mx: int
    N: int
    sample: np.ndarray
    fd_k: int  # Freedman-Diaconis bucket count of this snapshot

    def sampled_freq(self) -> np.ndarray:
        """Per-value counts of the sample over [mn, mx], scaled to N rows."""
        idx = np.clip(self.sample - self.mn, 0, self.mx - self.mn)
        return np.bincount(idx, minlength=self.mx - self.mn + 1) * (self.N / len(self.sample))


@dataclass
class Feedback:
    """Executed adapt-half queries: exact row counts and the model's estimates before this step's action."""

    queries: list
    true_counts: np.ndarray  # exact, may be 0
    pred_counts: np.ndarray  # clipped to >= 1, as scored


class DriftModel:
    """
    Adapter for one method. The harness sets self.B (bucket count) and self.out_dir before build().
    rebuild_rule: "10%" = rebuilt from the step's sample by the 10 % rule; "never" = never reads data;
                  "100k" = checkpoint after CHECKPOINT_ROWS changed rows (Hybrid FD CDF).
    uses_feedback: the model learns from the adapt half (update(), and rebuild() gets feedback).
    staged: Shock -> FT -> RB at every step (see the module docstring); needs mark() and rebuild()
            returning the number of retrained buckets.
    """

    rebuild_rule = "10%"
    uses_feedback = False
    staged = False

    def build(self, step: Step):
        raise NotImplementedError

    def rebuild(self, step: Step, feedback: Feedback):
        """Default: build from scratch on the step's sample. May return extra columns (dict)."""
        self.build(step)

    def update(self, feedback: Feedback):
        """May return the number of buckets the update touched (staged models: FT buckets)."""
        pass

    def mark(self, feedback: Feedback) -> int:
        """staged only: mark buckets FT did not fix (feedback = post-FT estimates); returns the number marked."""
        raise NotImplementedError

    def predict(self, q) -> float:
        raise NotImplementedError

    def n_buckets(self) -> int:
        raise NotImplementedError

    def n_values(self) -> int:
        raise NotImplementedError

    def info(self) -> dict:
        """Extra per-step columns (e.g. Hybrid's chosen CDFs)."""
        return {}

    def end_step(self, k: int, drift_pct: float):
        """Called once a step is fully scored (e.g. Hybrid writes its per-bucket CDF table)."""
        pass


class SampleHistogram(DriftModel):
    """A histogram built from the sample, rebuilt from scratch by the 10 % rule (EquiWidth, EquiDepth, ...)."""

    def __init__(self, build_fn, n_buckets_fn):
        self.build_fn = build_fn  # (step, B) -> histogram with predict() and n_values()
        self.n_buckets_fn = n_buckets_fn

    def build(self, step):
        self.h = self.build_fn(step, self.B)

    def predict(self, q):
        return self.h.predict(q)

    def n_buckets(self):
        return self.n_buckets_fn(self.h)

    def n_values(self):
        return self.h.n_values()


class RebuildRule:
    """Rows changed since the last (re)build, as a share of the table at that (re)build."""

    def __init__(self, n_rows, min_rows=None):
        self.min_rows = min_rows  # None: 10 % of the table; else an absolute number of rows
        self.reset(n_rows)

    def reset(self, n_rows):
        self.base, self.changed = float(n_rows), 0.0

    def add(self, rows):
        self.changed += rows

    @property
    def changed_pct(self):
        # Точность 0.1 % таблицы: в синтетике N шага гуляет на ±0.1 % из-за округления частот
        # в генераторе, и два шага по 5 % давали то 9.99 %, то 10.01 % — перестройка зависела от шума
        return round(100.0 * self.changed / self.base, 1)

    def due(self):
        if self.min_rows is not None:
            return self.changed >= self.min_rows
        return self.changed_pct > 100.0 * REBUILD_FRACTION


def drift_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--shift-dir", required=True, help="Folder with step_<i>.pkl snapshots")
    p.add_argument("--workload", required=True,
                   help="Query count of the per-step workloads step_<i>_workload_driven_<count>.csv (with --shift-workload)")
    p.add_argument("--static-workload-path", help="One workload evaluated at every step")
    p.add_argument("--shift-workload", action="store_true", help="Per-step workloads that drift with the data")
    p.add_argument("--drift-total-pct", type=float, default=100.0,
                   help="Total drift over all steps as %% of the step-0 table: 100 (displacement 0->100 %%) "
                        "or e.g. 200 (additive growth 0->+200 %%, final table = 3x)")
    p.add_argument("--drift-type", default="displacement",
                   help="Label only, written to every row: displacement / additive / churn")
    p.add_argument("--init-dist", default="normal", help="Label only: output folder gradual_<init>_to_<target>_<Model>")
    p.add_argument("--target-dist", default="zipf", help="Label only")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--experiment-name", default=None)
    add_buckets_arg(p)
    return p


def _scores(truth_safe, pred, N):
    """(median, p95, mean q-error, MAE in rows); truth and estimate both clipped to >= 1 row."""
    pred_safe = np.maximum(pred, 1.0)
    m = summarize(truth_safe / N, pred_safe / N)
    return m["QErr_median"], m["QErr_p95"], m["QErr_avg"], float(np.mean(np.abs(pred_safe - truth_safe)))


def _three_stages(model, step, due, feedback, eval_queries, eval_truth):
    """FT -> marking -> RB for a staged model (Shock is the caller's "before")."""
    t0 = time.perf_counter()
    ft_buckets = model.update(feedback)
    ft_time = time.perf_counter() - t0
    pred, ft_infer, _ = run_queries(model.predict, eval_queries, eval_truth)
    ft = _scores(eval_truth, pred, step.N)
    ft_values = model.n_values()

    # Buckets FT did not fix: the model's post-FT estimates on the same adapt queries.
    # Re-estimating them is part of the simulation, not of the method, so it is not timed.
    post, _, _ = run_queries(model.predict, feedback.queries, np.maximum(feedback.true_counts, 1.0))
    post_ft = Feedback(feedback.queries, feedback.true_counts, np.maximum(post, 1.0))
    t0 = time.perf_counter()
    marked = model.mark(post_ft)
    mark_time = time.perf_counter() - t0

    rebuilt, rb_time = 0, 0.0
    if due:
        t0 = time.perf_counter()
        rebuilt = model.rebuild(step, post_ft)
        rb_time = time.perf_counter() - t0
        pred, rb_infer, _ = run_queries(model.predict, eval_queries, eval_truth)
        rb = _scores(eval_truth, pred, step.N)
    else:  # no checkpoint: no rebuild, RB = FT
        rb, rb_infer = ft, ft_infer

    extra = {
        "checkpoint_triggered": bool(due), "rebuild_performed": bool(due and rebuilt > 0),
        "ft_buckets": ft_buckets,
        "ft_median_qerror": ft[0], "ft_p95_qerror": ft[1], "ft_avg_qerror": ft[2], "ft_mae_rows": ft[3],
        "ft_time_s": ft_time, "ft_infer_us_median": float(np.median(ft_infer)) * 1e6,
        "ft_n_values": ft_values, "ft_size_bytes": ft_values * BYTES_PER_VALUE,
        "marked_buckets": marked, "mark_time_s": mark_time,
        "rebuilt_buckets": rebuilt, "rb_time_s": rb_time,
        "rb_median_qerror": rb[0], "rb_p95_qerror": rb[1], "rb_avg_qerror": rb[2], "rb_mae_rows": rb[3],
        "rb_infer_us_median": float(np.median(rb_infer)) * 1e6,
    }
    action = "FT + rebuild" if due else "FT"
    return rb, rb_infer, extra, action, ft_time + mark_time + rb_time


def run_drift(args, model_name, model: DriftModel):
    out_dir = setup_out_dir(args, f"gradual_{args.init_dist}_to_{args.target_dist}_{model_name}")
    model.out_dir = out_dir
    shift_dir = Path(args.shift_dir)
    paths = sorted(shift_dir.glob("step_*.pkl"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 step_*.pkl in {shift_dir}")
    n_steps = len(paths) - 1
    total_pct = float(args.drift_total_pct)

    rows, queries = [], None
    for k, path in enumerate(paths):
        with open(path, "rb") as f:
            mn, mx, N, freq, sample, fd_k = pickle.load(f)
        step = Step(int(mn), int(mx), int(N), np.asarray(sample, dtype=np.int64), int(fd_k))

        if args.shift_workload:
            queries, _ = load_and_filter_workload(
                str(shift_dir / f"step_{k}_workload_driven_{args.workload}.csv"), mn, mx, freq)
        elif queries is None:
            # zero-truth queries are dropped once, on step 0; later steps keep the same queries
            queries, _ = load_and_filter_workload(str(args.static_workload_path), mn, mx, freq)

        truth = exact_counts(queries, freq, mn)
        truth_safe = np.maximum(truth, 1.0)
        adapt_idx, eval_idx = split_adapt_eval(len(queries))
        eval_queries = [queries[i] for i in eval_idx]

        before = (None, None, None, None)
        extra = {}
        shock_values, shock_infer_us = None, None
        if k == 0:
            model.B = resolve_buckets(args.buckets, step.fd_k)
            t0 = time.perf_counter()
            model.build(step)
            action, action_time = "initial build", time.perf_counter() - t0
            extra = {"checkpoint_triggered": False, "rebuild_performed": False}
            rule = RebuildRule(step.N, CHECKPOINT_ROWS if model.rebuild_rule == "100k" else None)
            rows_per_step = step.N * total_pct / 100.0 / n_steps
            changed_pct, changed_rows = 0.0, 0.0
        else:
            rule.add(rows_per_step)
            changed_pct, changed_rows = rule.changed_pct, rule.changed
            due = model.rebuild_rule in ("10%", "100k") and rule.due()

            # "before": the model as it is. Feedback models also need their estimates on the adapt half.
            scored = np.arange(len(queries)) if model.uses_feedback else eval_idx
            pred = np.ones(len(queries))
            pred[scored], infer_times, _ = run_queries(model.predict, [queries[i] for i in scored], truth_safe[scored])
            pred = np.maximum(pred, 1.0)
            before = _scores(truth_safe[eval_idx], pred[eval_idx], step.N)
            shock_values = model.n_values()
            pos = np.searchsorted(scored, eval_idx)  # eval queries' timings among the scored ones
            shock_infer_us = float(np.median(np.asarray(infer_times)[pos])) * 1e6
            feedback = None
            if model.uses_feedback:
                feedback = Feedback([queries[i] for i in adapt_idx], truth[adapt_idx], pred[adapt_idx])

            t0 = time.perf_counter()
            if model.staged:
                after, eval_infer, extra, action, action_time = _three_stages(
                    model, step, due, feedback, eval_queries, truth_safe[eval_idx])
                if due:
                    rule.reset(step.N)
            elif due:
                extra = model.rebuild(step, feedback) or {}
                action = "rebuild"
                rule.reset(step.N)
            elif model.uses_feedback:
                model.update(feedback)
                action = "feedback update"
            else:
                action = "none"
            if not model.staged:
                action_time = time.perf_counter() - t0 if action != "none" else 0.0
                # 10 % rule: the rebuild is the whole action; no checkpoint for "never" models
                extra = {"checkpoint_triggered": bool(due), "rebuild_performed": bool(due), **extra}

        if action == "none":  # only models without feedback get here, so "before" was scored on eval only
            after, eval_infer = before, infer_times
        elif k == 0 or not model.staged:  # a staged model's "after" comes from _three_stages
            pred_eval, eval_infer, _ = run_queries(model.predict, eval_queries, truth_safe[eval_idx])
            after = _scores(truth_safe[eval_idx], pred_eval, step.N)

        n_values = model.n_values()
        drift_pct = round(total_pct * k / n_steps, 2)
        rows.append({
            "step": k,
            "drift_type": args.drift_type,
            "drift_pct": drift_pct,
            "n_rows": step.N,
            "changed_since_rebuild_pct": round(changed_pct, 2),
            "changed_since_rebuild_rows": int(round(changed_rows)),
            "action": action,
            "action_time_s": action_time,
            "median_qerror": after[0],
            "p95_qerror": after[1],
            "avg_qerror": after[2],
            "before_median_qerror": before[0],
            "before_p95_qerror": before[1],
            "before_avg_qerror": before[2],
            "mae_rows": after[3],
            "before_mae_rows": before[3],
            "shock_infer_us_median": shock_infer_us,
            "shock_n_values": shock_values,
            "infer_us_median": float(np.median(eval_infer)) * 1e6,
            "n_eval_queries": len(eval_idx),
            "n_buckets": model.n_buckets(),
            "n_values": n_values,
            "size_bytes": n_values * BYTES_PER_VALUE,
            **extra,
            **model.info(),
        })
        model.end_step(k, drift_pct)
        print(f"[{model_name}] step {k:>2} drift {rows[-1]['drift_pct']:>5.1f}% "
              f"changed {changed_pct:5.1f}% -> {action:<15} {action_time:8.3f}s | "
              f"median {after[0]:.3f} (before {before[0] if before[0] is None else round(before[0], 3)}) "
              f"p95 {after[1]:.3f}")

    pd.DataFrame(rows).to_csv(out_dir / "drift_steps.csv", index=False)
    print(f"Saved {out_dir / 'drift_steps.csv'}")

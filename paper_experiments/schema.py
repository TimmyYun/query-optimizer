"""
Unified output contract for results/paper_experiments/**.

Every result row carries ID_COLUMNS. Units are part of the column name:
  *_bytes   memory in bytes (KiB / MiB only for display: fmt_bytes)
  *_s       seconds,  *_us  microseconds
  *_qerror  q-error, dimensionless, >= 1, lower is better
  *_rows    rows (MAE in rows)

Stages:
  STATIC  a model built once and scored (static benchmarks; step 0 of a drift run)
  SHOCK   drift step, the model as it was before this step's action
  FT      after fine-tuning from the adapt half (Hybrid FT; STHist feedback update)
  RB      after a rebuild that was actually performed (Hybrid: >= 1 marked bucket retrained at a
          checkpoint; 10 %-rule baselines: full rebuild). Never emitted when no rebuild ran —
          the final state is then FT (Hybrid) or SHOCK (baseline, action "none").
"""
import math

import pandas as pd

ID_COLUMNS = [
    "experiment_id", "dataset", "table", "column", "workload", "drift_type", "drift_pct",
    "bin_strategy", "bucket_count", "model", "model_version", "seed", "stage",
]
STAGES = ("STATIC", "SHOCK", "FT", "RB")
REBUILD_COLUMNS = ["checkpoint_triggered", "rebuild_performed", "rebuilt_buckets"]

METRIC_COLUMNS = [
    "median_qerror", "p95_qerror", "avg_qerror", "mae_rows",
    "stage_time_s", "infer_us_median", "n_values", "size_bytes",
]


def fmt_bytes(n) -> str:
    """Display helper: 1536 -> '1.50 KiB'. Stored values stay in bytes."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return ""
    n = float(n)
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def bin_strategy_label(buckets_arg: str) -> str:
    """--buckets value -> bin_strategy: '100' -> 'fixed100', 'fd' -> 'fd', 'fd*0.5' -> 'fd_x0.5'."""
    a = str(buckets_arg).strip().lower()
    if a == "fd":
        return "fd"
    if a.startswith("fd*"):
        return f"fd_x{a[3:]}"
    return f"fixed{a}"


def _get(row, key):
    v = row.get(key)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def drift_steps_to_long(df: pd.DataFrame, ids: dict, staged: bool) -> pd.DataFrame:
    """
    drift_steps.csv (one wide row per step) -> long rows, one per (step, stage).
    ids: values for ID_COLUMNS except drift_pct / stage (taken per row); bucket_count defaults to n_buckets.
    staged: True for Hybrid FD CDF (Shock -> FT -> RB), False for single-action baselines.
    """
    out = []
    for _, r in df.iterrows():
        r = r.to_dict()
        base = {**{c: ids.get(c) for c in ID_COLUMNS}, "drift_pct": r["drift_pct"]}
        if base.get("drift_type") is None:
            base["drift_type"] = r.get("drift_type")
        if base.get("bucket_count") is None:
            base["bucket_count"] = r.get("n_buckets")
        common = {
            "step": int(r["step"]), "n_rows": _get(r, "n_rows"), "action": r.get("action"),
            "checkpoint_triggered": bool(_get(r, "checkpoint_triggered") or False),
            "rebuild_performed": bool(_get(r, "rebuild_performed") or False),
            "rebuilt_buckets": _get(r, "rebuilt_buckets") if staged else (1 if _get(r, "rebuild_performed") else 0),
            "ft_buckets": _get(r, "ft_buckets"), "marked_buckets": _get(r, "marked_buckets"),
            "n_buckets": _get(r, "n_buckets"), "n_eval_queries": _get(r, "n_eval_queries"),
            "cdf_choice": _get(r, "cdf_choice"),
        }
        if staged and common["rebuilt_buckets"] is None:
            common["rebuilt_buckets"] = 0

        def add(stage, median, p95, avg, mae, t, infer, n_values):
            out.append({**base, "stage": stage, **common,
                        "median_qerror": median, "p95_qerror": p95, "avg_qerror": avg, "mae_rows": mae,
                        "stage_time_s": t, "infer_us_median": infer,
                        "n_values": n_values, "size_bytes": None if n_values is None else 8 * n_values})

        if int(r["step"]) == 0:
            add("STATIC", r["median_qerror"], r["p95_qerror"], r["avg_qerror"], _get(r, "mae_rows"),
                r["action_time_s"], r["infer_us_median"], _get(r, "n_values"))
            continue
        add("SHOCK", r["before_median_qerror"], r["before_p95_qerror"], r["before_avg_qerror"],
            _get(r, "before_mae_rows"), 0.0, _get(r, "shock_infer_us_median"), _get(r, "shock_n_values"))
        if staged:
            add("FT", r["ft_median_qerror"], r["ft_p95_qerror"], r["ft_avg_qerror"], _get(r, "ft_mae_rows"),
                _get(r, "ft_time_s"), _get(r, "ft_infer_us_median"), _get(r, "ft_n_values"))
            if common["rebuild_performed"]:
                add("RB", r["rb_median_qerror"], r["rb_p95_qerror"], r["rb_avg_qerror"], _get(r, "rb_mae_rows"),
                    _get(r, "rb_time_s"), _get(r, "rb_infer_us_median"), _get(r, "n_values"))
        else:
            action = r.get("action")
            if action == "rebuild":
                stage = "RB"
            elif action == "feedback update":
                stage = "FT"
            else:
                continue  # action "none": the step's only state is SHOCK
            add(stage, r["median_qerror"], r["p95_qerror"], r["avg_qerror"], _get(r, "mae_rows"),
                r["action_time_s"], r["infer_us_median"], _get(r, "n_values"))
    return pd.DataFrame(out)


def final_stage(long_df: pd.DataFrame) -> pd.DataFrame:
    """Last stage of every step (the state the model ends the step in)."""
    order = {s: i for i, s in enumerate(STAGES)}
    d = long_df.assign(_o=long_df["stage"].map(order))
    keys = [c for c in ID_COLUMNS if c not in ("stage",)] + ["step"]
    keys = [k for k in keys if k in d.columns]
    return d.sort_values("_o").groupby(keys, dropna=False, as_index=False).tail(1).drop(columns="_o")

#!/usr/bin/env python3
"""
Train regressors that map query -> selectivity for each dataset.
- Input: truth/truth_mapping.csv (from compute_true_selectivity.py), and data/*.csv for min/max.
- Output:
    results/
      <dataset_stem>/leaderboard.csv         # metrics per model
      <dataset_stem>/predictions_<model>.csv # per-query preds (test fold)
      <dataset_stem>/<model>.joblib          # serialized model
      <dataset_stem>/scaler.joblib           # if used
    results/leaderboard_all.csv              # concatenated summary across datasets

Optionally, if PyCaret is installed and --auto is passed, also produce:
    results/<dataset_stem>/pycaret_leaderboard.csv
    results/<dataset_stem>/pycaret_top10/...
"""

# python train_selectivity_regressors.py \
#   --truth truth/truth_mapping.csv \
#   --data-dir data \
#   --outdir results

# Baseline feature vector
#
# is_eq — 1 if equality, 0 if range.
#
# value_norm — for equality: (value − vmin) / (vmax − vmin).
#
# low_norm — for range: (low − vmin) / (vmax − vmin).
#
# high_norm — for range: (high − vmin) / (vmax − vmin).
#
# width_norm — (high − low) / (vmax − vmin) (0 for equality).
#
# center_norm — ((low + high)/2 − vmin) / (vmax − vmin) (or equality’s value).
#
# width_log1p — log(1 + (high − low)) (0 for equality).
#
# is_eq_x_value — interaction: is_eq * value_norm.

import argparse, os, time, json, io
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from dataclasses import dataclass

# --------------------------
# Utilities
# --------------------------

def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def model_size_bytes(obj) -> int:
    # estimate by serializing to in-memory buffer
    import joblib
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.tell()

def scan_min_max(csv_path: Path) -> tuple[int, int]:
    # Fast two-pass is fine, but for min/max just one pass chunked
    mn, mx = None, None
    for chunk in pd.read_csv(csv_path, header=None, names=["v"], dtype="int64",
                             chunksize=1_000_000, engine="c"):
        cmin, cmax = int(chunk["v"].min()), int(chunk["v"].max())
        mn = cmin if mn is None else min(mn, cmin)
        mx = cmax if mx is None else max(mx, cmax)
    if mn is None or mx is None:
        raise ValueError(f"Empty CSV: {csv_path}")
    return mn, mx

def q_error(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-12) -> np.ndarray:
    # standard definition with epsilon guards for zeros
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    ratio = np.maximum(y_true / y_pred, y_pred / y_true)
    return ratio

# --------------------------
# Feature engineering
# --------------------------

def build_features(df: pd.DataFrame, vmin: int, vmax: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Input df columns expected:
      kind in {"equality","range"}
      value, low, high, true_selectivity
    We create light, query-only features (no data peeking):
      - is_eq
      - value_norm (if eq)
      - low_norm, high_norm, width_norm, center_norm (if range)
      - width_log1p
    """
    dom = max(vmax - vmin, 1)
    out = pd.DataFrame(index=df.index)
    out["is_eq"] = (df["kind"] == "equality").astype(int)

    # Equality features
    v = df["value"].fillna((vmin + vmax) // 2).astype(float)
    out["value_norm"] = (v - vmin) / dom

    # Range features
    lo = df["low"].fillna(vmin).astype(float)
    hi = df["high"].fillna(vmin).astype(float)
    width = (hi - lo).clip(lower=0)
    center = (lo + hi) / 2.0

    out["low_norm"] = (lo - vmin) / dom
    out["high_norm"] = (hi - vmin) / dom
    out["width_norm"] = width / dom
    out["center_norm"] = (center - vmin) / dom
    out["width_log1p"] = np.log1p(width)

    # Optional simple interactions
    out["is_eq_x_value"] = out["is_eq"] * out["value_norm"]

    y = df["true_selectivity"].astype(float)
    return out, y

# --------------------------
# Baseline models
# --------------------------

def get_baseline_models():
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import (
        RandomForestRegressor, ExtraTreesRegressor,
        GradientBoostingRegressor, AdaBoostRegressor
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingRegressor

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-4, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.2, max_iter=10000),

        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "KNN_10": KNeighborsRegressor(n_neighbors=10, weights="distance"),

        "RF200": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "ExtraTrees200": ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "GBDT": GradientBoostingRegressor(random_state=42),
        "HGB": HistGradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),

        "SVR_rbf": SVR(kernel="rbf", C=10.0, gamma="scale"),
    }
    # Optional extras if available
    try:
        from xgboost import XGBRegressor
        models["XGB"] = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=4, random_state=42
        )
    except Exception:
        pass
    try:
        import lightgbm as lgb
        models["LGBM"] = lgb.LGBMRegressor(
            n_estimators=1000, num_leaves=64, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    except Exception:
        pass
    return models

def eval_model(name, model, X_train, y_train, X_test, y_test):
    # train
    (_, train_sec) = timed(model.fit, X_train, y_train)

    # predict
    y_pred, infer_sec = timed(model.predict, X_test)
    y_pred = np.asarray(y_pred).reshape(-1)

    qe = q_error(y_test.values, y_pred)
    metrics = {
        "model": name,
        "MAE": float(np.mean(np.abs(y_test - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "R2": float(1.0 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2 + 1e-12)),
        "QErr_median": float(np.median(qe)),
        "QErr_p95": float(np.percentile(qe, 95)),
        "Train_sec": train_sec,
        "Infer_sec_total": infer_sec,
        "Infer_ms_per_row": float(1000.0 * infer_sec / max(len(y_pred), 1)),
        "Model_bytes": model_size_bytes(model),
    }
    return metrics, y_pred

# --------------------------
# PyCaret option (auto mode)
# --------------------------

def run_pycaret(X, y, out_dir: Path, n_select=10, seed=42):
    try:
        from pycaret.regression import setup, compare_models, pull, predict_model, save_model
    except Exception as e:
        print("[auto] PyCaret not installed; skipping auto mode.")
        return None

    import pandas as pd
    df = pd.concat([X.reset_index(drop=True), pd.Series(y.values, name="target")], axis=1)

    setup(data=df, target="target", session_id=seed, fold=5, silent=True, verbose=False, html=False)
    best = compare_models(sort="MAE", n_select=n_select, turbo=True)
    lb = pull()
    lb.to_csv(out_dir / "pycaret_leaderboard.csv", index=False)

    saved = []
    for i, m in enumerate(best, start=1):
        name = f"pyc_{i}_{type(m).__name__}"
        save_model(m, str(out_dir / f"{name}"))  # creates .pkl files
        saved.append(name)
    (out_dir / "pycaret_top10_models.txt").write_text("\n".join(saved))
    return lb

# --------------------------
# Main driver
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Train query->selectivity regressors per dataset.")
    ap.add_argument("--truth", default="truth/truth_mapping.csv", help="Path to combined truth mapping CSV")
    ap.add_argument("--data-dir", default="data", help="Folder with original CSVs (for min/max)")
    ap.add_argument("--outdir", default="results", help="Where to save models & metrics")
    ap.add_argument("--test-size", type=float, default=0.3, help="Test split fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto", action="store_true", help="Also run PyCaret to try many algos and save top-10")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(args.truth)
    datasets = sorted(truth["dataset"].unique().tolist())
    all_rows = []

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    for ds in datasets:
        print(f"\n=== Dataset: {ds} ===")
        df = truth[truth["dataset"] == ds].copy()

        # min/max from the source CSV (for normalization)
        csv_path = Path(args.data_dir) / ds
        vmin, vmax = scan_min_max(csv_path)
        print(f"Domain [{vmin}, {vmax}]")

        X, y = build_features(df, vmin, vmax)

        # Standardize only non-binary features
        feat_cols = X.columns.tolist()
        bin_cols = ["is_eq"]
        cont_cols = [c for c in feat_cols if c not in bin_cols]

        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[cont_cols] = scaler.fit_transform(X[cont_cols])

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=args.test_size, random_state=args.seed
        )

        out_dir = out_root / Path(ds).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save scaler
        dump(scaler, out_dir / "scaler.joblib")

        # Baseline models
        models = get_baseline_models()
        leaderboard = []

        for name, mdl in models.items():
            print(f"[fit] {name}")
            metrics, y_pred = eval_model(name, mdl, X_train, y_train, X_test, y_test)
            leaderboard.append(metrics)

            # Save model + predictions
            dump(mdl, out_dir / f"{name}.joblib")
            preds = pd.DataFrame({
                "y_true": y_test.values,
                "y_pred": y_pred,
                "Q_error": q_error(y_test.values, y_pred)
            })
            preds.to_csv(out_dir / f"predictions_{name}.csv", index=False)

        # Auto mode (optional)
        if args.auto:
            print("[auto] Running PyCaret compare_models(top-10)…")
            lb = run_pycaret(X, y, out_dir, n_select=10, seed=args.seed)
            # We don’t mix PyCaret metrics into our leaderboard to avoid apples-oranges folds,
            # but we save its CSV in out_dir for inspection.

        # Save leaderboard for this dataset
        lb_df = pd.DataFrame(leaderboard)
        lb_df = lb_df.sort_values(["QErr_median", "MAE", "RMSE"])
        lb_df.to_csv(out_dir / "leaderboard.csv", index=False)

        # Tag with dataset and accumulate
        lb_df2 = lb_df.copy()
        lb_df2.insert(0, "dataset", ds)
        all_rows.append(lb_df2)

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(out_root / "leaderboard_all.csv", index=False)
        print(f"\n[done] Wrote {out_root/'leaderboard_all.csv'}")

if __name__ == "__main__":
    main()

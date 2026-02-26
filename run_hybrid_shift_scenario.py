#!/usr/bin/env python3
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models import EquiWidthHistogram, HybridEstimator, summarize
from benchmark_utils import load_and_filter_workload, save_benchmark_results


def load_dataset_meta(dataset_dir: str, dist: str):
    """Loads the meta.pkl containing dataset statistics and frequency arrays."""
    meta_path = Path(dataset_dir) / dist / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}. Run datasets.py first."
        )
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def detect_drift(
    hybrid_est: HybridEstimator,
    target_freq: np.ndarray,
    target_mn: int,
    rng: np.random.Generator,
    threshold: float = 1.5,
) -> list:
    """
    Evaluates the existing ML models on the new data frequency.
    Returns a list of bucket indices where the median Q-Error exceeds the threshold.
    """
    bad_buckets = []
    for i, b in enumerate(hybrid_est.buckets):
        if b.count == 0:
            continue

        val_queries = hybrid_est._generate_bucket_queries(
            b, 200, rng, target_freq, target_mn
        )
        if not val_queries:
            continue

        q_lo = np.array([q.low for q in val_queries])
        q_hi = np.array([q.high for q in val_queries])

        b_lo_idx, b_hi_idx = b.lo - target_mn, b.hi - target_mn
        b_ps = np.cumsum(target_freq[b_lo_idx : b_hi_idx + 1])
        width = b.hi - b.lo + 1

        q_m, _ = hybrid_est._eval_model_q_error_vec(
            hybrid_est.models.get(i), q_lo, q_hi, b, b_ps, b.lo, width
        )

        if q_m > threshold:
            bad_buckets.append(i)

    return bad_buckets


def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid Model Data Shift Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Base path to dataset directories (e.g., data/generated/60000000_hard)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        help="Direct path to the workload.csv file",
    )
    parser.add_argument(
        "--init-dist",
        type=str,
        default="normal",
        help="Initial distribution to train on",
    )
    parser.add_argument(
        "--target-dists",
        type=str,
        nargs="+",
        default=["uniform", "zipf", "anti_zipf"],
        help="Distributions to shift to",
    )
    parser.add_argument(
        "--points", type=int, default=20, help="Points per bucket for training"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=1.5,
        help="Q-Error threshold to trigger bucket finetuning",
    )
    args = parser.parse_args()

    out_dir = Path("results/shift_scenario")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Initialize a list to hold our summary table rows
    summary_records = []

    # ==========================================
    # Phase 1: Initial Training
    # ==========================================
    print(
        f"\n{'=' * 60}\nPHASE 1: Initial Training on '{args.init_dist.upper()}'\n{'=' * 60}"
    )

    init_mn, init_mx, init_N, init_freq, _, init_k, _, _ = load_dataset_meta(
        args.dataset, args.init_dist
    )
    print(f"Using Freedman-Diaconis Bin Count: {init_k}")

    init_hist = EquiWidthHistogram.build(init_mn, init_mx, init_k, init_freq)
    model_init = HybridEstimator(init_hist.buckets)

    t_train_init = model_init.train(init_freq, init_mn, args.points, rng)
    print(f"Initial Training Time: {t_train_init:.4f}s")

    q_init, y_true_init = load_and_filter_workload(
        args.workload, init_mn, init_mx, init_freq
    )

    t0 = time.perf_counter()
    y_pred_init = model_init.predict_batch(q_init) / init_N
    inf_time_init = time.perf_counter() - t0

    m_init = summarize(y_true_init, y_pred_init, f"Init_{args.init_dist}")
    print(
        f"Initial Performance: Median QErr = {m_init['QErr_median']:.4f}, p95 QErr = {m_init['QErr_p95']:.4f}"
    )

    # ==========================================
    # Phase 2 & 3: Shift and Finetune Loop
    # ==========================================
    for target_dist in args.target_dists:
        print(
            f"\n{'=' * 60}\nSHIFT SCENARIO: {args.init_dist} -> {target_dist}\n{'=' * 60}"
        )
        tgt_mn, tgt_mx, tgt_N, tgt_freq, _, _, _, _ = load_dataset_meta(
            args.dataset, target_dist
        )

        q_tgt, y_true_tgt = load_and_filter_workload(
            args.workload, tgt_mn, tgt_mx, tgt_freq
        )

        # ---------------------------------------------------------
        # Phase 2: Zero-Shot Degradation
        # ---------------------------------------------------------
        tgt_hist = EquiWidthHistogram.build(tgt_mn, tgt_mx, init_k, tgt_freq)

        model_shifted = HybridEstimator(
            tgt_hist.buckets, models=dict(model_init.models)
        )
        model_shifted._bake_vectorized_data()

        t0 = time.perf_counter()
        y_pred_shifted = model_shifted.predict_batch(q_tgt) / tgt_N
        inf_time_shifted = time.perf_counter() - t0

        m_shifted = summarize(y_true_tgt, y_pred_shifted, f"Shifted_to_{target_dist}")
        print(
            f"[Before Finetune] Median QErr: {m_shifted['QErr_median']:.4f} | p95 QErr: {m_shifted['QErr_p95']:.4f}"
        )

        # ---------------------------------------------------------
        # Phase 3: Drift Detection & Finetuning
        # ---------------------------------------------------------
        print("Detecting drifted buckets...")
        bad_buckets = detect_drift(
            model_shifted, tgt_freq, tgt_mn, rng, threshold=args.drift_threshold
        )
        percent_bad = (len(bad_buckets) / init_k) * 100
        print(
            f"Detected {len(bad_buckets)} / {init_k} buckets ({percent_bad:.1f}%) exceeding QErr > {args.drift_threshold}"
        )

        if len(bad_buckets) > 0:
            print(f"Finetuning {len(bad_buckets)} buckets...")
            t_finetune = model_shifted.train(
                tgt_freq, tgt_mn, args.points, rng, bucket_indices=bad_buckets
            )
            print(
                f"Finetune Time: {t_finetune:.4f}s (vs {t_train_init:.4f}s full train)"
            )
        else:
            print("No finetuning required!")
            t_finetune = 0.0

        t0 = time.perf_counter()
        y_pred_finetuned = model_shifted.predict_batch(q_tgt) / tgt_N
        inf_time_finetuned = time.perf_counter() - t0

        m_finetune = summarize(y_true_tgt, y_pred_finetuned, f"Finetuned_{target_dist}")
        print(
            f"[After Finetune]  Median QErr: {m_finetune['QErr_median']:.4f} | p95 QErr: {m_finetune['QErr_p95']:.4f}"
        )

        # Append exhaustive unrounded data to summary table
        summary_records.append(
            {
                "Dataset Count": init_N,
                "Workload Count": len(q_tgt),
                "Init Dist": args.init_dist,
                "Shift Dist": target_dist,
                # Init Metrics
                "Init Train Time (s)": t_train_init,
                "Init Infer Time (s)": inf_time_init,
                "Init Avg QErr": m_init["QErr_avg"],
                "Init 25% QErr": m_init["QErr_p25"],
                "Init Median QErr": m_init["QErr_median"],
                "Init 75% QErr": m_init["QErr_p75"],
                "Init 95% QErr": m_init["QErr_p95"],
                # Shift Metrics
                "Shift Infer Time (s)": inf_time_shifted,
                "Shift Avg QErr": m_shifted["QErr_avg"],
                "Shift 25% QErr": m_shifted["QErr_p25"],
                "Shift Median QErr": m_shifted["QErr_median"],
                "Shift 75% QErr": m_shifted["QErr_p75"],
                "Shift 95% QErr": m_shifted["QErr_p95"],
                # Finetune Metrics
                "FT Train Time (s)": t_finetune,
                "FT Infer Time (s)": inf_time_finetuned,
                "FT Avg QErr": m_finetune["QErr_avg"],
                "FT 25% QErr": m_finetune["QErr_p25"],
                "FT Median QErr": m_finetune["QErr_median"],
                "FT 75% QErr": m_finetune["QErr_p75"],
                "FT 95% QErr": m_finetune["QErr_p95"],
            }
        )

        # Save individual iteration results for charting
        workload_name = Path(args.workload).stem
        save_benchmark_results(
            out_dir,
            workload_name,
            f"Shift_{args.init_dist}_to_{target_dist}",
            q_tgt,
            y_true_tgt,
            y_pred_shifted,
            {
                "median_q_error": m_shifted["QErr_median"],
                "p95_q_error": m_shifted["QErr_p95"],
            },
        )
        save_benchmark_results(
            out_dir,
            workload_name,
            f"Finetuned_{target_dist}",
            q_tgt,
            y_true_tgt,
            y_pred_finetuned,
            {
                "median_q_error": m_finetune["QErr_median"],
                "p95_q_error": m_finetune["QErr_p95"],
                "train_time": t_finetune,
            },
        )

    # ==========================================
    # Final Summary Table Output
    # ==========================================
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        print("\n" + "=" * 120)
        print("SHIFT SCENARIO SUMMARY TABLE")
        print("=" * 120)
        print(df_summary.to_string(index=False))
        print("=" * 120)

        # Save to CSV
        summary_csv_path = out_dir / f"shift_summary_{args.init_dist}.csv"
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"\nSaved raw summary table to: {summary_csv_path}")


if __name__ == "__main__":
    main()

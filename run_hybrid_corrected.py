#!/usr/bin/env python3
"""
Static runner for Hybrid-Corrected (models/hybrid_corrected.py) — a labelled MODIFICATION of the
canonical Hybrid FD CDF: inclusive-interval fix only. Same CLI, protocol and outputs as
run_hybrid.py (it runs that file's run_logic with the estimator class swapped), written as
summary_Hybrid-Corrected.json / bucket_table_Hybrid-Corrected.csv.

    --variant corrected               (default) inference + validation fix
    --variant corrected-predict-only  diagnostic: inference fix only -> Hybrid-Corrected-PredictOnly
"""
import run_hybrid
from benchmark_utils import get_common_parser, run_benchmark_suite
from models.hybrid_corrected import VARIANTS

NAMES = {"corrected": "Hybrid-Corrected", "corrected-predict-only": "Hybrid-Corrected-PredictOnly"}


def main():
    parser = get_common_parser("Run Hybrid-Corrected (inclusive-interval fix of Hybrid FD CDF)")
    parser.add_argument("--variant", choices=list(NAMES), default="corrected")
    parser.add_argument("--points", type=int, default=200, help="Points per bucket")
    parser.add_argument("--ident", type=float, default=1e-4, help="Identity threshold")
    parser.add_argument("--penalty", type=float, default=1.5, help="Model penalty")
    args = parser.parse_args()
    # run_hybrid's options that the corrected variant keeps at their canonical defaults
    args.min_sample_points, args.partition, args.parametric_only, args.debug_plot = 0, "equiwidth", False, False

    run_hybrid.HybridEstimator = VARIANTS[args.variant]
    run_hybrid.model_name_for = lambda a: NAMES[a.variant]
    run_benchmark_suite(args, run_hybrid.run_logic)


if __name__ == "__main__":
    main()

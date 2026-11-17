#!/usr/bin/env python3
"""
Drift runner for Hybrid-Corrected (models/hybrid_corrected.py) — a labelled MODIFICATION of the
canonical Hybrid FD CDF: inclusive-interval fix only. Shock -> FT -> RB, checkpoint, marking, outputs:
exactly run_hybrid_drift.py (its HybridDrift with the estimator class swapped); the output folder is
gradual_<init>_to_<target>_Hybrid-Corrected.
"""
import run_hybrid_drift
from drift_harness import drift_parser, run_drift
from models.hybrid_corrected import VARIANTS

NAMES = {"corrected": "Hybrid-Corrected", "corrected-predict-only": "Hybrid-Corrected-PredictOnly"}


def main():
    parser = drift_parser("Hybrid-Corrected under data drift: Shock -> FT -> Rebuild (checkpoint 100 000 rows)")
    parser.add_argument("--points", type=int, default=200, help="CDF points sampled per bucket")
    parser.add_argument("--variant", choices=list(NAMES), default="corrected")
    args = parser.parse_args()
    run_hybrid_drift.HybridEstimator = VARIANTS[args.variant]
    run_drift(args, NAMES[args.variant], run_hybrid_drift.HybridDrift(args.points))


if __name__ == "__main__":
    main()

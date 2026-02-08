import json
import os
import pandas as pd

def aggregate_tuning():
    experiments = ["Baseline", "DenseScaling", "AggressiveModels", "Combined"]
    distributions = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    
    all_data = []
    
    for exp in experiments:
        exp_dir = f"results/tune_{exp}/10000000"
        for dist in distributions:
            summary_path = os.path.join(exp_dir, dist, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {})
                    # Extract Hybrid model results
                    hybrid_results = metrics.get("Hybrid")
                    if hybrid_results:
                        all_data.append({
                            "Experiment": exp,
                            "Distribution": dist,
                            "Avg Q-Error": hybrid_results["avg_q_error"],
                            "Median Q-Error": hybrid_results["median_q_error"],
                            "P95 Q-Error": hybrid_results["p95_q_error"],
                            "Train Time": hybrid_results.get("build_time_total", hybrid_results["build_time"])
                        })
    
    df = pd.DataFrame(all_data)
    print("\n>>> Tuning Comparison (Hybrid Model) <<<")
    # Pivot to see Medians by experiment and distribution
    pivot_median = df.pivot(index='Distribution', columns='Experiment', values='Median Q-Error')
    print("\n--- Median Q-Error ---")
    print(pivot_median)
    
    pivot_avg = df.pivot(index='Distribution', columns='Experiment', values='Avg Q-Error')
    print("\n--- Avg Q-Error ---")
    print(pivot_avg)
    
    pivot_p95 = df.pivot(index='Distribution', columns='Experiment', values='P95 Q-Error')
    print("\n--- P95 Q-Error ---")
    print(pivot_p95)

if __name__ == "__main__":
    aggregate_tuning()

import json
import os
import pandas as pd

def compare_60m():
    # Comparing Tuned vs Baseline (Exp 16 - though Exp 16 might have different structure now)
    # I'll just gather the Tuned 60M results first.
    dist_list = ["uniform", "normal", "zipf", "sparse_cluster", "anti_zipf"]
    rows = 60000000
    
    tuned_dir = "results/final_tuned_60m/60000000"
    
    print("\n>>> Final 60M Tuned Results (Hybrid Model) <<<")
    results = []
    for dist in dist_list:
        summary_path = os.path.join(tuned_dir, dist, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
                hybrid = data["metrics"]["Hybrid"]
                equi = data["metrics"]["Equi-Width"]
                results.append({
                    "Distribution": dist,
                    "Hybrid Median": hybrid["median_q_error"],
                    "Hybrid P95": hybrid["p95_q_error"],
                    "EquiWidth Median": equi["median_q_error"],
                    "Improvement (Median)": f"{((equi['median_q_error'] - hybrid['median_q_error']) / equi['median_q_error'] * 100):.2f}%"
                })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    compare_60m()

import json
import os
import pandas as pd

def get_metrics(exp_id):
    results = []
    base_path = f"results/{exp_id}/60000000"
    for dist in os.listdir(base_path):
        summary_path = os.path.join(base_path, dist, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
                hybrid_data = data.get('metrics', {}).get('Hybrid')
                if hybrid_data:
                    results.append({
                        "Distribution": dist,
                        "Median Q-Err": hybrid_data.get('median_q_error'),
                        "Build Time": hybrid_data.get('build_time'),
                        "Inference Time": hybrid_data.get('infer_time')
                    })
    return pd.DataFrame(results)

df_tourn = get_metrics("final_tuned_60m")
df_water = get_metrics("hybrid_fast_predict")

# Merge
comparison = pd.merge(df_tourn, df_water, on="Distribution", suffixes=("_Tourn", "_Water"))

# Calculate improvements
comparison["Train Speedup (%)"] = (1 - comparison["Build Time_Water"] / comparison["Build Time_Tourn"]) * 100
comparison["Inf Slowdown (%)"] = (comparison["Inference Time_Water"] / comparison["Inference Time_Tourn"] - 1) * 100

print("=== Hybrid Model: Tournament vs Waterfall (60M Rows) ===")
print(comparison[["Distribution", "Median Q-Err_Tourn", "Median Q-Err_Water", "Build Time_Tourn", "Build Time_Water", "Train Speedup (%)"]])
print("\n=== Inference Latency Check ===")
print(comparison[["Distribution", "Inference Time_Tourn", "Inference Time_Water", "Inf Slowdown (%)"]])

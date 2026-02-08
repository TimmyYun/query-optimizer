
import pandas as pd
import numpy as np
from pathlib import Path
from workload import load_workload_csv

def analyze_outliers():
    # ADJUST PATHS AS NEEDED
    results_path = Path("results/20260207_153135/1000000/zipf/100000.csv")
    workload_path = Path("workload/100000_skewed.csv")
    
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return
        
    df = pd.read_csv(results_path)
    
    # Filter for Hybrid
    # Check exact model name in CSV
    hybrid_df = df[df['Model'].str.contains('Hybrid')]
    if hybrid_df.empty:
        print("No Hybrid model results found.")
        print("Available models:", df['Model'].unique())
        return

    # Find worst outliers
    worst = hybrid_df.sort_values('Q_Error', ascending=False).head(20)
    
    # Load Workload
    try:
        queries = load_workload_csv(workload_path)
    except Exception as e:
        print(f"Failed to load workload: {e}")
        return
    
    print(f"Top 20 Worst Queries (Max Q-Error: {worst['Q_Error'].max()}):")
    print("-" * 100)
    print(f"{'ID':<6} | {'Range':<20} | {'Truth':<12} | {'Pred':<12} | {'Q-Error':<10}")
    print("-" * 100)
    
    for idx, row in worst.iterrows():
        try:
            q_id = int(row['Query_ID'])
            if q_id < len(queries):
                q = queries[q_id]
                range_str = f"[{q.low}, {q.high}]"
                print(f"{q_id:<6} | {range_str:<20} | {row['Truth']:<12.8f} | {row['Prediction']:<12.8f} | {row['Q_Error']:<10.1f}")
            else:
                print(f"{q_id:<6} | {'UNKNOWN':<20} | {row['Truth']:<12.8f} | {row['Prediction']:<12.8f} | {row['Q_Error']:<10.1f}")
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")

if __name__ == "__main__":
    analyze_outliers()

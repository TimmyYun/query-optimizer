import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def analyze_specifics():
    try:
        df = pd.read_excel("Results.xlsx")
        
        # Select relevant columns for clarity
        cols_of_interest = ['Dataset', 'Approach', 'Init QErr Median', 'Init MAE', 'Drift MAE', 'Final MAE', 'Init Memory (KB)']
        
        # Filter for Zipf and Sparse Cluster to check the "7000" claim
        subset = df[df['Dataset'].astype(str).str.contains("Zipf|Sparse", case=False, regex=True)]
        
        print(subset[cols_of_interest].to_string())

        # Also get summary for Operational Table (Training times etc)
        # Just grab the first row of Uniform for "Static" vs "Hybrid" general comparison if needed, 
        # or average across all datasets.
        print("\n--- Operational Averages (All Datasets) ---")
        grp = df.groupby('Approach')[['Init Training time (s)', 'Init Inference time (s)', 'Init Memory (KB)']].mean()
        print(grp.to_string())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_specifics()

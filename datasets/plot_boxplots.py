import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def generate_boxplots(csv_path="static_errors.csv", output_path="plots/static_boxplots.png", title="Q-Error Distribution"):
    # Try Parquet first, then CSV
    # If the user passed a specific path, use it. If default, check for parquet fallback.
    
    if csv_path == "static_errors.csv" and not os.path.exists(csv_path):
        if os.path.exists("static_errors.parquet"):
            csv_path = "static_errors.parquet"
            
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    if csv_path.endswith('.parquet'):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)

    # Ensure QErr is numeric
    if "QErr" in df.columns:
        df["QErr"] = pd.to_numeric(df["QErr"], errors='coerce')
        
    print(f"Columns: {df.columns}")
    print(f"Distributions: {df['Distribution'].unique()}")
    print(f"Models: {df['Model'].unique()}")

    # Combined Plot
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    sns.boxplot(
        data=df, 
        x="Distribution", 
        y="QErr", 
        hue="Model", 
        showfliers=False, 
        palette="Set2",
        width=0.8
    )

    plt.yscale("log")
    plt.title(title, fontsize=16)
    plt.ylabel("Q-Error (Log Scale)", fontsize=14)
    plt.xlabel("Distribution", fontsize=14)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save combined plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Combined Boxplot saved to {output_path}")
    plt.close()

    # Per-Distribution Plots
    distributions = df['Distribution'].unique()
    for dist in distributions:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        dist_df = df[df['Distribution'] == dist]
        
        sns.boxplot(
            data=dist_df, 
            x="Model", 
            y="QErr", 
            showfliers=False, 
            palette="Set2",
            width=0.6
        )
        
        plt.yscale("log")
        plt.title(f"{title} - {dist}", fontsize=16)
        plt.ylabel("Q-Error (Log Scale)", fontsize=14)
        plt.xlabel("Model", fontsize=14)
        
        # Construct path for individual plot
        path_obj = Path(output_path)
        dist_output_path = path_obj.parent / f"dist_{dist}.png"
        
        plt.savefig(dist_output_path, dpi=300, bbox_inches="tight")
        print(f"Distribution plot saved to {dist_output_path}")
        plt.close()

if __name__ == "__main__":
    generate_boxplots()

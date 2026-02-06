import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def generate_boxplots(csv_path="experiments/static_errors.csv", output_path="plots/static_boxplots.png", title="Q-Error Distribution"):
    # Try Parquet first, then CSV
    # If the user passed a specific path, use it. If default, check for parquet fallback.
    
    if csv_path == "experiments/static_errors.csv" and not os.path.exists(csv_path):
        if os.path.exists("experiments/static_errors.parquet"):
            csv_path = "experiments/static_errors.parquet"
            
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

    # Use catplot to create a faceted boxplot
    # col="Distribution": creates a subplot for each distribution
    # sharey=False: allows each subplot to have its own y-axis scale, preventing squishing
    # col_wrap: wraps columns to keep the plot compact
    
    g = sns.catplot(
        data=df, 
        x="Model", 
        y="QErr", 
        col="Distribution", 
        kind="box", 
        showfliers=False, 
        palette="Set2",
        sharey=False,
        col_wrap=3,
        height=4, 
        aspect=1.2
    )

    g.set(yscale="log")
    g.set_axis_labels("Model", "Q-Error (Log Scale)")
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(title, fontsize=16)
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Faceted boxplot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_boxplots()

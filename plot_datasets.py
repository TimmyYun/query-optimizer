
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_distributions():
    distributions = ['uniform', 'normal', 'zipf', 'sparse_cluster', 'anti_zipf']
    cache_dir = Path("datasets/files/generated")
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    
    for dist in distributions:
        print(f"Plotting {dist}...")
        ds_path = cache_dir / f"bench_{dist}.csv"
        
        if not ds_path.exists():
            print(f"Warning: {ds_path} not found. Skipping.")
            continue
            
        try:
            # Read first 100k rows for plotting speed if large, or full
            df = pd.read_csv(ds_path, header=None, names=['v'])
            
            plt.figure(figsize=(10, 6))
            plt.hist(df['v'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f"Distribution: {dist}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(axis='y', alpha=0.5)
            
            out_path = plot_dir / f"dist_{dist}.png"
            plt.savefig(out_path)
            plt.close()
            print(f"Saved plot to {out_path}")
            
        except Exception as e:
            print(f"Error plotting {dist}: {e}")

if __name__ == "__main__":
    plot_distributions()


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
            
            # Calculate statistics
            stats_text = (
                f"Count (N): {len(df)}\n"
                f"Min: {df['v'].min()}\n"
                f"Max: {df['v'].max()}\n"
                f"Mean: {df['v'].mean():.2f}\n"
                f"Std Dev: {df['v'].std():.2f}\n"
                f"Skew: {df['v'].skew():.2f}\n"
                f"Kurtosis: {df['v'].kurt():.2f}"
            )

            plt.figure(figsize=(12, 7))
            
            # Plot Histogram
            # Use log scale for Zipf to see the tail
            use_log = (dist == 'zipf')
            plt.hist(df['v'], bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=use_log)
            
            plt.title(f"Distribution: {dist}")
            plt.xlabel("Value")
            plt.ylabel("Frequency" + (" (Log Scale)" if use_log else ""))
            plt.grid(axis='y', alpha=0.5)
            
            # Fix X-Axis to global domain to show relative width (Anti-Zipf vs Uniform)
            plt.xlim(0, 200_000)

            # Add text box with statistics
            plt.gcf().text(0.75, 0.5, stats_text, fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Adjust layout to make room for text if needed (though .text with gcf coordinates overlays)
            plt.subplots_adjust(right=0.7) # Make room on the right for the stats box if we used axes coordinates, but here checking visuals
            
            out_path = plot_dir / f"dist_{dist}.png"
            plt.savefig(out_path, bbox_inches='tight') # bbox_inches=tight helps save the extra text
            plt.close()
            print(f"Saved plot to {out_path}")
            
        except Exception as e:
            print(f"Error plotting {dist}: {e}")

if __name__ == "__main__":
    plot_distributions()

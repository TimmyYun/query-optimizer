
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_data_distribution(vals, dist_name, output_path):
    """
    Plots the histogram and scatter plot of a given dataset.
    """
    print(f"Plotting distribution for {dist_name} to {output_path}...")
    
    # Use 1M values for histogram and stats to keep it fast
    if len(vals) > 1_000_000:
        plot_vals_hist = np.random.choice(vals, 1_000_000, replace=False)
    else:
        plot_vals_hist = vals

    # Use 10k values for scatter plot to avoid overplotting and slow rendering
    if len(vals) > 10_000:
        indices = np.random.choice(len(vals), 10_000, replace=False)
        plot_vals_scatter = vals[indices]
        scatter_indices = indices
    else:
        plot_vals_scatter = vals
        scatter_indices = np.arange(len(vals))

    # Calculate statistics
    stats_text = (
        f"Count (N): {len(vals)}\n"
        f"Min: {np.min(vals)}\n"
        f"Max: {np.max(vals)}\n"
        f"Mean: {np.mean(vals):.2f}\n"
        f"Std Dev: {np.std(vals):.2f}"
    )

    plt.figure(figsize=(10, 6))
    
    # Use log scale for Zipf to see the tail
    use_log = (dist_name.lower() == 'zipf')
    plt.hist(plot_vals_hist, bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=use_log)
    
    plt.title(f"Distribution: {dist_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency" + (" (Log Scale)" if use_log else ""))
    plt.grid(axis='y', alpha=0.3)
    
    # Standardize X-Axis range for synthetic data if not imdb/census
    if dist_name.lower() not in ['imdb', 'census']:
        plt.xlim(0, 200_000)
    
    # Add text box with statistics
    plt.gcf().text(0.78, 0.6, stats_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {output_path}")

def plot_distributions():
    distributions = ['uniform', 'normal', 'zipf', 'exponential', 'lognormal']
    cache_dir = Path("datasets/files/generated")
    plot_dir = Path("plots")
    
    for dist in distributions:
        # Check various possible row count folders or base folder
        # For simplicity, look in the new structure if possible
        # However, run_single_experiment will call this directly now.
        pass

if __name__ == "__main__":
    # Example usage
    # plot_distributions()
    pass

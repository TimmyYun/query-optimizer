
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_data_distribution(vals, dist_name, output_path):
    """
    Plots the histogram and statistics of a given dataset.
    """
    print(f"Plotting distribution for {dist_name} to {output_path}...")
    
    # Use first 1M values for plotting if too large to keep it fast
    if len(vals) > 1_000_000:
        plot_vals = np.random.choice(vals, 1_000_000, replace=False)
    else:
        plot_vals = vals

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
    plt.hist(plot_vals, bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=use_log)
    
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
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
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

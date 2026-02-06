import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_data_distribution(vals, dist_name, output_path):
    print(f"Plotting distribution for {dist_name} to {output_path}...")
    plot_vals = np.random.choice(vals, min(len(vals), 1_000_000), replace=False)
    stats_text = (f"Count (N): {len(vals)}\nMin: {np.min(vals)}\nMax: {np.max(vals)}\n"
                  f"Mean: {np.mean(vals):.2f}\nStd Dev: {np.std(vals):.2f}")

    plt.figure(figsize=(10, 6))
    use_log = (dist_name.lower() == 'zipf')
    plt.hist(plot_vals, bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=use_log)
    plt.title(f"Distribution: {dist_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency" + (" (Log Scale)" if use_log else ""))
    plt.grid(axis='y', alpha=0.3)
    if dist_name.lower() not in ['imdb', 'census']: plt.xlim(0, 200_000)
    plt.gcf().text(0.78, 0.6, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_boxplots(csv_path, output_path, title="Q-Error Distribution"):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if df.empty: return

    plt.figure(figsize=(12, 6))
    g = sns.catplot(data=df, x="Model", y="QErr", col="Distribution", kind="box", 
                    showfliers=False, palette="Set2", sharey=False, col_wrap=3, height=4, aspect=1.2)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Faceted boxplot saved to {output_path}")

def plot_model_comparison(csv_path: str, output_path: str, title: str):
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Model", y="QErr", showfliers=False, palette="Set2")
    plt.title(title)
    plt.ylabel("Q-Error")
    plt.yscale("log")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Single distribution boxplot saved to {output_path}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Setup
OUT_DIR = Path("experiments/artifacts_thesis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def load_data():
    df = pd.read_csv("final_benchmark.csv")
    # Clean up names for plotting
    df['Approach'] = df['Approach'].replace({
        'Static Equi-Width': 'Static',
        'EquiHist (Online)': 'EquiHist',
        'Hybrid (Repaired)': 'Hybrid'
    })
    return df

def plot_static_accuracy(df):
    # Filter for initial build (Rows inserted == 0)
    subset = df[df['Rows inserted'] == 0].copy()
    
    # Figure 1: Median Q-Error by Dataset
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=subset, x='Dataset', y='Init QErr Median', hue='Approach', palette='viridis')
    plt.yscale('log')
    plt.title('Static Accuracy: Median Q-Error (Lower is Better)')
    plt.ylabel('Median Q-Error (Log Scale)')
    plt.xlabel('Dataset Distribution')
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=8)
                   
    plt.savefig(OUT_DIR / "fig_1_static_accuracy.png", dpi=300)
    plt.close()
    print("Generated fig_1_static_accuracy.png")

def plot_drift_resilience(df):
    # Figure 2: Resilience to Drift (Normal Distribution)
    # Filter for Normal dataset and drift scenarios
    subset = df[(df['Dataset'] == 'Normal') & (df['Rows inserted'] > 0)].copy()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x='Drift Scenario', y='Drift MAE', hue='Approach', style='Approach', markers=True, markersize=8, palette='viridis', linewidth=2)
    
    plt.title('Drift Resilience: Error vs. Data Shift (Normal Dist)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('Drift Shift Magnitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(OUT_DIR / "fig_2_drift_resilience.png", dpi=300)
    plt.close()
    print("Generated fig_2_drift_resilience.png")

def plot_operational_overhead(df):
    # Figure 3: Operational Overhead (Memory and Latency)
    # Average across all unique (Dataset, Approach) keys from initial phase
    subset = df[df['Rows inserted'] == 0].groupby('Approach')[['Init Memory (KB)', 'Init Inference time (s)']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Memory
    sns.barplot(data=subset, x='Approach', y='Init Memory (KB)', ax=axes[0], palette='viridis')
    axes[0].set_title('Average Memory Footprint')
    axes[0].set_ylabel('Memory (KB)')
    axes[0].set_xlabel('')
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f} KB', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

    # Latency
    sns.barplot(data=subset, x='Approach', y='Init Inference time (s)', ax=axes[1], palette='viridis')
    axes[1].set_title('Average Inference Latency')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_xlabel('')
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height()*1000:.1f} ms', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_3_operational.png", dpi=300)
    plt.close()
    print("Generated fig_3_operational.png")

def main():
    print("Loading data...")
    try:
        df = load_data()
    except FileNotFoundError:
        print("Error: final_benchmark.csv not found.")
        return

    print("Generating figures...")
    plot_static_accuracy(df)
    plot_drift_resilience(df)
    plot_operational_overhead(df)
    print(f"All figures saved to {OUT_DIR}")

if __name__ == "__main__":
    main()

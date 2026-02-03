import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
DATA_FILE = "datasets/files/benchmarks/ablation_results.csv"

def plot_exp(df, exp_name, x_col, x_label, title, filename):
    subset = df[df["Exp"].str.startswith(exp_name)].copy()
    if subset.empty: return
    
    # Sort by numeric value of variation param
    subset["Var_Value"] = pd.to_numeric(subset["Var_Value"])
    subset = subset.sort_values("Var_Value")
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(subset["Var_Value"], subset["Static_MAE"], marker='o', label="Static (Stale)", linestyle='--')
    plt.plot(subset["Var_Value"], subset["EquiHist_MAE"], marker='s', label="EquiHist (Online)")
    plt.plot(subset["Var_Value"], subset["Hybrid_Repair_MAE"], marker='^', label="Hybrid (Repaired)", linewidth=2)
    
    plt.xlabel(x_label)
    plt.ylabel("MAE (Mean Absolute Error)")
    plt.title(title)
    plt.legend()
    plt.ylim(bottom=0)
    
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def main():
    if not Path(DATA_FILE).exists():
        print(f"Error: {DATA_FILE} not found")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    # Plot Exp B: NDV
    plot_exp(df, "ExpB", "ndv-threshold", "NDV Threshold (Exact Values)", "MAE vs Sparsity Handling", "plot_exp_b_ndv.png")
    
    # Plot Exp C: Drift
    plot_exp(df, "ExpC", "drift-rows", "Drift Intensity (Rows Inserted)", "MAE vs Drift Intensity (Robustness)", "plot_exp_c_drift.png")
    
    # Plot Exp D: LR
    plot_exp(df, "ExpD", "eh-lr", "Learning Rate", "EquiHist Stability vs Learning Rate", "plot_exp_d_lr.png")

if __name__ == "__main__":
    main()

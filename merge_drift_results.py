import pandas as pd
import os

def merge_csvs():
    # Input paths
    equihist_path = "results/drift_synthetic/equihist_drift/gradual_normal_to_zipf_EquiHist/equihist_drift_analysis.csv"
    equiwidth_path = "results/drift_synthetic/equiwidth_drift/gradual_normal_to_zipf_EquiWidth/equiwidth_drift_analysis.csv"
    hybrid_path = "results/archive/hybrid_drift/gradual_normal_to_zipf_FT_RB/adaptation_ft_rb.csv"
    
    output_path = "results/merged_drift_analysis.csv"
    
    all_data = []
    
    # 1. EquiHist
    if os.path.exists(equihist_path):
        df = pd.read_csv(equihist_path)
        # Shift %,Median Q-Error,P95 Q-Error,Avg Q-Error,Time
        df_processed = pd.DataFrame({
            'model': 'EquiHist',
            'drift': df['Shift %'],
            'median q error': df['Median Q-Error'],
            'p95 q error': df['P95 Q-Error'],
            'time': df['Time'],
            'avg q error': df.get('Avg Q-Error', None)
        })
        all_data.append(df_processed)
        
    # 2. EquiWidth
    if os.path.exists(equiwidth_path):
        df = pd.read_csv(equiwidth_path)
        # Shift %,Median Q-Error,P95 Q-Error,Avg Q-Error,Inference Time
        df_processed = pd.DataFrame({
            'model': 'EquiWidth',
            'drift': df['Shift %'],
            'median q error': df['Median Q-Error'],
            'p95 q error': df['P95 Q-Error'],
            'time': df['Inference Time'],
            'avg q error': df.get('Avg Q-Error', None)
        })
        all_data.append(df_processed)
        
    # 3. Hybrid (FT and RB)
    if os.path.exists(hybrid_path):
        df = pd.read_csv(hybrid_path)
        # Shift %,Shock_Med,Shock_P95,FT_Med,FT_P95,RB_Med,RB_P95,Rebuilt_Count,FT_Time,RB_Time
        
        # Hybrid (FT)
        df_ft = pd.DataFrame({
            'model': 'Hybrid (FT)',
            'drift': df['Shift %'],
            'median q error': df['FT_Med'],
            'p95 q error': df['FT_P95'],
            'time': df['FT_Time'],
            'rebuilt_count': df.get('Rebuilt_Count', None)
        })
        all_data.append(df_ft)
        
        # Hybrid (RB)
        df_rb = pd.DataFrame({
            'model': 'Hybrid (RB)',
            'drift': df['Shift %'],
            'median q error': df['RB_Med'],
            'p95 q error': df['RB_P95'],
            'time': df['RB_Time'],
            'rebuilt_count': df.get('Rebuilt_Count', None)
        })
        all_data.append(df_rb)

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        # Reorder columns to have model, drift, median q error, p95 q error, time first
        cols = ['model', 'drift', 'median q error', 'p95 q error', 'time']
        rest_cols = [c for c in merged_df.columns if c not in cols]
        merged_df = merged_df[cols + rest_cols]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"Merged CSV saved to {output_path}")
    else:
        print("No CSV files found to merge.")

if __name__ == "__main__":
    merge_csvs()

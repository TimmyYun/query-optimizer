import pandas as pd

def merge_results(suffix, output_path):
    # Load data
    df_hist = pd.read_csv(f'results/equihist_drift{suffix}/gradual_normal_to_zipf_EquiHist/equihist_drift_analysis.csv')
    df_width = pd.read_csv(f'results/equiwidth_drift{suffix}/gradual_normal_to_zipf_EquiWidth/equiwidth_drift_analysis.csv')
    df_hybrid = pd.read_csv(f'results/hybrid_drift{suffix}/gradual_normal_to_zipf_FT_RB/adaptation_ft_rb.csv')

    all_data = []

    # Process EquiHist
    for _, row in df_hist.iterrows():
        all_data.append({
            'drift': row['Shift %'],
            'model': 'EquiHist',
            'Median': row['Median Q-Error'],
            'P95': row['P95 Q-Error'],
            'Avg': row['Avg Q-Error'],
            'Time': row['Time'],
            'Count': None
        })

    # Process EquiWidth
    for _, row in df_width.iterrows():
        all_data.append({
            'drift': row['Shift %'],
            'model': 'EquiWidth',
            'Median': row['Median Q-Error'],
            'P95': row['P95 Q-Error'],
            'Avg': row['Avg Q-Error'],
            'Time': row['Inference Time'],
            'Count': None
        })

    # Process Hybrid
    for _, row in df_hybrid.iterrows():
        # Shock
        all_data.append({
            'drift': row['Shift %'],
            'model': 'Hybrid_Shock',
            'Median': row['Shock_Med'],
            'P95': row['Shock_P95'],
            'Avg': row['Shock_Avg'],
            'Time': row['Evaluation Time'],
            'Count': None
        })
        # FT
        all_data.append({
            'drift': row['Shift %'],
            'model': 'Hybrid_FT',
            'Median': row['FT_Med'],
            'P95': row['FT_P95'],
            'Avg': row['FT_Avg'],
            'Time': row['FT_Time'],
            'Count': row['Finetuned_Count']
        })
        # RB
        all_data.append({
            'drift': row['Shift %'],
            'model': 'Hybrid_RB',
            'Median': row['RB_Med'],
            'P95': row['RB_P95'],
            'Avg': row['RB_Avg'],
            'Time': row['RB_Time'],
            'Count': row['Rebuilt_Count']
        })

    df_combined = pd.DataFrame(all_data)

    # Sort
    df_combined['drift_int'] = df_combined['drift'].str.replace('%', '').astype(int)
    # Sort by drift, then by model (so they're grouped logically and in order: Shock, FT, RB)
    model_order = ['EquiHist', 'EquiWidth', 'Hybrid_Shock', 'Hybrid_FT', 'Hybrid_RB']
    df_combined['model'] = pd.Categorical(df_combined['model'], categories=model_order, ordered=True)
    df_combined = df_combined.sort_values(by=['drift_int', 'model']).drop(columns=['drift_int'])

    # Save to CSV
    df_combined.to_csv(output_path, index=False)
    print(f'Combined results saved to {output_path}')
    
if __name__ == '__main__':
    merge_results('', 'results/drift_synthetic/combined_all_drifts.csv')
    merge_results('_5%', 'results/drift_synthetic/combined_all_drifts_5%.csv')
    merge_results('_5%_workload', 'results/drift_synthetic/combined_all_drifts_5%_workload.csv')
    merge_results('_5%_workload_dataset', 'results/drift_synthetic/combined_all_drifts_5%_workload_dataset.csv')

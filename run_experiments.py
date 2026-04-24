import os
import subprocess
import json
import argparse
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def parse_results(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, 'r') as f:
        return json.load(f)

def format_row(model_name, res):
    if not res or 'metrics' not in res:
        return f"{model_name:<12} - - - - - - -"
    m = res['metrics']
    return f"{model_name:<12} {m.get('avg_q_error', 0):.3f} {m.get('p25_q_error', 0):.3f} {m.get('p75_q_error', 0):.3f} {m.get('p95_q_error', 0):.3f} {m.get('median_q_error', 0):.3f} {m.get('train_time', 0):.9f} {m.get('infer_time', 0):.9f}"

def collect_table(dataset_name, models, results_dir):
    lines = [dataset_name]
    for model in models:
        path = os.path.join(results_dir, f"summary_{model}.json")
        res = parse_results(path)
        lines.append(format_row(model, res))
    return lines

def main():
    parser = argparse.ArgumentParser(description="Run Static Experiments for Synthetic and DSB data")
    parser.add_argument('--scenario', type=str, required=True, 
                        choices=['simple_wide', 'simple_narrow', 'hard_wide', 'hard_narrow', 'hard_narrow_data_driven', 'dsb', 'all'],
                        help="The experiment scenario to run.")
    # The user has fixed datasets of 60,000,000 and workloads of 1,000,000.
    # No dynamic generation is needed or allowed.
    args = parser.parse_args()

    models = ["Equi-Width", "Equi-Hist", "Hybrid"]
    scenarios_to_run = []
    
    if args.scenario == 'all':
        scenarios_to_run = ['simple_wide', 'simple_narrow', 'hard_wide', 'hard_narrow', 'hard_narrow_data_driven', 'dsb']
    else:
        scenarios_to_run = [args.scenario]

    final_report = []
    final_report.append("Dataset Model Avg Q-Err 25% 75% 95% Median Train (s) Infer (s)")

    for sc in scenarios_to_run:
        if sc == 'dsb':
            print("\n" + "="*50)
            print(f"Running Scenario: DSB")
            print("="*50)
            datasets = ["ws_item_sk", "cr_item_sk", "cr_returned_time_sk"]
            
            # Ensure DSB data is prepared
            if not os.path.exists("data/dsb/ws_item_sk/meta.pkl"):
                run_cmd("poetry run python prepare_dsb.py")
                
            for ds in datasets:
                ds_path = f"data/dsb/{ds}"
                exp_name = f"dsb_static"
                
                workload_files = list(Path(ds_path).glob("workload_driven_*.csv"))
                if not workload_files:
                    print(f"No workload file found in {ds_path}")
                    continue
                    
                w_file = workload_files[0]
                
                for script, model in zip(["run_equiwidth.py", "run_equihist.py", "run_hybrid.py"], models):
                    run_cmd(f"poetry run python {script} --experiment-name {exp_name} --dataset {ds_path} --workload {w_file}")
                
                final_report.extend(collect_table(ds, models, f"results/{exp_name}/{ds}"))

        else:
            print("\n" + "="*50)
            print(f"Running Scenario: Synthetic - {sc.upper()}")
            print("="*50)
            
            distributions = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
            
            for dist in distributions:
                # 1. Resolve Dataset Path
                if "hard" in sc:
                    ds_path = f"data/generated/60000000_hard/{dist}"
                else:
                    ds_path = f"data/generated/60000000_simple/{dist}"
                    
                # 2. Resolve Workload Path
                if "data_driven" in sc:
                    workload_arg = f"{ds_path}/workload_driven_1000000.csv"
                elif "wide" in sc:
                    workload_arg = "workload/1000000/wide/workload.csv"
                else:
                    workload_arg = "workload/1000000/narrow/workload.csv"
                
                # Check if paths exist to prevent obscure FileNotFoundError crashes
                if not os.path.exists(ds_path):
                    print(f"ERROR: Dataset not found at {ds_path}. Skipping...")
                    continue
                if not os.path.exists(workload_arg):
                    print(f"ERROR: Workload not found at {workload_arg}. Skipping...")
                    continue
                
                # 3. Run models
                exp_name = f"synth_{sc}"
                for script, model in zip(["run_equiwidth.py", "run_equihist.py", "run_hybrid.py"], models):
                    run_cmd(f"poetry run python {script} --experiment-name {exp_name} --dataset {ds_path} --workload {workload_arg}")
                    
                final_report.extend(collect_table(f"{dist}_{sc}", models, f"results/{exp_name}/{dist}"))

    print("\n\n" + "="*80)
    print("FINAL EXPERIMENT RESULTS")
    print("="*80)
    for line in final_report:
        print(line)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

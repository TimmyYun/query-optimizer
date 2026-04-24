import os
import subprocess
import argparse

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Dynamic Drift Experiments")
    parser.add_argument('--scenario', type=str, required=True, 
                        choices=[
                            'drift_10_ds_zipf_wl_normal', 
                            'drift_5_ds_zipf_wl_normal', 
                            'drift_5_ds_zipf_wl_zipf', 
                            'drift_5_ds_add_zipf_wl_zipf', 
                            'all'
                        ],
                        help="The drift scenario to run.")
    args = parser.parse_args()

    scenarios = []
    if args.scenario == 'all':
        scenarios = [
            'drift_10_ds_zipf_wl_normal', 
            'drift_5_ds_zipf_wl_normal', 
            'drift_5_ds_zipf_wl_zipf', 
            'drift_5_ds_add_zipf_wl_zipf'
        ]
    else:
        scenarios = [args.scenario]

    # Map scenario to its configuration
    config = {
        'drift_10_ds_zipf_wl_normal': {
            'shift_dir': 'data/generated/60000000_hard/shift_normal_to_zipf_10%',
            'static_workload': 'data/generated/60000000_hard/normal/workload_driven_1000000.csv',
            'shift_workload': False,
            'workload_count': '1000000'
        },
        'drift_5_ds_zipf_wl_normal': {
            'shift_dir': 'data/generated/60000000_hard/shift_normal_to_zipf_5%',
            'static_workload': 'data/generated/60000000_hard/normal/workload_driven_1000000.csv',
            'shift_workload': False,
            'workload_count': '1000000'
        },
        'drift_5_ds_zipf_wl_zipf': {
            'shift_dir': 'data/generated/60000000_hard/shift_normal_to_zipf_5%',
            'static_workload': None,
            'shift_workload': True,
            'workload_count': '1000000'
        },
        'drift_5_ds_add_zipf_wl_zipf': {
            'shift_dir': 'data/generated/60000000_hard/shift_normal_to_zipf_5%_dataset',
            'static_workload': None,
            'shift_workload': True,
            'workload_count': '1000000'
        }
    }

    models = ["run_equiwidth_drift.py", "run_equihist_drift.py", "run_hybrid_drift.py"]

    for sc in scenarios:
        print("\n" + "="*50)
        print(f"Running Drift Scenario: {sc}")
        print("="*50)
        
        cfg = config[sc]
        shift_dir = cfg['shift_dir']
        
        if not os.path.exists(shift_dir):
            print(f"Warning: Shift directory {shift_dir} does not exist. Skipping.")
            continue
            
        for script in models:
            cmd = [
                f"poetry run python {script}",
                f"--experiment-name {sc}",
                f"--shift-dir {shift_dir}",
                f"--workload {cfg['workload_count']}" # Required param, fallback to string parsing if not static
            ]
            
            if cfg['shift_workload']:
                cmd.append("--shift-workload")
            
            if cfg['static_workload']:
                cmd.append(f"--static-workload-path {cfg['static_workload']}")
                
            run_cmd(" ".join(cmd))

if __name__ == "__main__":
    main()

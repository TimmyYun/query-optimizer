import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from registry import MODELS, DEFAULT_MODELS, STATIC_SCENARIOS


def run_cmd(cmd, dry_run=False):
    print(f"\n[RUNNING] {shlex.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)

def parse_results(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, 'r') as f:
        return json.load(f)

def format_row(model_name, res):
    if not res or 'metrics' not in res:
        return f"{model_name:<16} - - - - - - -"
    m = res['metrics']
    return f"{model_name:<16} {m.get('avg_q_error', 0):.3f} {m.get('p25_q_error', 0):.3f} {m.get('p75_q_error', 0):.3f} {m.get('p95_q_error', 0):.3f} {m.get('median_q_error', 0):.3f} {m.get('train_time', 0):.9f} {m.get('infer_time', 0):.9f}"

def collect_table(dataset_name, models, results_dir):
    lines = [dataset_name]
    for model in models:
        path = os.path.join(results_dir, f"summary_{model}.json")
        res = parse_results(path)
        lines.append(format_row(model, res))
    return lines

def resolve_workload(pattern, ds_path):
    """'{dataset}' -> dataset folder; a glob pattern resolves to its first match."""
    path = pattern.format(dataset=ds_path)
    matches = sorted(glob.glob(path))
    return matches[0] if matches else None

def main():
    parser = argparse.ArgumentParser(description="Run Static Experiments for Synthetic and DSB data")
    parser.add_argument('--scenario', type=str, required=True,
                        choices=[*STATIC_SCENARIOS, 'all'],
                        help="The experiment scenario to run (see registry.STATIC_SCENARIOS).")
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=DEFAULT_MODELS,
                        help="Subset of models to run (default: all except those marked default=False).")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the commands without running them.")
    args = parser.parse_args()

    scenarios_to_run = list(STATIC_SCENARIOS) if args.scenario == 'all' else [args.scenario]

    final_report = []
    final_report.append("Dataset Model Avg Q-Err 25% 75% 95% Median Train (s) Infer (s)")

    for sc in scenarios_to_run:
        cfg = STATIC_SCENARIOS[sc]
        exp_name = cfg['experiment']
        print("\n" + "="*50)
        print(f"Running Scenario: {sc} -> results/{exp_name}")
        print("="*50)

        if cfg.get('prepare') and not all(resolve_workload(cfg['workload'], ds) for ds in cfg['datasets']):
            run_cmd([sys.executable, *cfg['prepare']], args.dry_run)

        for ds_path in cfg['datasets']:
            workload = resolve_workload(cfg['workload'], ds_path)

            # Check if paths exist to prevent obscure FileNotFoundError crashes
            if not os.path.exists(ds_path):
                print(f"ERROR: Dataset not found at {ds_path}. Skipping...")
                continue
            if workload is None and not args.dry_run:
                print(f"ERROR: Workload not found at {cfg['workload'].format(dataset=ds_path)}. Skipping...")
                continue
            workload = workload or cfg['workload'].format(dataset=ds_path)

            for model in args.models:
                run_cmd([sys.executable, MODELS[model]['static'],
                         "--experiment-name", exp_name,
                         "--dataset", ds_path,
                         "--workload", workload], args.dry_run)

            ds_name = Path(ds_path).name
            final_report.extend(collect_table(f"{ds_name}_{sc}", args.models, f"results/{exp_name}/{ds_name}"))

    print("\n\n" + "="*80)
    print("FINAL EXPERIMENT RESULTS")
    print("="*80)
    for line in final_report:
        print(line)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

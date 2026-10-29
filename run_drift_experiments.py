import os
import shlex
import subprocess
import sys
import argparse

from registry import MODELS, DEFAULT_MODELS, DRIFT_SCENARIOS


def run_cmd(cmd, dry_run=False):
    print(f"\n[RUNNING] {shlex.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Dynamic Drift Experiments")
    parser.add_argument('--scenario', type=str, required=True,
                        choices=[*DRIFT_SCENARIOS, 'all'],
                        help="The drift scenario to run (see registry.DRIFT_SCENARIOS).")
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=DEFAULT_MODELS,
                        help="Subset of models to run (default: all except those marked default=False).")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the commands without running them.")
    args = parser.parse_args()

    scenarios = list(DRIFT_SCENARIOS) if args.scenario == 'all' else [args.scenario]

    for sc in scenarios:
        print("\n" + "="*50)
        print(f"Running Drift Scenario: {sc}")
        print("="*50)

        cfg = DRIFT_SCENARIOS[sc]
        shift_dir = cfg['shift_dir']

        if not os.path.exists(shift_dir):
            print(f"Warning: Shift directory {shift_dir} does not exist. Skipping.")
            continue

        for model in args.models:
            cmd = [
                sys.executable, MODELS[model]['drift'],
                "--experiment-name", sc,
                "--shift-dir", shift_dir,
                "--workload", cfg['workload_count'],
            ]

            if cfg['static_workload']:
                cmd += ["--static-workload-path", cfg['static_workload']]
            else:
                # Workload drifts with the data: step_N_workload_driven_<count>.csv
                cmd.append("--shift-workload")

            run_cmd(cmd, args.dry_run)

if __name__ == "__main__":
    main()

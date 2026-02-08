import subprocess
import os

def run_tuning():
    rows = 10000000
    eval_n = 100000
    
    configs = [
        ("Baseline", "--hybrid-points 20 --hybrid-ident 1e-4 --hybrid-penalty 1.5"),
        ("DenseScaling", "--hybrid-points 500 --hybrid-ident 1e-4 --hybrid-penalty 1.5"),
        ("AggressiveModels", "--hybrid-points 20 --hybrid-ident 1e-6 --hybrid-penalty 1.1"),
        ("Combined", "--hybrid-points 500 --hybrid-ident 1e-6 --hybrid-penalty 1.1")
    ]
    
    for name, params in configs:
        print(f"\n>>> Running Experiment: {name} <<<")
        cmd = f"poetry run python main.py --mode static --rows {rows} --dist all --eval-n {eval_n} --bin-method fd {params} --experiment-name tune_{name}"
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    run_tuning()

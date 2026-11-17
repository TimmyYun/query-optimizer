"""
Exact configuration and version manifest of every paper-experiment family.

Writes results/paper_experiments/methodology/manifest.json (provenance.write_manifest) and CONFIG_MANIFEST.md:
  - git commit, worktree hash, package versions, model versions (provenance)
  - fixed parameters of the protocol (seed streams, sample size, Hybrid / FT / RB constants)
  - datasets: sha256 of every meta.pkl / step_*.pkl used, N, domain, FD k, sample size
  - workloads: sha256, query count, min / max width (hi - lo)
  - per family: RUN (results present) or NOT_RUN, with the command that produces it
Re-runnable: file hashes and workload statistics are cached by (path, size, mtime) in
results/paper_experiments/methodology/.manifest_cache.json, so later parts (DSB, FD reduction) can be
added by simply running it again.

    poetry run python -m paper_experiments.methodology_manifest
"""
import glob
import json
import pickle
import re
from pathlib import Path

import pandas as pd

from paper_experiments.provenance import ROOT, file_sha256, git_info, model_version, write_manifest

OUT = Path("results/paper_experiments/methodology")
CACHE = OUT / ".manifest_cache.json"
MODELS = ["EquiWidth", "EquiDepth", "TKHist", "EntropyHist-MEA", "STHist", "Hybrid"]
DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
# labelled modifications: version = their own sources on top of the base model's
MODIFICATIONS = {"Hybrid-Corrected": ["models/hybrid_corrected.py", "run_hybrid_corrected.py"]}

PARAMS = {
    "seed": 42,
    "seed_streams": {
        "dataset generation": "default_rng(42) (data/datasets.py)",
        "dataset sample": "[42, 1] (data/datasets.py:744-748)",
        "synthetic drift steps": "one default_rng(42) over all steps (data/generate_shifts.py)",
        "workloads": "default_rng(42) (workload/workload.py)",
        "adapt/eval split": "default_rng(42).permutation (benchmark_utils.split_adapt_eval)",
        "Hybrid training": "rng = default_rng(42); per bucket default_rng(base_seed + i)",
    },
    "sample_size": 100_000,
    "sample_rule": "Bernoulli p = min(1, 1.5*S/N), truncated to S",
    "fd_rule": "h = 2*IQR(sample)/N^(1/3), k = min(int((mx-mn)/h), 2000)",
    "hybrid": {
        "spec": "hybrid-fd-cdf-canonical", "points_per_bucket": 200, "validation_queries": 500,
        "tier0_uniform_threshold": 1.3, "tier1_3_threshold": 1.05, "ridge_alpha": 1.0,
        "density_weight": "1 + 5*P/maxP", "tree_max_depth": 4,
        "fourier": {"bands": 32, "max_freq": 1000.0, "features": 66, "hidden": [64, 32], "max_iter": 500},
        "uniform_floor": "max(model_pred, 0.05*uniform_pred)",
    },
    "drift": {
        "ft_query_qerror_threshold": 1.2, "ft_alpha": 0.15, "ft_ratio_clip": [0.2, 5.0],
        "mark_median_qerror_threshold": 1.3, "mark_min_queries": 6,
        "checkpoint_rows": 100_000, "baseline_rebuild_rule": "> 10 % rows changed since last rebuild",
        "adapt_eval_split": "50/50, split_adapt_eval(seed=42)",
    },
}


class Cache:
    def __init__(self, path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {}

    def get(self, p: Path, fn):
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}|{fn.__name__}"
        if key not in self.d:
            self.d[key] = fn(p)
        return self.d[key]

    def save(self):
        self.path.write_text(json.dumps(self.d))


def sha(p: Path):
    return file_sha256(p.resolve().relative_to(ROOT))


def wl_stats(p: Path):
    d = pd.read_csv(p)
    w = d["high"] - d["low"]
    return {"queries": int(len(d)), "width_min": int(w.min()), "width_max": int(w.max()), "width_median": float(w.median())}


def meta_stats(p: Path):
    with open(p, "rb") as f:
        t = pickle.load(f)
    mn, mx, N, freq, sample, k = t[:6]
    return {"arity": len(t), "N": int(N), "mn": int(mn), "mx": int(mx), "fd_k": int(k), "sample_size": int(len(sample))}


def describe_dataset(c, p):
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "status": "NOT_RUN"}
    return {"path": str(p), "sha256": c.get(p, sha), **c.get(p, meta_stats)}


def describe_steps(c, folder):
    folder = Path(folder)
    steps = sorted(folder.glob("step_*.pkl"), key=lambda q: int(re.search(r"\d+", q.stem).group()))
    if not steps:
        return {"path": str(folder), "status": "NOT_RUN"}
    hashes = [c.get(s, sha) for s in steps]
    import hashlib
    first, last = c.get(steps[0], meta_stats), c.get(steps[-1], meta_stats)
    return {"path": str(folder), "steps": len(steps), "combined_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
            "N_first": first["N"], "N_last": last["N"], "fd_k_step0": first["fd_k"]}


def describe_workload(c, p):
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "status": "NOT_RUN"}
    return {"path": str(p), "sha256": c.get(p, sha), **c.get(p, wl_stats)}


def family_status(path_glob):
    hits = glob.glob(path_glob, recursive=True)
    return ("RUN", len(hits)) if hits else ("NOT_RUN", 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = Cache(CACHE)
    R = "results/paper_experiments"
    datasets = [describe_dataset(c, f"data/generated/60000000_{kind}/{d}/meta.pkl") for kind in ("simple", "hard") for d in DISTS]
    datasets += [describe_dataset(c, p) for p in sorted(glob.glob("data/dsb/*/meta.pkl"))]
    datasets += [describe_dataset(c, p) for p in sorted(glob.glob("data/dsb_expanded/*/meta.pkl"))]
    drift = [describe_steps(c, f) for f in ["data/generated/60000000_hard/shift_normal_to_zipf_10%",
                                            "data/generated/60000000_hard/shift_normal_to_zipf_5%",
                                            "data/generated/60000000_hard/shift_normal_to_zipf_5%_dataset"]]
    drift += [describe_steps(c, f) for f in sorted({str(Path(p).parent) for p in glob.glob("data/dsb_expanded/**/step_0.pkl", recursive=True)})]
    workloads = [
        {"id": "narrow_w999", **describe_workload(c, "workload/1000000/narrow/workload.csv")},
        {"id": "wide", **describe_workload(c, "workload/1000000/wide/workload.csv")},
        {"id": "narrow_w100_thesis", **describe_workload(c, f"{R}/methodology/workloads/narrow100_1000000.csv")},
    ]
    for d in DISTS:
        workloads.append({"id": f"hard/{d}:driven_w999", **describe_workload(c, f"data/generated/60000000_hard/{d}/workload_driven_1000000.csv")})
        workloads.append({"id": f"hard/{d}:driven_w100_thesis", **describe_workload(c, f"{R}/methodology/workloads/hard/{d}/driven100_1000000.csv")})
    for p in sorted(glob.glob("data/dsb_expanded/*/workload_*.csv")):
        workloads.append({"id": f"dsb_expanded:{Path(p).parent.name}:{Path(p).stem}", **describe_workload(c, p)})

    families = {
        "static_repro_w999_fd": (f"{R}/methodology/reproduction/static/fd/**/summary_*.json",
                                 "run_experiments.py --scenario <s> --buckets fd --out-dir results/paper_experiments/methodology/reproduction/static/fd"),
        "static_repro_w999_fixed100": (f"{R}/methodology/reproduction/static/fixed100/**/summary_*.json",
                                       "run_experiments.py --scenario <s> --buckets 100 --out-dir results/paper_experiments/methodology/reproduction/static/fixed100"),
        "static_repro_w100_thesis_fd": (f"{R}/methodology/reproduction/static_w100_thesis/fd/**/summary_*.json",
                                        "results/paper_experiments/methodology/logs/run_static_w100.sh fd fd"),
        "static_repro_w100_thesis_fixed100": (f"{R}/methodology/reproduction/static_w100_thesis/fixed100/**/summary_*.json",
                                              "results/paper_experiments/methodology/logs/run_static_w100.sh 100 fixed100"),
        "static_repro_hybrid_corrected": (f"{R}/methodology/reproduction/**/summary_Hybrid-Corrected.json",
                                          "results/paper_experiments/methodology/logs/run_static_corrected.sh <fd|100> <fd|fixed100>"),
        "drift_instrumentation_check": (f"{R}/methodology/reproduction/drift_check/**/drift_steps.csv",
                                        "results/paper_experiments/methodology/logs/slot_b.sh"),
        "dsb_expanded_static": (f"{R}/dsb_expanded/static/**/summary_*.json", "Agent 2 (paper_experiments/dsb_*.py)"),
        "dsb_expanded_displacement": (f"{R}/dsb_expanded/displacement/**/drift_steps.csv", "Agent 2"),
        "dsb_expanded_additive_200": (f"{R}/dsb_expanded/additive_200/**/drift_steps.csv", "Agent 2"),
        "fd_reduction": (f"{R}/fd_reduction/**/*.csv", "Agent 3 (paper_experiments/fd_reduction*.py)"),
        "memory_analysis": (f"{R}/memory_analysis/**/*.csv", "Agent 4 (paper_experiments/memory_analysis.py)"),
    }
    fam = {}
    for name, (pattern, cmd) in families.items():
        status, n = family_status(pattern)
        fam[name] = {"status": status, "result_files": n, "pattern": pattern, "command": cmd}
    c.save()

    mod_versions = {m: model_version(m, srcs) for m, srcs in MODIFICATIONS.items() if all(Path(x).exists() for x in srcs)}
    rec = write_manifest(OUT / "manifest.json", models=MODELS, parameters=PARAMS, modification_versions=mod_versions,
                         datasets=datasets,
                         drift_folders=drift, workloads=workloads, families=fam)
    gi = git_info()

    L = ["# Конфигурация и версии экспериментов (генерируется `paper_experiments/methodology_manifest.py`)", "",
         f"- git commit `{gi['git_commit']}` (branch `{gi['git_branch']}`, dirty = {gi['git_dirty']})",
         f"- worktree_sha256 `{gi['worktree_sha256']}` ({gi['worktree_files_hashed']} файлов исходников)",
         f"- created {rec['created_utc']}; packages: " + ", ".join(f"{k} {v}" for k, v in rec["packages"].items()
                                                                if k in ("python", "numpy", "scipy", "sklearn", "pandas")),
         "", "## Версии моделей", "", "| Model | model_version |", "|---|---|"]
    L += [f"| {m} | `{model_version(m)}` |" for m in MODELS]
    L += [f"| {m} (MODIFICATION) | `{v}` |" for m, v in mod_versions.items()]
    L += ["", "## Параметры протокола", "", "```json", json.dumps(PARAMS, indent=2, ensure_ascii=False), "```",
          "", "## Семейства экспериментов", "", "| Семейство | Статус | Файлов | Команда |", "|---|---|---|---|"]
    L += [f"| {k} | **{v['status']}** | {v['result_files']} | `{v['command']}` |" for k, v in fam.items()]
    L += ["", "## Датасеты", "", "| path | sha256 (16) | N | domain | FD k | sample |", "|---|---|---|---|---|---|"]
    for d in datasets:
        if d.get("status") == "NOT_RUN":
            L.append(f"| {d['path']} | NOT_RUN | | | | |")
        else:
            L.append(f"| {d['path']} | `{d['sha256'][:16]}` | {d['N']:,} | [{d['mn']}, {d['mx']}] | {d['fd_k']} | {d['sample_size']:,} |")
    L += ["", "## Папки дрейфа", "", "| path | steps | combined sha256 (16) | N first → last | FD k step 0 |", "|---|---|---|---|---|"]
    for d in drift:
        if d.get("status") == "NOT_RUN":
            L.append(f"| {d['path']} | NOT_RUN | | | |")
        else:
            L.append(f"| {d['path']} | {d['steps']} | `{d['combined_sha256'][:16]}` | {d['N_first']:,} → {d['N_last']:,} | {d['fd_k_step0']} |")
    L += ["", "## Workload'ы", "", "| id | path | sha256 (16) | queries | width min–max (median) |", "|---|---|---|---|---|"]
    for w in workloads:
        if w.get("status") == "NOT_RUN":
            L.append(f"| {w['id']} | {w['path']} | NOT_RUN | | |")
        else:
            L.append(f"| {w['id']} | {w['path']} | `{w['sha256'][:16]}` | {w['queries']:,} | {w['width_min']}–{w['width_max']} ({w['width_median']:.0f}) |")
    L += ["", "Ширина = hi − lo (запрос [lo, hi] включительно покрывает hi − lo + 1 значений).",
          "Синтетические per-step workload'ы дрейфа (`step_<i>_workload_driven_1000000.csv`) имеют W ∈ [1, 999], как и статические data-driven."]
    (OUT / "CONFIG_MANIFEST.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "manifest.json", "and", OUT / "CONFIG_MANIFEST.md")


if __name__ == "__main__":
    main()

"""
DSB expansion, part 4: job queue for the static and drift runs (existing runners via subprocess).

    poetry run python -m paper_experiments.dsb_run list  --kind static|drift   # print the jobs
    poetry run python -m paper_experiments.dsb_run run   --kind static|drift --parallel 6 [--bins fd fixed100]
                                                         [--scenarios displacement additive_200] [--columns ...]

Every job is one existing runner (registry.MODELS) on one column / workload / bucket setting:
  static  results/paper_experiments/dsb_expanded/static/<bin>/<workload>/<column>/summary_<Model>.json
  drift   results/paper_experiments/dsb_expanded/<scenario>/<bin>/<column>/gradual_<column>_to_zipfpop_<Model>/
Each process runs with OMP/OPENBLAS/MKL threads = 1. A job is DONE only if rc == 0 AND its output exists
AND its log has no "Failed to run" (benchmark_utils swallows per-dataset exceptions). Status of every
attempt is appended to results/paper_experiments/dsb_expanded/jobs_status.csv; DONE jobs are skipped on rerun.
A manifest.json (provenance) is written next to every configuration's output.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from paper_experiments.dsb_columns import ROOT, SELECTED, SELECTED_ALL
from paper_experiments.provenance import file_sha256, model_version, worker_env, write_manifest
from registry import DEFAULT_MODELS, MODELS

DATA = ROOT / "data" / "dsb_expanded"
RES = ROOT / "results" / "paper_experiments" / "dsb_expanded"
LOGS = RES / "logs"
STATUS = RES / "jobs_status.csv"
BINS = {"fd": "fd", "fixed100": "100"}
STATIC_WORKLOADS = ["dd_narrow", "uniform_narrow", "uniform_wide", "official"]
SCENARIOS = {"displacement": {"total_pct": 100, "drift_type": "displacement", "steps": 21},
             "additive_200": {"total_pct": 200, "drift_type": "additive", "steps": 41}}
TARGET_LABEL = "zipfpop"
N_DRIFT_QUERIES = "1000000"
# Labelled MODIFICATIONS of the canonical Hybrid (approved by the user, 2026-11-10): own runners, own model
# names, outputs side by side with the Original (never overwrite it). Queued only after the Original jobs.
EXTRA_MODELS = {
    "Hybrid-Corrected": {"static": "run_hybrid_corrected.py", "drift": "run_hybrid_corrected_drift.py",
                         "sources": ["models/hybrid_corrected.py", "run_hybrid_corrected.py",
                                     "run_hybrid_corrected_drift.py"]},
    # diagnostic: inference fix only; static, dd_narrow only
    "Hybrid-Corrected-PredictOnly": {"static": "run_hybrid_corrected.py",
                                     "args": ["--variant", "corrected-predict-only"],
                                     "sources": ["models/hybrid_corrected.py", "run_hybrid_corrected.py"]},
}


def spec(model: str) -> dict:
    return MODELS[model] if model in MODELS else EXTRA_MODELS[model]


STATUS_FIELDS = ["job_id", "kind", "bin", "workload_or_scenario", "column", "model", "rc", "outputs_ok",
                 "wall_s", "started", "finished", "log", "note"]


@lru_cache(maxsize=None)
def sha(path: str) -> str:
    return file_sha256(path)


def static_jobs(bins, columns, models, workloads=STATIC_WORKLOADS):
    jobs = []
    for b in bins:
        for wl in workloads:
            for col in columns:
                wl_path = DATA / col / f"workload_{wl}.csv"
                if not wl_path.exists():
                    continue  # official: NOT_APPLICABLE for most columns (data/dsb_expanded/official_status.json)
                for m in models:
                    out_root = RES / "static" / b
                    jobs.append({
                        "job_id": f"static|{b}|{wl}|{col}|{m}", "kind": "static", "bin": b,
                        "workload_or_scenario": wl, "column": col, "model": m,
                        "cmd": [sys.executable, spec(m)["static"], "--dataset", str((DATA / col).relative_to(ROOT)),
                                "--workload", str(wl_path.relative_to(ROOT)), "--out-dir", str(out_root.relative_to(ROOT)),
                                "--experiment-name", wl, "--buckets", BINS[b], *spec(m).get("args", [])],
                        "out_dir": out_root / wl / col, "expect": out_root / wl / col / f"summary_{m}.json",
                        "manifest": {"workload": wl, "workload_path": str(wl_path.relative_to(ROOT)),
                                     "dataset": col, "bin_strategy": b},
                    })
    return jobs


def drift_jobs(bins, columns, models, scenarios):
    jobs = []
    # priority: thesis columns first, displacement before additive, given bin order
    cols = [c for c in columns if c in SELECTED] + [c for c in columns if c not in SELECTED]
    for b in bins:
        for sc in scenarios:
            cfg = SCENARIOS[sc]
            for col in cols:
                shift_dir = DATA / col / sc
                for m in models:
                    out_root = RES / sc / b
                    model_dir = out_root / col / f"gradual_{col}_to_{TARGET_LABEL}_{m}"
                    jobs.append({
                        "job_id": f"drift|{b}|{sc}|{col}|{m}", "kind": "drift", "bin": b,
                        "workload_or_scenario": sc, "column": col, "model": m,
                        "cmd": [sys.executable, spec(m)["drift"], "--shift-dir", str(shift_dir.relative_to(ROOT)),
                                "--workload", N_DRIFT_QUERIES, "--shift-workload",
                                "--init-dist", col, "--target-dist", TARGET_LABEL,
                                "--out-dir", str(out_root.relative_to(ROOT)), "--experiment-name", col,
                                "--buckets", BINS[b], "--drift-total-pct", str(cfg["total_pct"]),
                                "--drift-type", cfg["drift_type"], *spec(m).get("args", [])],
                        "out_dir": model_dir, "expect": model_dir / "drift_steps.csv", "steps": cfg["steps"],
                        "manifest": {"scenario": sc, "drift_type": cfg["drift_type"], "drift_total_pct": cfg["total_pct"],
                                     "shift_dir": str(shift_dir.relative_to(ROOT)), "dataset": col, "bin_strategy": b,
                                     "workload": f"per-step step_<i>_workload_driven_{N_DRIFT_QUERIES}.csv"},
                    })
    return jobs


def done_ids():
    if not STATUS.exists():
        return set()
    with open(STATUS) as f:
        return {r["job_id"] for r in csv.DictReader(f) if r["rc"] == "0" and r["outputs_ok"] == "True"}


def outputs_ok(job, log_path: Path):
    if not job["expect"].exists():
        return False, "missing output"
    txt = log_path.read_text(errors="ignore")
    if "Failed to run" in txt or "Traceback" in txt:
        return False, "error in log"
    if job["kind"] == "drift":
        n = sum(1 for _ in open(job["expect"])) - 1
        if n != job["steps"]:
            return False, f"{n} steps, expected {job['steps']}"
    return True, ""


def write_job_manifest(job):
    job["out_dir"].mkdir(parents=True, exist_ok=True)
    cfg = dict(job["manifest"])
    if job["kind"] == "static":
        cfg["dataset_meta_sha256"] = sha(f"data/dsb_expanded/{job['column']}/meta.pkl")
        cfg["workload_sha256"] = sha(cfg["workload_path"])
    else:
        sd = ROOT / cfg["shift_dir"]
        cfg["step_pkl_sha256"] = {p.name: sha(str(p.relative_to(ROOT))) for p in sorted(sd.glob("step_*.pkl"))}
    extra_src = spec(job["model"]).get("sources", [])
    write_manifest(job["out_dir"] / f"manifest_{job['model']}.json", models=[job["model"]],
                   model=job["model"], model_version=model_version(job["model"], extra_sources=extra_src),
                   command=job["cmd"][1:],
                   threads=1, **cfg)


def run_job(job):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / (job["job_id"].replace("|", "__") + ".log")
    write_job_manifest(job)
    t0, started = time.time(), datetime.now().isoformat(timespec="seconds")
    with open(log, "w") as f:
        rc = subprocess.run(job["cmd"], cwd=ROOT, env=worker_env(1), stdout=f, stderr=subprocess.STDOUT).returncode
    ok, note = outputs_ok(job, log) if rc == 0 else (False, f"rc={rc}")
    return {"job_id": job["job_id"], "kind": job["kind"], "bin": job["bin"],
            "workload_or_scenario": job["workload_or_scenario"], "column": job["column"], "model": job["model"],
            "rc": rc, "outputs_ok": ok, "wall_s": round(time.time() - t0, 1), "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"), "log": str(log.relative_to(ROOT)), "note": note}


def append_status(row):
    new = not STATUS.exists()
    with open(STATUS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "run"])
    ap.add_argument("--kind", choices=["static", "drift"], required=True)
    ap.add_argument("--bins", nargs="+", default=list(BINS))
    ap.add_argument("--columns", nargs="+", default=list(SELECTED_ALL))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=[*MODELS, *EXTRA_MODELS])
    ap.add_argument("--workloads", nargs="+", default=STATIC_WORKLOADS)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = (static_jobs(args.bins, args.columns, args.models, args.workloads) if args.kind == "static"
            else drift_jobs(args.bins, args.columns, args.models, args.scenarios))
    done = done_ids()
    todo = [j for j in jobs if j["job_id"] not in done][: args.limit]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    if args.action == "list":
        for j in todo:
            print(j["job_id"], " ".join(j["cmd"][1:]))
        return
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(run_job, j): j for j in todo}
        for fu in as_completed(futs):
            row = fu.result()
            append_status(row)
            print(json.dumps({k: row[k] for k in ("job_id", "rc", "outputs_ok", "wall_s", "note")}), flush=True)


if __name__ == "__main__":
    main()

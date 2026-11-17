"""
Resumable job queue for long experiment batches.

    poetry run python -m paper_experiments.jobqueue --jobs <jobs.jsonl> --parallel N --log-dir <dir>

jobs.jsonl: one JSON object per line {"id": str, "cmd": [argv...], "expect": path}. A job whose
`expect` file exists is skipped, so a killed batch is resumed by running the same command again.
Each job's stdout+stderr goes to <log-dir>/<safe id>.log; a job is OK when it exits 0, `expect`
exists and the log has no "Traceback" / "Failed to run". Status rows are appended to
<log-dir>/status.csv. BLAS/OpenMP threads are limited to 1 per job (provenance.worker_env).
"""
import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from paper_experiments.provenance import ROOT, worker_env


def _run(job, log_dir: Path):
    log = log_dir / (re.sub(r"[^A-Za-z0-9_.=-]+", "_", job["id"])[:180] + ".log")
    t0, started = time.perf_counter(), datetime.now().isoformat(timespec="seconds")
    with open(log, "w") as f:
        rc = subprocess.run(job["cmd"], cwd=ROOT, env=worker_env(), stdout=f, stderr=subprocess.STDOUT).returncode
    txt = log.read_text(errors="ignore")
    ok = rc == 0 and (ROOT / job["expect"]).exists() and "Traceback" not in txt and "Failed to run" not in txt
    return {"id": job["id"], "rc": rc, "ok": ok, "wall_s": round(time.perf_counter() - t0, 1),
            "started": started, "finished": datetime.now().isoformat(timespec="seconds"), "log": str(log)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--log-dir", required=True)
    a = p.parse_args()
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in open(a.jobs) if line.strip()]
    todo = [j for j in jobs if not (ROOT / j["expect"]).exists()]
    print(f"{datetime.now():%H:%M:%S} {a.jobs}: {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    status = log_dir / "status.csv"
    new = not status.exists()
    with open(status, "a", newline="") as sf, ThreadPoolExecutor(a.parallel) as ex:
        w = csv.DictWriter(sf, fieldnames=["id", "rc", "ok", "wall_s", "started", "finished", "log"])
        if new:
            w.writeheader()
        futs = [ex.submit(_run, j, log_dir) for j in todo]
        n_fail = 0
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            w.writerow(r)
            sf.flush()
            n_fail += not r["ok"]
            print(f"{datetime.now():%H:%M:%S} [{i}/{len(todo)}] {'OK  ' if r['ok'] else 'FAIL'} {r['id']} {r['wall_s']}s", flush=True)
    print(f"{datetime.now():%H:%M:%S} QUEUE_DONE {a.jobs} failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()

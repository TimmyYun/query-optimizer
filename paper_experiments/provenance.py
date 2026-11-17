"""
Provenance of every experiment configuration.

The work tree is not committed while these experiments run, so the git commit alone does not
identify the code. Each manifest therefore records
  git_commit / git_dirty            HEAD and whether tracked files differ from it
  worktree_sha256                   hash over (path, sha256) of every tracked + untracked,
                                    non-ignored source file (*.py, *.toml, *.lock, *.sh) —
                                    two runs with the same value ran the same code
  source_sha256                     per-file hashes of the files a model depends on
  model_version                     "<Model>:<spec>:<sha10 of its source files>"
  packages                          python / numpy / scipy / sklearn / pandas versions
plus whatever the caller passes (seed, dataset, workload, parameters).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
SOURCE_SUFFIXES = (".py", ".toml", ".lock", ".sh")

# Files whose content defines each model's behaviour (estimator + its runner glue).
MODEL_SOURCES = {
    "EquiWidth": ["models/equi_width.py", "models/common.py"],
    "EquiDepth": ["models/equi_depth.py", "models/common.py"],
    "TKHist": ["models/tkhist.py", "models/common.py"],
    "EntropyHist-MEA": ["models/entropy_histogram.py", "models/common.py"],
    "STHist": ["models/st_histogram.py", "models/common.py"],
    "Hybrid": ["models/hybrid.py", "models/common.py"],
    # labelled MODIFICATION (inclusive-interval fix), see models/hybrid_corrected.py
    "Hybrid-Corrected": ["models/hybrid.py", "models/hybrid_corrected.py", "models/common.py",
                         "run_hybrid_corrected.py", "run_hybrid_corrected_drift.py"],
}
HARNESS_SOURCES = ["benchmark_utils.py", "drift_harness.py", "run_hybrid_drift.py", "run_hybrid.py"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(ROOT / path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def git_info() -> dict:
    files = [f for f in _git("ls-files", "-co", "--exclude-standard").splitlines()
             if f.endswith(SOURCE_SUFFIXES) and (ROOT / f).is_file()]
    tree = hashlib.sha256()
    for f in sorted(files):
        tree.update(f"{f}\0{file_sha256(f)}\n".encode())
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status),
        "worktree_sha256": tree.hexdigest(),
        "worktree_files_hashed": len(files),
    }


def sources_sha(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(f"{p}\0{file_sha256(p)}\n".encode())
    return h.hexdigest()[:10]


def model_version(model: str, extra_sources=()) -> str:
    """
    'Hybrid:hybrid-fd-cdf-canonical:<sha10>' — spec label + hash of the model's source files.
    Variants pass the base model name via MODEL_SOURCES key prefix (e.g. 'Hybrid-Merge' -> Hybrid
    sources) plus their own extra_sources, so a modification never shares a version with the original.
    """
    base = next((m for m in MODEL_SOURCES if model == m), None)
    if base is None:  # longest matching prefix: Hybrid-Corrected-Merge -> Hybrid-Corrected
        base = max((m for m in MODEL_SOURCES if model.startswith(m)), key=len, default=None)
    srcs = list(MODEL_SOURCES.get(base, [])) + list(extra_sources)
    spec = "src"
    if base == "Hybrid":
        from models.hybrid import HybridEstimator
        spec = HybridEstimator.SPEC_VERSION if model == "Hybrid" else f"MODIFICATION-of-{HybridEstimator.SPEC_VERSION}"
    elif base == "Hybrid-Corrected":
        from models.hybrid_corrected import HybridCorrectedEstimator
        spec = "MODIFICATION-" + HybridCorrectedEstimator.SPEC_VERSION.split(" ")[0]
    return f"{model}:{spec}:{sources_sha(srcs) if srcs else 'n/a'}"


def env_info() -> dict:
    import numpy
    import pandas
    import scipy
    import sklearn
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pandas": pandas.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "hostname": platform.node(),
        # latency is only comparable between runs made under similar load
        "loadavg_1_5_15": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }


def worker_env(threads: int = 1) -> dict:
    """Environment for experiment subprocesses: BLAS/OpenMP limited so parallel jobs do not oversubscribe."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    return env


def manifest(**config) -> dict:
    """Full provenance record: git + environment + model versions + the caller's configuration."""
    models = config.get("models") or list(MODEL_SOURCES)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.pop("seed", SEED),
        **git_info(),
        "source_sha256": {p: file_sha256(p) for p in sorted(set(HARNESS_SOURCES + [s for v in MODEL_SOURCES.values() for s in v]))
                          if (ROOT / p).exists()},
        "model_versions": {m: model_version(m) for m in models},
        "packages": env_info(),
        "config": config,
    }


def write_manifest(path, **config) -> dict:
    """Write manifest.json (or the given path). Returns the record."""
    path = Path(path)
    if path.is_dir() or path.suffix != ".json":
        path = path / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = manifest(**config)
    path.write_text(json.dumps(rec, indent=2, default=str))
    return rec

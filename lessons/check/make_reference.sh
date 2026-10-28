#!/usr/bin/env bash
# Пересчёт эталона. Запускать ТОЛЬКО на ветке feature/new_baseline (там есть models/tkhist.py и др.).
#   bash lessons/check/make_reference.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=lessons/check/_ref_parts
mkdir -p "$OUT"
MODELS="trivial equi_width tkhist equi_depth entropy_hist equi_hist st_hist hybrid"
for m in $MODELS; do
  poetry run python lessons/check/check_static.py "$m" --package reference_estimators --queries 200000 --json "$OUT/static_$m.json"
  for sc in replace add; do
    poetry run python lessons/check/check_drift.py "$m" --package reference_estimators --queries 50000 --scenario $sc --json "$OUT/drift_${sc}_$m.json"
  done
done
poetry run python - <<'PY'
import json, pathlib
out = pathlib.Path("lessons/check/_ref_parts")
static = {"_n_queries": 200000}
drift = {"_n_queries": 50000, "replace": {}, "add": {}}
for p in sorted(out.glob("static_*.json")):
    static[p.stem[len("static_"):]] = {r["dataset"]: r for r in json.loads(p.read_text())}
for p in sorted(out.glob("drift_*.json")):
    _, sc, m = p.stem.split("_", 2)
    drift[sc][m] = json.loads(p.read_text())
pathlib.Path("lessons/check/reference_static.json").write_text(json.dumps(static, indent=1))
pathlib.Path("lessons/check/reference_drift.json").write_text(json.dumps(drift, indent=1))
print("reference written")
PY

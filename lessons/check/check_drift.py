#!/usr/bin/env python3
"""
Проверка модели под дрейфом normal -> zipf (20 шагов по 5%) + сравнение с эталоном.

    poetry run python lessons/check/check_drift.py equi_width
    poetry run python lessons/check/check_drift.py hybrid --steps 0 5 10 15 20
    poetry run python lessons/check/check_drift.py st_hist --scenario add

Что модель может уметь (всё опционально, см. lessons/08_drift.md):
    refresh(sample, mn, mx, total_rows)          — ANALYZE на новом шаге
    observe(q, actual)                           — учиться по одному ответу
    adapt(queries, actual, sample, mn, mx, N)    — адаптация пачкой (Hybrid FT/RB)
"""
import argparse
import json

from bench import CHECK_DIR, run_drift
from check_static import parse_kw

REF_PATH = CHECK_DIR / "reference_drift.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--package", default="estimators")
    ap.add_argument("--queries", type=int, default=50_000, help="запросов на шаг (половина — фидбэк, половина — оценка)")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="какие шаги показывать (ANALYZE идёт на всех)")
    ap.add_argument("--scenario", choices=["replace", "add"], default="replace",
                    help="replace: zipf заменяет normal (жёстко); add: zipf подмешивается (мягко)")
    ap.add_argument("--kw", nargs="*")
    ap.add_argument("--ref-name", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ref_all = json.loads(REF_PATH.read_text()) if REF_PATH.exists() else {}
    ref = {r["step"]: r for r in ref_all.get(args.scenario, {}).get(args.ref_name or args.model, [])}

    rows = run_drift(args.package, args.model, args.queries, parse_kw(args.kw), args.steps, args.scenario)
    hdr = (f"{'shift':>6} {'shock med':>15} {'shock p95':>15} {'adapt med':>15} {'adapt p95':>15} "
           f"{'refresh,s':>9} {'feedbk,s':>9} {'inf,us':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        rr = ref.get(r["step"])

        def pair(key, nd):
            return f"{r[key]:.{nd}f}" + (f"/{rr[key]:.{nd}f}" if rr else "")

        print(f"{r['shift']:>6} {pair('shock_median', 3):>15} {pair('shock_p95', 2):>15} "
              f"{pair('adapted_median', 3):>15} {pair('adapted_p95', 2):>15} "
              f"{r['refresh_s']:>9.3f} {r['feedback_s']:>9.3f} {r['infer_us']:>7.1f}")
    print(f"\nbuild шага 0: {rows[0]['build_s']:.3f}s; формат: твоё/эталон")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

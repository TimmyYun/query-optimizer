#!/usr/bin/env python3
"""
Проверка модели на статических датасетах + сравнение с эталоном.

    poetry run python lessons/check/check_static.py equi_width
    poetry run python lessons/check/check_static.py hybrid --datasets normal zipf --queries 50000
    poetry run python lessons/check/check_static.py tkhist --kw top_k=10

Модель — модуль estimators/<model>.py с функцией build(sample, mn, mx, total_rows, n_buckets, **kw).
Эталон — reference_static.json (готовые реализации из ветки feature/new_baseline,
посчитаны на тех же запросах этим же скриптом).
"""
import argparse
import json
import math

from bench import CHECK_DIR, all_datasets, run_static

REF_PATH = CHECK_DIR / "reference_static.json"


def parse_kw(items):
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    return kw


def verdict(you, ref):
    """✓ — не хуже эталона (допуск 2% по медиане-1 и 5% по P95), ≈ — в пределах 15%, ✗ — хуже."""
    if ref is None:
        return " "
    dm = (you["median"] - 1) - (ref["median"] - 1)
    tol_m = 0.02 * (ref["median"] - 1) + 0.002
    if dm <= tol_m and you["p95"] <= ref["p95"] * 1.05:
        return "✓"
    if dm <= 0.15 * (ref["median"] - 1) + 0.01 and you["p95"] <= ref["p95"] * 1.15:
        return "≈"
    return "✗"


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="имя модуля в пакете: equi_width, tkhist, equi_depth, entropy_hist, equi_hist, st_hist, hybrid")
    ap.add_argument("--package", default="estimators")
    ap.add_argument("--datasets", nargs="*", help="по умолчанию все: 5 синтетических (по данным), 5 @random, 3 dsb/*")
    ap.add_argument("--queries", type=int, default=200_000)
    ap.add_argument("--kw", nargs="*", help="доп. параметры build: key=value")
    ap.add_argument("--ref-name", default=None, help="под каким именем искать эталон (по умолчанию = model)")
    ap.add_argument("--json", default=None, help="сохранить результаты в файл")
    args = ap.parse_args()

    ds = all_datasets()
    names = args.datasets or list(ds)
    ref_all = json.loads(REF_PATH.read_text()) if REF_PATH.exists() else {}
    ref = ref_all.get(args.ref_name or args.model, {})
    if ref and args.queries != ref_all.get("_n_queries"):
        print(f"! эталон считался на {ref_all.get('_n_queries')} запросах, у тебя {args.queries} — сравнение примерное")

    kw = parse_kw(args.kw)
    rows = []
    hdr = f"{'dataset':<26} {'median':>14} {'p95':>16} {'p99':>9} {'build,s':>8} {'inf,us':>7} {'floats':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    for name in names:
        meta, wl = ds[name]
        if not meta.exists() or not wl.exists():
            print(f"{name:<26} нет данных: {meta if not meta.exists() else wl}")
            continue
        r = run_static(args.package, args.model, name, meta, wl, args.queries, kw)
        rows.append(r)
        rr = ref.get(name)
        v = verdict(r, rr)
        med = f"{r['median']:.3f}" + (f"/{rr['median']:.3f}" if rr else "")
        p95 = f"{r['p95']:.2f}" + (f"/{rr['p95']:.2f}" if rr else "")
        print(f"{name:<26} {med:>14} {p95:>16} {r['p99']:>9.2f} {r['build_s']:>8.3f} {r['infer_us']:>7.1f} {fmt(r['n_floats']):>8} {v}")
    if ref:
        print("\nформат: твоё/эталон.  ✓ не хуже эталона   ≈ близко (до 15%)   ✗ заметно хуже")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

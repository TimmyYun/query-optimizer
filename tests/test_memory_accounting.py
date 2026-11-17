"""Memory accounting (paper_experiments.memory_analysis), the generic xlsx exporter and consolidation."""
import json

import numpy as np
import pandas as pd

from models import (EntropyHistogram1D, EquiDepthHistogram, EquiWidthHistogram, HistogramConfig, HybridEstimator,
                    STHistogram, STHistogramConfig, TKHistogram)
from models.common import Bucket
from paper_experiments import consolidate
from paper_experiments.excel_export import export_dir
from paper_experiments.memory_analysis import breakdown

N = 1_000_000


def _sample():
    rng = np.random.default_rng([42, 11])
    a = rng.integers(0, 5_000, size=20_000)
    b = 5_000 + np.minimum(rng.zipf(1.4, size=20_000), 4_999)
    return np.concatenate([a, b]).astype(np.int64)


def _hybrid(buckets, sample, mn, mx):
    h = HybridEstimator(buckets)
    est = np.bincount(sample - mn, minlength=mx - mn + 1) * (N / len(sample))
    h.train(est, mn, 50, np.random.default_rng(42))
    return h


def test_hybrid_uniform_grid_per_bucket_plus_global_equals_n_values():
    s = _sample()
    h = _hybrid(EquiWidthHistogram.build_from_sample(0, 9_999, 16, s, N).buckets, s, 0, 9_999)
    b = breakdown("Hybrid", h)  # asserts sum(components) == n_values == global + sum(per_bucket)
    assert h.uniform_grid
    assert b["global_values"] + b["per_bucket"].sum() == h.n_values()
    assert "bucket boundaries (non-equal width)" not in b["components"]


def test_hybrid_non_uniform_grid_counts_boundaries_once_per_bucket():
    s = _sample()
    edges = [0, 100, 1_000, 4_000, 5_000, 5_050, 6_000, 10_000]
    counts = [float(((s >= lo) & (s < hi)).sum()) * N / len(s) for lo, hi in zip(edges, edges[1:])]
    buckets = [Bucket(lo, hi - 1, c) for lo, hi, c in zip(edges, edges[1:], counts)]
    h = _hybrid(buckets, s, 0, 9_999)
    b = breakdown("Hybrid-Merge", h)
    assert not h.uniform_grid
    assert b["components"]["bucket boundaries (non-equal width)"] == len(buckets)
    assert b["global_values"] + b["per_bucket"].sum() == h.n_values()


def test_baseline_breakdowns_sum_to_n_values():
    s = _sample()
    models = {
        "EquiWidth": EquiWidthHistogram.build_from_sample(0, 9_999, 20, s, N),
        "TKHist": TKHistogram.build_from_sample(0, 9_999, 20, s, N),
        "EquiDepth": EquiDepthHistogram.build_from_sample(sample=s, total_rows=N, statistics_target=20),
        "EntropyHist-MEA": EntropyHistogram1D.build_from_sample(s, N, HistogramConfig(n_buckets=20)),
        "STHist": STHistogram(STHistogramConfig(bucket_count=20), total_rows=N, min_value=0, max_value=9_999),
    }
    for name, h in models.items():
        b = breakdown(name, h)
        assert sum(b["components"].values()) == h.n_values(), name
        assert b["global_values"] + b["per_bucket"].sum() + sum(b["nonbucket"].values()) == h.n_values(), name


def test_exporter_round_trip(tmp_path):
    from openpyxl import load_workbook
    d = tmp_path / "res"
    (d / "sub").mkdir(parents=True)
    pd.DataFrame({"model": ["A", "B"], "size_bytes": [2048, 4096], "median_qerror": [1.1, 1.2],
                  "infer_us_median": [3.0, 4.0]}).to_csv(d / "sub" / "summary.csv", index=False)
    pd.DataFrame({"Query_ID": [0], "Model": ["A"], "Prediction": [1.0], "Truth": [1.0], "Q_Error": [1.0]}).to_csv(
        d / "per_query_A.csv", index=False)
    res = export_dir(d, tmp_path / "out.xlsx")
    assert res["sheets"] == 1 and res["skipped"] == 1
    wb = load_workbook(tmp_path / "out.xlsx")
    assert wb.sheetnames[0] == "README"
    ws = wb[wb.sheetnames[1]]
    headers = [c.value for c in ws[1]]
    assert "size_bytes [bytes]" in headers and "size_bytes_KiB [KiB]" in headers
    assert "median_qerror [q-error, >=1, lower is better]" in headers and "infer_us_median [µs]" in headers
    kib = headers.index("size_bytes_KiB [KiB]") + 1
    assert ws.cell(row=2, column=kib).value == 2.0
    assert ws.freeze_panes == "A2"


def test_consolidate_marks_missing_configs(tmp_path, monkeypatch):
    base = tmp_path / "results" / "paper_experiments"
    run = base / "methodology" / "reproduction" / "static" / "fd" / "synth_hard_narrow" / "normal"
    run.mkdir(parents=True)
    (run / "summary_EquiWidth.json").write_text(json.dumps({"model": "EquiWidth", "wl_name": "workload", "metrics": {
        "n_queries": 10, "median_q_error": 1.5, "p95_q_error": 3.0, "avg_q_error": 2.0, "n_buckets": 5,
        "n_values": 7, "size_bytes": 56, "median_infer_time": 1e-6}}))
    monkeypatch.setattr(consolidate, "ROOT", tmp_path)
    monkeypatch.setattr(consolidate, "BASE", base)
    monkeypatch.setattr(consolidate, "EXP_ROOTS", {"fixed100": tmp_path / "e1", "fd": tmp_path / "e2"})
    static = consolidate.static_rows()
    assert len(static) == 1 and static["stage"].iloc[0] == "STATIC"
    comp = consolidate.completeness(static, consolidate.drift_runs())
    cfg = comp[(comp["part"] == "methodology_repro_static") & (comp["config"] == "synth_hard_narrow|normal|fd")]
    st = dict(zip(cfg["model"], cfg["status"]))
    assert st["EquiWidth"] == "OK" and st["Hybrid"] == "MISSING"
    other = comp[(comp["part"] == "methodology_repro_static") & (comp["config"] == "synth_hard_wide|zipf|fixed100")]
    assert set(other["status"]) == {"NOT_RUN"}
    assert (comp[comp["part"] == "fd_reduction"]["status"] == "NOT_RUN").all()

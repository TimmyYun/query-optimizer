"""
Single source of truth for what the orchestrators run.

Adding a method  -> one entry in MODELS.
Adding a dataset -> one entry in STATIC_SCENARIOS (and DRIFT_SCENARIOS if it drifts).

run_experiments.py and run_drift_experiments.py only read these dicts.
"""

# Key = model name exactly as the static runner writes it to summary_<Model>.json.
# "args": extra CLI arguments passed to both runners (a variant of the same method).
# "default": False keeps a method out of the orchestrators unless asked for via --models.
MODELS = {
    "EquiWidth": {"static": "run_equiwidth.py", "drift": "run_equiwidth_drift.py"},
    "Hybrid": {"static": "run_hybrid.py", "drift": "run_hybrid_drift.py"},
    "TKHist": {"static": "run_tkhist.py", "drift": "run_tkhist_drift.py"},
    "EquiDepth": {"static": "run_equidepth.py", "drift": "run_equidepth_drift.py"},
    "STHist": {"static": "run_sthist.py", "drift": "run_sthist_drift.py"},
    "EntropyHist-MEA": {"static": "run_entropyhist.py", "drift": "run_entropyhist_drift.py"},
    # Paper estimator interpolates continuously, so a point at a bucket edge gets zero mass;
    # [EXTENSION] counting integer cells fixes that on discrete data. Reported next to the paper version.
    "EntropyHist-MEA-Rng_discrete_linear": {
        "static": "run_entropyhist.py",
        "drift": "run_entropyhist_drift.py",
        "args": ["--range-estimator", "discrete_linear"],
    },
    # Исключён из дизайна эксперимента (слишком медленный); запускается через --models EquiHist
    "EquiHist": {"static": "run_equihist.py", "drift": "run_equihist_drift.py", "default": False},
}

DEFAULT_MODELS = [name for name, m in MODELS.items() if m.get("default", True)]

SYNTH_DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
# Real benchmarks, converted by prepare_benchmark.py (see its docstring for what is taken from where).
# DSB: web_sales columns that the official templates filter with BETWEEN; values in integer cents.
DSB_COLUMNS = ["ws_wholesale_cost", "ws_list_price"]
# STATS-CEB: integer columns with >= 8 distinct 1-D ranges in the official single-table sub-queries.
STATS_COLUMNS = [
    "posts.Score", "posts.ViewCount", "posts.CommentCount", "posts.AnswerCount", "posts.FavoriteCount",
    "users.Reputation", "users.Views", "users.UpVotes", "users.DownVotes",
]
STATS_DRIVEN_COUNT = 100_000

WIDE_WL = "workload/1000000/wide/workload.csv"
NARROW_WL = "workload/1000000/narrow/workload.csv"

# datasets: folders with meta.pkl.
# workload: CSV path; "{dataset}" is replaced by the dataset folder, glob patterns are allowed.
# prepare:  optional script (+ args) run first if any workload is missing.
STATIC_SCENARIOS = {
    "simple_wide": {
        "experiment": "synth_simple_wide",
        "datasets": [f"data/generated/60000000_simple/{d}" for d in SYNTH_DISTS],
        "workload": WIDE_WL,
    },
    "simple_narrow": {
        "experiment": "synth_simple_narrow",
        "datasets": [f"data/generated/60000000_simple/{d}" for d in SYNTH_DISTS],
        "workload": NARROW_WL,
    },
    "hard_wide": {
        "experiment": "synth_hard_wide",
        "datasets": [f"data/generated/60000000_hard/{d}" for d in SYNTH_DISTS],
        "workload": WIDE_WL,
    },
    "hard_narrow": {
        "experiment": "synth_hard_narrow",
        "datasets": [f"data/generated/60000000_hard/{d}" for d in SYNTH_DISTS],
        "workload": NARROW_WL,
    },
    "hard_narrow_data_driven": {
        "experiment": "synth_hard_narrow_data_driven",
        "datasets": [f"data/generated/60000000_hard/{d}" for d in SYNTH_DISTS],
        "workload": "{dataset}/workload_driven_1000000.csv",
    },
    # Official queries only. Needs the DSB toolkit run once (README §6.2); prepare converts what exists.
    "dsb": {
        "experiment": "dsb_static",
        "datasets": [f"data/dsb/{c}" for c in DSB_COLUMNS],
        "workload": "{dataset}/workload_official.csv",
        "prepare": ["prepare_benchmark.py", "--bench", "dsb"],
    },
    # Official 1-D predicates of stats_CEB_single_table_sub_query.sql (10-33 per column).
    "stats_ceb": {
        "experiment": "stats_ceb_official",
        "datasets": [f"data/stats-ceb/{c}" for c in STATS_COLUMNS],
        "workload": "{dataset}/workload_official.csv",
        "prepare": ["prepare_benchmark.py", "--bench", "stats-ceb"],
    },
    # Supplement, NOT the benchmark's queries: our data-driven generator on the official STATS data.
    "stats_ceb_driven": {
        "experiment": "stats_ceb_driven",
        "datasets": [f"data/stats-ceb/{c}" for c in STATS_COLUMNS],
        "workload": f"{{dataset}}/workload_driven_{STATS_DRIVEN_COUNT}.csv",
        "prepare": ["prepare_benchmark.py", "--bench", "stats-ceb"],
    },
}

# shift_dir: folder with step_*.pkl.
# static_workload: workload evaluated at every step, or None to use the per-step
#                  step_N_workload_driven_<count>.csv (workload drifts with the data).
# labels:   optional (init, target) for --init-dist/--target-dist, i.e. the output
#           folder gradual_<init>_to_<target>_<Model>; default ("normal", "zipf").
DRIFT_SCENARIOS = {
    "drift_10_ds_zipf_wl_normal": {
        "shift_dir": "data/generated/60000000_hard/shift_normal_to_zipf_10%",
        "static_workload": "data/generated/60000000_hard/normal/workload_driven_1000000.csv",
        "workload_count": "1000000",
    },
    "drift_5_ds_zipf_wl_normal": {
        "shift_dir": "data/generated/60000000_hard/shift_normal_to_zipf_5%",
        "static_workload": "data/generated/60000000_hard/normal/workload_driven_1000000.csv",
        "workload_count": "1000000",
    },
    "drift_5_ds_zipf_wl_zipf": {
        "shift_dir": "data/generated/60000000_hard/shift_normal_to_zipf_5%",
        "static_workload": None,
        "workload_count": "1000000",
    },
    "drift_5_ds_add_zipf_wl_zipf": {
        "shift_dir": "data/generated/60000000_hard/shift_normal_to_zipf_5%_dataset",
        "static_workload": None,
        "workload_count": "1000000",
    },
    # DSB: official refresh streams, one step per 5 % cumulative churn; official queries at every step
    **{
        f"dsb_drift_{c}": {
            "shift_dir": f"data/dsb/{c}/drift",
            "static_workload": f"data/dsb/{c}/workload_official.csv",
            "workload_count": "official",
            "labels": (c, "refresh"),
        }
        for c in DSB_COLUMNS
    },
}

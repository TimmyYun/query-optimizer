"""
Single source of truth for what the orchestrators run.

Adding a method  -> one entry in MODELS.
Adding a dataset -> one entry in STATIC_SCENARIOS (and DRIFT_SCENARIOS if it drifts).

run_experiments.py and run_drift_experiments.py only read these dicts.
"""

# Key = model name exactly as the static runner writes it to summary_<Model>.json.
# "default": False keeps a method out of the orchestrators unless asked for via --models.
MODELS = {
    "EquiWidth": {"static": "run_equiwidth.py", "drift": "run_equiwidth_drift.py"},
    "Hybrid": {"static": "run_hybrid.py", "drift": "run_hybrid_drift.py"},
    "TKHist": {"static": "run_tkhist.py", "drift": "run_tkhist_drift.py"},
    "EquiDepth": {"static": "run_equidepth.py", "drift": "run_equidepth_drift.py"},
    "STHist": {"static": "run_sthist.py", "drift": "run_sthist_drift.py"},
    "EntropyHist-MEA": {"static": "run_entropyhist.py", "drift": "run_entropyhist_drift.py"},
    # Исключён из дизайна эксперимента (слишком медленный); запускается через --models EquiHist
    "EquiHist": {"static": "run_equihist.py", "drift": "run_equihist_drift.py", "default": False},
}

DEFAULT_MODELS = [name for name, m in MODELS.items() if m.get("default", True)]

SYNTH_DISTS = ["normal", "zipf", "anti_zipf", "uniform", "sparse_cluster"]
DSB_COLUMNS = ["ws_item_sk", "cr_item_sk", "cr_returned_time_sk"]

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
    "dsb": {
        "experiment": "dsb_static",
        "datasets": [f"data/dsb/{c}" for c in DSB_COLUMNS],
        "workload": "{dataset}/workload_driven_*.csv",
        "prepare": ["prepare_dsb.py"],
    },
}

# shift_dir: folder with step_*.pkl.
# static_workload: workload evaluated at every step, or None to use the per-step
#                  step_N_workload_driven_<count>.csv (workload drifts with the data).
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
}

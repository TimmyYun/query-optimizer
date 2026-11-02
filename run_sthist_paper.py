#!/usr/bin/env python3
"""
Воспроизведение 1-D экспериментов статьи Aboulnaga & Chaudhuri (SIGMOD 1999)
для STHistogram в исходной постановке (не на данных бенчмарка):

  table2         — §5.1/§5.2, Table 2: Zipf z in {0, 0.5, 1, 2, 3},
                   с реструктуризацией и без; ошибка на тестовом ворклоаде
                   до и после обучения на 2000 запросах.
  data_shift     — §5.5: R2 = R1 без случайных 25% строк + столько же новых
                   строк Zipf(z=1); старая гистограмма -> фидбэк по R2.
  workload_shift — §5.4: локальность 80/20 переезжает на другие 20% домена.

Настройки по умолчанию — §5.1: T=100000, домен [1, 1000], V=200 различных
значений, B=100, R=200, m=0.025%, s=10%, alpha=0.5.

Метрика статьи — средняя относительная ошибка |act - est| / act по запросам
с act > 0 (в процентах). Дополнительно (не из статьи): median/p95/p99,
число нулевых запросов, их абсолютная ошибка и q-error со сглаживанием max(., 1).

Разделение ролей: данные и точные ответы живут только здесь (оракул);
гистограмма получает (T, min, max) и тройки (low, high, actual).
Train и test ворклоады генерируются независимыми сидами; тест не обучает.

    poetry run python run_sthist_paper.py --experiment-name sthist_paper
    poetry run python run_sthist_paper.py --experiment-name sthist_paper --seeds 42 43 44 45 46
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_utils import setup_out_dir
from models import STHistogram, STHistogramConfig, q_error_vec

# Table 2 статьи (тестовый ворклоад, средняя относительная ошибка, %)
PAPER_TABLE2 = {
    0.0: (3.34, 3.05),
    0.5: (4.44, 4.54),
    1.0: (9.39, 8.94),
    2.0: (130.52, 95.09),
    3.0: (306.79, 271.75),
}

# Метки независимых потоков случайности: rng_for(seed, experiment, stream)
DATA, DATA_NEW, TRAIN, TEST, TRAIN_NEW, TEST_NEW, LOCALITY = range(7)


# ----------------------------------------------------------------------
# Оракул: данные, ворклоады, точные ответы
# ----------------------------------------------------------------------


def zipf_frequencies(rng, T, dmin, dmax, V, z):
    """
    T строк на V случайных значениях домена; частоты по Zipf(z), ранги
    назначаются значениям случайно (§5.1). Округление до целых — методом
    наибольших остатков, чтобы сумма была ровно T.
    """
    D = dmax - dmin + 1
    values = rng.choice(D, size=V, replace=False)
    weights = rng.permutation(1.0 / np.arange(1, V + 1) ** z)
    quota = T * weights / weights.sum()
    counts = np.floor(quota).astype(np.int64)
    leftover = T - counts.sum()
    counts[np.argsort(-(quota - counts), kind="stable")[:leftover]] += 1

    freq = np.zeros(D, dtype=np.int64)
    freq[values] = counts
    return freq


def replace_rows(rng, freq, fraction, V, z):
    """§5.5: удалить случайные `fraction` строк и вставить столько же строк Zipf(z)."""
    n_replace = int(round(freq.sum() * fraction))
    kept = rng.multivariate_hypergeometric(freq, freq.sum() - n_replace)
    inserted = zipf_frequencies(rng, n_replace, 0, len(freq) - 1, V, z)
    return kept + inserted


def random_workload(rng, n, dmin, dmax):
    """Оба конца диапазона — равномерно по домену, затем сортировка."""
    ends = np.sort(rng.integers(dmin, dmax + 1, size=(n, 2)), axis=1)
    return ends[:, 0], ends[:, 1]


def locality_workload(rng, n, dmin, dmax, loc_lo, loc_hi, p_local=0.8):
    """
    §5.4: каждый конец с вероятностью p_local берётся из [loc_lo, loc_hi],
    иначе — равномерно из остальной части домена.
    """
    local = rng.integers(loc_lo, loc_hi + 1, size=(n, 2))
    # Вне локальности: равномерно по D - L значениям со сдвигом через окно
    outside = rng.integers(dmin, dmax + 1 - (loc_hi - loc_lo + 1), size=(n, 2))
    outside = np.where(outside >= loc_lo, outside + (loc_hi - loc_lo + 1), outside)
    ends = np.where(rng.random((n, 2)) < p_local, local, outside)
    ends = np.sort(ends, axis=1)
    return ends[:, 0], ends[:, 1]


def exact_counts(freq, dmin, lows, highs):
    ps = np.concatenate([[0], np.cumsum(freq)])
    return (ps[highs - dmin + 1] - ps[lows - dmin]).astype(np.float64)


# ----------------------------------------------------------------------
# Обучение и оценка
# ----------------------------------------------------------------------


def train(hist, lows, highs, actual):
    """Последовательный фидбэк. Возвращает prequential относительную ошибку (до обновления)."""
    rel_errors = np.full(len(lows), np.nan)
    for i, (lo, hi, act) in enumerate(zip(lows, highs, actual)):
        est = hist.observe_feedback(int(lo), int(hi), float(act)).estimate
        if act > 0:
            rel_errors[i] = abs(act - est) / act
    return rel_errors


def evaluate(hist, lows, highs, actual):
    """Оценка без обучения."""
    est = np.array([hist.estimate_range(int(lo), int(hi)) for lo, hi in zip(lows, highs)])
    pos = actual > 0
    rel = np.abs(actual[pos] - est[pos]) / actual[pos] * 100
    qe = q_error_vec(np.maximum(actual, 1.0), np.maximum(est, 1.0))
    return {
        "rel_err_mean_pct": float(rel.mean()),
        "rel_err_median_pct": float(np.median(rel)),
        "rel_err_p95_pct": float(np.percentile(rel, 95)),
        "rel_err_p99_pct": float(np.percentile(rel, 99)),
        "n_zero_queries": int((~pos).sum()),
        "zero_abs_err_mean": float(est[~pos].mean()) if (~pos).any() else 0.0,
        "qerr_median": float(np.median(qe)),
        "qerr_p95": float(np.percentile(qe, 95)),
    }


def prefixed(prefix, metrics):
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def new_histogram(args, restructure=True):
    cfg = STHistogramConfig(
        bucket_count=args.buckets,
        restructure_every=args.restructure_every,
        alpha=args.alpha,
        merge_threshold=args.merge_threshold,
        split_fraction=args.split_fraction,
        enable_restructuring=restructure,
    )
    return STHistogram(cfg, total_rows=args.rows, min_value=args.dmin, max_value=args.dmax)


def rng_for(seed, experiment, stream):
    return np.random.default_rng([seed, experiment, stream])


# ----------------------------------------------------------------------
# Эксперименты
# ----------------------------------------------------------------------


def run_table2(args, out_dir):
    records = []
    for seed in args.seeds:
        for zi, z in enumerate(args.zipf):
            freq = zipf_frequencies(rng_for(seed, zi, DATA), args.rows, args.dmin, args.dmax, args.distinct, z)
            tr_lo, tr_hi = random_workload(rng_for(seed, zi, TRAIN), args.train_queries, args.dmin, args.dmax)
            te_lo, te_hi = random_workload(rng_for(seed, zi, TEST), args.test_queries, args.dmin, args.dmax)
            tr_act = exact_counts(freq, args.dmin, tr_lo, tr_hi)
            te_act = exact_counts(freq, args.dmin, te_lo, te_hi)

            # Одни и те же данные и ворклоады для обеих веток абляции
            for restructure in (False, True):
                hist = new_histogram(args, restructure)
                before = evaluate(hist, te_lo, te_hi, te_act)
                preq = train(hist, tr_lo, tr_hi, tr_act)
                after = evaluate(hist, te_lo, te_hi, te_act)

                paper = PAPER_TABLE2.get(float(z), (np.nan, np.nan))[int(restructure)]
                records.append({
                    "seed": seed,
                    "z": z,
                    "restructure": restructure,
                    "paper_rel_err_pct": paper,
                    **prefixed("before", before),
                    **prefixed("after", after),
                    "prequential_rel_err_pct": float(np.nanmean(preq) * 100),
                    "final_mass_ratio": hist.total_mass / args.rows,
                    "events": json.dumps(dict(hist.events)),
                })
                print(
                    f"seed={seed} z={z:<4} restructure={restructure!s:<5} | test rel.err: "
                    f"before={before['rel_err_mean_pct']:8.2f}%  after={after['rel_err_mean_pct']:8.2f}%  "
                    f"(paper: {paper:.2f}%)"
                )

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "table2.csv", index=False)

    # При z >= 2 ошибка сильно зависит от сида — сравнивать с статьёй по среднему
    agg = (
        df.groupby(["z", "restructure"])
        .agg(
            paper=("paper_rel_err_pct", "first"),
            after_mean=("after_rel_err_mean_pct", "mean"),
            after_min=("after_rel_err_mean_pct", "min"),
            after_max=("after_rel_err_mean_pct", "max"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "table2_agg.csv", index=False)
    print(f"\nTable 2 over {len(args.seeds)} seed(s), test rel.err after training, %:")
    print(agg.to_string(index=False, float_format="%.2f"))
    return df


def run_data_shift(args, out_dir):
    exp, seed = 100, args.seeds[0]  # отдельное пространство сидов
    freq_r1 = zipf_frequencies(rng_for(seed, exp, DATA), args.rows, args.dmin, args.dmax, args.distinct, args.base_z)
    freq_r2 = replace_rows(rng_for(seed, exp, DATA_NEW), freq_r1, args.replace_fraction, args.distinct, args.insert_z)
    assert freq_r2.sum() == freq_r1.sum()  # T не меняется -> T_reference не трогаем

    def workload(stream, freq):
        lo, hi = random_workload(rng_for(seed, exp, stream), args.train_queries, args.dmin, args.dmax)
        return lo, hi, exact_counts(freq, args.dmin, lo, hi)

    r1_train = workload(TRAIN, freq_r1)
    r1_test = workload(TEST, freq_r1)
    # Новые запросы по R2 — независимые сиды; тест R2 — тот же диапазонный ворклоад, что и тест R1
    r2_train = workload(TRAIN_NEW, freq_r2)
    r2_test = (r1_test[0], r1_test[1], exact_counts(freq_r2, args.dmin, r1_test[0], r1_test[1]))

    hist = new_histogram(args)
    rows = [{"phase": "R1 untrained", **evaluate(hist, *r1_test)}]
    preq_r1 = train(hist, *r1_train)
    rows.append({"phase": "R1 trained", **evaluate(hist, *r1_test)})
    rows.append({"phase": "R2 stale (after shift, no feedback)", **evaluate(hist, *r2_test)})
    preq_r2 = train(hist, *r2_train)
    rows.append({"phase": "R2 refined", **evaluate(hist, *r2_test)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "data_shift.csv", index=False)
    print(df[["phase", "rel_err_mean_pct", "qerr_median"]].to_string(index=False))

    plot_recovery(
        [preq_r1, preq_r2],
        ["R1 feedback", "R2 feedback (after 25% replacement)"],
        out_dir / "data_shift_recovery.png",
        "Data shift (§5.5): prequential error",
    )
    return df


def run_workload_shift(args, out_dir):
    exp, seed = 200, args.seeds[0]
    freq = zipf_frequencies(rng_for(seed, exp, DATA), args.rows, args.dmin, args.dmax, args.distinct, args.base_z)

    # Две непересекающиеся области по 20% домена
    D = args.dmax - args.dmin + 1
    L = int(D * 0.2)
    starts = rng_for(seed, exp, LOCALITY).choice(np.arange(0, D - L + 1, L), size=2, replace=False)
    w1 = (int(args.dmin + starts[0]), int(args.dmin + starts[0] + L - 1))
    w2 = (int(args.dmin + starts[1]), int(args.dmin + starts[1] + L - 1))
    print(f"W1 locality = {w1}, W2 locality = {w2}")

    def workload(stream, loc):
        lo, hi = locality_workload(rng_for(seed, exp, stream), args.train_queries, args.dmin, args.dmax, *loc)
        return lo, hi, exact_counts(freq, args.dmin, lo, hi)

    w1_train, w1_test = workload(TRAIN, w1), workload(TEST, w1)
    w2_train, w2_test = workload(TRAIN_NEW, w2), workload(TEST_NEW, w2)

    hist = new_histogram(args)
    preq_w1 = train(hist, *w1_train)
    rows = [
        {"phase": "trained on W1, test W1", **evaluate(hist, *w1_test)},
        {"phase": "trained on W1, test W2", **evaluate(hist, *w2_test)},
    ]
    preq_w2 = train(hist, *w2_train)
    rows += [
        {"phase": "trained on W1+W2, test W2", **evaluate(hist, *w2_test)},
        {"phase": "trained on W1+W2, test W1", **evaluate(hist, *w1_test)},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "workload_shift.csv", index=False)
    print(df[["phase", "rel_err_mean_pct", "qerr_median"]].to_string(index=False))

    plot_recovery(
        [preq_w1, preq_w2],
        ["W1 feedback", "W2 feedback (locality moved)"],
        out_dir / "workload_shift_recovery.png",
        "Workload shift (§5.4): prequential error",
    )
    return df


def plot_recovery(phases, labels, path, title, window=100):
    """Скользящее среднее prequential относительной ошибки по фазам подряд."""
    fig, ax = plt.subplots(figsize=(9, 4))
    offset = 0
    for errs, label in zip(phases, labels):
        rolling = pd.Series(errs * 100).rolling(window, min_periods=1).mean()
        ax.plot(np.arange(offset, offset + len(errs)), rolling, label=label)
        offset += len(errs)
        ax.axvline(offset, color="grey", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("feedback query #")
    ax.set_ylabel(f"rel. error, % (rolling mean, w={window})")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reproduce 1-D ST-histogram experiments (Aboulnaga & Chaudhuri 1999)")
    parser.add_argument("--experiments", nargs="+", default=["table2", "data_shift", "workload_shift"],
                        choices=["table2", "data_shift", "workload_shift"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Table 2 is averaged over these seeds; shift experiments use the first one")
    # Данные (§5.1)
    parser.add_argument("--rows", type=int, default=100_000, help="T")
    parser.add_argument("--dmin", type=int, default=1)
    parser.add_argument("--dmax", type=int, default=1000)
    parser.add_argument("--distinct", type=int, default=200, help="V distinct values")
    parser.add_argument("--zipf", type=float, nargs="+", default=[0, 0.5, 1, 2, 3])
    parser.add_argument("--base-z", type=float, default=1.0, help="Zipf z of base data for shift experiments")
    parser.add_argument("--insert-z", type=float, default=1.0, help="§5.5: Zipf z of inserted rows")
    parser.add_argument("--replace-fraction", type=float, default=0.25, help="§5.5: fraction of rows replaced")
    # Ворклоад
    parser.add_argument("--train-queries", type=int, default=2000)
    parser.add_argument("--test-queries", type=int, default=2000)
    # Гистограмма (§5.1)
    parser.add_argument("--buckets", type=int, default=100)
    parser.add_argument("--restructure-every", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--merge-threshold", type=float, default=0.00025)
    parser.add_argument("--split-fraction", type=float, default=0.10)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    out_dir = setup_out_dir(args, "sthist_paper")
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if "table2" in args.experiments:
        print("\n>>> Table 2: Zipf skew x restructuring <<<")
        run_table2(args, out_dir)
    if "data_shift" in args.experiments:
        print("\n>>> §5.5: 25% delete + Zipf insert <<<")
        run_data_shift(args, out_dir)
    if "workload_shift" in args.experiments:
        print("\n>>> §5.4: locality shift <<<")
        run_workload_shift(args, out_dir)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Глобальный лимит домена
DOMAIN_MAX = 1_000_000


def plot_query_distribution(dist_dir: Path, workload_file: Path):
    dist_name = dist_dir.name
    print(f"[{dist_name}] Processing workload: {workload_file.name}...")

    # 1. Загрузка бакетов гистограммы (из datasets.py)
    buckets_file = dist_dir / "histogram_buckets.csv"
    if not buckets_file.exists():
        print(f"[{dist_name}] Skipped: buckets file not found.")
        return

    buckets_df = pd.read_csv(buckets_file)
    bin_starts = buckets_df["bin_start"].values
    bin_ends = buckets_df["bin_end"].values
    bin_counts = buckets_df["count"].values

    # 2. Загрузка Data-Driven ворклоада
    queries_df = pd.read_csv(workload_file, usecols=["low", "high"])
    q_low = queries_df["low"].values
    q_high = queries_df["high"].values

    n_queries = len(q_low)
    n_buckets = len(bin_starts)

    # 3. Векторизованный расчет пересечений (Broadcasting)
    # Создаем матрицы (Q, B), где Q - запросы, B - бакеты
    Q_low = q_low[:, np.newaxis]
    Q_high = q_high[:, np.newaxis]
    B_start = bin_starts[np.newaxis, :]
    B_end = bin_ends[np.newaxis, :]

    # overlap_matrix[i, j] == True, если запрос i пересекает бакет j
    overlap_matrix = (Q_low < B_end) & (Q_high > B_start)

    # 4. Определение валидных запросов (Selectivity > 0)
    # Запрос валиден, если он пересекает хотя бы один бакет, в котором есть данные
    non_empty_buckets_mask = bin_counts > 0
    overlap_with_data = overlap_matrix & non_empty_buckets_mask[np.newaxis, :]
    valid_queries_mask = np.any(overlap_with_data, axis=1)

    valid_count = np.sum(valid_queries_mask)
    print(f"[{dist_name}] Valid queries: {valid_count} / {n_queries}")

    # 5. Подсчет "хитов" по бакетам для валидных запросов
    valid_overlap_matrix = overlap_matrix[valid_queries_mask]
    bucket_hit_counts = np.sum(valid_overlap_matrix, axis=0)

    # 6. Визуализация
    plt.figure(figsize=(12, 6))

    # Рисуем гистограмму распределения запросов по индексам бакетов
    plt.bar(
        range(n_buckets),
        bucket_hit_counts,
        width=1.0,
        align="edge",
        color="salmon",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.2,
    )

    plt.xlabel("Bucket Index (from histogram_buckets.csv)")
    plt.ylabel("Number of Intersecting Queries")
    plt.title(
        f"Query Distribution over Buckets: {dist_name}\n"
        f"(Workload: {workload_file.name} | Valid Queries: {valid_count})"
    )
    plt.grid(axis="y", alpha=0.3)

    # Добавляем инфо-текст
    plt.figtext(0.02, 0.02, f"Total Queries in CSV: {n_queries}", fontsize=9)

    # Сохраняем в папку рядом с ворклоадом
    output_path = dist_dir / f"plot_distribution_{workload_file.stem}.png"
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[{dist_name}] Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot query distribution for generated workloads."
    )
    parser.add_argument(
        "--rows", type=str, default="60000000_hard", help="Dataset folder name."
    )
    args = parser.parse_args()

    base_dir = Path("data/generated") / args.rows
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found.")
        return

    # Проходим по всем папкам распределений (zipf, normal, etc.)
    for dist_dir in base_dir.iterdir():
        if dist_dir.is_dir():
            # Ищем файлы ворклоадов внутри папки
            workload_files = list(dist_dir.glob("workload_driven_*.csv"))

            for wf in workload_files:
                try:
                    plot_query_distribution(dist_dir, wf)
                except Exception as e:
                    print(f"Error processing {dist_dir.name} with {wf.name}: {e}")


if __name__ == "__main__":
    main()

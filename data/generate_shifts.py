#!/usr/bin/env python3
import sys
from pathlib import Path

# Фиксим проблему с путями: добавляем корень проекта
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import io

from data.datasets import DatasetManager, DOMAIN_MAX


def align_frequency(freq, mn):
    """Выравнивает массив частот по глобальному домену [0, DOMAIN_MAX]."""
    aligned = np.zeros(DOMAIN_MAX + 1, dtype=np.float64)
    length = min(len(freq), DOMAIN_MAX + 1 - mn)
    aligned[mn : mn + length] = freq[:length]
    return aligned


def generate_shift_sequence(
    rows: str, init_dist: str, target_dist: str, steps: int = 20, seed: int = 42
):
    dm = DatasetManager()
    rng = np.random.default_rng(seed)

    # Загружаем исходные метаданные (rows теперь принимает строку, например "60000000_hard")
    init_dir = dm.base_path / "generated" / rows / init_dist
    target_dir = dm.base_path / "generated" / rows / target_dist

    if not init_dir.exists() or not target_dir.exists():
        raise FileNotFoundError(
            f"Missing base datasets in {dm.base_path}/generated/{rows}"
        )

    with open(init_dir / "meta.pkl", "rb") as f:
        mn_i, mx_i, N_i, freq_i, sample_i, k_i, skew_i, kurt_i = pickle.load(f)

    with open(target_dir / "meta.pkl", "rb") as f:
        mn_t, mx_t, N_t, freq_t, sample_t, k_t, skew_t, kurt_t = pickle.load(f)

    # Выравниваем частоты
    f_init = align_frequency(freq_i, mn_i)
    f_target = align_frequency(freq_t, mn_t)

    max_y_freq = np.max(f_init + f_target)

    out_dir = dm.base_path / "generated" / rows / f"shift_{init_dist}_to_{target_dist}_5%_workload"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {steps + 1} shift steps: {init_dist} -> {target_dist} ({rows} rows)"
    )

    frames = []
    TARGET_SAMPLE_SIZE = 100_000

    for step in range(steps + 1):
        alpha = step / steps
        print(f"Processing Shift {alpha:>4.0%}...")

        # 1. Смешиваем частоты (Ground Truth)
        # Суммируем 100% начального и постепенно добавляем до 100% от целевого
        mixed_freq_float = f_init + alpha * f_target
        mixed_freq = np.round(mixed_freq_float).astype(np.int64)

        active_indices = np.nonzero(mixed_freq)[0]
        step_mn = int(active_indices[0]) if len(active_indices) > 0 else 0
        step_mx = int(active_indices[-1]) if len(active_indices) > 0 else 0
        trimmed_freq = mixed_freq[step_mn : step_mx + 1]

        step_N = int(np.sum(mixed_freq))

        # ========================================================
        # 2. ЧЕСТНЫЙ RESERVOIR SAMPLING (Копия из datasets.py)
        # ========================================================
        # Разворачиваем частоты в сырой массив данных (имитация базы данных)
        # np.repeat берет индексы домена и дублирует их согласно mixed_freq
        domain_vals = np.arange(len(mixed_freq))
        vals = np.repeat(domain_vals, mixed_freq)

        # Считаем вероятность p, точно как в build_frequency_and_sample
        p = min(1.0, float(TARGET_SAMPLE_SIZE * 1.5) / float(max(step_N, 1)))

        # Генерируем маску вероятности
        mask = rng.random(vals.size) < p
        s = vals[mask]

        # Обрезаем точно до 100k
        if s.size > TARGET_SAMPLE_SIZE:
            mixed_sample = rng.choice(s, size=TARGET_SAMPLE_SIZE, replace=False)
        else:
            mixed_sample = s
        # ========================================================

        step_k = int((1.0 - alpha) * k_i + alpha * k_t)

        # Сохраняем шаг
        step_file = out_dir / f"step_{step}.pkl"
        with open(step_file, "wb") as f:
            pickle.dump(
                (step_mn, step_mx, step_N, trimmed_freq, mixed_sample, step_k), f
            )

        # --- ГЕНЕРАЦИЯ КАДРА ДЛЯ GIF ---
        plt.figure(figsize=(10, 6))
        vis_bins = 500
        chunk_size = max(1, len(mixed_freq) // vis_bins)
        vis_freq = np.array(
            [
                np.sum(mixed_freq[i : i + chunk_size])
                for i in range(0, len(mixed_freq), chunk_size)
            ]
        )
        x_axis = np.linspace(0, len(mixed_freq), len(vis_freq))

        r, g, b = alpha, 0.2, 1.0 - alpha
        plt.fill_between(x_axis, vis_freq, color=(r, g, b), alpha=0.7)
        plt.yscale("log")
        plt.title(
            f"Distribution Shift: {init_dist.capitalize()} -> {target_dist.capitalize()} ({alpha:.0%})"
        )
        plt.xlabel("Domain Value")
        plt.ylabel("Frequency (Log Scale)")
        plt.ylim(1, max_y_freq * 1.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

    gif_path = out_dir / f"animation_{init_dist}_to_{target_dist}.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=600, loop=0
    )
    print(f"\nAwesome! Sequence and animation saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate distribution shifts.")
    parser.add_argument(
        "--rows",
        type=str,
        default="60000000",
        help="Name of the folder inside data/generated/",
    )
    parser.add_argument("--init", type=str, default="normal")
    parser.add_argument("--target", type=str, default="zipf")
    args = parser.parse_args()

    generate_shift_sequence(args.rows, args.init, args.target)

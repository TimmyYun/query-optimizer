"""The sample must be representative of the generated data."""
import numpy as np

from data.datasets import build_frequency_and_sample, gen_values, save_csv_column


def test_simple_zipf_sample_covers_tail(tmp_path):
    # Same seeds as DatasetManager.prepare_dataset: generation 42, sample [42, 1].
    # With seed 42 for both, the mask repeated the generation's uniforms and the
    # sample held only values < 1000 although ~4.5 % of rows are above.
    n = 2_000_000
    vals = gen_values(np.random.default_rng(42), "zipf", n, 0, 1_000_000, use_micro_dist=False)
    path = tmp_path / "data.csv"
    save_csv_column(vals, path)

    mn, mx = int(vals.min()), int(vals.max())
    _, sample = build_frequency_and_sample(path, mn, mx, n, 20_000, [42, 1])

    true_tail = (vals >= 1000).mean()
    sample_tail = (sample >= 1000).mean()
    assert true_tail > 0.03
    assert abs(sample_tail - true_tail) < 0.01

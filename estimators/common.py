"""
Урок 0: общие кирпичики, которыми пользуются все методы и стенд.

Заполни функции вместо `raise NotImplementedError`. Проверка:
    poetry run python -m pytest tests/lessons/test_00_common.py -v
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RangeQuery:
    """Запрос WHERE x BETWEEN low AND high — обе границы ВКЛЮЧИТЕЛЬНО."""

    low: int
    high: int


def q_error(y_true, y_pred) -> np.ndarray:
    """
    Во сколько раз ошиблись, для каждого запроса: max(pred/true, true/pred).

    y_true, y_pred — списки или массивы одинаковой длины (число строк).
    Перед делением обе стороны прижать снизу к 1 строке (np.maximum(..., 1.0)),
    иначе при true = 0 получится деление на ноль.

    Пример: q_error([100], [50]) -> array([2.])
    """
    raise NotImplementedError


def true_counts(freq, mn, lows, highs) -> np.ndarray:
    """
    Точное число строк для каждого запроса [lows[i], highs[i]].

    freq[j] — сколько строк со значением mn + j (это истина из meta.pkl).
    Без цикла по запросам — через префиксные суммы:
        ps = [0, freq[0], freq[0]+freq[1], ...]        (длина len(freq) + 1)
        count(a, b) = ps[b - mn + 1] - ps[a - mn]
    Части запроса вне [mn, mx] дают 0: индексы обрезать (np.clip) до [0, len(freq)].

    Пример: freq = [5, 0, 2, 7], mn = 10  (значения 10..13)
        [10, 10] -> 5,   [11, 13] -> 9,   [8, 10] -> 5,   [20, 30] -> 0
    """
    raise NotImplementedError


def summarize(qe) -> dict:
    """
    Сводка по массиву q-error: median, p25, p75, p95, avg, max (обычные float).
    Подсказка: np.median, np.percentile(qe, 95), np.mean, np.max.
    """
    raise NotImplementedError

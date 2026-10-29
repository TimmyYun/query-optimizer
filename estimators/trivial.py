"""
Урок 0: оценка БЕЗ статистики — точка отсчёта для всех методов.

Считаем, что N строк размазаны ровно по всему домену [mn, mx].
Проверка на реальных данных:
    poetry run python lessons/check/check_static.py trivial
"""


class Trivial:
    def __init__(self, mn, mx, total_rows):
        self.mn = mn
        self.mx = mx
        self.total_rows = total_rows
        self.bucket_ranges = [(mn, mx)]  # один "бакет" на весь домен

    def predict(self, q) -> float:
        """
        Оценка числа строк для q.low..q.high (включительно).

        1. Обрезать запрос по домену: lo = max(q.low, mn), hi = min(q.high, mx).
        2. Если lo > hi — запрос мимо данных, ответ 0.
        3. Иначе: total_rows * (сколько значений покрыто) / (сколько значений в домене).
           Сколько значений в [lo, hi] — помни про включительные границы.
        """
        raise NotImplementedError

    def n_floats(self) -> int:
        return 3  # mn, mx, N


def build(sample, mn, mx, total_rows, n_buckets):
    # sample и n_buckets не нужны: у тривиальной оценки нет статистики
    return Trivial(mn, mx, total_rows)

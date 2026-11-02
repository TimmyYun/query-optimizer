import bisect
import math

import numpy as np

from .common import RangeQuery


class EquiDepthHistogram:
    """
    Equi-Depth гистограмма + MCV-список, как в PostgreSQL (pg_stats).

    Построение повторяет ANALYZE (src/backend/commands/analyze.c,
    compute_scalar_stats / analyze_mcv_list):
      * выборка из 300 * statistics_target строк;
      * most_common_vals / most_common_freqs — значения, встречающиеся в
        выборке значимо чаще среднего (не более statistics_target штук);
      * histogram_bounds — statistics_target + 1 границ, делящих ОСТАЛЬНЫЕ
        (не-MCV) значения выборки на бакеты одинаковой частоты;
      * n_distinct — оценка Haas–Stokes (Duj1).

    Оценка повторяет планировщик PostgreSQL 17 (selfuncs.c: scalarineqsel,
    ineq_histogram_selectivity, var_eq_const; clausesel.c) и
    https://www.postgresql.org/docs/current/row-estimation-examples.html:
      sel(x <= c) = mcv_selectivity + histogram_selectivity * histogram_fraction
      sel(a <= x <= b) = sel(x <= b) + sel(x >= a) - 1
      sel(x = c) = mcf[c], либо (1 - sum(mcf)) / (n_distinct - num_mcv)
    Точечный запрос [c, c] оценивается как "x = c".
    NULL-ов в бенчмарке нет, поэтому null_frac = 0.

    actual_min / actual_max имитируют B-tree индекс по столбцу: когда поиск по
    гистограмме доходит до крайней границы, планировщик подставляет реальные
    min/max из индекса (get_actual_variable_range) и не отсекает маленькие
    оценки. Без индекса оценки по одной стороне ограничены снизу
    0.01 / num_buckets, и узкий диапазон в крайнем бакете схлопывается в ~0.
    """

    DEFAULT_RANGE_INEQ_SEL = 0.005  # selfuncs.h

    def __init__(
        self,
        total_rows,
        mcv_vals,
        mcv_freqs,
        hist_bounds,
        n_distinct,
        actual_min=None,
        actual_max=None,
    ):
        self.total_rows = total_rows
        self.n_distinct = n_distinct
        # Реальные min/max столбца из индекса; None -> индекса нет
        self.actual_min = actual_min
        self.actual_max = actual_max

        # MCV храним отсортированными по значению: сумма частот MCV на
        # отрезке = разность префиксных сумм.
        order = np.argsort(mcv_vals, kind="stable")
        self.mcv_vals = [int(v) for v in np.asarray(mcv_vals)[order]]
        self.mcv_freqs = [float(f) for f in np.asarray(mcv_freqs)[order]]
        self.mcv_prefix = [0.0]
        for f in self.mcv_freqs:
            self.mcv_prefix.append(self.mcv_prefix[-1] + f)
        self.sumcommon = self.mcv_prefix[-1]

        self.hist_bounds = [float(b) for b in hist_bounds]

    @property
    def num_mcv(self) -> int:
        return len(self.mcv_vals)

    @property
    def n_buckets(self) -> int:
        return max(0, len(self.hist_bounds) - 1)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _estimate_n_distinct(ndistinct, nmultiple, samplerows, totalrows):
        """Оценка числа различных значений в таблице (Haas–Stokes Duj1)."""
        if nmultiple == 0:
            # Все значения в выборке уникальны -> считаем столбец уникальным
            return float(totalrows)
        if nmultiple == ndistinct:
            # Каждое значение встретилось >= 2 раз -> видели все значения
            return float(ndistinct)
        f1 = ndistinct - nmultiple
        n, N = samplerows, totalrows
        est = (n * ndistinct) / ((n - f1) + f1 * n / N)
        est = min(max(est, ndistinct), N)
        return math.floor(est + 0.5)

    @staticmethod
    def _analyze_mcv_list(mcv_counts, n_distinct, samplerows, totalrows):
        """
        Сколько кандидатов оставить в MCV-списке (analyze_mcv_list).
        Отбрасываем наименее частые значения, пока частота значения не станет
        значимо (2 стандартных отклонения) выше средней частоты не-MCV значений.
        """
        num_mcv = len(mcv_counts)
        if samplerows >= totalrows:
            return num_mcv

        sumcount = float(sum(mcv_counts[: num_mcv - 1]))
        while num_mcv > 0:
            # Частота наименее частого кандидата, если бы его не было в MCV
            selec = min(max(1.0 - sumcount / samplerows, 0.0), 1.0)
            otherdistinct = n_distinct - (num_mcv - 1)
            if otherdistinct > 1:
                selec /= otherdistinct

            # Дисперсия гипергеометрического распределения для count в выборке
            N, n = totalrows, samplerows
            K = N * mcv_counts[num_mcv - 1] / n
            variance = n * K * (N - K) * (N - n) / (N * N * (N - 1))
            stddev = math.sqrt(variance)

            if mcv_counts[num_mcv - 1] > selec * samplerows + 2 * stddev + 0.5:
                break  # значимо чаще среднего -> оставляем его и все более частые
            num_mcv -= 1
            if num_mcv == 0:
                break
            sumcount -= mcv_counts[num_mcv - 1]
        return num_mcv

    @staticmethod
    def build_from_sample(
        sample: np.ndarray,
        total_rows: int,
        statistics_target: int = 100,
        seed: int = 42,
        actual_min: int = None,
        actual_max: int = None,
    ) -> "EquiDepthHistogram":
        """
        ANALYZE: из sample берётся 300 * statistics_target строк (как в
        PostgreSQL), по ним строятся MCV, histogram_bounds и n_distinct.
        actual_min / actual_max передаются, если по столбцу есть индекс.
        """
        sample = np.asarray(sample, dtype=np.int64)
        if len(sample) == 0:
            return EquiDepthHistogram(total_rows, [], [], [], 0)

        target_rows = 300 * statistics_target
        if len(sample) > target_rows:
            rng = np.random.default_rng(seed)
            sample = rng.choice(sample, size=target_rows, replace=False)
        samplerows = len(sample)

        values, counts = np.unique(sample, return_counts=True)
        ndistinct = len(values)
        nmultiple = int(np.sum(counts > 1))
        n_distinct = EquiDepthHistogram._estimate_n_distinct(
            ndistinct, nmultiple, samplerows, total_rows
        )

        # Кандидаты в MCV: значения, встретившиеся > 1 раза, по убыванию частоты
        # (при равенстве — меньшее значение первым), не более statistics_target
        dup = np.nonzero(counts > 1)[0]
        dup = dup[np.argsort(-counts[dup], kind="stable")]
        track = dup[:statistics_target]

        if len(track) == ndistinct and nmultiple == ndistinct:
            # Все значения выборки попали в список и помещаются -> храним все
            num_mcv = len(track)
        else:
            num_mcv = EquiDepthHistogram._analyze_mcv_list(
                counts[track].tolist(), n_distinct, samplerows, total_rows
            )
        mcv_idx = track[:num_mcv]
        mcv_vals = values[mcv_idx]
        mcv_freqs = counts[mcv_idx] / samplerows

        # Гистограмма строится по выборке без MCV-значений
        hist_bounds = []
        num_hist = ndistinct - num_mcv
        if num_hist > statistics_target:
            num_hist = statistics_target + 1
        if num_hist >= 2:
            rest = np.sort(sample[~np.isin(sample, mcv_vals)])
            nvals = len(rest)
            # Равномерно расставленные позиции, включая min и max (analyze.c)
            delta, deltafrac = divmod(nvals - 1, num_hist - 1)
            pos, posfrac = 0, 0
            for _ in range(num_hist):
                hist_bounds.append(rest[pos])
                pos += delta
                posfrac += deltafrac
                if posfrac >= num_hist - 1:
                    pos += 1
                    posfrac -= num_hist - 1

        return EquiDepthHistogram(
            total_rows,
            mcv_vals,
            mcv_freqs,
            hist_bounds,
            n_distinct,
            actual_min,
            actual_max,
        )

    # --------------------------------------------------------------- estimate
    def _eq_selec_non_mcv(self) -> float:
        """1 / (n_distinct - num_mcv): частота одного не-MCV значения в гистограмме."""
        other_distinct = self.n_distinct - self.num_mcv
        return 1.0 / other_distinct if other_distinct > 1 else 0.0

    def _mcv_selectivity(self, c: int, isgt: bool, iseq: bool) -> float:
        """Сумма частот MCV, удовлетворяющих x < c / x <= c / x > c / x >= c."""
        n_lt = bisect.bisect_left(self.mcv_vals, c)  # число MCV < c
        n_le = bisect.bisect_right(self.mcv_vals, c)  # число MCV <= c
        if not isgt:
            return self.mcv_prefix[n_le if iseq else n_lt]
        return self.sumcommon - self.mcv_prefix[n_lt if iseq else n_le]

    def _histogram_selectivity(self, c: int, isgt: bool, iseq: bool) -> float:
        """
        Доля не-MCV популяции, удовлетворяющей неравенству
        (ineq_histogram_selectivity): (целые бакеты до c + линейная доля
        бакета с c) / num_buckets. Возвращает -1, если гистограммы нет.
        """
        bounds = self.hist_bounds
        n = len(bounds)
        if n < 2:
            return -1.0
        has_index = self.actual_min is not None

        # Крайние границы; при наличии индекса заменяются реальными min/max,
        # как только бинарный поиск до них доходит (get_actual_variable_range)
        first, last = bounds[0], bounds[-1]
        have_end = False
        if has_index and n == 2:
            first, last = self.actual_min, self.actual_max
            have_end = True

        # Бинарный поиск правой границы бакета, содержащего c
        lobound, hibound = 0, n
        while lobound < hibound:
            probe = (lobound + hibound) // 2
            if probe == 0:
                if has_index and n > 2:
                    first = self.actual_min
                    have_end = True
                value = first
            elif probe == n - 1:
                if has_index and n > 2:
                    last = self.actual_max
                    have_end = True
                value = last
            else:
                value = bounds[probe]

            if isgt:  # оператор > / >=, результат инвертируется
                ltcmp = not (value >= c if iseq else value > c)
            else:  # оператор < / <=
                ltcmp = value <= c if iseq else value < c
            if ltcmp:
                lobound = probe + 1
            else:
                hibound = probe

        if lobound <= 0:
            histfrac = 0.0  # c ниже гистограммы
        elif lobound >= n:
            histfrac = 1.0  # c выше гистограммы
        else:
            i = lobound
            low = first if i - 1 == 0 else bounds[i - 1]
            high = last if i == n - 1 else bounds[i]

            eq_selec = 0.0
            if i == 1 or isgt == iseq:
                eq_selec = self._eq_selec_non_mcv()

            # Линейная интерполяция внутри бакета
            if high <= low:
                binfrac = 0.5
            elif c <= low:
                binfrac = 0.0
            elif c >= high:
                binfrac = 1.0
            else:
                binfrac = (c - low) / (high - low)

            # histfrac — оценка доли "x <= c"
            histfrac = (i - 1 + binfrac) / (n - 1)
            # Первый бакет уже остальных на eq_selec: "x <= first" = eq_selec
            if i == 1:
                histfrac += eq_selec * (1.0 - binfrac)
            # Для "<" и ">=" вычитаем долю "x = c"
            if isgt == iseq:
                histfrac -= eq_selec

        hist_selec = 1.0 - histfrac if isgt else histfrac

        if have_end:
            return min(max(hist_selec, 0.0), 1.0)
        # Границы гистограммы приблизительны: не доверяем совсем маленьким и
        # совсем большим оценкам (сотая доля разрешения гистограммы)
        cutoff = 0.01 / (n - 1)
        return min(max(hist_selec, cutoff), 1.0 - cutoff)

    def _scalarineqsel(self, c: int, isgt: bool, iseq: bool) -> float:
        """sel = mcv_selectivity + histogram_selectivity * histogram_fraction."""
        hist_selec = self._histogram_selectivity(c, isgt, iseq)
        selec = 1.0 - self.sumcommon  # доля популяции, описанная гистограммой
        if hist_selec >= 0.0:
            selec *= hist_selec
        else:
            selec *= 0.5  # нет гистограммы: половина не-MCV популяции
        selec += self._mcv_selectivity(c, isgt, iseq)
        return min(max(selec, 0.0), 1.0)

    def _selectivity_eq(self, c: int) -> float:
        """sel(x = c) (var_eq_const): частота из MCV, иначе равномерно по остальным."""
        i = bisect.bisect_left(self.mcv_vals, c)
        if i < self.num_mcv and self.mcv_vals[i] == c:
            return self.mcv_freqs[i]
        selec = 1.0 - self.sumcommon
        other_distinct = self.n_distinct - self.num_mcv
        if other_distinct > 1:
            selec /= other_distinct
        # Не-MCV значение не может быть частым, чем наименее частое MCV
        if self.num_mcv > 0:
            selec = min(selec, min(self.mcv_freqs))
        return min(max(selec, 0.0), 1.0)

    def selectivity(self, q: RangeQuery) -> float:
        # Точечный запрос на целочисленном столбце = предикат "x = c"
        if q.low == q.high:
            return self._selectivity_eq(q.low)

        # low <= x <= high: пара неравенств (clauselist_selectivity)
        s_lo = self._scalarineqsel(q.low, isgt=True, iseq=True)  # x >= low
        s_hi = self._scalarineqsel(q.high, isgt=False, iseq=True)  # x <= high
        s = s_hi + s_lo - 1.0

        if s <= 0.0:
            # Слабо отрицательное — ошибка округления на очень узком диапазоне
            s = self.DEFAULT_RANGE_INEQ_SEL if s < -0.01 else 1.0e-10
        return min(max(s, 0.0), 1.0)

    def predict(self, q: RangeQuery) -> float:
        return self.selectivity(q) * self.total_rows

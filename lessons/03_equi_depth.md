# Урок 3. EquiDepth — как на самом деле считает PostgreSQL

**Ветка:** `lesson/03-equi-depth` (от `lesson/02-tkhist`)
**Пишешь:** `estimators/equi_depth.py`
**Тесты:** `tests/lessons/test_03_equi_depth.py`
**Эталон:** `feature/new_baseline:models/equi_depth.py`
**Первоисточник:** PostgreSQL 17 — `src/backend/commands/analyze.c` (`compute_scalar_stats`,
`analyze_mcv_list`), `src/backend/utils/adt/selfuncs.c` (`scalarineqsel`,
`ineq_histogram_selectivity`, `var_eq_const`), `src/backend/optimizer/path/clausesel.c`
(`clauselist_selectivity` — пара неравенств), документация
<https://www.postgresql.org/docs/current/row-estimation-examples.html>
**Время:** 4–6 часов (самый «инженерный» урок)

---

## Зачем

Это бейзлайн, который рецензент примет без вопросов: «а что делает настоящая СУБД?».
Две новые идеи по сравнению с уроками 1–2:
1. **Равная глубина**: границы бакетов ставятся так, чтобы в каждом было **одинаковое число строк**.
   В густых местах бакеты узкие, в пустых — широкие. Сетка подстраивается под данные.
2. **MCV-список** — глобальный (не по бакетам) список частых значений, отделённый от гистограммы.

Плюс много мелких решений PostgreSQL, каждое из которых влияет на хвост ошибок.

## Теория

### ANALYZE (построение)

1. **Выборка**: `300 * statistics_target` строк (по умолчанию target = 100 → 30 000).
   У нас выборка уже 100k — берём из неё 30 000 случайных (seed 42).
2. **n_distinct** — оценка Haas–Stokes (Duj1) по выборке:
   - `d` — различных значений в выборке, `f1` — встретившихся ровно 1 раз, `n` — размер выборки;
   - если `f1 == d` (все уникальны) → колонка считается уникальной, `n_distinct = N`;
   - если `f1 == 0` (всё повторилось) → видели всё, `n_distinct = d`;
   - иначе `n·d / ((n − f1) + f1·n/N)`, обрезать до `[d, N]`, округлить.
3. **MCV**: кандидаты — значения, встретившиеся > 1 раза, по убыванию частоты, не больше target.
   `analyze_mcv_list` отбрасывает с хвоста кандидатов, частота которых **незначимо** выше
   средней частоты остальных (порог: среднее + 2 стандартных отклонения гипергеометрического
   распределения + 0.5). Если все значения выборки помещаются — храним все.
4. **histogram_bounds**: из выборки **без MCV-значений** — `target + 1` границ на равных позициях
   отсортированного массива (включая min и max). Получается `target` бакетов по `~n/target` строк.

### Оценка (планировщик)

- `sel(x ≤ c)` = (сумма частот MCV ≤ c) + (доля не-MCV популяции ≤ c по гистограмме) × (1 − Σ MCV)
- доля по гистограмме (`ineq_histogram_selectivity`): бинарный поиск бакета с `c`, затем
  `(i − 1 + binfrac) / (n_bounds − 1)` — целые бакеты слева + **линейная интерполяция** внутри;
- **отсечка**: одностороннюю оценку не пускают ближе `0.01 / (n_bounds − 1)` к 0 и 1
  (границы гистограммы приблизительны). Если есть индекс по колонке — PostgreSQL берёт реальные
  min/max (`get_actual_variable_range`) и отсечку не применяет;
- **диапазон** `a ≤ x ≤ b`: `sel = sel(x ≤ b) + sel(x ≥ a) − 1` (`clauselist_selectivity`).
  Если вышло ≤ 0: при `< −0.01` — `DEFAULT_RANGE_INEQ_SEL = 0.005`, иначе `1e−10`;
- **равенство** `x = c` (`var_eq_const`): если `c` в MCV — его частота, иначе
  `(1 − Σ MCV) / (n_distinct − num_MCV)`, но не больше наименьшей MCV-частоты.
  Наш точечный запрос `[c, c]` оцениваем как `x = c` (PostgreSQL на `BETWEEN c AND c` применил бы
  формулу диапазона — это отклонение, запиши его в заметки).

## Что написать (по подшагам — после каждого гоняй тесты)

```python
def estimate_n_distinct(ndistinct, nmultiple, samplerows, totalrows) -> int: ...
def build(sample, mn, mx, total_rows, n_buckets, statistics_target=100, actual_min=None, actual_max=None): ...

class EquiDepthPG:
    mcv_vals, mcv_freqs, hist_bounds, n_distinct
    def selectivity(self, q) -> float
    def predict(self, q) -> float           # selectivity * total_rows
    def n_floats(self) -> int               # 2 * num_mcv + len(hist_bounds) + 1
```

- **3a.** `estimate_n_distinct` → `test_n_distinct_haas_stokes`.
- **3b.** Подвыборка, MCV-кандидаты, `analyze_mcv_list` → `test_mcv_catches_heavy_value` (пока без predict можно проверить `mcv_vals`).
- **3c.** `histogram_bounds` → `test_bounds_split_sample_into_equal_parts`.
- **3d.** `_histogram_selectivity(c, isgt, iseq)` и `_scalarineqsel` → `test_range_on_uniform`.
- **3e.** диапазон и равенство → все тесты, затем `check_static.py equi_depth`.

`n_buckets` из стенда **игнорируется** — PostgreSQL берёт `statistics_target`. Поэтому EquiDepth
хранит ~100 чисел, а Equi-Width — 875: при сравнении это нужно оговаривать (или прогнать
`--kw statistics_target=875`).

## Ловушки

- **`<` против `≤`** в бинарном поиске и вычитание `eq_selec` для `<` / `≥` — источник ошибок на единицу.
  Сверяйся с `ineq_histogram_selectivity` построчно.
- **Первый бакет «уже» на `eq_selec`** — в PostgreSQL отдельная поправка для `i == 1`.
- **Отсечка `0.01/(n−1)`** делает узкие диапазоны у краёв гистограммы почти нулевыми или
  `DEFAULT_RANGE_INEQ_SEL` — посмотри, что это делает с `zipf@random`.
- MCV хранят частоты **в долях выборки**, а не в строках.
- `actual_min/actual_max` в эталоне взяты из **истинных** min/max данных (имитация индекса) — это
  метаданные, но подумай, честно ли это в нашем «только выборка» правиле.

## Проверка

```bash
poetry run python -m pytest tests/lessons/test_03_equi_depth.py -v
poetry run python lessons/check/check_static.py equi_depth
```

Эталон (медиана / P95):

| | uniform | normal | sparse_cluster | zipf | zipf@random | dsb/cr_item_sk |
|---|---|---|---|---|---|---|
| Equi-Width | 1.044 / 1.12 | 1.410 / 2.59 | 4.23 / 20.5 | 1.047 / 11.1 | 17.0 / 128 | 1.021 / 2.08 |
| **EquiDepth** | 1.034 / 1.11 | 1.410 / 2.49 | 1.83 / 8.7 | **1.000 / 1.02** | **71.9 / 709** | 1.016 / 1.06 |

Память: ~100 чисел. Построение ~1 мс, инференс ~2.3 мкс (Python-бинарный поиск ×2).

## Поиграть

1. **Что сломалось на `zipf@random`?** Возьми 20 худших запросов: истина, оценка,
   `s_hi`, `s_lo`, их сумма. Какие ветки сработали — отсечка, `DEFAULT_RANGE_INEQ_SEL`, `1e−10`?
   Это лучшее объяснение того, почему PostgreSQL плох на узких запросах в разреженном хвосте.
2. Добавь `with_index=True` (`actual_min=mn, actual_max=mx`) — как изменился `zipf@random`?
3. `statistics_target` = 10, 100, 1000, 10000 — график «P95 от памяти» рядом с Equi-Width и TKHist.
4. Где EquiDepth лучше Equi-Width при **в 9 раз меньшей** памяти и почему именно там?

## Вопросы для самопроверки

1. Почему на `zipf` по данным EquiDepth почти идеален (1.000), а Equi-Width — 1.047 / 11?
2. Зачем отделять MCV от гистограммы, а не просто строить equi-depth по всей выборке?
3. Почему `sel(a ≤ x ≤ b) = sel(x ≤ b) + sel(x ≥ a) − 1`, и когда эта формула даёт отрицательное?
4. Почему `normal` (горбы внутри каждой тысячи) EquiDepth не лечит?
5. Что в этой реализации — «PostgreSQL», а что — наше решение (подвыборка из 100k, точечные запросы)?

## Итог урока → `lessons/notes/03.md`

Плюсы: то, что реально работает в СУБД; сетка подстраивается под скос; MCV ловит пики; очень мало памяти.
Минусы: внутри бакета — всё та же равномерность (линейная интерполяция); отсечки и дефолты
разрушают узкие запросы в разреженных областях; Python-инференс медленнее.

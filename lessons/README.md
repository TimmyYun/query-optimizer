# Курс: оценка селективности с нуля — от Equi-Width до Hybrid

Цель: заново написать каждый метод из диссертации своими руками, так чтобы к концу было
понятно **что** делает каждый метод, **почему** он так устроен, **где** он выигрывает и
**почему** проигрывает. Один урок = одна ветка = один рабочий метод, проверенный тестами
и на реальных датасетах.

---

## Что я понял про проект и ветки

| ветка | что там | статус |
|---|---|---|
| `dev` | основа: Equi-Width, EquiHist, Hybrid, харнесс (`benchmark_utils.py`), генераторы данных и ворклоадов, drift-раннеры | от неё начинаем |
| `feature/new_baseline` | +4 бейзлайна из литературы (TKHist, EquiDepth/PostgreSQL, STHist, EntropyHist), абляции Hybrid, отчёты `EXPERIMENT_RESULTS.md`, `METHODS_EXPLAINED.md` | **эталон** для этого курса |
| `merge/merged_DSB_setup` | DSB-генератор, `setup_env.sh`, `RUN_TESTS.md`, детерминированный сид в Hybrid | вошла в `feature/new_baseline` |
| `origin/stats-ceb` | подготовка датасета STATS-CEB (`data/STATS-CEB/`), `NewReadme.md` | не влита; третий реальный бенчмарк |
| `origin/nurbek` | старая архитектура (`optimizer/`, `main.py`), ранние абляции | история, не используем |

Куда всё движется: диссертация показывает, что **Hybrid** (равноширинные бакеты + маленькая
модель формы внутри бакета) точнее классических гистограмм на **узких** запросах и устойчивее
под **дрейфом данных**, и сравнивает его с честным набором бейзлайнов из литературы на синтетике
(5 распределений, 60M строк) и реальных данных (DSB, позже STATS-CEB). Сейчас код бейзлайнов
написан «кем-то» — курс нужен, чтобы каждую строчку ты мог защитить сам.

---

## Порядок уроков

| # | ветка | файл | что пишешь | главный вопрос урока |
|---|---|---|---|---|
| 0 | `lesson/00-harness` | [00_harness.md](00_harness.md) | `estimators/common.py`, `trivial.py` | что такое «оценка», «истина» и «ошибка» |
| 1 | `lesson/01-equi-width` | [01_equi_width.md](01_equi_width.md) | `equi_width.py` | как устроена любая гистограмма и где врёт «равномерность» |
| 2 | `lesson/02-tkhist` | [02_tkhist.md](02_tkhist.md) | `tkhist.py` | что даёт точное хранение частых значений |
| 3 | `lesson/03-equi-depth` | [03_equi_depth.md](03_equi_depth.md) | `equi_depth.py` | как на самом деле считает PostgreSQL |
| 4 | `lesson/04-entropy-hist` | [04_entropy_hist.md](04_entropy_hist.md) | `entropy_hist.py` | можно ли выбрать границы бакетов «оптимально» |
| 5 | `lesson/05-equi-hist` | [05_equi_hist.md](05_equi_hist.md) | `equi_hist.py` | учиться не на данных, а на ответах запросов |
| 6 | `lesson/06-st-hist` | [06_st_hist.md](06_st_hist.md) | `st_hist.py` | гистограмма, которая сама меняет свои границы |
| 7 | `lesson/07-hybrid` | [07_hybrid.md](07_hybrid.md) | `hybrid.py` | модель формы внутри бакета — наш метод |
| 8 | `lesson/08-drift` | [08_drift.md](08_drift.md) | `refresh()` во всех | что происходит, когда данные меняются |
| 9 | `lesson/09-hybrid-drift` | [09_hybrid_drift.md](09_hybrid_drift.md) | `Hybrid.adapt()` | дешёвая адаптация: FT и RB |
| 10 | `lesson/10-final` | [10_final.md](10_final.md) | сводный прогон, таблица | кто, где и почему выигрывает |

Порядок выбран так, что каждый урок использует предыдущий: Equi-Width — каркас для TKHist,
EquiHist и Hybrid; EquiDepth и EntropyHist — два способа «двигать границы»; EquiHist и STHist —
два способа учиться по фидбэку; Hybrid собирает всё вместе.

---

## Правила игры (одинаковые для всех методов)

1. **Модель никогда не видит `freq`.** `freq` — точная частота каждого значения, это *истина*.
   Модель строится только из `sample` (100 000 строк) и метаданных `mn, mx, N, k`.
   Контракт `build(...)` ниже даже не принимает `freq` — так нарушить правило невозможно.
   Исключение — методы, которые учатся на **ответах запросов** (EquiHist, STHist, FT у Hybrid):
   им точный ответ приходит через `observe` / `adapt`, как в СУБД после выполнения запроса.
2. **`predict` возвращает число строк**, не селективность.
3. **Сид 42** везде, где есть случайность. Два одинаковых `build` → одинаковые ответы.
4. **`predict` скалярный** — один запрос за вызов. Время инференса меряется поштучно, как в харнессе.

### Контракт `estimators/`

```python
# estimators/<name>.py
def build(sample, mn, mx, total_rows, n_buckets, **params) -> Estimator: ...

class Estimator:
    def predict(self, q) -> float: ...            # q.low, q.high — включительно; ответ в строках
    bucket_ranges: list[tuple[int, int]]          # (lo, hi) каждого бакета — для тестов и отладки
    def n_floats(self) -> int: ...                # сколько чисел хранит модель (память)

    # опционально, появляется в уроках 5, 6, 8, 9:
    def observe(self, q, actual) -> None: ...     # выучить один точный ответ
    def refresh(self, sample, mn, mx, total_rows) -> None: ...      # ANALYZE: новая выборка
    def adapt(self, queries, actual, sample, mn, mx, total_rows) -> dict: ...  # адаптация пачкой
```

`n_buckets` — число бакетов по Фридману–Диаконису из `meta.pkl` (поле `k`). Метод может его
игнорировать (EquiDepth берёт `statistics_target` как PostgreSQL), но тогда это надо понимать
при сравнении памяти.

---

## Подготовка (один раз, в уроке 0)

```bash
poetry add --group dev pytest      # на dev pytest не объявлен в pyproject.toml
poetry run python -m pytest tests/lessons -q    # 38 skipped — всё на месте
```

Данные: `meta.pkl` и `step_*.pkl` закоммичены, но **CSV ворклоадов в git нет** (`*.csv` в
`.gitignore`). У тебя локально они уже есть; на чистом клоне — `./setup_env.sh --workloads`
(ветка `feature/new_baseline`) или команды из `CLAUDE.md`.

## Как проверять каждый шаг

Три уровня, от быстрого к медленному:

```bash
# 1. Юнит-тесты на игрушечных примерах (секунды). Ответы посчитаны вручную.
poetry run python -m pytest tests/lessons/test_01_equi_width.py -v

# 2. Реальные датасеты: 5 синтетических распределений (60M строк) x 2 ворклоада + 3 колонки DSB,
#    200k запросов на датасет, сравнение с эталоном (1-3 минуты).
poetry run python lessons/check/check_static.py equi_width
poetry run python lessons/check/check_static.py equi_width --datasets normal zipf --queries 50000  # быстро

# 3. Дрейф normal -> zipf, 21 шаг (с урока 8).
poetry run python lessons/check/check_drift.py equi_width --steps 0 5 10 15 20
```

Все тесты пропускаются (`skipped`), пока соответствующего модуля нет, поэтому
`pytest tests/lessons -q` можно запускать в любой момент — увидишь, сколько уже пройдено.

### Как читать вывод `check_static.py`

```
dataset                    median         p95       p99  build,s  inf,us   floats
normal                1.410/1.410   2.59/2.59      4.99    0.004     1.0      875 ✓
```

`твоё/эталон`. **✓** — не хуже эталона, **≈** — в пределах 15%, **✗** — заметно хуже.
Лучше эталона — тоже ✓ (а в Hybrid это ожидаемо: в эталоне есть ошибки, см. урок 7 и 9).

### Что такое эталон и почему ему нельзя слепо верить

Эталон — реализации из `feature/new_baseline`, обёрнутые в тот же контракт
(`lessons/check/reference_estimators/`) и прогнанные этим же стендом
(`lessons/check/reference_*.json`). Пересчитать: `bash lessons/check/make_reference.sh` на ветке
`feature/new_baseline`.

Эталон **не идеален**. На нём падают 2 юнит-теста Hybrid (урок 7), и в уроках отдельно
перечислены найденные в нём ошибки. Если ты исправил ошибку и стал лучше эталона — это успех,
запиши это в заметки урока.

---

## Рабочий процесс с ветками

```bash
git switch lesson/00-course            # курс: эти файлы, тесты, стенд, эталон
git switch -c lesson/00-harness        # урок 0 ответвляется от курса
# ... пишешь estimators/common.py, гоняешь тесты ...
git add estimators lessons/notes/00.md
git commit -m "lesson 00: harness — q-error, truth, trivial estimator"

git switch -c lesson/01-equi-width     # следующий урок — от предыдущего
```

Каждая ветка — цепочка от предыдущей, так что в `lesson/07-hybrid` лежат все методы 0–7.
В конце `lesson/10-final` вливается в `dev` одним PR.

**В каждой ветке, кроме кода, — `lessons/notes/NN.md`** со своими словами:
1. Что делает метод — в 3–5 предложениях, без формул.
2. Таблица своих чисел с `check_static.py` (копия вывода).
3. Ответы на «вопросы для самопроверки» из урока.
4. Плюсы / минусы / «когда бы я его выбрал».
5. Что пошло не так и как починил.

Эти заметки к концу курса — готовый черновик главы «Бейзлайны» диссертации.

---

## Итоговая таблица (заполняется по ходу курса)

| метод | что хранит | на чём учится | меняет границы? | адаптация к дрейфу | плюс | минус |
|---|---|---|---|---|---|---|
| Equi-Width | | | | | | |
| TKHist | | | | | | |
| EquiDepth (PG) | | | | | | |
| EntropyHist | | | | | | |
| EquiHist | | | | | | |
| STHist | | | | | | |
| Hybrid | | | | | | |

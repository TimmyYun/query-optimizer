#!/usr/bin/env python3
"""
Collect every result into one Excel workbook.

    poetry run python export_results.py                      # -> results/all_results.xlsx
    poetry run python export_results.py --out paper.xlsx --all

Sheets, in order:
    index    every benchmark in registry.py: run or not, where its results are, how to run it
    Выводы   plain-language summary: best method, Hybrid's place, errors in %, adaptation cost
    Колонки  what every column of the data sheets means
    <one sheet per logical benchmark>  full table + pivot of the median q-error on the right

A logical benchmark is one experiment folder results/<experiment>/:
    static  (registry.STATIC_SCENARIOS)  rows = summary_<Model>.json of every dataset
    drift   (registry.DRIFT_SCENARIOS)   rows = every step x model x stage of the drift CSVs
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from merge_drift import collect_drift_rows
from registry import DRIFT_SCENARIOS, STATIC_SCENARIOS

STATIC_COLUMNS = {
    "n_queries": "Queries",
    "median_q_error": "Median Q-Error",
    "p25_q_error": "P25 Q-Error",
    "p75_q_error": "P75 Q-Error",
    "p95_q_error": "P95 Q-Error",
    "avg_q_error": "Avg Q-Error",
    "build_time": "Build (s)",
    "update_time_total": "Feedback Updates (s)",
    "train_time": "Train = Build + Updates (s)",
    "infer_time": "Inference, all queries (s)",
    "avg_infer_time": "Inference / query, mean (s)",
    "median_infer_time": "Inference / query, median (s)",
    "p95_infer_time": "Inference / query, p95 (s)",
    "n_buckets": "Buckets",
    "n_floats": "Floats stored (Hybrid)",
}
DRIFT_COLUMNS = {
    "drift": "Drift %", "step": "Step", "model": "Model", "stage": "Stage", "label": "Line",
    "Median": "Median Q-Error", "P95": "P95 Q-Error", "Avg": "Avg Q-Error",
    "Infer_Med": "Inference / query, median (s)", "Infer_Avg": "Inference / query, mean (s)",
    "Adapt_Time": "Adaptation cost at this step (s)", "Rebuilt_Buckets": "Rebuilt buckets",
    "Initial_Build": "Initial build, step 0 (s)",
}
GLOSSARY = [
    ("Все листы", "Model", "Метод оценки (см. RESULTS_GUIDE.md, раздел 2)."),
    ("Все листы", "Median Q-Error", "Медиана q-error = max(оценка/истина, истина/оценка). 1.00 — точно; 1.10 — типичная ошибка 10 %; 2.0 — в 2 раза. Основная метрика."),
    ("Все листы", "P95 Q-Error", "95-й процентиль: насколько плохи худшие 5 % запросов. Для оптимизатора хвост часто важнее медианы."),
    ("Все листы", "Avg Q-Error", "Среднее. Если оно намного больше P95, значит, есть единичные катастрофические ошибки; ищи их причину."),
    ("Статика", "Dataset", "Распределение (синтетика) или колонка реальной таблицы."),
    ("Статика", "Workload", "Файл запросов."),
    ("Статика", "Queries", "Сколько запросов в оценке. Запросы с истинным ответом 0 отброшены. Меньше ~50 — различия во втором знаке ненадёжны."),
    ("Статика", "P25 / P75 Q-Error", "Квартили: разброс типичной ошибки."),
    ("Статика", "Build (s)", "Время построения модели по сэмплу. У Hybrid сюда входит обучение моделей во всех бакетах."),
    ("Статика", "Feedback Updates (s)", "Время обучения по фидбэку запросов (STHist, Equi-Hist), иначе 0."),
    ("Статика", "Train = Build + Updates (s)", "Полная стоимость подготовки модели."),
    ("Статика", "Inference, all queries (s)", "Сумма времени всех вызовов predict."),
    ("Статика", "Inference / query, mean / median / p95 (s)", "Время одной оценки. Все методы замеряются одним циклом на Python; сравнивай методы между собой, а не с СУБД."),
    ("Статика", "Buckets", "Число бакетов после построения (у EntropyHist — эффективное, у STHist — после реструктуризаций)."),
    ("Статика", "Floats stored (Hybrid)", "Сколько чисел хранят модели Hybrid: мера памяти. У гистограммы примерно 2 × Buckets."),
    ("Дрейф", "Run", "Префикс прогона, например gradual_ws_list_price_to_refresh."),
    ("Дрейф", "Drift %", "Синтетика: доля нового распределения в смеси. DSB: накопленный churn (удалённые + вставленные строки) / исходное число строк."),
    ("Дрейф", "Step", "Имя чекпоинта DSB (initial, churn_5pct, ...); у синтетики пусто."),
    ("Дрейф", "Stage, Line", "Стадия адаптации и линия графика = Model_Stage. Shock — до адаптации; FT — подстройка масс по фидбэку; RB — переобучение плохих бакетов; Online/Post — STHist во время/после обучения; Static/Rebuild — EntropyHist без обновления / пересобранная."),
    ("Дрейф", "Q-Error на листе дрейфа", "Считается на eval-половине запросов, которую модель не видит (кроме STHist_Online, prequential по adapt-половине)."),
    ("Дрейф", "Adaptation cost at this step (s)", "Сколько секунд модель потратила на этом шаге, чтобы оказаться в этой стадии, с накоплением внутри шага. Гистограммы: пересборка по сэмплу (ANALYZE). Hybrid: Shock = обновление масс, FT = + подстройка, RB = + переобучение бакетов. STHist: обучение по фидбэку. EntropyHist_Static: 0."),
    ("Дрейф", "Rebuilt buckets", "Hybrid_RB: сколько бакетов переобучено на этом шаге."),
    ("Дрейф", "Initial build, step 0 (s)", "Построение модели на шаге 0; в стоимость стадий не входит."),
    ("Выводы", "Типичная ошибка, %", "(Median Q-Error − 1) × 100: на сколько процентов ошибается оценка у типичного запроса."),
    ("Выводы", "Ошибка худших 5 %, %", "(P95 Q-Error − 1) × 100."),
    ("Выводы", "Средний ранг", "Среднее место метода по всем датасетам листа (1 — лучший по медиане). Ничьи до 4-го знака делят место."),
]
SHEET_NAME_MAX = 31  # Excel limit
NOT_COMPARABLE = {"STHist_Online"}  # scored on the adapt half, the others on the eval half


def pct(q):
    return None if pd.isna(q) else round((q - 1) * 100, 2)


# ---------- data sheets ----------

def static_table(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(exp_dir.rglob("summary_*.json")):
        s = json.loads(path.read_text())
        m = s["metrics"]
        row = {"Dataset": str(path.parent.relative_to(exp_dir)), "Workload": s["wl_name"], "Model": s["model"]}
        row.update({label: m.get(key) for key, label in STATIC_COLUMNS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(axis=1, how="all").sort_values(["Dataset", "Model"])
    return df


def drift_table(exp_dir: Path) -> pd.DataFrame:
    rows = [{"Run": prefix, **r} for prefix, rs in collect_drift_rows(exp_dir).items() for r in rs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns=DRIFT_COLUMNS)
    return df.dropna(axis=1, how="all").sort_values(["Run", "Line", "Drift %"])


# ---------- plain-language summary ----------

def rank_by_median(sub, key):
    return sub["Median Q-Error"].round(4).rank(method="min").astype(int).rename(key)


def static_findings(sheet, df):
    """One row per dataset: who is best, where Hybrid stands, in words."""
    out = []
    for ds, sub in df.groupby("Dataset"):
        sub = sub.assign(rank=rank_by_median(sub, "rank")).sort_values("rank")
        best, tail = sub.iloc[0], sub.loc[sub["P95 Q-Error"].idxmin()]
        hyb = sub[sub["Model"] == "Hybrid"]
        n, q = len(sub), int(sub["Queries"].min()) if "Queries" in sub else None
        if hyb.empty:
            text = f"Лучший — {best.Model} ({pct(best['Median Q-Error'])} %). Hybrid не запускался."
            h_pct = h_rank = None
        else:
            h = hyb.iloc[0]
            h_pct, h_rank = pct(h["Median Q-Error"]), int(h["rank"])
            if h_rank == 1:
                second = sub[sub["Model"] != "Hybrid"].iloc[0]
                if int(second["rank"]) == 1:
                    text = f"Hybrid делит первое место с {second.Model} (типичная ошибка {h_pct} %)."
                else:
                    text = (f"Hybrid точнее всех: типичная ошибка {h_pct} % против "
                            f"{pct(second['Median Q-Error'])} % у {second.Model}.")
            else:
                text = (f"Лучший — {best.Model} ({pct(best['Median Q-Error'])} %); "
                        f"Hybrid на {h_rank}-м месте из {n} ({h_pct} %).")
        if q is not None and q < 50:
            text += f" Всего {q} запросов — различия ненадёжны."
        out.append({
            "Бенчмарк": sheet, "Датасет": ds,
            "Лучший по медиане": best.Model, "Его типичная ошибка, %": pct(best["Median Q-Error"]),
            "Лучший по хвосту (P95)": tail.Model, "Его ошибка худших 5 %, %": pct(tail["P95 Q-Error"]),
            "Hybrid: типичная ошибка, %": h_pct, "Hybrid: место": h_rank, "Методов": n,
            "Запросов": q, "Вывод": text,
        })
    return out


def static_ranking(sheet, df):
    """Per method across the sheet's datasets: mean rank, wins, mean errors, cost."""
    df = df.copy()
    df["rank"] = df.groupby("Dataset")["Median Q-Error"].transform(lambda m: m.round(4).rank(method="min")).astype(int)
    g = df.groupby("Model")
    res = pd.DataFrame({
        "Средний ранг": g["rank"].mean().round(2),
        "Побед (1-е место)": g["rank"].apply(lambda r: int((r == 1).sum())),
        "Средняя типичная ошибка, %": g["Median Q-Error"].mean().map(pct),
        "Средняя ошибка худших 5 %, %": g["P95 Q-Error"].mean().map(pct),
        "Подготовка, с (среднее)": g["Train = Build + Updates (s)"].mean().round(4),
        "Оценка одного запроса, мкс (среднее)": (g["Inference / query, mean (s)"].mean() * 1e6).round(2),
    }).sort_values("Средний ранг").reset_index()
    res.insert(0, "Бенчмарк", sheet)
    return res.to_dict("records")


def drift_findings(sheet, df):
    """Per line: error over all steps and at the end, adaptation cost over the run."""
    rows = []
    for line, sub in df.groupby("Line"):
        sub = sub.sort_values("Drift %")
        cost = sub.loc[sub["Drift %"] > 0, "Adaptation cost at this step (s)"] if "Adaptation cost at this step (s)" in sub else None
        init = sub["Initial build, step 0 (s)"].dropna() if "Initial build, step 0 (s)" in sub else pd.Series(dtype=float)
        rows.append({
            "Бенчмарк": sheet, "Линия": line, "Сравнима с остальными": line not in NOT_COMPARABLE,
            "Типичная ошибка, % (среднее по шагам)": pct(sub["Median Q-Error"].mean()),
            "Типичная ошибка на последнем шаге, %": pct(sub["Median Q-Error"].iloc[-1]),
            "Худший шаг, %": pct(sub["Median Q-Error"].max()),
            "Ошибка худших 5 %, % (среднее по шагам)": pct(sub["P95 Q-Error"].mean()),
            "Адаптация за все шаги, с": None if cost is None else round(float(cost.sum()), 4),
            "Начальное построение, с": None if init.empty else round(float(init.iloc[0]), 4),
        })
    res = pd.DataFrame(rows)
    comp = res["Сравнима с остальными"]
    res.loc[comp, "Место"] = res.loc[comp, "Типичная ошибка, % (среднее по шагам)"].rank(method="min").astype(int)
    res = res.sort_values(["Сравнима с остальными", "Место"], ascending=[False, True])
    best = res[res["Сравнима с остальными"]].iloc[0]
    rb = res[res["Линия"] == "Hybrid_RB"]
    text = f"Лучше всех на всём дрейфе держится {best['Линия']} (в среднем {best['Типичная ошибка, % (среднее по шагам)']} %)."
    if not rb.empty and rb.iloc[0]["Линия"] != best["Линия"]:
        text += f" Hybrid_RB на {int(rb.iloc[0]['Место'])}-м месте ({rb.iloc[0]['Типичная ошибка, % (среднее по шагам)']} %)."
    res["Вывод по бенчмарку"] = [text] + [""] * (len(res) - 1)
    res["Сравнима с остальными"] = res["Сравнима с остальными"].map({True: "да", False: "нет (adapt-половина)"})
    return res.to_dict("records")


# ---------- formatting ----------

def style_sheet(ws, n_cols, wrap_cols=()):
    header = PatternFill("solid", fgColor="DDE4EE")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    for col in range(1, n_cols + 1):
        letter = get_column_letter(col)
        name = str(ws[letter][0].value)
        width = max((len(str(c.value)) for c in ws[letter][:200] if c.value is not None), default=8)
        ws.column_dimensions[letter].width = 70 if name in wrap_cols else min(max(10, width + 2), 45)
        for c in ws[letter][1:]:
            if name in wrap_cols:
                c.alignment = Alignment(wrap_text=True, vertical="top")
            elif isinstance(c.value, float) and "%" not in name:
                c.number_format = "0.000000000" if "(s)" in name else "0.0000"


def write_df(writer, sheet, df, wrap_cols=()):
    df.to_excel(writer, sheet_name=sheet, index=False)
    style_sheet(writer.sheets[sheet], len(df.columns), wrap_cols)


def write_data_sheet(writer, sheet, df, pivot):
    df.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]
    if pivot is not None and not pivot.empty:
        start = len(df.columns) + 2
        ws.cell(row=1, column=start, value="Median Q-Error").font = Font(bold=True)
        pivot.to_excel(writer, sheet_name=sheet, startrow=1, startcol=start - 1)
    style_sheet(ws, ws.max_column)


# ---------- main ----------

def plan_benchmarks(root: Path, include_other: bool):
    present = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    plan = [(n, "static", STATIC_SCENARIOS[n]["experiment"],
             f"poetry run python run_experiments.py --scenario {n}") for n in STATIC_SCENARIOS]
    plan += [(n, "drift", n, f"poetry run python run_drift_experiments.py --scenario {n}") for n in DRIFT_SCENARIOS]
    if include_other:
        known = {p[2] for p in plan}
        plan += [(n, "other", n, "") for n in sorted(present - known)]
    return plan, present


def describe(name, kind):
    if kind == "static":
        cfg = STATIC_SCENARIOS[name]
        return f"{len(cfg['datasets'])} датасетов, запросы {cfg['workload']}"
    if kind == "drift":
        cfg = DRIFT_SCENARIOS[name]
        wl = cfg["static_workload"] or "свои на каждом шаге (меняются вместе с данными)"
        return f"шаги {cfg['shift_dir']}; запросы {wl}; оценка на eval-половине"
    return "нет в registry.py"


def main():
    parser = argparse.ArgumentParser(description="All results -> one xlsx: index, summary, glossary, one sheet per benchmark")
    parser.add_argument("--results-dir", default="results", help="Root of the results tree")
    parser.add_argument("--out", default="results/all_results.xlsx", help="Output workbook")
    parser.add_argument("--all", action="store_true",
                        help="Also include experiment folders that are not scenarios in registry.py (e.g. smoke)")
    args = parser.parse_args()

    root = Path(args.results_dir)
    plan, present = plan_benchmarks(root, args.all)

    index, findings, ranking, drift_sum, sheets = [], [], [], [], []
    for name, kind, folder, cmd in plan:
        exp_dir = root / folder
        entry = {"Лист": None, "Бенчмарк": name, "Тип": kind, "Что": describe(name, kind),
                 "Папка": str(exp_dir), "Статус": "не запускался", "Строк": 0, "Методы": "",
                 "Как запустить": cmd}
        if folder in present:
            is_drift = kind == "drift" or (kind == "other" and not any(exp_dir.rglob("summary_*.json")))
            df = drift_table(exp_dir) if is_drift else static_table(exp_dir)
            if not df.empty:
                sheet = folder[:SHEET_NAME_MAX]
                entry.update({"Лист": sheet, "Статус": "есть", "Строк": len(df),
                              "Методы": ", ".join(sorted(df["Model"].unique()))})
                if is_drift:
                    pivot = df.pivot_table(index="Drift %", columns="Line", values="Median Q-Error")
                    drift_sum += drift_findings(sheet, df)
                else:
                    pivot = df.pivot_table(index="Dataset", columns="Model", values="Median Q-Error")
                    findings += static_findings(sheet, df)
                    ranking += static_ranking(sheet, df)
                sheets.append((sheet, df, pivot))
            else:
                entry["Статус"] = "папка есть, результатов нет (прогон не завершился?)"
        index.append(entry)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        write_df(writer, "index", pd.DataFrame(index), wrap_cols=("Что", "Как запустить", "Методы"))

        # Выводы: three blocks stacked on one sheet
        sheet = "Выводы"
        row = 0
        for title, block, wrap in [
            ("Статика: кто лучший на каждом датасете", findings, ("Вывод",)),
            ("Статика: общий рейтинг методов по каждому бенчмарку", ranking, ()),
            ("Дрейф: ошибка по всем шагам и стоимость адаптации", drift_sum, ("Вывод по бенчмарку",)),
        ]:
            if not block:
                continue
            bdf = pd.DataFrame(block)
            pd.DataFrame([[title]]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
            bdf.to_excel(writer, sheet_name=sheet, startrow=row + 1, index=False)
            ws = writer.sheets[sheet]
            for c in ws[row + 1]:
                c.font = Font(bold=True, size=12)
            for c in ws[row + 2]:
                c.font = Font(bold=True)
                c.fill = PatternFill("solid", fgColor="DDE4EE")
                c.alignment = Alignment(wrap_text=True, vertical="top")
            for r in ws.iter_rows(min_row=row + 3, max_row=row + 2 + len(bdf)):
                for c in r:
                    if str(ws.cell(row=row + 2, column=c.column).value) in wrap:
                        c.alignment = Alignment(wrap_text=True, vertical="top")
            row += len(bdf) + 4
        if row:
            ws = writer.sheets[sheet]
            for col in range(1, ws.max_column + 1):
                ws.column_dimensions[get_column_letter(col)].width = 22
            for name in ("Вывод", "Вывод по бенчмарку"):
                for col in range(1, ws.max_column + 1):
                    if any(ws.cell(row=r, column=col).value == name for r in range(1, ws.max_row + 1)):
                        ws.column_dimensions[get_column_letter(col)].width = 70

        write_df(writer, "Колонки", pd.DataFrame(GLOSSARY, columns=["Где", "Колонка", "Что означает"]),
                 wrap_cols=("Что означает",))
        for sheet, df, pivot in sheets:
            write_data_sheet(writer, sheet, df, pivot)

    for e in index:
        print(f"{e['Бенчмарк']:30s} {e['Статус']:12s} {e['Строк']:5d}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

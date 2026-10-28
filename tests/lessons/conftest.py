"""
Тесты уроков. Каждый файл проверяет один модуль estimators/<name>.py и
пропускается (skip), пока модуля нет — можно запускать всё сразу:

    poetry run python -m pytest tests/lessons -q
    poetry run python -m pytest tests/lessons/test_01_equi_width.py -v

Проверить сами тесты на эталоне (только на ветке feature/new_baseline):

    LESSONS_PKG=reference_estimators poetry run python -m pytest tests/lessons -q
"""
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "lessons" / "check")):
    if p not in sys.path:
        sys.path.insert(0, p)

PKG = os.environ.get("LESSONS_PKG", "estimators")


def need(module):
    """Импорт estimators.<module>; если файла ещё нет — тест пропускается."""
    try:
        return importlib.import_module(f"{PKG}.{module}")
    except ModuleNotFoundError as e:
        if e.name in (PKG, f"{PKG}.{module}"):
            pytest.skip(f"{PKG}/{module}.py ещё не написан")
        raise


def maybe(module):
    """Как need, но без skip: None, если модуля нет (для сравнений между уроками)."""
    try:
        return importlib.import_module(f"{PKG}.{module}")
    except ModuleNotFoundError:
        return None


@dataclass(frozen=True)
class Q:
    low: int
    high: int


def random_queries(rng, mn, mx, n, max_width=None):
    max_width = max_width or (mx - mn + 1)
    lo = rng.integers(mn, mx + 1, n)
    w = rng.integers(0, max_width, n)
    return [Q(int(a), int(min(mx, a + b))) for a, b in zip(lo, w)]

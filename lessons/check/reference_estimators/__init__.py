"""
Эталон: готовые реализации из ветки feature/new_baseline (models/*), обёрнутые
в контракт уроков. Импортируется ТОЛЬКО на той ветке — там лежит models/tkhist.py и т.д.
Нужен, чтобы пересчитать reference_*.json (см. make_reference.sh) и проверить сами тесты:

    LESSONS_PKG=reference_estimators poetry run python -m pytest tests/lessons -q
"""

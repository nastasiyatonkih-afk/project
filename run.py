"""Одна команда, которая делает весь проект с нуля.

    python run.py

Порядок ровно такой, в каком строится любой аналитический пайплайн:
данные -> метрики -> статистика -> графики. Каждый шаг можно запустить и
отдельно, если нужно перепроверить только его.
"""

from __future__ import annotations

import time

STEPS = [
    ("1/4  генерация данных", "analytics.data"),
    ("2/4  продуктовые метрики", "analytics.metrics"),
    ("3/4  анализ A/B-теста", "analytics.ab_test"),
    ("4/4  графики", "analytics.charts"),
]


def main() -> None:
    import importlib

    for title, module in STEPS:
        print("\n" + "=" * 76)
        print(title)
        print("=" * 76)
        t0 = time.perf_counter()
        mod = importlib.import_module(module)
        (mod.main if hasattr(mod, "main") else mod.analyze)()
        print(f"\n[{time.perf_counter() - t0:.1f} с]")

    print("\nГотово. Отчёт: reports/otchet.md, графики: reports/figures/")


if __name__ == "__main__":
    main()

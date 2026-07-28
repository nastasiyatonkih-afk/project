"""Запускает запросы из sql/queries.sql прямо по csv-файлам.

DuckDB умеет читать csv как таблицы, поэтому никакой базы поднимать не надо:
это тот же SQL, который вы написали бы в хранилище, только данные лежат в
файлах рядом.

Запуск:
    python -m analytics.sql_demo
    python -m analytics.sql_demo "Экономика"     # только подходящие блоки
"""

from __future__ import annotations

import re
import sys

import duckdb
import pandas as pd

from analytics.data import DATA

SQL_FILE = DATA.parent / "sql" / "queries.sql"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def connect() -> duckdb.DuckDBPyConnection:
    """Открывает in-memory базу и вешает на csv-файлы обычные имена таблиц."""
    con = duckdb.connect()
    for name in ("users", "orders", "ab_test"):
        path = str(DATA / f"{name}.csv").replace("\\", "/")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
    return con


def split_queries(text: str) -> list[tuple[str, str]]:
    """Режет файл на пары (название, sql) по строкам вида «-- @Название»."""
    parts = re.split(r"^--\s*@(.+)$", text, flags=re.MULTILINE)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def main() -> None:
    needle = sys.argv[1].lower() if len(sys.argv) > 1 else None
    con = connect()
    for title, sql in split_queries(SQL_FILE.read_text(encoding="utf-8")):
        if needle and needle not in title.lower():
            continue
        print(f"\n### {title}\n")
        print(con.execute(sql).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()

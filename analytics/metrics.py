"""Продуктовые метрики: считаем на pandas, без промежуточных слоёв.

Здесь только одно правило, но оно важное: у каждой метрики должно быть
записанное определение, и считаться она должна ровно в одном месте.
Половина споров в продуктовой команде — это спор двух людей, которые
называют «retention» две разные вещи.

Определения, которые используются в этом проекте:

* Конверсия в первый заказ — доля зарегистрировавшихся, кто сделал хотя бы
  один заказ. Без ограничения по времени, потому что нас интересует
  качество трафика, а не скорость.
* Retention месяца N — доля пользователей когорты, сделавших хотя бы один
  заказ в N-й месяц жизни (месяц 0 — месяц регистрации). Пользователи, у
  которых N-й месяц ещё не наступил, из знаменателя убираются: иначе
  свежие когорты выглядят хуже старых просто потому, что они моложе.
* AOV (average order value) — средняя сумма заказа.
* Маржа с заказа — наценка 22% плюс плата клиента за доставку минус наша
  себестоимость доставки. Это то, что реально остаётся компании.
* LTV 90 — суммарная маржа с пользователя за первые 90 дней. Считается
  только по тем, кто прожил 90 дней, иначе получится занижено.
* ROMI = (LTV 90 − CAC) / CAC. Больше нуля — канал окупается.

Запуск:
    python -m analytics.metrics
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from analytics.data import DATA, DATA_END, LATE_DELIVERY_MIN

LTV_WINDOW_DAYS = 90

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает три csv и приводит даты к типу даты (иначе pandas видит строки)."""
    users = pd.read_csv(DATA / "users.csv", parse_dates=["reg_date"])
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    ab = pd.read_csv(DATA / "ab_test.csv", parse_dates=["ab_start_date"])
    return users, orders, ab


def headline(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Главные цифры одной строкой — то, что показывают на первом слайде."""
    buyers = orders["user_id"].nunique()
    per_user = orders.groupby("user_id").size()
    return pd.DataFrame(
        [
            {
                "регистраций": len(users),
                "покупателей": buyers,
                "конверсия в 1-й заказ": buyers / len(users),
                "заказов": len(orders),
                "заказов на покупателя": per_user.mean(),
                "AOV, руб.": orders["amount_rub"].mean(),
                "маржа с заказа, руб.": orders["margin_rub"].mean(),
                "выручка, млн руб.": orders["amount_rub"].sum() / 1e6,
                "маржа, млн руб.": orders["margin_rub"].sum() / 1e6,
                "доля долгих доставок": (orders["delivery_minutes"] > LATE_DELIVERY_MIN).mean(),
            }
        ]
    )


def first_order_funnel(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Воронка повторных заказов: регистрация -> 1-й -> 2-й -> 3-й -> 4-й.

    Для доставки это и есть главная воронка. Первый заказ покупается
    маркетингом, а вот второй и третий — это уже продукт и операции.
    """
    n_orders = orders.groupby("user_id").size()
    steps = [("регистрация", len(users))]
    for k in range(1, 5):
        steps.append((f"{k}-й заказ", int((n_orders >= k).sum())))

    df = pd.DataFrame(steps, columns=["шаг", "пользователей"])
    df["конверсия из предыдущего"] = df["пользователей"] / df["пользователей"].shift(1)
    df["доля от регистраций"] = df["пользователей"] / len(users)
    df["потеряли"] = (df["пользователей"].shift(1) - df["пользователей"]).astype("Int64")
    return df


def _month_index(a: pd.Series, b: pd.Series) -> pd.Series:
    """Сколько полных календарных месяцев прошло между двумя датами."""
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


def cohort_retention(users: pd.DataFrame, orders: pd.DataFrame, max_month: int = 5) -> pd.DataFrame:
    """Матрица удержания: строки — месяц регистрации, столбцы — месяц жизни.

    Знаменатель считается честно: если у когорты N-й месяц ещё не наступил,
    ячейка остаётся пустой, а не занижает картину.
    """
    u = users[["user_id", "reg_date"]].copy()
    u["cohort"] = u["reg_date"].dt.to_period("M")

    o = orders.merge(u, on="user_id")
    o["month_index"] = _month_index(o["reg_date"], o["order_date"])

    active = (
        o[o["month_index"].between(0, max_month)]
        .groupby(["cohort", "month_index"])["user_id"]
        .nunique()
        .rename("активных")
        .reset_index()
    )
    size = u.groupby("cohort")["user_id"].nunique().rename("размер когорты")

    data_end = pd.Timestamp(DATA_END)
    rows = []
    for cohort, n in size.items():
        # сколько месяцев жизни этой когорты мы реально успели увидеть
        observed = (data_end.year - cohort.year) * 12 + (data_end.month - cohort.month)
        for m in range(max_month + 1):
            if m > observed:
                rows.append({"cohort": cohort, "month_index": m, "retention": np.nan})
                continue
            hit = active[(active["cohort"] == cohort) & (active["month_index"] == m)]
            rows.append(
                {
                    "cohort": cohort,
                    "month_index": m,
                    "retention": (hit["активных"].iloc[0] / n) if len(hit) else 0.0,
                }
            )

    matrix = pd.DataFrame(rows).pivot(index="cohort", columns="month_index", values="retention")
    matrix.insert(0, "размер когорты", size)
    return matrix


def channel_economics(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Экономика каналов привлечения: сколько платим и сколько возвращается.

    Ключевой момент — окно в 90 дней. В расчёт берём только пользователей,
    которые зарегистрировались минимум за 90 дней до конца данных. Иначе
    свежие когорты «съедят» LTV и все каналы будут выглядеть убыточными.
    """
    cutoff = pd.Timestamp(DATA_END) - pd.Timedelta(days=LTV_WINDOW_DAYS)
    mature = users[users["reg_date"] <= cutoff].copy()

    o = orders.merge(mature[["user_id", "reg_date"]], on="user_id")
    o = o[(o["order_date"] - o["reg_date"]).dt.days < LTV_WINDOW_DAYS]

    per_user = o.groupby("user_id").agg(
        заказов=("order_id", "count"),
        маржа=("margin_rub", "sum"),
        выручка=("amount_rub", "sum"),
    )
    m = mature.merge(per_user, on="user_id", how="left").fillna(
        {"заказов": 0, "маржа": 0.0, "выручка": 0.0}
    )

    agg = m.groupby("channel").agg(
        регистраций=("user_id", "count"),
        CAC=("cac_rub", "mean"),
        конверсия_в_заказ=("заказов", lambda s: (s > 0).mean()),
        заказов_на_юзера=("заказов", "mean"),
        LTV90=("маржа", "mean"),
        выручка_на_юзера=("выручка", "mean"),
    )
    agg["ROMI"] = np.where(agg["CAC"] > 0, (agg["LTV90"] - agg["CAC"]) / agg["CAC"], np.nan)
    agg["прибыль_на_юзера"] = agg["LTV90"] - agg["CAC"]
    agg["бюджет_млн"] = agg["регистраций"] * agg["CAC"] / 1e6
    return agg.sort_values("ROMI", ascending=False).reset_index()


def delivery_impact(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Как время первой доставки влияет на всё остальное.

    Берём первый заказ каждого пользователя, делим людей на две группы —
    «привезли быстро» и «привезли долго» — и сравниваем, сколько заказов они
    сделали дальше. Это наблюдательное сравнение, не эксперимент: люди в
    группах могли отличаться и до этого (например, адресами на окраинах).
    Поэтому вывод формулируем как гипотезу, которую надо проверить тестом.
    """
    first = orders.sort_values("order_date").groupby("user_id").first().reset_index()
    first["долгая доставка"] = first["delivery_minutes"] > LATE_DELIVERY_MIN

    totals = orders.groupby("user_id").agg(
        всего_заказов=("order_id", "count"),
        маржа=("margin_rub", "sum"),
    )
    df = first[["user_id", "долгая доставка"]].merge(totals, on="user_id")

    out = df.groupby("долгая доставка").agg(
        пользователей=("user_id", "count"),
        заказов_на_юзера=("всего_заказов", "mean"),
        доля_с_повторным=("всего_заказов", lambda s: (s >= 2).mean()),
        маржа_на_юзера=("маржа", "mean"),
    )
    out.index = ["до 45 минут", "дольше 45 минут"]
    return out.reset_index(names="первая доставка")


def rfm(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """RFM-сегментация: Recency, Frequency, Monetary.

    Три вопроса про каждого клиента: как давно заказывал, как часто и на
    сколько. Каждому признаку ставим балл от 1 до 3 по квантилям, дальше
    сегменты — это просто читаемые названия для сочетаний баллов.
    """
    end = pd.Timestamp(DATA_END)
    agg = orders.groupby("user_id").agg(
        recency=("order_date", lambda s: (end - s.max()).days),
        frequency=("order_id", "count"),
        monetary=("margin_rub", "sum"),
    )
    # Чем меньше recency, тем лучше — поэтому баллы для неё разворачиваем.
    agg["R"] = pd.qcut(agg["recency"], 3, labels=[3, 2, 1]).astype(int)
    agg["F"] = pd.qcut(agg["frequency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    agg["M"] = pd.qcut(agg["monetary"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)

    def segment(r):
        if r.R == 3 and r.F == 3:
            return "Чемпионы"
        if r.R == 3 and r.F <= 2:
            return "Новички и редкие"
        if r.R == 2 and r.F >= 2:
            return "Лояльные"
        if r.R == 1 and r.F == 3:
            return "В зоне риска"
        if r.R == 1:
            return "Спящие"
        return "Прочие"

    agg["сегмент"] = agg.apply(segment, axis=1)
    out = agg.groupby("сегмент").agg(
        пользователей=("recency", "count"),
        дней_с_последнего_заказа=("recency", "median"),
        заказов=("frequency", "mean"),
        маржа_на_юзера=("monetary", "mean"),
    )
    out["доля"] = out["пользователей"] / out["пользователей"].sum()
    out["маржа_всего_млн"] = out["пользователей"] * out["маржа_на_юзера"] / 1e6
    return out.sort_values("маржа_всего_млн", ascending=False).reset_index()


# --- вывод отчёта ----------------------------------------------------------
def to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Простая markdown-таблица (чтобы не тянуть зависимость tabulate)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:,.3f}".replace(",", " "))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SECTIONS = [
    ("Главные метрики", headline),
    ("Воронка повторных заказов", first_order_funnel),
    ("Удержание по когортам регистрации", cohort_retention),
    ("Экономика каналов привлечения (окно 90 дней)", channel_economics),
    ("Влияние времени первой доставки", delivery_impact),
    ("RFM-сегменты", rfm),
]


def main() -> None:
    users, orders, _ = load()
    out = ["# Отчёт по продуктовым метрикам", "",
           "_Файл сгенерирован командой `python -m analytics.metrics`._", ""]
    for title, fn in SECTIONS:
        df = fn(users, orders)
        show_index = fn is cohort_retention
        print(f"\n### {title}\n")
        print(df.to_string(index=show_index))
        out += [f"## {title}", "", to_markdown(df, index=show_index), ""]

    path = DATA.parent / "reports" / "otchet.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()

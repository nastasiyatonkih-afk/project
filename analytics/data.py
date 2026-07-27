"""Генерация данных сервиса доставки продуктов «Полка».

Зачем это нужно. Открытых данных российского сервиса доставки в свободном
доступе нет, поэтому данные синтетические. Но они сделаны так, чтобы в них
были те же проблемы, которые аналитик разбирает на реальной работе:

1. Канал «VK Реклама» даёт много дешёвых регистраций, но эти пользователи
   почти не возвращаются. По количеству регистраций канал выглядит лучшим,
   по деньгам (ROMI) — худшим.
2. Если первый заказ приехал дольше 45 минут, человек заказывает дальше
   заметно реже. Это операционная проблема, которая выглядит как проблема
   продукта.
3. A/B-тест «снизить порог бесплатной доставки»: заказов становится больше,
   а денег — меньше, потому что падает средний чек и мы платим за доставку.

Все случайные числа зафиксированы через SEED, поэтому повторный запуск даёт
ровно те же файлы.

Запуск:
    python -m analytics.data
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# --- параметры мира --------------------------------------------------------
SEED = 2026
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

N_USERS = 25_000
REG_START = dt.date(2025, 9, 1)    # начало привлечения
REG_END = dt.date(2026, 2, 28)     # последняя регистрация
DATA_END = dt.date(2026, 3, 31)    # последний день, за который есть данные

# A/B-тест: снижаем порог бесплатной доставки с 2000 руб. до 1000 руб.
AB_START = dt.date(2026, 1, 15)
AB_HORIZON_DAYS = 28               # сколько дней смотрим после старта

FREE_DELIVERY_CONTROL = 2000       # руб., порог бесплатной доставки в контроле
FREE_DELIVERY_TEST = 1000          # руб., порог в тестовой группе
DELIVERY_FEE = 149                 # руб., платит клиент, если порог не набран
DELIVERY_COST = 149                # руб., во сколько доставка обходится нам
MARGIN_RATE = 0.22                 # маржа с товаров, доля от суммы заказа
LATE_DELIVERY_MIN = 45             # минут; дольше — считаем доставку долгой

# город -> (доля пользователей, множитель времени доставки, множитель чека)
CITIES = {
    "Москва": (0.42, 1.00, 1.12),
    "Санкт-Петербург": (0.18, 1.05, 1.04),
    "Екатеринбург": (0.09, 1.12, 0.94),
    "Новосибирск": (0.08, 1.15, 0.92),
    "Казань": (0.07, 1.08, 0.93),
    "Нижний Новгород": (0.06, 1.10, 0.90),
    "Ростов-на-Дону": (0.05, 1.18, 0.88),
    "Краснодар": (0.05, 1.14, 0.91),
}

# канал -> (доля, CAC в рублях, сдвиг «качества» пользователя)
# «Качество» — скрытая характеристика: насколько человек склонен заказывать
# дальше. Именно из-за неё каналы отличаются не только ценой привлечения.
CHANNELS = {
    "Яндекс.Директ": (0.24, 620, +0.10),
    "VK Реклама": (0.26, 310, -0.50),
    "органика": (0.20, 0, +0.40),
    "реферальная программа": (0.12, 450, +0.55),
    "Telegram-блогеры": (0.10, 540, -0.05),
    "наружная реклама": (0.08, 700, +0.05),
}

rng = np.random.default_rng(SEED)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_users() -> pd.DataFrame:
    """Пользователи: когда зарегистрировались, откуда пришли, из какого города."""
    n_days = (REG_END - REG_START).days + 1

    # Регистраций со временем становится больше, в выходные — меньше.
    days = np.arange(n_days)
    dates = [REG_START + dt.timedelta(days=int(d)) for d in days]
    intensity = 1.004 ** days
    intensity *= np.where([d.weekday() >= 5 for d in dates], 0.88, 1.0)
    intensity /= intensity.sum()

    reg_day = rng.choice(days, size=N_USERS, p=intensity)

    channels = list(CHANNELS)
    channel = rng.choice(channels, size=N_USERS, p=[CHANNELS[c][0] for c in channels])
    cities = list(CITIES)
    city = rng.choice(cities, size=N_USERS, p=[CITIES[c][0] for c in cities])
    platform = rng.choice(["Android", "iOS"], size=N_USERS, p=[0.58, 0.42])

    # Скрытое «качество» пользователя — главный источник различий между
    # сегментами. В реальных данных мы его не видим, видим только следствия.
    quality = (
        rng.normal(0, 1, N_USERS)
        + np.array([CHANNELS[c][2] for c in channel])
        + np.where(platform == "iOS", 0.12, 0.0)
    )

    users = pd.DataFrame(
        {
            "user_id": np.arange(1, N_USERS + 1),
            "reg_date": [REG_START + dt.timedelta(days=int(d)) for d in reg_day],
            "city": city,
            "channel": channel,
            "platform": platform,
            "cac_rub": [CHANNELS[c][1] for c in channel],
            "quality": quality,
        }
    )
    return users.sort_values("reg_date", ignore_index=True)


def assign_ab_test(users: pd.DataFrame, orders_base: pd.DataFrame) -> pd.DataFrame:
    """Делим на группы активных пользователей.

    Два условия попадания в тест:

    * зарегистрировался минимум за 21 день до старта — чтобы у каждого была
      история до эксперимента, по которой можно проверить, что группы
      изначально одинаковые;
    * сделал хотя бы один заказ за 28 дней до старта — то есть на момент
      старта он живой. Если катить тест на всех подряд, 90% выборки будут
      люди, которые всё равно ничего не закажут: они не дадут эффекта, но
      добавят шума, и тест не увидит даже реального изменения.

    Оба условия смотрят только на прошлое, поэтому эксперимент на них
    повлиять не может — иначе отбор сам стал бы источником различий.
    """
    pre_start = AB_START - dt.timedelta(days=28)
    active = set(
        orders_base.loc[
            (orders_base["order_date"] >= pre_start) & (orders_base["order_date"] < AB_START),
            "user_id",
        ]
    )
    eligible = (users["reg_date"] <= AB_START - dt.timedelta(days=21)) & users["user_id"].isin(active)
    ids = users.loc[eligible, "user_id"].to_numpy()
    group = rng.choice(["control", "test"], size=ids.size, p=[0.5, 0.5])
    return pd.DataFrame({"user_id": ids, "ab_group": group, "ab_start_date": AB_START})


def make_orders(users: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    """Заказы каждого пользователя.

    Логика по шагам, ровно в том порядке, в котором это происходит в жизни:

    1. Человек регистрируется. С какой-то вероятностью делает первый заказ.
    2. Первый заказ приезжает за какое-то время. Если долго — впечатление
       испорчено: человек уходит быстрее (жизнь короче на 32%).
    3. У каждого своя частота заказов и своя «продолжительность жизни» в
       сервисе. Заказы идут, пока жизнь не кончилась или пока не кончились
       данные.
    4. Если пользователь в тестовой группе A/B, после старта теста его
       частота заказов выше, а средний чек ниже.

    Важная деталь: «жизнь» пользователя не зависит от того, когда у нас
    заканчиваются данные. Если бы она зависела (например, если бы мы делили
    число заказов на остаток окна наблюдения), то у февральской когорты
    retention первого месяца получился бы ниже, чем у сентябрьской, — просто
    из-за формулы, а не из-за поведения людей.
    """
    ab_map = ab.set_index("user_id")["ab_group"].to_dict() if len(ab) else {}
    rows = []

    for u in users.itertuples():
        # У каждого пользователя свой генератор случайных чисел, привязанный
        # к его user_id. Благодаря этому поведение человека до эксперимента
        # не зависит от того, кто попал в тест, — и функцию можно спокойно
        # вызвать дважды: сначала чтобы выбрать активных, потом уже с
        # группами.
        r = np.random.default_rng([SEED, int(u.user_id)])

        # --- шаг 1: будет ли первый заказ ---------------------------------
        p_first = _sigmoid(0.45 + 0.75 * u.quality)
        if r.random() > p_first:
            continue

        delay = int(r.exponential(2.2))            # дней от регистрации до 1-го заказа
        first_date = u.reg_date + dt.timedelta(days=delay)
        if first_date > DATA_END:
            continue

        city_time_mult = CITIES[u.city][1]
        city_amount_mult = CITIES[u.city][2]

        # --- шаг 2: как часто и как долго он будет заказывать --------------
        rate_per_month = np.exp(0.15 + 0.55 * u.quality) * 1.6
        months_left = (DATA_END - first_date).days / 30.0

        first_delivery = r.lognormal(np.log(34 * city_time_mult), 0.35)
        if first_delivery > LATE_DELIVERY_MIN:
            # Долгая первая доставка — главный операционный убийца retention.
            rate_per_month *= 0.68

        group = ab_map.get(u.user_id)
        n_extra = r.poisson(max(rate_per_month * months_left * 0.55, 0))

        # --- шаг 3: раскидываем заказы по времени -------------------------
        dates = [first_date]
        cursor = first_date
        for _ in range(n_extra):
            gap = max(1, int(r.exponential(30.0 / max(rate_per_month, 0.3))))
            cursor = cursor + dt.timedelta(days=gap)
            if cursor > DATA_END:
                break
            dates.append(cursor)

        # --- шаг 4: сумма, время доставки и деньги за каждый заказ ---------
        # В тесте люди перестают «добивать» корзину до 2000 руб., поэтому
        # средний чек внутри окна эксперимента ниже на 9%.
        for i, d in enumerate(sorted(dates)):
            in_ab_window = (
                group is not None
                and AB_START <= d < AB_START + dt.timedelta(days=AB_HORIZON_DAYS)
            )
            amount_mult = 0.91 if (in_ab_window and group == "test") else 1.0
            amount = float(
                r.lognormal(np.log(1350 * city_amount_mult * amount_mult), 0.45)
            )
            minutes = (
                first_delivery if i == 0
                else float(r.lognormal(np.log(34 * city_time_mult), 0.35))
            )
            threshold = (
                FREE_DELIVERY_TEST
                if (in_ab_window and group == "test")
                else FREE_DELIVERY_CONTROL
            )
            fee = 0 if amount >= threshold else DELIVERY_FEE
            rows.append(
                {
                    "user_id": u.user_id,
                    "order_date": d,
                    "amount_rub": round(amount, 2),
                    "delivery_minutes": round(minutes, 1),
                    "delivery_fee_rub": fee,
                    "free_delivery_threshold": threshold,
                    "items": int(np.clip(r.poisson(amount / 190), 1, 60)),
                }
            )

    orders = pd.DataFrame(rows).sort_values(["user_id", "order_date"], ignore_index=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))
    # Маржа с заказа: наценка на товары + то, что клиент заплатил за
    # доставку, минус то, во сколько доставка обошлась нам.
    orders["margin_rub"] = (
        MARGIN_RATE * orders["amount_rub"] + orders["delivery_fee_rub"] - DELIVERY_COST
    ).round(2)
    return orders


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    users = make_users()
    # Первый проход — «мир без эксперимента». Он нужен только чтобы понять,
    # кто был активен до старта теста и, значит, попадает в выборку.
    orders_base = make_orders(users, pd.DataFrame(columns=["user_id", "ab_group"]))
    ab = assign_ab_test(users, orders_base)
    # Второй проход — итоговые данные, уже с эффектом эксперимента.
    orders = make_orders(users, ab)

    users.drop(columns="quality").to_csv(DATA / "users.csv", index=False, encoding="utf-8")
    orders.to_csv(DATA / "orders.csv", index=False, encoding="utf-8")
    ab.to_csv(DATA / "ab_test.csv", index=False, encoding="utf-8")

    print(f"users     {len(users):>8,}")
    print(f"orders    {len(orders):>8,}")
    print(f"ab_test   {len(ab):>8,}")
    print(f"-> {DATA}")


if __name__ == "__main__":
    main()

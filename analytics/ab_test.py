"""A/B-тест: «снизить порог бесплатной доставки с 2000 до 1000 рублей».

Порядок проверок здесь важнее самих формул. Сначала мы выясняем, можно ли
вообще верить этому тесту, и только потом считаем эффект:

1. Сошлось ли деление на группы (SRM). Если в группах не 50/50, значит
   что-то сломалось в разбиении или в логировании, и любые цифры дальше
   бессмысленны — группы просто разные по составу.
2. Одинаковы ли группы ДО эксперимента (A/A-проверка). У нас есть история
   за 28 дней до старта: если она уже отличается, рандомизация не сработала.
3. Только теперь — эффект на метрики, с доверительными интервалами.
4. И главный вопрос: что стало с деньгами. Заказов может стать больше, а
   денег меньше — это самая частая ловушка в тестах со скидками.

Запуск:
    python -m analytics.ab_test
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

from analytics.data import (
    AB_HORIZON_DAYS,
    AB_START,
    DELIVERY_COST,
    FREE_DELIVERY_CONTROL,
    FREE_DELIVERY_TEST,
)
from analytics.metrics import load

ALPHA = 0.05          # уровень значимости
SRM_ALPHA = 0.001     # для SRM порог строже: это проверка данных, а не гипотезы
POWER = 0.80


# --- шаг 1: проверка деления на группы -------------------------------------
def srm_check(counts: dict[str, int]) -> dict:
    """Sample Ratio Mismatch — «не сошлось деление на группы».

    Мы планировали 50/50. Считаем хи-квадрат: насколько сильно фактические
    размеры групп отличаются от ожидаемых. Порог берём 0.001, а не 0.05,
    потому что эта проверка запускается на каждом тесте, и на пороге 0.05
    она ложно срабатывала бы в 5% случаев — команда перестала бы её читать.
    """
    keys = sorted(counts)
    observed = np.array([counts[k] for k in keys], dtype=float)
    expected = np.full(len(keys), observed.sum() / len(keys))
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=len(keys) - 1))
    return {
        "группы": {k: int(counts[k]) for k in keys},
        "доли": {k: counts[k] / observed.sum() for k in keys},
        "chi2": chi2,
        "p_value": p,
        "ок": p >= SRM_ALPHA,
    }


# --- шаг 3: сравнение групп ------------------------------------------------
def compare_means(control: np.ndarray, test: np.ndarray, name: str) -> dict:
    """Сравнение средних (t-тест Уэлча) + доверительный интервал разницы.

    Уэлч, а не обычный t-тест: он не требует равных дисперсий в группах.
    Равные дисперсии — предположение, которое почти никогда не выполняется,
    а проверять его отдельным тестом только добавляет проблем.
    """
    control = np.asarray(control, dtype=float)
    test = np.asarray(test, dtype=float)
    diff = test.mean() - control.mean()
    se = np.sqrt(control.var(ddof=1) / control.size + test.var(ddof=1) / test.size)
    df = se**4 / (
        (control.var(ddof=1) / control.size) ** 2 / (control.size - 1)
        + (test.var(ddof=1) / test.size) ** 2 / (test.size - 1)
    )
    crit = stats.t.ppf(1 - ALPHA / 2, df)
    p = float(stats.ttest_ind(test, control, equal_var=False).pvalue)
    return {
        "метрика": name,
        "контроль": float(control.mean()),
        "тест": float(test.mean()),
        "разница": float(diff),
        "разница_%": float(diff / control.mean()) if control.mean() else np.nan,
        "ci_low": float(diff - crit * se),
        "ci_high": float(diff + crit * se),
        "p_value": p,
        "значимо": p < ALPHA,
        # MDE — какой минимальный эффект мы вообще могли бы заметить на
        # таком размере групп. Без него фраза «эффекта нет» ничего не значит.
        "MDE": float(
            (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER))
            * np.sqrt(
                np.concatenate([control, test]).var(ddof=1)
                * (1 / control.size + 1 / test.size)
            )
        ),
    }


def compare_shares(x_c: int, n_c: int, x_t: int, n_t: int, name: str) -> dict:
    """Сравнение долей (z-тест для двух пропорций).

    Для теста используем объединённую оценку доли (так правильно при
    нулевой гипотезе), а для доверительного интервала — раздельные
    (интервал должен описывать то, что мы измерили, а не гипотезу).
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    se_ci = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    z = diff / se_test if se_test else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    crit = stats.norm.ppf(1 - ALPHA / 2)
    return {
        "метрика": name,
        "контроль": p_c,
        "тест": p_t,
        "разница": diff,
        "разница_%": diff / p_c if p_c else np.nan,
        "ci_low": diff - crit * se_ci,
        "ci_high": diff + crit * se_ci,
        "p_value": p,
        "значимо": p < ALPHA,
        "MDE": float(
            (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER))
            * np.sqrt(2 * p_pool * (1 - p_pool) / min(n_c, n_t))
        ),
    }


# --- сборка таблицы для анализа -------------------------------------------
def build_user_table(users: pd.DataFrame, orders: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на пользователя: что было до теста и что стало после.

    Все окна отсчитываются от даты старта эксперимента, а не от регистрации.
    Если считать от регистрации, в «период после» попадёт поведение, которого
    человек ещё не мог показать, — и эффект размажется.
    """
    start = pd.Timestamp(AB_START)
    end = start + pd.Timedelta(days=AB_HORIZON_DAYS)
    pre_start = start - pd.Timedelta(days=AB_HORIZON_DAYS)

    o = orders.merge(ab[["user_id", "ab_group"]], on="user_id")
    after = o[(o["order_date"] >= start) & (o["order_date"] < end)]
    before = o[(o["order_date"] >= pre_start) & (o["order_date"] < start)]

    def agg(df, suffix):
        return df.groupby("user_id").agg(
            **{
                f"заказов{suffix}": ("order_id", "count"),
                f"выручка{suffix}": ("amount_rub", "sum"),
                f"маржа{suffix}": ("margin_rub", "sum"),
                f"чек{suffix}": ("amount_rub", "mean"),
            }
        )

    table = ab[["user_id", "ab_group"]].copy()
    table = table.merge(agg(after, "_после"), on="user_id", how="left")
    table = table.merge(agg(before, "_до"), on="user_id", how="left")
    fill = {c: 0.0 for c in table.columns if c.startswith(("заказов", "выручка", "маржа"))}
    table = table.fillna(fill)  # у «чека» пропуск остаётся: чека без заказов не бывает
    return table.merge(users[["user_id", "city", "channel", "platform"]], on="user_id")


# --- главный отчёт ---------------------------------------------------------
def analyze(verbose: bool = True) -> dict:
    users, orders, ab = load()
    t = build_user_table(users, orders, ab)
    say = print if verbose else (lambda *a, **k: None)

    say("=" * 76)
    say("A/B-тест: порог бесплатной доставки "
        f"{FREE_DELIVERY_CONTROL} руб. (контроль) -> {FREE_DELIVERY_TEST} руб. (тест)")
    say(f"Старт {AB_START}, окно наблюдения {AB_HORIZON_DAYS} дней")
    say("=" * 76)

    # 1. SRM
    srm = srm_check(t["ab_group"].value_counts().to_dict())
    say("\n1. ПРОВЕРКА ДЕЛЕНИЯ НА ГРУППЫ (SRM)")
    doli = "  ".join(f"{k}={v:.2%}" for k, v in srm["доли"].items())
    razm = "  ".join(f"{k}={v:,}" for k, v in srm["группы"].items())
    say(f"   {razm}   ({doli})   chi2={srm['chi2']:.2f}  p={srm['p_value']:.3f}")
    say("   " + ("OK: деление сошлось" if srm["ок"] else "СТОП: деление не сошлось"))
    if not srm["ок"]:
        return {"srm": srm, "вывод": "тест невалиден"}

    # 2. A/A-проверка на периоде до эксперимента
    say("\n2. ПРОВЕРКА ГРУПП ДО ЭКСПЕРИМЕНТА (A/A)")
    aa = []
    for col in ["заказов_до", "маржа_до"]:
        r = compare_means(
            t.loc[t.ab_group == "control", col], t.loc[t.ab_group == "test", col], col
        )
        aa.append(r)
        flag = "различий нет" if not r["значимо"] else "ЕСТЬ РАЗЛИЧИЯ — рандомизация под вопросом"
        say(f"   {col:<14} контроль={r['контроль']:7.3f}  тест={r['тест']:7.3f}  "
            f"p={r['p_value']:.3f}   {flag}")

    # 3. Эффект на продуктовые метрики
    say("\n3. ЭФФЕКТ НА МЕТРИКИ")
    results = []
    for col, name in [
        ("заказов_после", "заказов на пользователя"),
        ("выручка_после", "выручка на пользователя, руб."),
        ("маржа_после", "маржа на пользователя, руб."),
    ]:
        results.append(
            compare_means(
                t.loc[t.ab_group == "control", col], t.loc[t.ab_group == "test", col], name
            )
        )

    buyers = t["заказов_после"] > 0
    results.append(
        compare_shares(
            int(buyers[t.ab_group == "control"].sum()), int((t.ab_group == "control").sum()),
            int(buyers[t.ab_group == "test"].sum()), int((t.ab_group == "test").sum()),
            "доля сделавших заказ",
        )
    )
    # Средний чек считаем только по тем, кто заказал. Это сравнение
    # «выживших», поэтому читаем его осторожно: состав групп здесь уже
    # не случайный, и небольшой сдвиг может быть эффектом отбора.
    results.append(
        compare_means(
            t.loc[(t.ab_group == "control") & buyers, "чек_после"].dropna(),
            t.loc[(t.ab_group == "test") & buyers, "чек_после"].dropna(),
            "средний чек, руб. (только заказавшие)",
        )
    )

    for r in results:
        say(f"   {r['метрика']:<38} {r['контроль']:9.3f} -> {r['тест']:9.3f}   "
            f"{r['разница_%']:+7.1%}   95% ДИ [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]   "
            f"p={r['p_value']:.2e}   MDE={r['MDE']:.3f}")

    # 4. Деньги
    say("\n4. ЧТО СТАЛО С ДЕНЬГАМИ")
    margin = next(r for r in results if r["метрика"].startswith("маржа"))
    orders_r = next(r for r in results if r["метрика"].startswith("заказов"))
    n_test = int((t.ab_group == "test").sum())
    say(f"   заказов на пользователя      {orders_r['разница_%']:+.1%}")
    say(f"   маржа на пользователя        {margin['разница_%']:+.1%} "
        f"({margin['разница']:+.1f} руб.)")
    say(f"   в пересчёте на {n_test:,} пользователей теста: "
        f"{margin['разница'] * n_test / 1000:+.0f} тыс. руб. за {AB_HORIZON_DAYS} дней")
    free_share = (
        t.loc[t.ab_group == "test", "заказов_после"].sum(),
        t.loc[t.ab_group == "control", "заказов_после"].sum(),
    )
    say(f"   всего заказов: контроль {free_share[1]:,.0f}, тест {free_share[0]:,.0f}")
    say(f"   каждая бесплатная доставка стоит нам {DELIVERY_COST} руб.")

    вывод = _decide(orders_r, margin)
    say(f"\n5. ВЫВОД: {вывод}")

    return {
        "srm": srm,
        "aa": pd.DataFrame(aa),
        "результаты": pd.DataFrame(results),
        "вывод": вывод,
        "таблица": t,
    }


def _decide(orders_r: dict, margin: dict) -> str:
    """Решение принимается по деньгам, а не по количеству заказов.

    Отдельно следим за формулировкой про заказы: если эффект незначим,
    писать «заказов стало больше» нельзя, даже когда среднее выросло.
    """
    if orders_r["значимо"]:
        про_заказы = f"Заказов стало больше ({orders_r['разница_%']:+.1%})"
    else:
        про_заказы = (
            f"Заказов, возможно, стало чуть больше ({orders_r['разница_%']:+.1%}), "
            f"но этот эффект незначим (p={orders_r['p_value']:.2f}, "
            f"MDE={orders_r['MDE']:.2f} заказа) — утверждать рост нельзя"
        )
    if margin["значимо"] and margin["разница"] < 0:
        return (
            f"НЕ РАСКАТЫВАТЬ в текущем виде. {про_заказы}, "
            f"а маржа на пользователя упала "
            f"({margin['разница_%']:+.1%}, p={margin['p_value']:.1e}). "
            "Мы заплатили за доставку и потеряли часть чека, не получив "
            "взамен значимого роста частоты заказов. Следующий шаг — тест с "
            "промежуточным порогом (например, 1500 руб.) или бесплатная "
            "доставка только для тех, кто и так близок к порогу."
        )
    if margin["значимо"] and margin["разница"] > 0:
        return f"РАСКАТЫВАТЬ: маржа на пользователя выросла на {margin['разница_%']:+.1%}."
    return (
        "НЕДОСТАТОЧНО ДАННЫХ: изменение маржи незначимо "
        f"(p={margin['p_value']:.2f}, MDE={margin['MDE']:.1f} руб.). "
        "Нужен более длинный тест или больше пользователей."
    )


if __name__ == "__main__":
    analyze()

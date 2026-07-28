"""Тесты.

Здесь проверяется не «работает ли код», а «правильные ли цифры он выдаёт».
Ошибка в аналитике обычно не падает с исключением — она просто тихо выдаёт
неверное число, и на его основе принимают решение. Поэтому тесты устроены
так:

* статистические функции проверяются на данных с заранее известным ответом
  (в том числе на A/A-данных, где эффекта нет по построению);
* метрики проверяются на свойствах, которые обязаны выполняться всегда
  (воронка не может расти, удержание не может быть больше 100%);
* ключевые цифры пересчитываются вторым способом — на SQL — и сравниваются
  с pandas. Если два независимых расчёта сошлись, вероятность ошибки резко
  падает.

Запуск:
    python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.ab_test import build_user_table, compare_means, compare_shares, srm_check
from analytics.data import DATA, DELIVERY_COST, MARGIN_RATE
from analytics.metrics import (
    channel_economics,
    cohort_retention,
    delivery_impact,
    first_order_funnel,
    load,
    rfm,
)

pytestmark = pytest.mark.skipif(
    not (DATA / "orders.csv").exists(),
    reason="нет данных — сначала запустите `python -m analytics.data`",
)


@pytest.fixture(scope="module")
def данные():
    return load()


# --- статистика ------------------------------------------------------------
def test_srm_пропускает_честное_деление():
    rng = np.random.default_rng(0)
    n = int(rng.binomial(100_000, 0.5))
    assert srm_check({"control": n, "test": 100_000 - n})["ок"]


def test_srm_ловит_перекос_на_большой_выборке():
    """51/49 на 200 тысячах — это уже не случайность."""
    res = srm_check({"control": 98_000, "test": 102_000})
    assert not res["ок"]
    assert res["p_value"] < 1e-6


def test_srm_не_срабатывает_на_маленькой_выборке():
    """Те же 51/49, но на 200 наблюдениях — обычный шум, тревожить незачем."""
    assert srm_check({"control": 98, "test": 102})["ок"]


def test_сравнение_долей_совпадает_с_хи_квадрат():
    x_c, n_c, x_t, n_t = 480, 2000, 545, 2010
    got = compare_shares(x_c, n_c, x_t, n_t, "тест")
    _, expected_p, _, _ = stats.chi2_contingency(
        [[x_c, n_c - x_c], [x_t, n_t - x_t]], correction=False
    )
    assert got["p_value"] == pytest.approx(expected_p, rel=1e-9)


def test_сравнение_средних_совпадает_со_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 3000), rng.normal(0.15, 2.5, 1200)
    got = compare_means(a, b, "тест")
    assert got["p_value"] == pytest.approx(stats.ttest_ind(b, a, equal_var=False).pvalue)
    assert got["ci_low"] < got["разница"] < got["ci_high"]


def test_доверительный_интервал_накрывает_истинную_разницу():
    """Прогоняем 200 экспериментов с известным эффектом 0.3.

    Интервал должен накрывать истину примерно в 95% случаев — это и есть
    смысл слов «95-процентный доверительный интервал».
    """
    rng = np.random.default_rng(2)
    накрыл = 0
    for _ in range(200):
        a, b = rng.normal(0, 1, 800), rng.normal(0.3, 1, 800)
        r = compare_means(a, b, "тест")
        накрыл += r["ci_low"] <= 0.3 <= r["ci_high"]
    assert 0.90 <= накрыл / 200 <= 1.0


def test_на_A_A_данных_эффект_находится_в_5_процентах_случаев():
    """Ложные срабатывания должны быть на уровне 5%, а не выше."""
    rng = np.random.default_rng(3)
    сработало = sum(
        compare_means(rng.normal(0, 1, 1000), rng.normal(0, 1, 1000), "x")["значимо"]
        for _ in range(1000)
    )
    assert сработало / 1000 == pytest.approx(0.05, abs=0.02)


# --- метрики ---------------------------------------------------------------
def test_воронка_только_убывает(данные):
    users, orders, _ = данные
    f = first_order_funnel(users, orders)
    assert f["пользователей"].is_monotonic_decreasing
    assert (f["конверсия из предыдущего"].dropna() <= 1).all()


def test_удержание_это_доля(данные):
    users, orders, _ = данные
    m = cohort_retention(users, orders).drop(columns="размер когорты")
    values = m.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    assert (values >= 0).all() and (values <= 1).all()


def test_ненаступившие_месяцы_остаются_пустыми(данные):
    """У февральской когорты не может быть 5-го месяца жизни — там NaN,
    а не ноль. Ноль занизил бы среднее и испортил сравнение когорт."""
    users, orders, _ = данные
    m = cohort_retention(users, orders).drop(columns="размер когорты")
    последняя = m.iloc[-1]
    assert последняя.isna().sum() > 0
    assert not m.iloc[0].isna().any()


def test_romi_считается_по_формуле(данные):
    users, orders, _ = данные
    ch = channel_economics(users, orders)
    платные = ch[ch["CAC"] > 0]
    ожидаемо = (платные["LTV90"] - платные["CAC"]) / платные["CAC"]
    assert np.allclose(платные["ROMI"], ожидаемо)


def test_у_органики_нет_romi(данные):
    """CAC = 0, делить нельзя. Должен быть NaN, а не бесконечность."""
    users, orders, _ = данные
    ch = channel_economics(users, orders)
    assert ch.loc[ch["channel"] == "органика", "ROMI"].isna().all()


def test_маржа_считается_по_формуле(данные):
    """Пересчитываем маржу из суммы заказа и платы за доставку."""
    _, orders, _ = данные
    ожидаемо = (
        MARGIN_RATE * orders["amount_rub"] + orders["delivery_fee_rub"] - DELIVERY_COST
    )
    assert np.allclose(orders["margin_rub"], ожидаемо.round(2))


def test_долгая_доставка_снижает_число_заказов(данные):
    """Эффект, ради которого этот датасет и сделан: он должен быть виден."""
    users, orders, _ = данные
    d = delivery_impact(users, orders).set_index("первая доставка")
    assert d.loc["дольше 45 минут", "заказов_на_юзера"] < d.loc["до 45 минут", "заказов_на_юзера"]


def test_rfm_покрывает_всех_покупателей(данные):
    users, orders, _ = данные
    r = rfm(users, orders)
    assert r["пользователей"].sum() == orders["user_id"].nunique()
    assert r["доля"].sum() == pytest.approx(1.0)


# --- A/B -------------------------------------------------------------------
def test_группы_ab_не_отличались_до_эксперимента(данные):
    """A/A-проверка на реальных данных проекта.

    Если этот тест падает, значит рандомизация или отбор в эксперимент
    сломаны, и все выводы по тесту недействительны.
    """
    users, orders, ab = данные
    t = build_user_table(users, orders, ab)
    r = compare_means(
        t.loc[t.ab_group == "control", "заказов_до"],
        t.loc[t.ab_group == "test", "заказов_до"],
        "заказов до теста",
    )
    assert not r["значимо"], f"группы отличались ещё до старта, p={r['p_value']:.4f}"


def test_у_каждого_участника_ab_ровно_одна_группа(данные):
    _, _, ab = данные
    assert ab["user_id"].is_unique
    assert set(ab["ab_group"]) == {"control", "test"}


# --- сверка pandas и SQL ---------------------------------------------------
def test_sql_и_pandas_дают_одинаковую_воронку(данные):
    """Два независимых расчёта одной метрики должны сойтись."""
    from analytics.sql_demo import SQL_FILE, connect, split_queries

    users, orders, _ = данные
    ожидаемо = first_order_funnel(users, orders)

    con = connect()
    запросы = dict(split_queries(SQL_FILE.read_text(encoding="utf-8")))
    got = con.execute(запросы["Воронка повторных заказов"]).df()
    con.close()

    assert got["пользователей"].tolist() == ожидаемо["пользователей"].tolist()


def test_sql_и_pandas_дают_одинаковую_экономику_каналов(данные):
    from analytics.sql_demo import SQL_FILE, connect, split_queries

    users, orders, _ = данные
    ожидаемо = channel_economics(users, orders).set_index("channel")["LTV90"]

    con = connect()
    запросы = dict(split_queries(SQL_FILE.read_text(encoding="utf-8")))
    got = con.execute(запросы["Экономика каналов привлечения"]).df().set_index("канал")["LTV90"]
    con.close()

    объединено = pd.concat([ожидаемо, got], axis=1).dropna()
    assert np.allclose(объединено.iloc[:, 0], объединено.iloc[:, 1], atol=0.1)

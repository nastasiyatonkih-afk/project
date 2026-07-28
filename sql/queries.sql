-- Те же метрики, но на SQL.
--
-- Зачем дублировать то, что уже посчитано на pandas: в реальной работе
-- аналитик почти всё считает в SQL прямо в хранилище, а Python подключает
-- там, где нужна статистика или графики. Эти запросы показывают, что
-- цифры из отчёта воспроизводятся одним запросом к сырым данным.
--
-- Запуск: python -m analytics.sql_demo
-- Каждый блок начинается со строки «-- @Название» — по ней скрипт делит файл.

-- @Главные метрики
SELECT
    count(DISTINCT u.user_id)                                        AS регистраций,
    count(DISTINCT o.user_id)                                        AS покупателей,
    round(count(DISTINCT o.user_id) * 1.0 / count(DISTINCT u.user_id), 4) AS конверсия_в_заказ,
    count(o.order_id)                                                AS заказов,
    round(avg(o.amount_rub), 1)                                      AS средний_чек,
    round(avg(o.margin_rub), 1)                                      AS маржа_с_заказа,
    round(sum(o.amount_rub) / 1e6, 2)                                AS выручка_млн,
    round(sum(o.margin_rub) / 1e6, 2)                                AS маржа_млн
FROM users u
LEFT JOIN orders o USING (user_id);

-- @Воронка повторных заказов
-- Оконная функция lag даёт конверсию из предыдущего шага, не заставляя
-- джойнить таблицу саму с собой.
WITH n_orders AS (
    SELECT user_id, count(*) AS cnt FROM orders GROUP BY user_id
),
steps AS (
    SELECT 0 AS шаг, 'регистрация' AS название, (SELECT count(*) FROM users) AS пользователей
    UNION ALL SELECT 1, '1-й заказ', (SELECT count(*) FROM n_orders WHERE cnt >= 1)
    UNION ALL SELECT 2, '2-й заказ', (SELECT count(*) FROM n_orders WHERE cnt >= 2)
    UNION ALL SELECT 3, '3-й заказ', (SELECT count(*) FROM n_orders WHERE cnt >= 3)
    UNION ALL SELECT 4, '4-й заказ', (SELECT count(*) FROM n_orders WHERE cnt >= 4)
)
SELECT
    название,
    пользователей,
    round(пользователей * 1.0 / lag(пользователей) OVER (ORDER BY шаг), 4) AS конверсия_из_предыдущего,
    round(пользователей * 1.0 / first_value(пользователей) OVER (ORDER BY шаг), 4) AS доля_от_регистраций,
    lag(пользователей) OVER (ORDER BY шаг) - пользователей AS потеряли
FROM steps
ORDER BY шаг;

-- @Удержание по когортам
-- Главная тонкость — знаменатель. Пара (когорта, месяц) попадает в расчёт
-- только если этот месяц у когорты уже наступил, иначе свежие когорты
-- выглядели бы хуже старых просто потому, что они моложе.
WITH cohorts AS (
    SELECT user_id, date_trunc('month', reg_date)::DATE AS cohort FROM users
),
bounds AS (SELECT max(order_date)::DATE AS data_end FROM orders),
grid AS (
    SELECT c.user_id, c.cohort, m.month_index
    FROM cohorts c
    CROSS JOIN bounds b
    CROSS JOIN (SELECT unnest(generate_series(0, 5))::INTEGER AS month_index) m
    WHERE m.month_index <= date_diff('month', c.cohort, b.data_end)
),
активность AS (
    SELECT DISTINCT c.user_id, date_diff('month', c.cohort, o.order_date) AS month_index
    FROM orders o JOIN cohorts c USING (user_id)
)
SELECT
    g.cohort,
    g.month_index,
    count(*)                                          AS размер_когорты,
    count(a.user_id)                                  AS активных,
    round(count(a.user_id) * 1.0 / count(*), 4)       AS удержание
FROM grid g
LEFT JOIN активность a
       ON a.user_id = g.user_id AND a.month_index = g.month_index
GROUP BY g.cohort, g.month_index
ORDER BY g.cohort, g.month_index;

-- @Экономика каналов привлечения
-- В расчёт берём только пользователей, проживших 90 дней: иначе свежие
-- когорты занизят LTV и все каналы будут выглядеть убыточными.
WITH bounds AS (SELECT max(order_date)::DATE AS data_end FROM orders),
зрелые AS (
    SELECT u.* FROM users u, bounds b
    WHERE u.reg_date <= b.data_end - 90
),
за_90_дней AS (
    SELECT z.user_id, count(o.order_id) AS заказов, coalesce(sum(o.margin_rub), 0) AS маржа
    FROM зрелые z
    LEFT JOIN orders o
           ON o.user_id = z.user_id
          AND date_diff('day', z.reg_date, o.order_date) < 90
    GROUP BY z.user_id
)
SELECT
    z.channel                                              AS канал,
    count(*)                                               AS регистраций,
    max(z.cac_rub)                                         AS CAC,
    round(avg((d.заказов > 0)::INT), 4)                    AS конверсия_в_заказ,
    round(avg(d.заказов), 3)                               AS заказов_на_юзера,
    round(avg(d.маржа), 1)                                 AS LTV90,
    round((avg(d.маржа) - max(z.cac_rub)) / nullif(max(z.cac_rub), 0), 3) AS ROMI
FROM зрелые z
JOIN за_90_дней d USING (user_id)
GROUP BY z.channel
ORDER BY ROMI DESC NULLS LAST;

-- @Влияние времени первой доставки
-- row_number() = 1 — стандартный способ достать «первую строку в группе».
WITH первый_заказ AS (
    SELECT user_id, delivery_minutes
    FROM orders
    QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY order_date, order_id) = 1
),
итоги AS (
    SELECT user_id, count(*) AS всего_заказов, sum(margin_rub) AS маржа
    FROM orders GROUP BY user_id
)
SELECT
    CASE WHEN f.delivery_minutes > 45 THEN 'дольше 45 минут' ELSE 'до 45 минут' END AS первая_доставка,
    count(*)                                              AS пользователей,
    round(avg(i.всего_заказов), 3)                        AS заказов_на_юзера,
    round(avg((i.всего_заказов >= 2)::INT), 4)            AS доля_с_повторным,
    round(avg(i.маржа), 1)                                AS маржа_на_юзера
FROM первый_заказ f
JOIN итоги i USING (user_id)
GROUP BY 1
ORDER BY 1;

-- @RFM-сегменты
-- ntile(3) делит пользователей на три равные части по каждому признаку.
-- Для recency порядок обратный: чем меньше дней прошло, тем выше балл.
WITH bounds AS (SELECT max(order_date)::DATE AS data_end FROM orders),
базовые AS (
    SELECT
        o.user_id,
        date_diff('day', max(o.order_date)::DATE, (SELECT data_end FROM bounds)) AS recency,
        count(*)          AS frequency,
        sum(o.margin_rub) AS monetary
    FROM orders o GROUP BY o.user_id
),
баллы AS (
    SELECT *,
        ntile(3) OVER (ORDER BY recency DESC)   AS R,
        ntile(3) OVER (ORDER BY frequency)      AS F,
        ntile(3) OVER (ORDER BY monetary)       AS M
    FROM базовые
)
SELECT
    CASE
        WHEN R = 3 AND F = 3  THEN 'Чемпионы'
        WHEN R = 3 AND F <= 2 THEN 'Новички и редкие'
        WHEN R = 2 AND F >= 2 THEN 'Лояльные'
        WHEN R = 1 AND F = 3  THEN 'В зоне риска'
        WHEN R = 1            THEN 'Спящие'
        ELSE 'Прочие'
    END                                    AS сегмент,
    count(*)                               AS пользователей,
    round(median(recency), 0)              AS дней_с_последнего_заказа,
    round(avg(frequency), 2)               AS заказов,
    round(avg(monetary), 0)                AS маржа_на_юзера,
    round(sum(monetary) / 1e6, 2)          AS маржа_всего_млн
FROM баллы
GROUP BY 1
ORDER BY маржа_всего_млн DESC;

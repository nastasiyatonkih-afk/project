"""Графики для отчёта.

Несколько правил, которых я придерживаюсь во всех графиках проекта:

* один график — одна мысль, и она вынесена в заголовок;
* никогда не две оси Y на одной картинке. Если метрики в разных единицах —
  это две картинки, а не две шкалы (иначе связь между линиями можно
  «нарисовать» любую, просто подобрав масштаб);
* цвет закреплён за сущностью, а не за местом в рейтинге: контроль всегда
  синий, тест всегда оранжевый, во всех графиках;
* красным помечается только то, что действительно плохо, и рядом всегда
  есть подпись словами — цвет не должен быть единственным носителем смысла
  (иначе график нечитаем для людей с дальтонизмом и в ч/б печати).

Запуск:
    python -m analytics.charts
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analytics import ab_test as AB
from analytics import metrics as M
from analytics.data import DATA, LATE_DELIVERY_MIN

FIGURES = DATA.parent / "reports" / "figures"

# Палитра проверена на различимость при дальтонизме.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, BAD = "#0ca30c", "#d03b3b"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5, "text.color": INK,
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.titlesize": 11.5, "axes.titleweight": "600", "axes.titlecolor": INK,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "figure.dpi": 130,
    }
)


def _clean(ax, keep_x_grid: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_x_grid:
        ax.xaxis.grid(False)


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  reports/figures/{name}")


# --- 1. воронка ------------------------------------------------------------
def fig_funnel(users, orders) -> None:
    df = M.first_order_funnel(users, orders)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    colors = [BLUE_RAMP[min(i + 1, 6)] for i in range(len(df))]
    colors[1] = BAD  # переход «регистрация -> 1-й заказ» — самая большая потеря
    ax.bar(df["шаг"], df["пользователей"], color=colors, width=0.66)

    for i, r in df.iterrows():
        ax.text(i, r["пользователей"] + len(users) * 0.018, f"{r['пользователей']:,}".replace(",", " "),
                ha="center", fontsize=9, color=INK2)
        if i:
            ax.text(i, r["пользователей"] / 2, f"{r['конверсия из предыдущего']:.0%}",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)

    ax.annotate(
        f"здесь теряем больше всего:\n{df.loc[1, 'потеряли']:,} человек так и не заказали".replace(",", " "),
        xy=(1.35, df.loc[1, "пользователей"]),
        xytext=(2.1, len(users) * 0.82), fontsize=9, color=BAD,
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1.2),
    )
    ax.set_title("Воронка повторных заказов: первый заказ покупает маркетинг,\nвторой и третий — продукт и операции")
    ax.set_ylabel("пользователей")
    ax.set_ylim(0, len(users) * 1.1)
    _clean(ax)
    fig.tight_layout()
    _save(fig, "01_funnel.png")


# --- 2. когорты ------------------------------------------------------------
def fig_cohorts(users, orders) -> None:
    m = M.cohort_retention(users, orders)
    size = m.pop("размер когорты")
    data = m.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    im = ax.imshow(data, cmap=SEQ, aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(m.shape[1]), [f"месяц {c}" for c in m.columns])
    ax.set_yticks(range(m.shape[0]),
                  [f"{c}  (n={int(n):,})".replace(",", " ") for c, n in zip(m.index.astype(str), size)])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=MUTED)
            else:
                ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if data[i, j] > np.nanmax(data) * 0.55 else INK)
    ax.set_title("Удержание по месяцам жизни\n(доля когорты, сделавшая заказ в N-й месяц; «—» — месяц ещё не наступил)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="удержание")
    fig.tight_layout()
    _save(fig, "02_cohorts.png")


# --- 3. каналы -------------------------------------------------------------
def fig_channels(users, orders) -> None:
    ch = M.channel_economics(users, orders)
    paid = ch[ch["CAC"] > 0].sort_values("ROMI")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    order = ch.sort_values("регистраций")
    a1.barh(order["channel"], order["регистраций"], color=BLUE, height=0.62)
    for i, v in enumerate(order["регистраций"]):
        a1.text(v * 1.02, i, f"{v:,}".replace(",", " "), va="center", fontsize=9, color=INK2)
    a1.set_title("Сколько регистраций даёт канал")
    a1.set_xlim(0, order["регистраций"].max() * 1.2)
    _clean(a1, keep_x_grid=True)
    a1.yaxis.grid(False)

    colors = [BAD if v < 0 else AQUA for v in paid["ROMI"]]
    a2.barh(paid["channel"], paid["ROMI"], color=colors, height=0.62)
    a2.axvline(0, color=AXIS, lw=1.2)
    for i, v in enumerate(paid["ROMI"]):
        # Подписи всегда справа от нуля — так они не наезжают на названия
        # каналов, когда столбик уходит влево.
        label = f"{v:+.0%}" + ("   не окупается" if v < 0 else "")
        a2.text(max(v, 0) + 0.06, i, label, va="center", ha="left", fontsize=9,
                color=BAD if v < 0 else INK2)
    a2.set_title("ROMI за 90 дней: (LTV − CAC) / CAC")
    a2.set_xlim(min(paid["ROMI"].min() - 0.25, -0.3), paid["ROMI"].max() + 0.75)
    a2.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    _clean(a2, keep_x_grid=True)
    a2.yaxis.grid(False)

    fig.suptitle("«VK Реклама» — самый крупный канал по числу регистраций и единственный, который не окупается",
                 fontsize=10.5, color=INK2, y=1.03)
    fig.tight_layout()
    _save(fig, "03_channels.png")


# --- 4. время доставки -----------------------------------------------------
def fig_delivery(users, orders) -> None:
    d = M.delivery_impact(users, orders)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    first = orders.sort_values("order_date").groupby("user_id").first()
    a1.hist(first["delivery_minutes"].clip(upper=110), bins=45, color=BLUE)
    a1.axvline(LATE_DELIVERY_MIN, color=BAD, ls="--", lw=1.6)
    a1.text(LATE_DELIVERY_MIN + 2, a1.get_ylim()[1] * 0.9,
            f"{LATE_DELIVERY_MIN} минут\nграница «долго»", color=BAD, fontsize=9)
    a1.set_title("Время первой доставки")
    a1.set_xlabel("минут")
    a1.set_ylabel("пользователей")
    _clean(a1)

    x = np.arange(len(d))
    bars = a2.bar(x, d["заказов_на_юзера"], color=[BLUE, BAD], width=0.55)
    for b, v, m in zip(bars, d["заказов_на_юзера"], d["маржа_на_юзера"]):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.12,
                f"{v:.2f} заказа\n{m:,.0f} руб. маржи".replace(",", " "),
                ha="center", fontsize=9, color=INK2)
    delta = d["заказов_на_юзера"].iloc[1] / d["заказов_на_юзера"].iloc[0] - 1
    a2.set_xticks(x, d["первая доставка"])
    a2.set_ylabel("заказов на пользователя за всё время")
    a2.set_ylim(0, d["заказов_на_юзера"].max() * 1.35)
    a2.set_title(f"Если первая доставка задержалась,\nчеловек заказывает на {abs(delta):.0%} меньше")
    _clean(a2)

    fig.tight_layout()
    _save(fig, "04_delivery.png")


# --- 5. RFM ----------------------------------------------------------------
def fig_rfm(users, orders) -> None:
    r = M.rfm(users, orders).sort_values("маржа_всего_млн")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

    a1.barh(r["сегмент"], r["доля"], color=BLUE, height=0.6)
    for i, v in enumerate(r["доля"]):
        a1.text(v + 0.004, i, f"{v:.1%}", va="center", fontsize=9, color=INK2)
    a1.set_title("Доля пользователей")
    a1.set_xlim(0, r["доля"].max() * 1.25)
    a1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _clean(a1, keep_x_grid=True)
    a1.yaxis.grid(False)

    a2.barh(r["сегмент"], r["маржа_всего_млн"], color=ORANGE, height=0.6)
    for i, v in enumerate(r["маржа_всего_млн"]):
        a2.text(v + r["маржа_всего_млн"].max() * 0.02, i, f"{v:.1f} млн", va="center",
                fontsize=9, color=INK2)
    a2.set_title("Сколько маржи приносит сегмент")
    a2.set_xlim(0, r["маржа_всего_млн"].max() * 1.3)
    _clean(a2, keep_x_grid=True)
    a2.yaxis.grid(False)

    fig.suptitle("RFM: на кого тратить бюджет на удержание", fontsize=10.5, color=INK2, y=1.02)
    fig.tight_layout()
    _save(fig, "05_rfm.png")


# --- 6. A/B тест -----------------------------------------------------------
def fig_ab(res: dict) -> None:
    r = res["результаты"].iloc[::-1].reset_index(drop=True)
    t = res["таблица"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    for i, row in r.iterrows():
        lo = row["ci_low"] / row["контроль"]
        hi = row["ci_high"] / row["контроль"]
        est = row["разница_%"]
        if not row["значимо"]:
            color, note = MUTED, "нет значимого эффекта"
        elif est < 0:
            color, note = BAD, "стало хуже"
        else:
            color, note = GOOD, "стало лучше"
        a1.plot([lo, hi], [i, i], color=color, lw=2.6, solid_capstyle="round")
        a1.plot([est], [i], "o", color=color, markersize=7, zorder=3)
        a1.text(hi + 0.02, i, f"{est:+.1%}   {note}", va="center", fontsize=9, color=INK2)
    a1.axvline(0, color=AXIS, lw=1.2)
    a1.set_yticks(range(len(r)), r["метрика"])
    a1.set_xlim(-0.45, 0.75)
    a1.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    a1.set_xlabel("изменение относительно контроля, 95% доверительный интервал")
    a1.set_title("Что изменилось в тестовой группе")
    _clean(a1, keep_x_grid=True)
    a1.yaxis.grid(False)

    # Разложение: маржа на пользователя = заказы x маржа с заказа.
    per_group = []
    for g in ["control", "test"]:
        sub = t[t.ab_group == g]
        orders_per_user = sub["заказов_после"].mean()
        margin_per_user = sub["маржа_после"].mean()
        per_group.append(
            {
                "группа": "контроль" if g == "control" else "тест",
                "заказов на юзера": orders_per_user,
                "маржа с заказа": margin_per_user / orders_per_user if orders_per_user else 0,
                "маржа на юзера": margin_per_user,
            }
        )
    pg = pd.DataFrame(per_group)

    labels = ["заказов\nна пользователя", "маржа\nс заказа, руб.", "маржа\nна пользователя, руб."]
    x = np.arange(3)
    # Показываем в процентах от контроля, чтобы три разные величины
    # поместились на одну ось и было видно, что чем компенсируется.
    ctrl = pg.iloc[0][["заказов на юзера", "маржа с заказа", "маржа на юзера"]].to_numpy(float)
    test = pg.iloc[1][["заказов на юзера", "маржа с заказа", "маржа на юзера"]].to_numpy(float)
    a2.bar(x - 0.19, np.ones(3), width=0.36, color=BLUE, label="контроль")
    a2.bar(x + 0.19, test / ctrl, width=0.36, color=ORANGE, label="тест")
    a2.axhline(1, color=AXIS, lw=1.2)
    for i in range(3):
        a2.text(i + 0.19, test[i] / ctrl[i] + 0.02, f"{test[i] / ctrl[i] - 1:+.0%}",
                ha="center", fontsize=9.5, color=BAD if test[i] < ctrl[i] else GOOD,
                fontweight="bold")
        a2.text(i - 0.19, 1.02, f"{ctrl[i]:,.0f}".replace(",", " ") if ctrl[i] > 10 else f"{ctrl[i]:.2f}",
                ha="center", fontsize=8.5, color=MUTED)
    a2.set_xticks(x, labels)
    a2.set_ylabel("к контролю")
    a2.set_ylim(0, 1.45)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.set_title("Заказов чуть больше, но каждый заказ приносит\nгораздо меньше — итог отрицательный")
    a2.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    _clean(a2)

    fig.suptitle("A/B-тест: порог бесплатной доставки 2000 → 1000 руб.",
                 fontsize=11, color=INK, y=1.04, fontweight="600")
    fig.tight_layout()
    _save(fig, "06_ab_test.png")


def main() -> None:
    users, orders, _ = M.load()
    print("графики:")
    fig_funnel(users, orders)
    fig_cohorts(users, orders)
    fig_channels(users, orders)
    fig_delivery(users, orders)
    fig_rfm(users, orders)
    fig_ab(AB.analyze(verbose=False))


if __name__ == "__main__":
    main()

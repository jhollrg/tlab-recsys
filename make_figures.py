"""Рисунки для README. Запуск: python make_figures.py  (нужен только matplotlib)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.unicode_minus": False})
BLUE, GREY, DARK = "#2a6fd6", "#b9b9ae", "#222222"


def fig_split():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    days = [0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21]
    for d in days:
        ax.add_patch(Rectangle((d, 0), 0.9, 1, fc="#dbe7fa", ec=BLUE, lw=1.4))
    ax.annotate("", xy=(-0.6, 1.9), xytext=(19, 1.9), arrowprops=dict(arrowstyle="->", color="#777"))
    ax.plot([19, 19], [1.0, 1.9], ls=":", color="#777")
    ax.text(9, 2.05, "признаки: все события до T (пример для первой test-отсечки)", ha="center", color="#555")
    ax.text(19, 2.05, "T", ha="center", fontweight="bold")
    for x0, x1, name, note in [(0, 15.9, "train", "6 отсечек с шагом 3 суток"), (16, 18.9, "val", "подбор"),
                               (19, 21.9, "test", "итог")]:
        ax.plot([x0, x1], [-0.25, -0.25], color="#777", lw=1.3)
        ax.text((x0 + x1) / 2, -0.75, name, ha="center", fontweight="bold")
        ax.text((x0 + x1) / 2, -1.25, note, ha="center", color="#555", fontsize=10)
    ax.text(0, 1.25, "каждая клетка: окно меток [T, T + 1 сутки)", color=BLUE, fontsize=10)
    ax.set_xlim(-1, 22.5); ax.set_ylim(-1.7, 2.6); ax.axis("off")
    ax.set_title("Признаки берутся только до T, метки только за сутки после T, окна не пересекаются",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_split.png", dpi=160); plt.close(fig)


def fig_windows():
    w = ["1d", "2d", "3d", "5d", "7d", "10d", "14d", "21d", "30d", "60d", "90d", "180d", "all"]
    v = [.018, .012, .016, .032, .027, .025, .027, .031, .034, .057, .242, .253, .225]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(w, [x * 100 for x in v], color=[GREY] * 10 + [BLUE] * 3, width=0.65)
    ax.axhline(100 / 13, ls="--", color="#777", lw=1.2)
    ax.text(0, 100 / 13 + 0.8, "поровну между окнами: 7.7%", color="#555", fontsize=10)
    ax.set_ylabel("доля суммарного gain, %")
    ax.text(4, -8.5, "окна 1-30 дней: 22%", ha="center", fontsize=11)
    ax.text(11, -8.5, "окна от 90 дней: 72%", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("72% gain счётчиков приходится на окна от 90 дней", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_windows.png", dpi=160); plt.close(fig)


def fig_logo():
    easy = [("ua: пользователь-артист", -.0137, -.0158, -.0116), ("ui: пользователь-трек", -.0133, -.0151, -.0118),
            ("item: статистики трека", -.0106, -.0126, -.0086), ("user: статистики пользователя", -.0064, -.0091, -.0038),
            ("artist + album", -.0048, -.0061, -.0036), ("recency + age", -.0036, -.0051, -.0021),
            ("frac_org", -.0034, -.0047, -.0019), ("события skip", -.0016, -.0024, -.0007),
            ("nuniq_item", -.0014, -.0025, -.0003), ("окно 90 дней", -.0013, -.0023, -.0002),
            ("отношения (rate)", -.0009, -.0017, -.00004), ("ual: пользователь-альбом", -.0002, -.0011, .0007)]
    hard = {"ua: пользователь-артист": (-.0276, -.0310, -.0244), "ui: пользователь-трек": (-.0148, -.0165, -.0132),
            "item: статистики трека": (-.0143, -.0169, -.0119), "user: статистики пользователя": (-.0065, -.0089, -.0042),
            "recency + age": (-.0031, -.0049, -.0015), "frac_org": (-.0030, -.0047, -.0013)}
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(easy)
    for i, (name, d, lo, hi) in enumerate(easy):
        y = n - 1 - i
        dy = 0.15 if name in hard else 0
        ax.plot([lo, hi], [y + dy] * 2, color=GREY, lw=2); ax.plot(d, y + dy, "o", color=GREY, ms=7,
                                                                 label="лёгкий протокол" if i == 0 else None)
        if name in hard:
            h, l2, h2 = hard[name]
            ax.plot([l2, h2], [y - 0.15] * 2, color=BLUE, lw=2)
            ax.plot(h, y - 0.15, "o", color=BLUE, ms=7, label="трудный протокол" if name.startswith("ua") else None)
    ax.axvline(0, color="#777", lw=1)
    ax.set_yticks(range(n)); ax.set_yticklabels([e[0] for e in easy][::-1])
    ax.set_xlabel("изменение AUC при удалении группы из пула (95% ДИ), 1 seed")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Связи пользователя с артистом и треком труднее всего заменить", loc="left", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_logo.png", dpi=160); plt.close(fig)


def fig_main():
    rows = [("AUC", "лёгкий", .0055, .0038, .0072), ("AUC", "трудный", .0060, .0037, .0085),
            ("NDCG@10", "лёгкий", .0068, .0026, .0108), ("NDCG@10", "трудный", .0144, .0074, .0209)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, m in zip(axes, ["AUC", "NDCG@10"]):
        for k, (mm, p, d, lo, hi) in enumerate([r for r in rows if r[0] == m]):
            c = GREY if p == "лёгкий" else BLUE
            ax.errorbar(d, k, xerr=[[d - lo], [hi - d]], fmt="o", color=c, capsize=4, lw=2, ms=7)
            ax.text(hi + 0.0004, k, f"+{d:.4f}", va="center", fontsize=10)
        ax.axvline(0, color="#777", lw=1)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["лёгкий", "трудный"]); ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(-0.001, 0.027 if m == "NDCG@10" else 0.011)
        ax.set_title(f"прирост {m}", loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("Пул из 487 признаков против ручного набора из 134 (разность и 95% ДИ)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_main.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    fig_split(); fig_windows(); fig_logo(); fig_main()
    print("ok ->", OUT)

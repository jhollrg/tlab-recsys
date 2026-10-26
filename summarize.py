"""Сводка всех результатов в Markdown-таблицы (для отчёта). Таблицы строятся из JSON, а не переписываются руками.
    python summarize.py --work-dirs work work_hard          # -> печать + <work_dir>/summary.md
"""
import argparse
import json
import os


def ms(v):
    return f"{v[0]:.4f} ± {v[1]:.4f}" if isinstance(v, (list, tuple)) else f"{v:.4f}"


def ci(v):
    sig = v["lo"] > 0 or v["hi"] < 0
    s = f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
    return f"**{s}**" if sig else s


METRICS = ("auc", "logloss", "ndcg@10", "recall@10")
HDR = "| AUC | LogLoss | NDCG@10 | Recall@10 |"


def methods_table(methods, extra_cols=("train_seconds",)):
    out = ["| Набор | #призн. " + HDR + " обучение, с |", "|---|---|---|---|---|---|---|"]
    for n, r in methods.items():
        t = r["test"]
        tr = r.get("train_seconds")
        out.append(f"| {n} | {r['n_features']} | " + " | ".join(ms(t[m]) for m in METRICS)
                   + f" | {'' if tr is None else f'{tr:.0f}'} |")
    return out


def diff_table(d, title):
    out = [f"\n{title}\n", "| Набор " + HDR, "|---|---|---|---|---|"]
    for n, dd in d.items():
        out.append(f"| {n} | " + " | ".join(ci(dd[m]) for m in METRICS) + " |")
    return out


def load(wd, name):
    for n in (name, name.replace(".json", "_syn.json")):      # синтетические прогоны пишут *_syn.json
        p = os.path.join(wd, n)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def summarize(wd):
    L = [f"# Сводка: `{wd}`\n"]
    r1, r2, rw, r3 = (load(wd, f) for f in ("results.json", "results_stage2.json", "results_windows.json", "results_stage3.json"))
    cfg = (r1 or r2 or r3 or {}).get("cfg", {})
    if cfg:
        L.append(f"Пользователей: {cfg.get('n_users')}; негативы pop/hist/art: {cfg.get('n_neg_pop')}/{cfg.get('n_neg_hist')}/"
                 f"{cfg.get('n_neg_art', 0)}; горизонт {cfg.get('horizon_days')} д.\n")
    if r1:
        b = r1["baseline_hand"]["test"]
        L.append(f"Test: {b['n_groups']} групп, {b['n_rows']} строк, {b['group_size']:.0f} кандидатов на группу, "
                 f"доля положительных {b['pos_rate']:.3f}. Расчёт признаков пула ({r1['n_pool']}): {r1['feature_seconds_total']:.0f} с.")
        sel = []
        for k in ("auto_importance", "auto_greedy"):
            if k in r1 and "select_seconds" in r1[k]:
                sel.append(f"{k}: {r1[k].get('n_features', len(r1[k].get('features', [])))} признаков, отбор {r1[k]['select_seconds']:.0f} с")
        if sel:
            L.append("Отбор: " + "; ".join(sel) + ".\n")
        L.append("Эвристики (один признак, test AUC): " + ", ".join(
            f"`{k}` {v['test']['auc']:.3f}" for k, v in r1["heuristics"].items()) + "\n")
    if r2:
        L.append("## Этап 2: seeds + подбор гиперпараметров\n")
        L += methods_table(r2["methods"])
        L += diff_table(r2["bootstrap_test_vs_hand"], "Разность с hand, 95% ДИ (жирным - ДИ не содержит 0):")
        if r2.get("ablation"):
            L += ["\nАбляции hand + группа (1 seed, без ДИ):\n", "| Группа | +призн. | Δval LL | Δtest LL | Δtest AUC | Δtest NDCG@10 |",
                  "|---|---|---|---|---|---|"]
            for a in r2["ablation"]:
                L.append(f"| {a['group']} | {a['n_added']} | {a['d_val_logloss']:+.4f} | {a['d_test_logloss']:+.4f} | "
                         f"{a['d_test_auc']:+.4f} | {a['d_test_ndcg10']:+.4f} |")
    if rw:
        L.append("\n## Окна\n")
        L += methods_table(rw["methods"])
        if "bootstrap_vs_A" in rw:
            L += diff_table(rw["bootstrap_vs_A"], "Разность с A (ручные окна), 95% ДИ:")
        L.append("\nДоля gain счётчиков по окнам: " + ", ".join(f"{k} {v:.3f}" for k, v in rw["gain_by_window"].items()))
    if r3:
        L.append("\n## Этап 3: операторы, сжатие, LOGO\n")
        L += methods_table(r3["methods"])
        L += diff_table(r3["bootstrap"]["hand"], "Разность с hand, 95% ДИ:")
        L += diff_table(r3["bootstrap"]["pool"], "Разность с pool, 95% ДИ:")
        if "logo" in r3:
            d = r3["logo"]["bootstrap_vs_pool"]
            d = dict(sorted(d.items(), key=lambda x: x[1]["auc"]["diff"]))
            L += diff_table(d, "Leave-one-group-out: pool минус группа, разность с pool (отрицательный ΔAUC - группа незаменима):")
        if "timing" in r3:
            t = r3["timing"]
            L += [f"\nВремя расчёта признаков на одну отсечку ({t['n_candidates']} кандидатов, {t['n_events']} событий):\n",
                  "| Набор | спецификаций | секунд |", "|---|---|---|"]
            for n, v in t["sets"].items():
                L.append(f"| {n} | {v['n_specs']} | {v['seconds']:.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dirs", nargs="+", default=["work"])
    a = ap.parse_args()
    for wd in a.work_dirs:
        if not os.path.isdir(wd):
            print(f"нет папки {wd}, пропускаю")
            continue
        md = summarize(wd)
        open(os.path.join(wd, "summary.md"), "w").write(md)
        print(md)

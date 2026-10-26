"""Этап 3: расширение пространства признаков новыми операторами и проверка, откуда берётся прирост.
  1) новые операторы DSL: затухающие счётчики dcnt<H> (вместо окон) и тренды cnt(короткое)/cnt(длинное);
  2) «окна vs затухание» при равном числе признаков: hand vs hand_decay;
  3) сжатие: top-k признаков по важности (отбор на train/val);
  4) leave-one-group-out от пула: какие группы незаменимы (с 95% ДИ, в отличие от «hand + группа» этапа 2);
  5) стоимость: время расчёта каждого набора признаков на одну отсечку.
Все наборы - на одних и тех же кандидатах (кэш __ext; кандидаты детерминированы и совпадают с этапом 1).
    python run_stage3.py --work-dir work --seeds 2
"""
import argparse
import json
import os
import time

import numpy as np

from yfe.config import Cfg
from yfe.features import (pool_specs, hand_specs, decay_specs, trend_specs, hand_decay_specs,
                          compute_features, Spec, Ratio)
from yfe.pipeline import build_all
from yfe.models import train_lgbm
from yfe.exp import fit_eval, fmt
from yfe.stats import bootstrap_diff, print_diffs

ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--topk", type=int, default=100)
ap.add_argument("--skip-logo", action="store_true")
ap.add_argument("--skip-timing", action="store_true")
ap.add_argument("--quick", action="store_true", help="только ключевые наборы и группы LOGO (для трудного протокола)")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

res0 = json.load(open(os.path.join(a.work_dir, "results_syn.json" if a.synthetic else "results.json")))
cfg = Cfg(**res0["cfg"])
uniq = lambda L: list(dict.fromkeys(L))
N = lambda L: [s.name for s in L]
hand, hd = hand_specs(), hand_decay_specs()
pool = uniq(pool_specs() + hand)
dec, trd = decay_specs(), trend_specs()
ext = uniq(pool + dec + trd)
data, info = build_all(cfg, uniq(ext + hd), synthetic=a.synthetic, name="ext")
tr, va, te = data["train"], data["val"], data["test"]
n0 = res0["baseline_hand"]["test"]["n_rows"]
print(f"кандидаты test: {te.height} строк (этап 1: {n0}) -> {'совпадают' if te.height == n0 else 'ОТЛИЧАЮТСЯ, сравнивать с этапом 1 нельзя'}")
seeds = list(range(a.seeds))

sets = {"hand": N(hand), "hand_decay": N(hd), "pool": N(pool), "pool+decay": N(uniq(pool + dec)),
        "pool+trend": N(uniq(pool + trd)), "pool+decay+trend": N(ext)}
# сжатие: важность считается на модели, обученной на train с early stopping по val (test не используется)
if a.quick:
    sets = {k: sets[k] for k in ("hand", "pool", "pool+decay+trend")}
for src in (("pool",) if a.quick else ("pool", "pool+decay+trend")):
    f = sets[src]
    m = train_lgbm(tr, va, f, seed=0)
    g = dict(zip(f, m.feature_importance("gain")))
    sets[f"top{a.topk} из {src}"] = sorted(f, key=lambda n: -g[n])[:a.topk]

out = {"cfg": res0["cfg"], "seeds": seeds, "methods": {}, "feature_seconds_build": info["feature_seconds"]}
preds = {}
for n, f in sets.items():
    r, p = fit_eval(tr, va, te, f, None, seeds)
    out["methods"][n] = r
    preds[n] = p
    print(f"[{n}] {len(f)} признаков, test AUC {fmt(r['test']['auc'])}, LL {fmt(r['test']['logloss'])}", flush=True)
out["bootstrap"] = bootstrap_diff(te, preds, ["hand", "pool"], K=10, n_boot=a.boot)

print("\n=== test, mean±std по seed (параметры по умолчанию) ===")
print(f"{'набор':<28}{'#feat':>6}  {'AUC':<14}{'LogLoss':<14}{'NDCG@10':<14}{'Recall@10':<14}{'train,s':>8}")
for n, r in out["methods"].items():
    t = r["test"]
    print(f"{n:<28}{r['n_features']:>6}  {fmt(t['auc']):<14}{fmt(t['logloss']):<14}{fmt(t['ndcg@10']):<14}"
          f"{fmt(t['recall@10']):<14}{r['train_seconds']:>8.0f}")
print_diffs(out["bootstrap"]["hand"], "\n=== разность с hand, 95% ДИ (bootstrap по пользователям; * = ДИ не содержит 0) ===")
print_diffs(out["bootstrap"]["pool"], "\n=== разность с pool ===")

# ---------------------------------------------------------------- leave-one-group-out
if not a.skip_logo:
    def _a(s):
        return s.num if isinstance(s, Ratio) else s
    G = {
        "сущность ui (прямая связь)": lambda s: _a(s).entity == "ui",
        "сущность ua (user-artist)": lambda s: _a(s).entity == "ua",
        "сущность ual (user-album)": lambda s: _a(s).entity == "ual",
        "сущность user": lambda s: _a(s).entity == "user",
        "сущность item": lambda s: _a(s).entity == "item",
        "сущности artist+album": lambda s: _a(s).entity in ("artist", "album"),
        "окно 90 дней": lambda s: _a(s).window == 90,
        "события skip": lambda s: _a(s).kind == "skip",
        "агрегат frac_org": lambda s: getattr(s, "agg", None) == "frac_org",
        # проверка гипотезы об артефакте сэмплирования: органичность трека/артиста (может выдавать «популярный негатив»)
        # против органичности пользователя и пар (вкус: пользователь сам ищет этого артиста/трек)
        "frac_org трека/артиста/альбома": lambda s: getattr(s, "agg", None) == "frac_org" and s.entity in ("item", "artist", "album"),
        "frac_org пользователя и пар": lambda s: getattr(s, "agg", None) == "frac_org" and s.entity in ("user", "ui", "ua", "ual"),
        "агрегат recency+age": lambda s: getattr(s, "agg", None) in ("recency", "age"),
        "агрегат nuniq_item": lambda s: getattr(s, "agg", None) == "nuniq_item",
        "отношения (rate)": lambda s: isinstance(s, Ratio),
    }
    if a.quick:
        keep = ("сущность ui (прямая связь)", "сущность ua (user-artist)", "сущность item", "сущность user",
                "агрегат frac_org", "frac_org трека/артиста/альбома", "frac_org пользователя и пар", "агрегат recency+age")
        G = {k: G[k] for k in keep}
    base_r, base_p = fit_eval(tr, va, te, N(pool), None, [0])
    lp = {"pool": base_p}
    rows = {}
    for g, fn in G.items():
        f = [s.name for s in pool if not fn(s)]
        r, p = fit_eval(tr, va, te, f, None, [0])
        lp[f"pool − {g}"] = p
        rows[g] = {"n_removed": len(pool) - len(f), "test": r["test"], "val": r["val"]}
        print(f"[LOGO] − {g}: убрано {len(pool) - len(f)}, test AUC {r['test']['auc'][0]:.4f}", flush=True)
    out["logo"] = {"rows": rows, "bootstrap_vs_pool": bootstrap_diff(te, lp, "pool", K=10, n_boot=a.boot)}
    print_diffs(out["logo"]["bootstrap_vs_pool"], "\n=== pool минус группа (seed 0), разность с pool: отрицательный ΔAUC = группа незаменима ===")

# ---------------------------------------------------------------- стоимость признаков
if not a.skip_timing:
    from yfe.data import load_yambda, make_synthetic, prepare_events
    from yfe.split import make_cutoffs, build_candidates
    ev, meta = make_synthetic(seed=cfg.seed) if a.synthetic else load_yambda(cfg)
    ev = prepare_events(ev, meta, cfg)
    T = make_cutoffs(int(ev["ts"].max()), cfg)["test"][0]
    cands = build_candidates(ev, meta.unique(subset="item_id"), T, cfg, np.random.default_rng(0))
    tm = {}
    for n, L in (("hand", hand), ("hand_decay", hd), ("pool", pool), ("decay", dec), ("trend", trd),
                 ("pool+decay+trend", ext)):
        t0 = time.time()
        compute_features(ev, cands, T, L)
        tm[n] = {"n_specs": len(L), "seconds": time.time() - t0}
    out["timing"] = {"cutoff": T, "n_candidates": cands.height, "n_events": ev.height, "sets": tm}
    print(f"\n=== время расчёта признаков на одну отсечку ({cands.height:,} кандидатов, {ev.height:,} событий) ===")
    for n, v in tm.items():
        print(f"  {n:<20}{v['n_specs']:>5} спецификаций  {v['seconds']:6.1f} с")

path = os.path.join(a.work_dir, "results_stage3_syn.json" if a.synthetic else "results_stage3.json")
json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
print("results ->", path)

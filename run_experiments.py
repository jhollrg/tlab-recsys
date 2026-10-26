"""Запуск: baseline (ручные признаки) vs авто-отбор из пула. Пример:
    python run_experiments.py --n-users 2000 --work-dir work
    python run_experiments.py --synthetic --work-dir work_syn          # проверка пайплайна без данных
"""
import argparse
import json
import os
import time

from yfe.config import Cfg
from yfe.features import pool_specs, hand_specs
from yfe.pipeline import build_all
from yfe.models import train_lgbm, predict, evaluate, heuristic_eval
from yfe.search import importance_selection, greedy_family_selection

ap = argparse.ArgumentParser()
ap.add_argument("--n-users", type=int, default=2000)
ap.add_argument("--size", default="50m")
ap.add_argument("--data-dir", default=None)
ap.add_argument("--work-dir", default="work")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--skip-greedy", action="store_true")
ap.add_argument("--neg-pop", type=int, default=20)
ap.add_argument("--neg-hist", type=int, default=20)
ap.add_argument("--neg-art", type=int, default=0, help="трудные негативы из топ-артистов пользователя (0 = выкл.)")
a = ap.parse_args()

cfg = Cfg(n_users=a.n_users, size=a.size, data_dir=a.data_dir, work_dir=a.work_dir, seed=a.seed,
          n_neg_pop=a.neg_pop, n_neg_hist=a.neg_hist, n_neg_art=a.neg_art)
if a.synthetic:
    cfg.n_train_cutoffs, cfg.train_stride_days, cfg.min_hist_events = 4, 2, 20
pool = list(dict.fromkeys(pool_specs() + hand_specs()))
hand = hand_specs()
data, info = build_all(cfg, pool, synthetic=a.synthetic, rebuild=a.rebuild)
tr, va, te = data["train"], data["val"], data["test"]
res = {"cfg": cfg.dict(), "n_pool": len(pool), "n_hand": len(hand), "feature_seconds_total": info["feature_seconds"]}

# 0. эвристики без обучения
res["heuristics"] = {c: {"val": heuristic_eval(va, c), "test": heuristic_eval(te, c)}
                     for c in ("item.listen.cnt.7d", "ui.listen.cnt.all", "ua.listen.cnt.30d")}

# 1. baseline: ручные признаки
hf = [s.name for s in hand]
m = train_lgbm(tr, va, hf, seed=a.seed)
res["baseline_hand"] = {"n_features": len(hf), "val": evaluate(va, predict(m, va, hf)),
                        "test": evaluate(te, predict(m, te, hf)), "train_seconds": m.train_seconds,
                        "best_iter": m.best_iteration}
# 2. весь пул без отбора
pf = [s.name for s in pool]
mp = train_lgbm(tr, va, pf, seed=a.seed)
res["full_pool"] = {"n_features": len(pf), "val": evaluate(va, predict(mp, va, pf)),
                    "test": evaluate(te, predict(mp, te, pf)), "train_seconds": mp.train_seconds}
# 3. отбор по важности
t0 = time.time()
imp = importance_selection(tr, va, te, pf, seed=a.seed)
imp["select_seconds"] = time.time() - t0
res["auto_importance"] = {k: v for k, v in imp.items() if k != "importance"}
# 4. жадный отбор семейств
if not a.skip_greedy:
    t0 = time.time()
    g = greedy_family_selection(tr, va, pool, imp["importance"], seed=a.seed)
    if g["features"]:
        mg = train_lgbm(tr, va, g["features"], seed=a.seed)
        g.update(val=evaluate(va, predict(mg, va, g["features"])), test=evaluate(te, predict(mg, te, g["features"])),
                 n_features=len(g["features"]), select_seconds=time.time() - t0)
    res["auto_greedy"] = g

os.makedirs(a.work_dir, exist_ok=True)
out = os.path.join(a.work_dir, "results_syn.json" if a.synthetic else "results.json")
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)

print("\n=== summary (test) ===")
print(f"{'method':<20}{'#feat':>7}{'AUC':>8}{'LogLoss':>9}{'NDCG@10':>9}{'Recall@10':>10}")
for k in ("baseline_hand", "full_pool", "auto_importance", "auto_greedy"):
    r = res.get(k)
    if r and "test" in r:
        t = r["test"]
        n = r.get("n_features", len(r.get("features", [])))
        print(f"{k:<20}{n:>7}{t['auc']:>8.4f}{t['logloss']:>9.4f}{t['ndcg@10']:>9.4f}{t['recall@10']:>10.4f}")
print("results ->", out)

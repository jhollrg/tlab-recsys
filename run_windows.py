"""Центральный вопрос задания: «почему именно 7 дней?». Сравниваем окна, выбранные человеком (1/7/30/all),
с мелкой сеткой окон, где окна отбираются данными.
  A: hand                       - ручные признаки, счётчики в окнах 1/7/30/all
  B: сетка                      - те же признаки, но счётчики по сетке из 13 окон (1,2,3,5,7,10,14,21,30,60,90,180,all)
  C: top-k окон из сетки        - тот же бюджет признаков, что у A, но окна выбраны по важности (на train/val)
  R: случайные окна из сетки    - контроль: тот же бюджет, окна случайные
    python run_windows.py --work-dir work --seeds 3
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from yfe.config import Cfg
from yfe.features import hand_specs, Spec
from yfe.pipeline import build_all
from yfe.models import train_lgbm
from yfe.exp import fit_eval, fmt
from yfe.stats import bootstrap_diff, print_diffs

ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--rand", type=int, default=3, help="число случайных контролей")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

res0 = json.load(open(os.path.join(a.work_dir, "results_syn.json" if a.synthetic else "results.json")))
cfg = Cfg(**res0["cfg"])
GRID = (1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, None)
gspecs = [Spec(e, k, "cnt", w) for e in ("user", "item", "artist", "ui", "ua")
          for k in ("listen", "lplus", "like", "dislike") for w in GRID]
hand = hand_specs()
specs = list(dict.fromkeys(hand + gspecs))
data, _ = build_all(cfg, specs, synthetic=a.synthetic, name="win")
tr, va, te = data["train"], data["val"], data["test"]

hand_cnt = [s.name for s in hand if isinstance(s, Spec) and s.agg == "cnt"]
other = [s.name for s in hand if not (isinstance(s, Spec) and s.agg == "cnt")]
grid_names = [s.name for s in gspecs]
A = other + hand_cnt
B = other + grid_names
m = train_lgbm(tr, va, B, seed=0)
imp = dict(zip(B, m.feature_importance("gain")))
C = other + sorted(grid_names, key=lambda n: -imp[n])[:len(hand_cnt)]
sets = {"A hand (окна 1/7/30/all)": A, f"B сетка ({len(GRID)} окон)": B, "C top-k окон по важности": C}
seeds = list(range(a.seeds))

out = {"cfg": res0["cfg"], "n_cnt_hand": len(hand_cnt), "methods": {}}
preds = {}
for n, f in sets.items():
    r, p = fit_eval(tr, va, te, f, None, seeds)
    out["methods"][n] = r
    preds[n] = p
rng = np.random.default_rng(0)
rr = []
for i in range(a.rand):
    pick = list(rng.choice(grid_names, len(hand_cnt), replace=False))
    r, p = fit_eval(tr, va, te, other + pick, None, [0])
    rr.append(r["test"])
    if i == 0:
        preds["R случайные окна (1-я выборка)"] = p
out["methods"]["R случайные окна (среднее)"] = {
    "n_features": len(other) + len(hand_cnt),
    "test": {k: (float(np.mean([t[k][0] for t in rr])), float(np.std([t[k][0] for t in rr]))) for k in rr[0]},
    "val": {}}

# важность по окнам и по (сущность.событие) в модели B
by_win, by_fam = defaultdict(float), defaultdict(float)
tot = sum(imp[n] for n in grid_names) or 1.0
for n in grid_names:
    e, k, _, w = n.split(".")
    by_win[w] += imp[n] / tot
    by_fam[f"{e}.{k}"] += imp[n] / tot
out["gain_by_window"] = {w: by_win[w] for w in [("all" if g is None else f"{g}d") for g in GRID]}
out["gain_by_family"] = dict(sorted(by_fam.items(), key=lambda x: -x[1]))
out["chosen_windows_C"] = [n for n in C if n in set(grid_names)]

print("\n=== test, mean±std по seed ===")
print(f"{'набор':<30}{'#feat':>6}  {'AUC':<14}{'LogLoss':<14}{'NDCG@10':<14}{'Recall@10':<14}")
for n, r in out["methods"].items():
    t = r["test"]
    print(f"{n:<30}{r['n_features']:>6}  {fmt(t['auc']):<14}{fmt(t['logloss']):<14}{fmt(t['ndcg@10']):<14}{fmt(t['recall@10']):<14}")
print("\nДоля gain по окнам (модель B, только счётчики):")
print("  " + "  ".join(f"{w}:{v:.3f}" for w, v in out["gain_by_window"].items()))
print("Доля gain по (сущность.событие):")
print("  " + "  ".join(f"{k}:{v:.3f}" for k, v in list(out["gain_by_family"].items())[:10]))
out["bootstrap_vs_A"] = bootstrap_diff(te, preds, "A hand (окна 1/7/30/all)", K=10, n_boot=a.boot)
print_diffs(out["bootstrap_vs_A"], "\nРазность с A (hand), 95% ДИ bootstrap по пользователям:")
path = os.path.join(a.work_dir, "results_windows_syn.json" if a.synthetic else "results_windows.json")
json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
print("results ->", path)

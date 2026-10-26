"""Этап 2: те же данные и признаки (из кэша), но корректная статистика.
  1) одинаковый подбор гиперпараметров на val для каждого набора признаков;
  2) несколько seed обучения -> среднее ± std;
  3) парный bootstrap по пользователям (95% ДИ разности с бейзлайном) на test;
  4) абляции: бейзлайн + одна группа новых признаков из пула.
Наборы признаков (importance/greedy) берутся из results.json этапа 1 - поиск заново не запускается.
    python run_stage2.py --work-dir work --seeds 3 --tune 6
"""
import argparse
import json
import os

from yfe.config import Cfg
from yfe.features import pool_specs, hand_specs, Spec, Ratio
from yfe.pipeline import build_all
from yfe.models import train_lgbm, predict, evaluate
from yfe.exp import tune, fit_eval, fmt
from yfe.stats import bootstrap_diff

ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--tune", type=int, default=6, help="число случайных конфигураций (+ параметры по умолчанию)")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--skip-ablation", action="store_true")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

res0 = json.load(open(os.path.join(a.work_dir, "results_syn.json" if a.synthetic else "results.json")))
cfg = Cfg(**res0["cfg"])
hand = hand_specs()
pool = list(dict.fromkeys(pool_specs() + hand))
data, _ = build_all(cfg, pool, synthetic=a.synthetic)
tr, va, te = data["train"], data["val"], data["test"]

hn = [s.name for s in hand]
sets = {"hand": hn, "pool": [s.name for s in pool], "importance": res0["auto_importance"]["features"]}
if res0.get("auto_greedy", {}).get("features"):
    sets["greedy"] = res0["auto_greedy"]["features"]
seeds = list(range(a.seeds))

out = {"cfg": res0["cfg"], "seeds": seeds, "tune_trials": a.tune, "methods": {}}
preds = {}
for name, feats in sets.items():
    params, trials = tune(tr, va, feats, a.tune) if a.tune else ({}, [])
    r, p = fit_eval(tr, va, te, feats, params, seeds)
    r["tuning"] = trials
    out["methods"][name] = r
    preds[name] = p
    print(f"[{name}] {len(feats)} feats, params={params}  test AUC {fmt(r['test']['auc'])}", flush=True)

out["bootstrap_test_vs_hand"] = bootstrap_diff(te, preds, "hand", K=10, n_boot=a.boot)

print("\n=== test, mean±std по seed (гиперпараметры подобраны на val) ===")
print(f"{'method':<12}{'#feat':>6}  {'AUC':<14}{'LogLoss':<14}{'NDCG@10':<14}{'Recall@10':<14}{'train,s':>8}")
for n, r in out["methods"].items():
    t = r["test"]
    print(f"{n:<12}{r['n_features']:>6}  {fmt(t['auc']):<14}{fmt(t['logloss']):<14}{fmt(t['ndcg@10']):<14}"
          f"{fmt(t['recall@10']):<14}{r['train_seconds']:>8.0f}")
print("\n=== разность с hand (усреднённые по seed прогнозы), 95% ДИ bootstrap по пользователям ===")
for n, d in out["bootstrap_test_vs_hand"].items():
    print(n)
    for m, v in d.items():
        sig = "*" if (v["lo"] > 0 or v["hi"] < 0) else " "
        print(f"   {m:<10} {v['diff']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]  P(лучше)={v['p_better']:.2f} {sig}")

# ---------------------------------------------------------------- абляции
if not a.skip_ablation:
    def _a(s):
        return s.num if isinstance(s, Ratio) else s
    GROUPS = {
        "окно 90 дней": lambda s: _a(s).window == 90,
        "сущности album / user-album": lambda s: _a(s).entity in ("album", "ual"),
        "события skip": lambda s: _a(s).kind == "skip",
        "агрегат frac_org": lambda s: getattr(s, "agg", None) == "frac_org",
        "агрегат age": lambda s: getattr(s, "agg", None) == "age",
        "агрегат nuniq_item": lambda s: getattr(s, "agg", None) == "nuniq_item",
        "агрегат recency": lambda s: getattr(s, "agg", None) == "recency",
        "новые отношения (rate)": lambda s: isinstance(s, Ratio),
        "mean_ratio (доля прослуш.)": lambda s: getattr(s, "agg", None) == "mean_ratio",
    }
    base_m = train_lgbm(tr, va, hn, seed=0)
    bv, bt = evaluate(va, predict(base_m, va, hn)), evaluate(te, predict(base_m, te, hn))
    rows = []
    print("\n=== абляции: hand + группа (seed 0, параметры по умолчанию) ===")
    print(f"{'группа':<30}{'+feat':>6}{'Δval LL':>10}{'Δtest LL':>10}{'Δtest AUC':>11}{'Δtest NDCG@10':>15}")
    for g, pred_fn in GROUPS.items():
        extra = [s.name for s in pool if pred_fn(s) and s.name not in set(hn)]
        if not extra:
            continue
        f = hn + extra
        m = train_lgbm(tr, va, f, seed=0)
        v, t = evaluate(va, predict(m, va, f)), evaluate(te, predict(m, te, f))
        row = {"group": g, "n_added": len(extra), "d_val_logloss": v["logloss"] - bv["logloss"],
               "d_test_logloss": t["logloss"] - bt["logloss"], "d_test_auc": t["auc"] - bt["auc"],
               "d_test_ndcg10": t["ndcg@10"] - bt["ndcg@10"]}
        rows.append(row)
        print(f"{g:<30}{len(extra):>6}{row['d_val_logloss']:>+10.4f}{row['d_test_logloss']:>+10.4f}"
              f"{row['d_test_auc']:>+11.4f}{row['d_test_ndcg10']:>+15.4f}", flush=True)
    out["ablation"] = rows

path = os.path.join(a.work_dir, "results_stage2_syn.json" if a.synthetic else "results_stage2.json")
json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
print("results ->", path)

"""Диагностика постановки: насколько задача «лёгкая» и что в ней доминирует. Работает по кэшу признаков.
    python diagnostics.py --work-dir work
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from yfe.config import Cfg
from yfe.features import pool_specs, hand_specs
from yfe.pipeline import build_all

ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()
res0 = json.load(open(os.path.join(a.work_dir, "results_syn.json" if a.synthetic else "results.json")))
cfg = Cfg(**res0["cfg"])
pool = list(dict.fromkeys(pool_specs() + hand_specs()))
data, _ = build_all(cfg, pool, synthetic=a.synthetic)
va, te = data["val"], data["test"]

for nm, d in (("val", va), ("test", te)):
    y = d["label"].to_numpy()
    seen = (d["ui.listen.cnt.all"].to_numpy() > 0)
    seen_art = (d["ua.listen.cnt.all"].to_numpy() > 0)
    gs = d.group_by(["cutoff", "uid"]).len()["len"]
    print(f"[{nm}] строк {d.height:,}; групп {gs.len():,}; кандидатов на группу: медиана {gs.median():.0f}; доля положительных {y.mean():.3f}")
    print(f"   повторное потребление: трек уже слушали раньше  среди положительных {seen[y == 1].mean():.3f}, среди отрицательных {seen[y == 0].mean():.3f}")
    print(f"   косвенная связь: слушали этого артиста раньше   среди положительных {seen_art[y == 1].mean():.3f}, среди отрицательных {seen_art[y == 0].mean():.3f}")
    # проверка артефакта сэмплирования: отрицательные берутся только из треков, встречавшихся до T, поэтому
    # положительный трек, которого до T не слушал никто в выборке, определяется по нулевому счётчику трека
    cold = (d["item.listen.cnt.all"].to_numpy() == 0)
    auc_cold = roc_auc_score(y, cold.astype(float)) if 0 < cold.sum() < len(y) else float("nan")
    print(f"   артефакт кандидатов: трек не слушал никто до T  среди положительных {cold[y == 1].mean():.3f}, среди отрицательных {cold[y == 0].mean():.3f}; "
          f"AUC индикатора «трек новый» {auc_cold:.3f}")

y = va["label"].to_numpy()
rows = []
for s in pool:
    x = va[s.name].fill_null(-1.0).to_numpy()
    if np.nanstd(x) > 0:
        rows.append((roc_auc_score(y, x), s.name))
rows.sort(key=lambda r: -abs(r[0] - 0.5))
print("\nTop-15 одиночных признаков по |AUC-0.5| на val (AUC<0.5 = признак работает «наоборот»):")
for auc, n in rows[:15]:
    print(f"   {auc:.3f}  {n}")

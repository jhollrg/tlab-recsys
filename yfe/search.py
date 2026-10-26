"""Автоматический отбор признаков из пула. Все решения принимаются ТОЛЬКО по train/val; test - для итоговой оценки.

Методы:
  importance_selection  - обучить на всём пуле, взять top-k по gain, k выбрать по валидации.
  greedy_family_selection - жадное добавление «семейств» (entity.kind.agg со всеми окнами) по приросту val AUC
                            на быстрой прокси-модели; окна внутри семейства отбираются затем по важности.
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from .models import train_lgbm, predict, evaluate


def _gain(m, feats):
    g = m.feature_importance("gain")
    return dict(zip(feats, g / max(g.sum(), 1e-12)))


def importance_selection(tr, va, te, feats, ks=(25, 50, 100, 200), seed=0, params=None):
    m0 = train_lgbm(tr, va, feats, params, seed)
    imp = _gain(m0, feats)
    order = sorted(feats, key=lambda f: -imp[f])
    best = None
    trials = []
    for k in [k for k in ks if k < len(feats)] + [len(feats)]:
        # для k == всем признакам берём исходный порядок колонок: m0 обучена именно на нём
        sel = list(feats) if k == len(feats) else order[:k]
        m = m0 if k == len(feats) else train_lgbm(tr, va, sel, params, seed)
        r = evaluate(va, predict(m, va, sel))
        trials.append({"k": k, **{x: r[x] for x in ("logloss", "auc", "ndcg@10")}})
        if best is None or r["logloss"] < best[0]:
            best = (r["logloss"], k, sel, m)
    _, k, sel, m = best
    return {"features": sel, "k": k, "trials": trials, "val": evaluate(va, predict(m, va, sel)),
            "test": evaluate(te, predict(m, te, sel)), "train_seconds": m.train_seconds, "importance": imp}


def _proxy_auc(tr_X, tr_y, va_X, va_y, seed):
    p = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed, num_threads=0)
    m = lgb.train(p, lgb.Dataset(tr_X, tr_y), 120)
    return roc_auc_score(va_y, m.predict(va_X))


def greedy_family_selection(tr, va, specs, importance, max_steps=15, shortlist=40, min_gain=5e-4,
                            proxy_rows=300_000, seed=0, log=print):
    """Жадный прямой отбор семейств. Кандидаты - shortlist семейств с наибольшей суммарной важностью."""
    fam_feats = {}
    for s in specs:
        fam_feats.setdefault(s.family, []).append(s.name)
    fam_imp = {f: sum(importance.get(n, 0.0) for n in ns) for f, ns in fam_feats.items()}
    cand = sorted(fam_imp, key=lambda f: -fam_imp[f])[:shortlist]
    rng = np.random.default_rng(seed)
    if tr.height > proxy_rows:
        tr = tr[rng.choice(tr.height, proxy_rows, replace=False)]
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
    chosen, cur, history = [], 0.5, []
    for step in range(max_steps):
        best = None
        for f in cand:
            if f in chosen:
                continue
            cols = [n for c in chosen + [f] for n in fam_feats[c]]
            auc = _proxy_auc(tr.select(cols).to_numpy(), ytr, va.select(cols).to_numpy(), yva, seed)
            if best is None or auc > best[0]:
                best = (auc, f)
        if best is None or best[0] - cur < min_gain:
            break
        cur = best[0]
        chosen.append(best[1])
        history.append({"step": step + 1, "added": best[1], "val_auc_proxy": cur})
        log(f"greedy step {step+1}: +{best[1]}  val AUC(proxy)={cur:.4f}")
    feats = [n for c in chosen for n in fam_feats[c]]
    return {"families": chosen, "features": feats, "history": history}

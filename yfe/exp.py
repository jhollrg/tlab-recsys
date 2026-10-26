"""Общие функции экспериментов: подбор гиперпараметров на val, прогон по нескольким seed."""
import numpy as np

from .models import train_lgbm, predict, evaluate

KEYS = ("auc", "logloss", "ndcg@10", "recall@10", "ndcg@50", "recall@50")


def sample_params(n, seed=0):
    """Одинаковые n конфигураций для всех наборов признаков (равный бюджет подбора)."""
    rng = np.random.default_rng(seed)
    grid = dict(num_leaves=[15, 31, 63, 127], min_data_in_leaf=[20, 50, 100, 200],
                feature_fraction=[0.3, 0.5, 0.7, 0.9], lambda_l2=[0.0, 1.0, 10.0])
    return [{k: rng.choice(v).item() for k, v in grid.items()} for _ in range(n)]


def tune(tr, va, feats, n_trials, seed=0):
    """Случайный поиск по val logloss; в кандидаты всегда входят параметры по умолчанию."""
    best, trials = None, []
    for p in [{}] + sample_params(n_trials, seed):
        m = train_lgbm(tr, va, feats, p, seed)
        ll = evaluate(va, predict(m, va, feats))["logloss"]
        trials.append({"params": p, "val_logloss": ll, "best_iter": m.best_iteration, "sec": round(m.train_seconds, 1)})
        if best is None or ll < best[0]:
            best = (ll, p)
    return best[1], trials


def fit_eval(tr, va, te, feats, params=None, seeds=(0,)):
    """Возвращает (summary, усреднённый по seed прогноз на test)."""
    runs, pt = [], []
    for s in seeds:
        m = train_lgbm(tr, va, feats, params, s)
        a, b = predict(m, va, feats), predict(m, te, feats)
        runs.append({"seed": s, "val": evaluate(va, a), "test": evaluate(te, b),
                     "train_seconds": m.train_seconds, "best_iter": m.best_iteration})
        pt.append(b)

    def agg(split):
        return {k: (float(np.mean([r[split][k] for r in runs])), float(np.std([r[split][k] for r in runs])))
                for k in KEYS}
    return ({"n_features": len(feats), "params": params or {}, "runs": runs, "val": agg("val"), "test": agg("test"),
             "train_seconds": float(np.mean([r["train_seconds"] for r in runs]))}, np.mean(pt, axis=0))


def fmt(ms):
    return f"{ms[0]:.4f}±{ms[1]:.4f}"

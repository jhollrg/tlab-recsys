"""Парный bootstrap по пользователям: насколько различие двух моделей превышает шум выборки.

Пользователь = единица ресемплирования (его группы из разных отсечек коррелированы). Для всех метрик используются
веса пользователей, так что модели сравниваются на ОДНИХ И ТЕХ ЖЕ ресемплах."""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from .models import group_table


def bootstrap_diff(df, preds: dict, base: str, K=10, n_boot=300, seed=0) -> dict:
    y = df["label"].to_numpy()
    uu, inv = np.unique(df["uid"].to_numpy(), return_inverse=True)
    nu = len(uu)
    per = {}
    for n, p in preds.items():
        g = group_table(df, p, K)
        gu = np.searchsorted(uu, g["uid"].to_numpy())
        per[n] = (np.bincount(gu, weights=g["ndcg"].to_numpy(), minlength=nu),
                  np.bincount(gu, weights=g["recall"].to_numpy(), minlength=nu),
                  np.bincount(gu, minlength=nu).astype(float))
    rng = np.random.default_rng(seed)
    names = list(preds)
    mk = ["auc", "logloss", f"ndcg@{K}", f"recall@{K}"]
    draws = {n: {m: [] for m in mk} for n in names}
    for _ in range(n_boot):
        w = rng.multinomial(nu, np.full(nu, 1.0 / nu)).astype(float)
        wr = w[inv]
        for n in names:
            p = np.asarray(preds[n], dtype=np.float64)
            nd, rc, ng = per[n]
            d = draws[n]
            d["auc"].append(roc_auc_score(y, p, sample_weight=wr))
            d["logloss"].append(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=wr, labels=[0, 1]))
            d[mk[2]].append((w * nd).sum() / (w * ng).sum())
            d[mk[3]].append((w * rc).sum() / (w * ng).sum())
    def vs(b0):
        out = {}
        for n in names:
            if n == b0:
                continue
            o = {}
            for m in mk:
                diff = np.array(draws[n][m]) - np.array(draws[b0][m])
                better = (diff < 0) if m == "logloss" else (diff > 0)   # у logloss меньше = лучше
                o[m] = {"diff": float(diff.mean()), "lo": float(np.percentile(diff, 2.5)),
                        "hi": float(np.percentile(diff, 97.5)), "p_better": float(better.mean())}
            out[n] = o
        return out
    # base - имя модели или список имён (тогда {база: {модель: ...}}); ресемплы общие для всех сравнений
    return vs(base) if isinstance(base, str) else {b0: vs(b0) for b0 in base}


def print_diffs(d, title):
    print(title)
    for n, dd in d.items():
        cells = []
        for m, v in dd.items():
            sig = "*" if (v["lo"] > 0 or v["hi"] < 0) else " "
            cells.append(f"{m} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]{sig}")
        print(f"  {n:<28} " + "  ".join(cells))

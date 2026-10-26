import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, deterministic=True, force_row_wise=True)


def _xy(df, feats):
    return df.select(feats).to_numpy(), df["label"].to_numpy()


def train_lgbm(tr, va, feats, params=None, seed=0, rounds=2000, early=50):
    """Early stopping идёт по валидации; тест здесь не используется."""
    p = {**LGB_PARAMS, **(params or {}), "seed": seed}
    Xt, yt = _xy(tr, feats)
    Xv, yv = _xy(va, feats)
    t0 = time.time()
    m = lgb.train(p, lgb.Dataset(Xt, yt), rounds,
                  valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(early, verbose=False)])
    m.train_seconds = time.time() - t0
    return m


def predict(m, df, feats):
    return m.predict(df.select(feats).to_numpy(), num_iteration=m.best_iteration or None)


def group_table(df: pl.DataFrame, pred, K=10) -> pl.DataFrame:
    """Метрики ранжирования по каждой группе (cutoff, uid): ndcg, recall @K."""
    d = (df.select("cutoff", "uid", "label").with_columns(pl.Series("p", np.asarray(pred, dtype=np.float64)))
           .with_columns(rank=pl.col("p").rank("ordinal", descending=True).over(["cutoff", "uid"]).cast(pl.Float64)))
    g = (d.group_by(["cutoff", "uid"])
          .agg(npos=pl.col("label").sum(),
               hits=(pl.col("label") * (pl.col("rank") <= K)).sum(),
               dcg=(pl.col("label") * (pl.col("rank") <= K) / (pl.col("rank") + 1).log(2)).sum()))
    idcg_tab = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, K + 2)))])
    npos = g["npos"].to_numpy()
    return g.with_columns(pl.Series("ndcg", g["dcg"].to_numpy() / idcg_tab[np.minimum(npos, K)]),
                          pl.Series("recall", g["hits"].to_numpy() / npos))


def evaluate(df: pl.DataFrame, pred, ks=(10, 50)) -> dict:
    """LogLoss/AUC по всем строкам + ранжирование внутри каждой группы (cutoff, uid)."""
    y = df["label"].to_numpy()
    pred = np.asarray(pred, dtype=np.float64)
    res = {"auc": float(roc_auc_score(y, pred))}
    res["logloss"] = float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))) if (pred.min() >= 0 and pred.max() <= 1) else float("nan")
    for K in ks:
        g = group_table(df, pred, K)
        res[f"ndcg@{K}"] = float(g["ndcg"].mean())
        res[f"recall@{K}"] = float(g["recall"].mean())
        res[f"hit@{K}"] = float((g["hits"] > 0).mean())
    res["n_groups"] = int(g.height)
    res["n_rows"] = int(df.height)
    res["pos_rate"] = float(y.mean())
    res["group_size"] = float(df.height / g.height)
    return res


def heuristic_eval(df, col):
    """Эвристика без обучения: ранжируем по одному признаку (референс снизу)."""
    return evaluate(df, df[col].fill_null(-1.0).to_numpy())

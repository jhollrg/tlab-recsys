"""Временное разбиение и формирование кандидатов.

Схема для каждой отсечки T:
  * признаки используют только события с ts < T (см. features.py);
  * положительные примеры: треки, прослушанные >= thr% в окне [T, T + horizon);
  * отрицательные: (а) популярные за последние pop_window_days ДО T, (б) треки из прошлой истории пользователя;
    положительные из негативов исключаются. Один и тот же набор кандидатов используется всеми методами.
Сплиты по времени: train-отсечки < val-отсечки < test-отсечки; окна меток не пересекаются между сплитами.
"""
import numpy as np
import polars as pl

from .config import Cfg, DAY


def make_cutoffs(ts_max: int, cfg: Cfg) -> dict:
    H = cfg.horizon_days * DAY
    assert cfg.train_stride_days >= cfg.horizon_days, "окна train-меток не должны пересекаться"
    end = ts_max + 1
    test = [end - (j + 1) * H for j in range(cfg.n_test_cutoffs)][::-1]
    val = [test[0] - (j + 1) * H for j in range(cfg.n_val_cutoffs)][::-1]
    train = [val[0] - H - i * cfg.train_stride_days * DAY for i in range(cfg.n_train_cutoffs)][::-1]
    out = {"train": train, "val": val, "test": test}
    # окно метки каждой отсечки должно заканчиваться до начала следующего сплита
    assert max(train) + H <= min(val) and max(val) + H <= min(test)
    assert min(train) > 0, "слишком мало истории для такого числа train-отсечек"
    return out


def build_candidates(ev: pl.DataFrame, meta: pl.DataFrame, T: int, cfg: Cfg, rng: np.random.Generator) -> pl.DataFrame:
    H = cfg.horizon_days * DAY
    hist = ev.filter(pl.col("ts") < T)
    win = ev.filter((pl.col("ts") >= T) & (pl.col("ts") < T + H))

    pos = win.filter(pl.col("k_lplus")).select("uid", "item_id").unique().sort(["uid", "item_id"])
    enough = hist.group_by("uid").len().filter(pl.col("len") >= cfg.min_hist_events).select("uid")
    users = pos.select("uid").unique().join(enough, on="uid").sort("uid")   # явный порядок: от него зависят случайные выборки
    pos = pos.join(users, on="uid")

    # негативы 1: популярное до T (веса ~ count^0.75)
    pop = (hist.filter(pl.col("ts") >= T - cfg.pop_window_days * DAY).filter(pl.col("k_listen"))
               .group_by("item_id").len().sort("item_id"))
    u = users["uid"].to_numpy()
    parts = []
    if pop.height and cfg.n_neg_pop:
        w = pop["len"].to_numpy().astype(np.float64) ** 0.75
        draw = rng.choice(pop["item_id"].to_numpy(), size=len(u) * cfg.n_neg_pop, p=w / w.sum())
        parts.append(pl.DataFrame({"uid": np.repeat(u, cfg.n_neg_pop).astype(np.uint32),
                                   "item_id": draw.astype(np.uint32)}))
    # негативы 2: случайные треки из истории пользователя
    if cfg.n_neg_hist:
        uh = hist.join(users, on="uid", how="semi").select("uid", "item_id").unique().sort(["uid", "item_id"])
        uh = uh.with_columns(pl.Series("r", rng.random(uh.height)))
        parts.append(uh.sort(["uid", "r"]).group_by("uid", maintain_order=True)
                       .head(cfg.n_neg_hist).select("uid", "item_id"))
    # негативы 3 (опционально): треки из топ-5 артистов пользователя за 30 дней, веса ~ count^0.75.
    # Это "трудные" негативы: тематически близкие к вкусам пользователя, не просто популярные.
    if cfg.n_neg_art:
        h30 = hist.filter((pl.col("ts") >= T - 30 * DAY) & pl.col("k_listen") & pl.col("artist_id").is_not_null())
        top = (h30.join(users, on="uid", how="semi").group_by(["uid", "artist_id"]).len()
                  .sort(["uid", "len", "artist_id"], descending=[False, True, False])
                  .group_by("uid", maintain_order=True).head(5).select("uid", "artist_id"))
        ai = h30.group_by(["artist_id", "item_id"]).len().rename({"len": "w"})
        ca = top.join(ai, on="artist_id").sort(["uid", "artist_id", "item_id"])
        if ca.height:
            w = ca["w"].to_numpy().astype(np.float64) ** 0.75
            key = np.log(rng.random(ca.height) + 1e-12) / w      # ключи Efraimidis-Spirakis: взвешенная выборка без возвращения
            ca = ca.with_columns(pl.Series("key", key))
            parts.append(ca.sort(["uid", "key"], descending=[False, True])
                           .group_by("uid", maintain_order=True).head(cfg.n_neg_art).select("uid", "item_id"))
    neg = (pl.concat(parts).unique(maintain_order=True).join(pos, on=["uid", "item_id"], how="anti")
             .with_columns(pl.lit(0, dtype=pl.Int8).alias("label")))
    cands = pl.concat([pos.with_columns(pl.lit(1, dtype=pl.Int8).alias("label")), neg.select(pos.columns + ["label"])])
    cands = cands.sort(["uid", "item_id"])   # детерминированный вход для shuffle ниже
    # перемешиваем, чтобы при равных предсказаниях порядок не давал преимущества положительным
    cands = cands.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 30)))
    cands = cands.with_row_index("rid").join(meta, on="item_id", how="left").sort("rid")
    return cands.with_columns(pl.lit(T, dtype=pl.Int32).alias("cutoff"))

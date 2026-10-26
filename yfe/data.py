"""Загрузка Yambda, сэмплирование пользователей и подготовка таблицы событий.

Единая схема событий после загрузки:
    uid, item_id, ts (Int32, 5-секундные бины), organic, kind, ratio
затем prepare_events добавляет artist_id, album_id и булевы флаги k_* (тип события).
"""
import os
import numpy as np
import polars as pl

from .config import Cfg, DAY


# ---------------------------------------------------------------- загрузка
def _get_file(cfg: Cfg, rel: str) -> str:
    if cfg.data_dir:
        p = os.path.join(cfg.data_dir, rel)
        if os.path.exists(p):
            return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download("yandex/yambda", rel, repo_type="dataset", local_dir=cfg.data_dir)


def _load_map(cfg: Cfg, name: str, out_col: str) -> pl.DataFrame:
    df = pl.read_parquet(_get_file(cfg, f"{name}.parquet"))
    cols = [c for c in df.columns if c != "item_id"]
    id_col = next((c for c in cols if c.endswith("_id")), cols[0])  # artist_id / album_id
    return (df.select(pl.col("item_id").cast(pl.UInt32), pl.col(id_col).cast(pl.UInt32).alias(out_col))
              .unique(subset="item_id", keep="first"))


def load_yambda(cfg: Cfg):
    """Возвращает (events, meta). История выбранных пользователей берётся целиком."""
    lf = pl.scan_parquet(_get_file(cfg, f"flat/{cfg.size}/multi_event.parquet"))
    uids = lf.select(pl.col("uid").unique()).collect()["uid"].sort()
    if cfg.n_users and cfg.n_users < len(uids):
        rng = np.random.default_rng(cfg.seed)
        uids = pl.Series("uid", np.sort(rng.choice(uids.to_numpy(), cfg.n_users, replace=False)), dtype=pl.UInt32)
    ev = (lf.filter(pl.col("uid").is_in(uids.implode()))
            .select(pl.col("uid").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32),
                    pl.col("timestamp").cast(pl.Int32).alias("ts"),
                    pl.col("is_organic").cast(pl.UInt8).alias("organic"),
                    pl.col("event_type").cast(pl.Utf8).alias("kind"),
                    pl.col("played_ratio_pct").cast(pl.Float32).alias("ratio"))
            .collect())
    meta = (_load_map(cfg, "artist_item_mapping", "artist_id")
            .join(_load_map(cfg, "album_item_mapping", "album_id"), on="item_id", how="full", coalesce=True))
    return ev, meta


# ---------------------------------------------------------------- подготовка
def prepare_events(ev: pl.DataFrame, meta: pl.DataFrame, cfg: Cfg) -> pl.DataFrame:
    is_listen = pl.col("kind") == "listen"
    return (ev.join(meta, on="item_id", how="left")
              .with_columns(
                  k_listen=is_listen,
                  k_lplus=is_listen & (pl.col("ratio") >= cfg.pos_ratio_thr),
                  k_skip=is_listen & (pl.col("ratio") < 25),
                  k_like=pl.col("kind") == "like",
                  k_dislike=pl.col("kind") == "dislike",
                  k_org_listen=is_listen & (pl.col("organic") == 1))
              .drop("kind")
              .sort(["uid", "ts"]))


# ---------------------------------------------------------------- синтетика для тестов
def make_synthetic(n_users=300, n_items=1500, n_artists=150, days=60, seed=0):
    """Маленький датасет со структурой: у пользователей есть любимые артисты, что делает признаки полезными."""
    rng = np.random.default_rng(seed)
    item_artist = rng.integers(0, n_artists, n_items)
    item_album = np.arange(n_items) // 5
    pop = 1.0 / (np.arange(1, n_items + 1) ** 0.9)
    pop /= pop.sum()
    by_artist = [np.where(item_artist == a)[0] for a in range(n_artists)]
    rows = []
    for u in range(n_users):
        fav = rng.choice(n_artists, 5, replace=False)
        fav_items = np.concatenate([by_artist[a] for a in fav if len(by_artist[a])])
        n = int(rng.integers(300, 1200))
        ts = np.sort(rng.integers(0, days * DAY, n))
        mode = rng.random(n)
        items = np.where(mode < 0.45, rng.choice(fav_items, n),
                         np.where(mode < 0.6, rng.choice(fav_items[:10], n), rng.choice(n_items, n, p=pop)))
        is_fav = np.isin(item_artist[items], fav)
        ratio = np.clip(np.where(is_fav, rng.normal(80, 25, n), rng.normal(40, 30, n)), 1, 100).round()
        org = rng.integers(0, 2, n)
        rows.append(np.column_stack([np.full(n, u), items, ts, org, np.zeros(n), ratio]))
        lk = np.where((ratio > 70) & (rng.random(n) < 0.05))[0]
        rows.append(np.column_stack([np.full(len(lk), u), items[lk], ts[lk] + 1, org[lk], np.ones(len(lk)),
                                     np.full(len(lk), np.nan)]))
        dl = np.where((ratio < 15) & (rng.random(n) < 0.05))[0]
        rows.append(np.column_stack([np.full(len(dl), u), items[dl], ts[dl] + 1, org[dl], np.full(len(dl), 2.0),
                                     np.full(len(dl), np.nan)]))
    a = np.concatenate(rows)
    kinds = np.array(["listen", "like", "dislike"])[a[:, 4].astype(int)]
    ev = pl.DataFrame({"uid": a[:, 0].astype(np.uint32), "item_id": a[:, 1].astype(np.uint32),
                       "ts": a[:, 2].astype(np.int32), "organic": a[:, 3].astype(np.uint8),
                       "kind": kinds, "ratio": a[:, 5].astype(np.float32)})
    meta = pl.DataFrame({"item_id": np.arange(n_items, dtype=np.uint32),
                         "artist_id": item_artist.astype(np.uint32), "album_id": item_album.astype(np.uint32)})
    return ev, meta

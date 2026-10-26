"""Первый запуск на реальных данных: проверяем схему и допущения (глобальный timestamp, маппинги артист/альбом).
    python inspect_data.py --n-users 200
"""
import argparse
import polars as pl
from yfe.config import Cfg, DAY
from yfe.data import load_yambda, prepare_events

ap = argparse.ArgumentParser()
ap.add_argument("--n-users", type=int, default=200)
ap.add_argument("--data-dir", default=None)
a = ap.parse_args()
cfg = Cfg(n_users=a.n_users, data_dir=a.data_dir)
ev, meta = load_yambda(cfg)
print("events", ev.shape, "\n", ev.head(3))
print("meta", meta.shape, meta.columns, "| items with artist:", meta["artist_id"].drop_nulls().len(),
      "| with album:", meta["album_id"].drop_nulls().len())
print("event kinds:\n", ev["kind"].value_counts().sort("count", descending=True))
print(f"ts range: {ev['ts'].min()} .. {ev['ts'].max()}  (= {ev['ts'].max()/DAY:.0f} days)")
# допущение: timestamp глобальный. Если у разных пользователей максимум ts сильно различается - допущение неверно.
per_user = ev.group_by("uid").agg(pl.col("ts").min().alias("mn"), pl.col("ts").max().alias("mx"))
print("per-user first ts (days) quantiles:", [round(per_user["mn"].quantile(q) / DAY, 1) for q in (0.1, 0.5, 0.9)])
print("per-user last  ts (days) quantiles:", [round(per_user["mx"].quantile(q) / DAY, 1) for q in (0.1, 0.5, 0.9)])
p = prepare_events(ev, meta, cfg)
print("share of events with artist:", round(p["artist_id"].is_not_null().mean(), 3),
      "| Listen+ share among listens:", round(p["k_lplus"].sum() / p["k_listen"].sum(), 3))
print("events per user: median", int(p.group_by("uid").len()["len"].median()))

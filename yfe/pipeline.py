"""Сборка датасета: события -> отсечки -> кандидаты -> признаки (кэш в parquet)."""
import json
import os
import time
import numpy as np
import polars as pl

from .config import Cfg
from .data import load_yambda, prepare_events, make_synthetic
from .split import make_cutoffs, build_candidates
from .features import compute_features, pool_specs, hand_specs

SPLITS = ("train", "val", "test")


def build_all(cfg: Cfg, specs=None, synthetic=False, rebuild=False, log=print, name=""):
    """Возвращает (dict split -> DataFrame, info). Кэш: work_dir/<tag>[__name]/<split>.parquet.
    name - отдельный кэш для другого набора признаков при тех же кандидатах (при том же seed кандидаты идентичны)."""
    if specs is None:
        specs = list(dict.fromkeys(pool_specs() + hand_specs()))
    names = [s.name for s in specs]
    root = os.path.join(cfg.work_dir, ("syn_" if synthetic else "") + cfg.tag() + (f"__{name}" if name else ""))
    os.makedirs(root, exist_ok=True)
    info_path = os.path.join(root, "info.json")
    paths = {s: os.path.join(root, f"{s}.parquet") for s in SPLITS}
    if not rebuild and os.path.exists(info_path) and all(os.path.exists(p) for p in paths.values()):
        info = json.load(open(info_path))
        if set(names) <= set(info["features"]):
            log(f"[cache] {root}")
            return {s: pl.read_parquet(p) for s, p in paths.items()}, info

    t0 = time.time()
    ev, meta = make_synthetic(seed=cfg.seed) if synthetic else load_yambda(cfg)
    ev = prepare_events(ev, meta, cfg)
    log(f"events: {ev.height:,} rows, users: {ev['uid'].n_unique():,}, load+prep {time.time()-t0:.0f}s")
    cuts = make_cutoffs(int(ev["ts"].max()), cfg)
    rng = np.random.default_rng(cfg.seed)
    meta_u = meta.unique(subset="item_id")
    out, t_feat = {}, 0.0
    for split in SPLITS:
        parts = []
        for T in cuts[split]:
            cands = build_candidates(ev, meta_u, T, cfg, rng)
            t1 = time.time()
            parts.append(compute_features(ev, cands, T, specs))
            t_feat += time.time() - t1
        df = pl.concat(parts).drop("rid")
        out[split] = df
        df.write_parquet(paths[split])
        log(f"{split}: cutoffs={cuts[split]} rows={df.height:,} pos={df['label'].mean():.3f}")
    info = {"features": names, "cutoffs": cuts, "feature_seconds": t_feat, "n_events": ev.height,
            "n_users": int(ev["uid"].n_unique()), "cfg": cfg.dict()}
    json.dump(info, open(info_path, "w"), indent=1)
    log(f"features computed in {t_feat:.0f}s for {len(names)} specs")
    return out, info

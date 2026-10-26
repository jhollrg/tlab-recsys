"""Тесты на отсутствие утечек. Запуск: pytest -q"""
import os
import sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yfe.config import Cfg, DAY
from yfe.data import make_synthetic, prepare_events
from yfe.split import make_cutoffs, build_candidates
from yfe.features import compute_features, pool_specs, hand_specs


def _setup():
    cfg = Cfg(n_train_cutoffs=3, train_stride_days=2, min_hist_events=20)
    ev, meta = make_synthetic(n_users=80, days=40)
    return cfg, prepare_events(ev, meta, cfg), meta


def test_cutoffs_do_not_overlap():
    cfg, ev, _ = _setup()
    c = make_cutoffs(int(ev["ts"].max()), cfg)
    H = cfg.horizon_days * DAY
    assert max(c["train"]) + H <= min(c["val"]) and max(c["val"]) + H <= min(c["test"])
    assert min(c["test"]) + H * cfg.n_test_cutoffs - 1 >= int(ev["ts"].max()) - H  # тест - последние дни


def test_features_ignore_future():
    """Главный тест: признаки, посчитанные на полных данных, идентичны признакам на данных,
    у которых удалено всё с ts >= T (включая окно метки)."""
    cfg, ev, meta = _setup()
    T = make_cutoffs(int(ev["ts"].max()), cfg)["val"][1]
    cands = build_candidates(ev, meta, T, cfg, np.random.default_rng(0))
    specs = list(dict.fromkeys(pool_specs() + hand_specs()))
    full = compute_features(ev, cands, T, specs)
    trunc = compute_features(ev.filter(pl.col("ts") < T), cands, T, specs)
    assert full.shape == trunc.shape and full.height > 0
    for c in [s.name for s in specs]:
        a, b = full[c].to_numpy(), trunc[c].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"утечка из будущего в признаке {c}"


def test_negatives_exclude_positives_and_use_history_only():
    cfg, ev, meta = _setup()
    T = make_cutoffs(int(ev["ts"].max()), cfg)["val"][0]
    c = build_candidates(ev, meta, T, cfg, np.random.default_rng(1))
    assert c.group_by(["uid", "item_id"]).len()["len"].max() == 1       # нет дублей пар
    H = cfg.horizon_days * DAY
    win = ev.filter((pl.col("ts") >= T) & (pl.col("ts") < T + H) & pl.col("k_lplus")).select("uid", "item_id")
    neg = c.filter(pl.col("label") == 0).select("uid", "item_id")
    assert neg.join(win, on=["uid", "item_id"], how="inner").height == 0
    pos = c.filter(pl.col("label") == 1).select("uid", "item_id")
    assert pos.join(win, on=["uid", "item_id"], how="anti").height == 0


def test_candidate_order_is_shuffled():
    cfg, ev, meta = _setup()
    T = make_cutoffs(int(ev["ts"].max()), cfg)["val"][0]
    c = build_candidates(ev, meta, T, cfg, np.random.default_rng(2))
    first = c["label"].to_numpy()[: c.height // 2].mean()
    assert 0 < first < 1 and abs(first - c["label"].mean()) < 0.1


def test_hard_negatives_valid_and_leak_free():
    cfg, ev, meta = _setup()
    cfg.n_neg_art = 20
    T = make_cutoffs(int(ev["ts"].max()), cfg)["val"][1]
    c = build_candidates(ev, meta, T, cfg, np.random.default_rng(3))
    assert c.group_by(["uid", "item_id"]).len()["len"].max() == 1
    H = cfg.horizon_days * DAY
    win = ev.filter((pl.col("ts") >= T) & (pl.col("ts") < T + H) & pl.col("k_lplus")).select("uid", "item_id")
    assert c.filter(pl.col("label") == 0).join(win, on=["uid", "item_id"], how="inner").height == 0
    # все негативы - треки, которые уже встречались в прослушиваниях ДО T (иначе негативы смотрели бы в будущее)
    seen_before = ev.filter((pl.col("ts") < T) & pl.col("k_listen")).select("item_id").unique()
    neg = c.filter(pl.col("label") == 0)
    assert neg.height > 0
    assert neg.join(seen_before, on="item_id", how="anti").height == 0


def test_bootstrap_detects_difference():
    from yfe.stats import bootstrap_diff
    rng = np.random.default_rng(0)
    n = 4000
    df = pl.DataFrame({"cutoff": np.zeros(n, dtype=np.int32), "uid": np.repeat(np.arange(200), 20).astype(np.uint32),
                       "label": np.tile((np.arange(20) < 6).astype(np.int8), 200)})
    y = df["label"].to_numpy()
    good = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.15, n), 0.01, 0.99)
    bad = np.clip(rng.random(n), 0.01, 0.99)
    same = bootstrap_diff(df, {"a": good, "b": good.copy()}, "a", n_boot=60)
    assert abs(same["b"]["auc"]["diff"]) < 1e-12
    d = bootstrap_diff(df, {"a": bad, "b": good}, "a", n_boot=60)
    assert d["b"]["auc"]["lo"] > 0 and d["b"]["logloss"]["hi"] < 0


def test_candidates_are_deterministic():
    """Один seed -> один и тот же набор кандидатов (в том же порядке)."""
    cfg, ev, meta = _setup()
    cfg.n_neg_art = 10
    T = make_cutoffs(int(ev["ts"].max()), cfg)["val"][1]
    hs = [build_candidates(ev, meta, T, cfg, np.random.default_rng(0)).select("uid", "item_id", "label").hash_rows().sum()
          for _ in range(5)]
    assert len(set(hs)) == 1


def test_new_operators_leak_free_and_correct():
    """Затухающие счётчики и тренды: (1) не видят будущего; (2) dcnt совпадает с ручным расчётом по numpy."""
    from yfe.features import ext_specs, hand_decay_specs, Spec
    cfg, ev, meta = _setup()
    T = make_cutoffs(int(ev["ts"].max()), cfg)["val"][1]
    cands = build_candidates(ev, meta, T, cfg, np.random.default_rng(0))
    specs = list(dict.fromkeys(ext_specs() + hand_decay_specs()))
    full = compute_features(ev, cands, T, specs)
    trunc = compute_features(ev.filter(pl.col("ts") < T), cands, T, specs)
    for c in [s.name for s in specs]:
        assert np.allclose(full[c].to_numpy(), trunc[c].to_numpy(), equal_nan=True), f"утечка в {c}"
    s = Spec("ui", "lplus", "dcnt7", None)
    got = compute_features(ev, cands, T, [s])
    r = got.row(0, named=True)
    h = ev.filter((pl.col("uid") == r["uid"]) & (pl.col("item_id") == r["item_id"]) & (pl.col("ts") < T) & pl.col("k_lplus"))
    want = float(np.sum(0.5 ** ((T - h["ts"].to_numpy().astype(np.float64)) / DAY / 7.0)))
    assert abs(r[s.name] - want) < 1e-6 * max(1.0, want)
    assert (got[s.name].to_numpy() >= 0).all() and not got[s.name].is_null().any()

"""Декларативное пространство признаков и движок их расчёта.

Признак = Spec(entity, kind, agg, window):
  entity  - по чему группируем историю: user / item / artist / album / ui (пара user-item) / ua (user-artist) / ual (user-album)
  kind    - какие события учитываем: listen / lplus (прослушано >= 50%) / skip (< 25%) / like / dislike / org_listen
  agg     - cnt / mean_ratio / recency / age / nuniq_item / frac_org /
            dcnt<H> - экспоненциально затухающий счётчик с полупериодом H дней (альтернатива окнам)
  window  - окно в сутках до отсечки T (None = вся история)
Ratio(num, den) = num / (den + 1) - сглаженные доли и "конверсии".

Утечек нет по построению: движок использует только события с ts < T (см. compute_features и tests/test_leakage.py).
"""
from dataclasses import dataclass
import numpy as np
import polars as pl

from .config import DAY

ENT_KEYS = {"user": ["uid"], "item": ["item_id"], "artist": ["artist_id"], "album": ["album_id"],
            "ui": ["uid", "item_id"], "ua": ["uid", "artist_id"], "ual": ["uid", "album_id"]}
POOL_KINDS = ("listen", "lplus", "skip", "like", "dislike")
POOL_WINDOWS = (1, 7, 30, 90, None)


@dataclass(frozen=True)
class Spec:
    entity: str
    kind: str
    agg: str
    window: int | None

    @property
    def name(self) -> str:
        w = "all" if self.window is None else f"{self.window}d"
        return f"{self.entity}.{self.kind}.{self.agg}.{w}"

    @property
    def family(self) -> str:
        return f"{self.entity}.{self.kind}.{self.agg}"


@dataclass(frozen=True)
class Ratio:
    num: Spec
    den: Spec

    @property
    def name(self) -> str:
        return f"rate[{self.num.name}|{self.den.name}]"

    @property
    def family(self) -> str:
        return f"rate[{self.num.family}|{self.den.family}]"


def is_valid(s: Spec) -> bool:
    if s.agg.startswith("dcnt"):          # затухающий счётчик dcnt<H>: окно не нужно, вес события 2^(-age/H)
        return s.window is None
    if s.agg == "mean_ratio":
        return s.kind == "listen"
    if s.agg == "nuniq_item":
        return s.entity in ("user", "artist", "album") and s.kind in ("listen", "lplus", "like")
    if s.agg == "recency":
        return s.window is None
    if s.agg == "age":
        return s.window is None and s.kind == "listen"
    if s.agg == "frac_org":
        return s.kind == "listen"
    return True


def pool_specs(windows=POOL_WINDOWS, kinds=POOL_KINDS):
    """Большой пул кандидатов: из него и выбирает как ручной набор, так и автоматический поиск."""
    out = []
    for e in ENT_KEYS:
        for k in kinds:
            for a in ("cnt", "mean_ratio", "recency", "age", "nuniq_item", "frac_org"):
                for w in windows:
                    s = Spec(e, k, a, w)
                    if is_valid(s):
                        out.append(s)
    for e in ENT_KEYS:
        for w in windows:
            for nk in ("lplus", "like", "skip", "dislike"):
                out.append(Ratio(Spec(e, nk, "cnt", w), Spec(e, "listen", "cnt", w)))
    for pe in ("ui", "ua", "ual"):   # доля пары/артиста/альбома в прослушиваниях пользователя
        for w in windows:
            out.append(Ratio(Spec(pe, "listen", "cnt", w), Spec("user", "listen", "cnt", w)))
    return out


def hand_specs():
    """Ручной набор: то, что обычно придумывает инженер (окна 1/7/30/вся история)."""
    ents = ("user", "item", "artist", "ui", "ua")
    out = []
    for e in ents:
        for k in ("listen", "lplus", "like", "dislike"):
            for w in (1, 7, 30, None):
                out.append(Spec(e, k, "cnt", w))
        for w in (7, 30, None):
            out.append(Spec(e, "listen", "mean_ratio", w))
    for e in ("user", "item", "ui", "ua"):
        out += [Spec(e, "listen", "recency", None), Spec(e, "lplus", "recency", None)]
    out += [Spec("user", "listen", "age", None), Spec("item", "listen", "age", None)]
    for e in ("user", "artist"):
        out += [Spec(e, "listen", "nuniq_item", 7), Spec(e, "listen", "nuniq_item", 30)]
    for e in ents:
        for w in (7, 30, None):
            out.append(Ratio(Spec(e, "lplus", "cnt", w), Spec(e, "listen", "cnt", w)))
    for e in ("user", "item", "artist"):
        for w in (30, None):
            out.append(Ratio(Spec(e, "like", "cnt", w), Spec(e, "listen", "cnt", w)))
    for pe in ("ui", "ua"):
        for w in (7, 30):
            out.append(Ratio(Spec(pe, "listen", "cnt", w), Spec("user", "listen", "cnt", w)))
    return out


DECAY_HL = (1, 3, 7, 14, 30, 90, 365)


def decay_specs(ents=tuple(ENT_KEYS), kinds=("listen", "lplus", "like", "skip"), hls=DECAY_HL):
    """Новый оператор: затухающие счётчики. Вопрос «какое окно» заменяется вопросом «какой полупериод»."""
    return [Spec(e, k, f"dcnt{h}", None) for e in ents for k in kinds for h in hls]


def trend_specs(ents=tuple(ENT_KEYS), kinds=("listen", "lplus"), pairs=((1, 7), (7, 30), (30, 90))):
    """Новый оператор: тренд - отношение короткого окна к длинному (растёт ли интерес)."""
    return [Ratio(Spec(e, k, "cnt", a), Spec(e, k, "cnt", b)) for e in ents for k in kinds for a, b in pairs]


def ext_specs():
    return decay_specs() + trend_specs()


def hand_decay_specs(hls=(1, 7, 30, 365)):
    """Ручной набор, в котором 80 оконных счётчиков (окна 1/7/30/all) заменены на 80 затухающих (полупериоды 1/7/30/365).
    Остальные признаки те же - сравнение «окна vs затухание» при равном числе признаков."""
    hand = hand_specs()
    other = [s for s in hand if not (isinstance(s, Spec) and s.agg == "cnt")]
    dec = [Spec(e, k, f"dcnt{h}", None) for e in ("user", "item", "artist", "ui", "ua")
           for k in ("listen", "lplus", "like", "dislike") for h in hls]
    return other + dec


# ------------------------------------------------------------------ движок
def _agg_expr(s: Spec, T: int) -> pl.Expr:
    k = pl.col(f"k_{s.kind}")
    if s.agg == "cnt":
        e = k.sum().cast(pl.Float32)
    elif s.agg.startswith("dcnt"):
        hl = float(s.agg[4:])
        age = (pl.lit(T, dtype=pl.Int64) - pl.col("ts").cast(pl.Int64)).cast(pl.Float64) / DAY
        e = (k.cast(pl.Float64) * (age * (-np.log(2.0) / hl)).exp()).sum()
    elif s.agg == "mean_ratio":
        e = pl.col("ratio").filter(k).mean()
    elif s.agg == "recency":
        e = (pl.lit(T, dtype=pl.Int32) - pl.col("ts").filter(k).max()) / DAY
    elif s.agg == "age":
        e = (pl.lit(T, dtype=pl.Int32) - pl.col("ts").filter(k).min()) / DAY
    elif s.agg == "nuniq_item":
        e = pl.col("item_id").filter(k).n_unique().cast(pl.Float32)
    elif s.agg == "frac_org":
        e = pl.col("organic").filter(k).cast(pl.Float32).mean()
    else:
        raise ValueError(s.agg)
    return e.cast(pl.Float32).alias(s.name)


def compute_features(ev: pl.DataFrame, cands: pl.DataFrame, T: int, specs) -> pl.DataFrame:
    """cands: uid, item_id, artist_id, album_id, rid (+ любые доп. колонки). Возвращает cands + признаки.
    Используются ТОЛЬКО события ev.ts < T."""
    base = {}
    for s in specs:
        for b in ((s.num, s.den) if isinstance(s, Ratio) else (s,)):
            base[b.name] = b
    hist = ev.filter(pl.col("ts") < T)
    groups = {}
    for b in base.values():
        groups.setdefault((b.entity, b.window), []).append(b)

    out = cands
    wcache = {}
    for (ent, win), lst in groups.items():
        keys = ENT_KEYS[ent]
        if win not in wcache:
            wcache[win] = hist if win is None else hist.filter(pl.col("ts") >= T - win * DAY)
        keyset = cands.select(keys).drop_nulls().unique()
        h = wcache[win].join(keyset, on=keys, how="semi")
        g = h.group_by(keys).agg([_agg_expr(b, T) for b in lst])
        out = out.join(g, on=keys, how="left")
        zero = [b.name for b in lst if b.agg in ("cnt", "nuniq_item") or b.agg.startswith("dcnt")]
        if zero:
            out = out.with_columns([pl.col(c).fill_null(0.0) for c in zero])
    ratios = [(pl.col(s.num.name) / (pl.col(s.den.name) + 1.0)).cast(pl.Float32).alias(s.name)
              for s in specs if isinstance(s, Ratio)]
    if ratios:
        out = out.with_columns(ratios)
    out = out.sort("rid")
    drop = [n for n in base if n not in {s.name for s in specs}]
    return out.drop(drop)

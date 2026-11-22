"""Twenty more in-game inputs, used by candidate A4 of the vexil-auto loop.

Port of the phase 2 definitions (``feat_a.py``: ``lh_diff``; ``feat_b.py``: kills in a window,
buildings by tier, Aegis, first blood, couriers, Tormentor, rank and share of gold and XP;
``feat_c.py``: ``item_nw_diff``). One implementation, ``extra_table``, computes every minute
of a match at once from events dated at most ``60 * t`` seconds, so a cut copy of a match
gives the same values at minute t. Stored values are 32-bit floats, as in the phase 2 tables
the loop scored on. All differences are Radiant minus Dire.

Item values need the OpenDota item constants. They are frozen in ``models/constants/items.json``
(pulled once, hashed in the registry of the candidates), so a later patch cannot change them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from vexil.features.a1 import _RAX
from vexil.matchdata import Match, Player, is_radiant, n_rows

EXTRA_INPUTS = (
    "lh_diff",
    "kd60",
    "kd120",
    "kd300",
    "t1_diff",
    "t2_diff",
    "t3_diff",
    "t4_diff",
    "rax_count_diff",
    "aegis_active",
    "fb_team",
    "courier_diff",
    "tormentor_diff",
    "xp_adv_d5",
    "gold_rank2_diff",
    "gold_rank45_diff",
    "top3_share_diff",
    "xp_min_diff",
    "xp_adv_dec",
    "item_nw_diff",
)

ITEMS_FILE = Path(__file__).resolve().parents[3] / "models" / "constants" / "items.json"
_LEVEL = re.compile(r"^(.*)_(\d)$")


def load_items(path: Path = ITEMS_FILE) -> dict:
    return json.loads(Path(path).read_text())


def item_value(key: str, items: dict) -> float:
    """Gold a purchase-log entry adds to the value of a player's items.

    Assembled items are logged next to their components and recipes are not logged, so an
    assembled item adds its cost minus the cost of its non-recipe components. An upgrade level
    without components (dagon_2 ...) adds the cost difference to the level below. Consumables,
    Roshan drops and unknown keys add 0.
    """
    it = items.get(key)
    if it is None or "consumable" in (it.get("qual") or "") or key.startswith("recipe"):
        return 0.0
    if key.endswith("_roshan"):
        return 0.0
    cost = float(it.get("cost") or 0)
    comp = it.get("components")
    if comp:
        parts = sum(
            float((items.get(c) or {}).get("cost") or 0) for c in comp if not c.startswith("recipe")
        )
        return max(0.0, cost - parts)
    level = _LEVEL.match(key)
    if level:
        n = int(level.group(2))
        below = level.group(1) if n == 2 else f"{level.group(1)}_{n - 1}"
        if below in items:
            return max(0.0, cost - float(items[below].get("cost") or 0))
    return cost


def _count(times, limits) -> np.ndarray:
    return np.searchsorted(np.sort(np.asarray(times, dtype=float)), limits, side="right")


def _player_items(player: Player, now: np.ndarray, items: dict) -> np.ndarray:
    log = player.get("purchase_log") or []
    times = np.array([e["time"] for e in log], dtype=float)
    values = np.array([item_value(e["key"], items) for e in log], dtype=float)
    order = np.argsort(times, kind="stable")
    csum = np.concatenate([[0.0], np.cumsum(values[order])])
    return csum[np.searchsorted(times[order], now, side="right")]


def extra_table(
    match: Match, players: list[Player], items: dict, n: int | None = None
) -> dict[str, np.ndarray]:
    """The twenty inputs for minutes 0 .. n-1 (all usable minutes by default), as float32."""
    n = n_rows(match) if n is None else n
    t = np.arange(n)
    now = 60.0 * t
    x = np.asarray(match["radiant_xp_adv"][:n], dtype=float)
    rad = [p for p in players if is_radiant(p)]
    dire = [p for p in players if not is_radiant(p)]
    out: dict[str, np.ndarray] = {}

    out["lh_diff"] = np.array([p["lh_t"][:n] for p in rad], dtype=float).sum(0) - np.array(
        [p["lh_t"][:n] for p in dire], dtype=float
    ).sum(0)

    rk = [e["time"] for p in rad for e in (p.get("kills_log") or [])]
    dk = [e["time"] for p in dire for e in (p.get("kills_log") or [])]

    def kd(limits):
        return _count(rk, limits) - _count(dk, limits)

    for w in (60, 120, 300):
        out[f"kd{w}"] = kd(now) - kd(now - w)

    objs = [
        o for o in (match.get("objectives") or []) if not str(o.get("key", "")).endswith("_fort")
    ]
    bk = [o for o in objs if o.get("type") == "building_kill"]
    for tier in (1, 2, 3, 4):
        tag = f"_tower{tier}"
        out[f"t{tier}_diff"] = _count(
            [o["time"] for o in bk if tag in o["key"] and "_badguys_" in o["key"]], now
        ) - _count([o["time"] for o in bk if tag in o["key"] and "_goodguys_" in o["key"]], now)
    out["rax_count_diff"] = _count(
        [o["time"] for o in bk if _RAX.match(o["key"]) and "_badguys_" in o["key"]], now
    ) - _count([o["time"] for o in bk if _RAX.match(o["key"]) and "_goodguys_" in o["key"]], now)

    aeg = sorted(
        (o["time"], i, 1 if o["player_slot"] < 128 else -1)
        for i, o in enumerate(objs)
        if o.get("type") in ("CHAT_MESSAGE_AEGIS", "CHAT_MESSAGE_AEGIS_STOLEN")
    )
    active = np.zeros(n)
    if aeg:
        times = np.array([a[0] for a in aeg], dtype=float)
        pos = np.searchsorted(times, now, side="right")
        age = now - times[np.maximum(pos - 1, 0)]
        on = (pos > 0) & (age < 300)
        side = np.array([a[2] for a in aeg])[np.maximum(pos - 1, 0)]
        active = np.where(on, side, 0.0)
    out["aegis_active"] = active

    fb = sorted(
        (o["time"], 1 if o.get("player_slot", 0) < 128 else -1)
        for o in objs
        if o.get("type") == "CHAT_MESSAGE_FIRSTBLOOD"
    )
    out["fb_team"] = np.where(now >= fb[0][0], fb[0][1], 0.0) if fb else np.zeros(n)

    def team(kind, tm):
        return [o["time"] for o in objs if o.get("type") == kind and o.get("team") == tm]

    out["courier_diff"] = _count(team("CHAT_MESSAGE_COURIER_LOST", 3), now) - _count(
        team("CHAT_MESSAGE_COURIER_LOST", 2), now
    )
    out["tormentor_diff"] = _count(team("CHAT_MESSAGE_MINIBOSS_KILL", 2), now) - _count(
        team("CHAT_MESSAGE_MINIBOSS_KILL", 3), now
    )
    out["xp_adv_d5"] = x - x[np.maximum(t - 5, 0)]

    gr = -np.sort(-np.array([p["gold_t"][:n] for p in rad], dtype=float), axis=0)
    gd = -np.sort(-np.array([p["gold_t"][:n] for p in dire], dtype=float), axis=0)
    xr = np.array([p["xp_t"][:n] for p in rad], dtype=float)
    xd = np.array([p["xp_t"][:n] for p in dire], dtype=float)
    out["gold_rank2_diff"] = gr[1] - gd[1]
    out["gold_rank45_diff"] = gr[3:].sum(0) - gd[3:].sum(0)

    def share(a):
        return np.divide(a[:3].sum(0), a.sum(0), out=np.zeros(n), where=a.sum(0) > 0)

    out["top3_share_diff"] = share(gr) - share(gd)
    out["xp_min_diff"] = xr.min(0) - xd.min(0)
    out["xp_adv_dec"] = x * np.exp(-t / 20)

    nw = np.zeros(n)
    for p in players:
        nw += (1.0 if is_radiant(p) else -1.0) * _player_items(p, now, items)
    out["item_nw_diff"] = nw
    return {name: out[name].astype(np.float32) for name in EXTRA_INPUTS}

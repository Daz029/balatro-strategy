"""Static effect-descriptor vectors for shop items, derived from engine data.

The shop agent represents item identity two ways, concatenated (grilled
decision — CLAUDE.md shop-agent design):

* a **learned embedding** per center key (``nn.Embedding`` in
  ``shop_policy``, indexed by ``observation.center_key_id`` — ONE table for
  every item type, so "I own Lusty Joker" and "the shop offers Lusty Joker"
  share a vector by construction), and
* these **hand-derived descriptors**: coarse mechanical facts extracted
  from the engine's own prototype configs. They exist for cold-start and
  pool-transfer — a joker the agent has rarely seen still arrives as
  "+mult-per-heart, rare, $8" instead of a random vector. The embedding
  absorbs everything the descriptors miss; precision here is NOT critical,
  coverage of broad effect families is.

``DESCRIPTOR_MATRIX`` is a frozen ``(NUM_CENTER_KEYS + 1, DESCRIPTOR_DIM)``
float32 table (row 0 = padding/unknown, e.g. playing cards) registered as a
non-trainable buffer in the policy net. Row layout is APPEND-ONLY like every
other frozen contract in this project.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jackdaw.engine.data.prototypes import BOOSTERS, CENTER_POOLS, JOKERS
from jackdaw.env.observation import NUM_CENTER_KEYS, center_key_id

DESCRIPTOR_DIM = 24

# Descriptor feature layout (frozen, append-only):
#   0  rarity / 4                     (jokers; 0 otherwise)
#   1  base cost / 10
#   2  is_joker
#   3  is_tarot
#   4  is_planet
#   5  is_spectral
#   6  is_voucher
#   7  is_booster
#   8  flat +mult / 20
#   9  flat +chips / 100
#  10  (x_mult - 1) / 3               (static xmult jokers)
#  11  suit-conditional flag
#  12  conditioned suit ordinal / 3   (H=0 D=1 C=2 S=3, when flagged)
#  13  suit mult value / 10
#  14  hand-type-conditional flag     (Type Mult family)
#  15  conditional t_mult / 20
#  16  conditional t_chips / 100
#  17  per-trigger dollars / 10       (economy family)
#  18  probabilistic flag             (extra.odds present)
#  19  1 / odds                       (trigger probability, when flagged)
#  20  scaling flag                   (accumulates over the run)
#  21  scaling rate (coarse) / 5
#  22  hand/discard-size mod / 3
#  23  blueprint_compat               (copyable by Blueprint/Brainstorm)

_SUIT_ORD = {"Hearts": 0, "Diamonds": 1, "Clubs": 2, "Spades": 3}

# Standard consumable shop costs (vanilla): used only as a coarse
# descriptor feature, actual purchase cost always comes from the live card.
_POOL_BASE_COST = {"Tarot": 3.0, "Planet": 3.0, "Spectral": 4.0, "Voucher": 10.0}

# extra-dict keys that indicate per-trigger accumulation (scaling jokers)
_SCALING_EXTRA_KEYS = frozenset(
    {"chip_mod", "mult_mod", "increase", "h_mod", "hand_add", "discard_sub", "chips"}
)


def _num(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) else 0.0


def _joker_descriptor(key: str) -> np.ndarray:
    proto = JOKERS[key]
    cfg: dict[str, Any] = proto.config or {}
    extra = cfg.get("extra")
    extra_d: dict[str, Any] = extra if isinstance(extra, dict) else {}

    v = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    v[0] = proto.rarity / 4.0
    v[1] = proto.cost / 10.0
    v[2] = 1.0

    v[8] = (_num(cfg.get("mult")) + _num(extra_d.get("mult"))) / 20.0
    v[9] = _num(extra_d.get("chips")) / 100.0
    xmult = max(_num(cfg.get("Xmult")), _num(extra_d.get("Xmult")))
    if xmult > 1.0:
        v[10] = (xmult - 1.0) / 3.0

    suit = extra_d.get("suit")
    if suit in _SUIT_ORD:
        v[11] = 1.0
        v[12] = _SUIT_ORD[suit] / 3.0
        v[13] = _num(extra_d.get("s_mult")) / 10.0

    if "type" in cfg or "poker_hand" in extra_d:
        v[14] = 1.0
        v[15] = _num(cfg.get("t_mult")) / 20.0
        v[16] = _num(cfg.get("t_chips")) / 100.0

    v[17] = _num(extra_d.get("dollars")) / 10.0

    odds = _num(extra_d.get("odds"))
    if odds > 0:
        v[18] = 1.0
        v[19] = 1.0 / odds

    # Scaling: a bare numeric `extra` is the engine's idiom for a
    # per-trigger increment (Ride the Bus, Hologram, ...); several named
    # extra keys mean the same. Coarse by design.
    scaling_rate = 0.0
    if isinstance(extra, (int, float)):
        scaling_rate = float(extra)
    else:
        for k in _SCALING_EXTRA_KEYS & set(extra_d):
            scaling_rate = max(scaling_rate, _num(extra_d[k]))
    if scaling_rate > 0 and (xmult > 0 or isinstance(extra, (int, float)) or extra_d):
        v[20] = 1.0
        v[21] = min(scaling_rate, 5.0) / 5.0

    v[22] = (_num(cfg.get("h_size")) + _num(extra_d.get("h_size")) + _num(cfg.get("d_size"))) / 3.0
    v[23] = float(getattr(proto, "blueprint_compat", False))
    return v


def _simple_descriptor(pool: str, cost: float, type_slot: int) -> np.ndarray:
    v = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    v[1] = cost / 10.0
    v[type_slot] = 1.0
    return v


def build_descriptor_matrix() -> np.ndarray:
    """(NUM_CENTER_KEYS + 1, DESCRIPTOR_DIM) — row per center-key id, row 0 pad."""
    m = np.zeros((NUM_CENTER_KEYS + 1, DESCRIPTOR_DIM), dtype=np.float32)

    for key in CENTER_POOLS.get("Joker", []):
        m[center_key_id(key)] = _joker_descriptor(key)

    for pool, type_slot in (("Tarot", 3), ("Planet", 4), ("Spectral", 5)):
        for key in CENTER_POOLS.get(pool, []):
            m[center_key_id(key)] = _simple_descriptor(pool, _POOL_BASE_COST[pool], type_slot)

    for key in CENTER_POOLS.get("Voucher", []):
        m[center_key_id(key)] = _simple_descriptor("Voucher", _POOL_BASE_COST["Voucher"], 6)

    for key, proto in BOOSTERS.items():
        m[center_key_id(key)] = _simple_descriptor("Booster", float(proto.cost), 7)

    m[0] = 0.0  # padding/unknown stays exactly zero
    return m


DESCRIPTOR_MATRIX: np.ndarray = build_descriptor_matrix()

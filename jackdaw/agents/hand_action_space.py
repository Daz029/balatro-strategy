"""Canonical fixed action space for the hand-play (ante-play) agent.

The full-run gym wrapper (``jackdaw/env/gymnasium_wrapper.py``) re-enumerates
legal actions into a fresh ``Discrete(500)`` table every step, randomly
subsampled when combos exceed budget -- action index *i* means something
different on every step. That is unusable for behavior cloning (a
demonstration label must map to a fixed index) and for BC-to-PPO weight
transfer (the policy head's output units need stable meanings).

For hand-play only, the space is small enough to fix canonically:

    {PlayHand, Discard} x all 1..5-card subsets of the 8 padded hand
    positions = 2 x 218 = 436 actions.

Index layout (APPEND-ONLY -- see below):

    [0, 218)    PlayHand with combo ``COMBOS[i]``
    [218, 436)  Discard  with combo ``COMBOS[i - 218]``

``COMBOS`` enumerates subsets in (size, lexicographic) order:
(0,), (1,), ... (7,), (0,1), (0,2), ... (3,4,5,6,7).

**Append-only contract**: indices 0..435 are frozen forever. When the
shop-merge later adds action families (UseConsumable x targets, ...), they
append at 436+ and the policy head grows new rows; existing rows -- and any
trained checkpoint -- keep their meaning. Never reorder or renumber.

Card *ordering* is deliberately not part of the action space: the agent
picks a subset, and the environment plays it in the engine-optimal order
via ``jackdaw.engine.play_ordering.best_play_order`` (decided in a
/grilling session -- see CLAUDE.md's ante-play track).

No torch dependency here; this module is shared by the BC data loader, the
gym env, and the training scripts.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from jackdaw.env.action_space import ActionType

MAX_HAND_CARDS = 8
MIN_SELECT = 1
MAX_SELECT = 5

# All 1..5-card subsets of range(8), in (size, lexicographic) order.
COMBOS: tuple[tuple[int, ...], ...] = tuple(
    combo
    for size in range(MIN_SELECT, MAX_SELECT + 1)
    for combo in combinations(range(MAX_HAND_CARDS), size)
)
NUM_COMBOS = len(COMBOS)  # 218
NUM_HAND_ACTIONS = 2 * NUM_COMBOS  # 436

_COMBO_TO_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(COMBOS)}

# Max hand index touched by each combo, for vectorized legality checks:
# a combo is in-range iff its max index < current hand size.
_COMBO_MAX_INDEX: np.ndarray = np.array([c[-1] for c in COMBOS], dtype=np.int64)


def action_to_combo(action: int) -> tuple[int, tuple[int, ...]]:
    """Decode a canonical action index into (action_type, card indices).

    ``action_type`` is ``ActionType.PlayHand`` or ``ActionType.Discard``;
    the card indices are 0-based positions into the current hand, sorted
    ascending.
    """
    if not 0 <= action < NUM_HAND_ACTIONS:
        raise ValueError(f"action {action} outside [0, {NUM_HAND_ACTIONS})")
    if action < NUM_COMBOS:
        return (int(ActionType.PlayHand), COMBOS[action])
    return (int(ActionType.Discard), COMBOS[action - NUM_COMBOS])


def combo_to_action(action_type: int, card_indices: tuple[int, ...] | list[int]) -> int:
    """Encode (action_type, card indices) into the canonical action index.

    ``card_indices`` may be unsorted; duplicates are an error. Used to map
    demonstration labels (and engine actions) onto policy-head indices.
    """
    combo = tuple(sorted(int(i) for i in card_indices))
    idx = _COMBO_TO_INDEX.get(combo)
    if idx is None:
        raise ValueError(f"card indices {card_indices!r} are not a valid 1-5 card combo")
    if action_type == ActionType.PlayHand:
        return idx
    if action_type == ActionType.Discard:
        return NUM_COMBOS + idx
    raise ValueError(f"action_type {action_type} is not PlayHand/Discard")


def mask_to_action(action_type: int, card_target_mask: np.ndarray) -> int:
    """Encode a demo-shard label (action_type, multi-hot mask over the padded
    hand width) into the canonical action index."""
    (indices,) = np.nonzero(np.asarray(card_target_mask, dtype=bool))
    return combo_to_action(action_type, tuple(int(i) for i in indices))


def legal_action_mask(hand_size: int, hands_left: int, discards_left: int) -> np.ndarray:
    """Boolean legality mask of shape ``(NUM_HAND_ACTIONS,)``.

    A play combo is legal iff hands remain and every index is within the
    current hand; a discard combo additionally requires discards remaining.
    Matches the engine's own legality rules for the SELECTING_HAND phase
    (``jackdaw/env/action_space.py::get_action_mask``): all in-range combos
    of 1-5 cards are always selectable.
    """
    mask = np.zeros(NUM_HAND_ACTIONS, dtype=bool)
    in_range = _COMBO_MAX_INDEX < hand_size
    if hands_left > 0:
        mask[:NUM_COMBOS] = in_range
    if discards_left > 0:
        mask[NUM_COMBOS:] = in_range
    return mask

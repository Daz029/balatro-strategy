"""Deterministic scripted hand-play policy for the shop agent's env.

The shop env auto-resolves hand phases through a ``hand_policy`` callable
(``game_state -> engine Action``). The real partner is a trained h-agent
checkpoint, but that can't serve as a test fixture (checkpoints change with
every retrain, and torch inference in unit tests is slow), and it doesn't
exist until BC lands. This module is the permanent scripted stand-in:

* **fast** (one engine hand-detection pass, no search, no scoring rollout),
* **deterministic** (same state -> same action, forever),
* **deliberately simple** — it plays the engine-detected best hand, and
  spends a discard first only when the best hand is weak. It knows nothing
  about jokers, blind thresholds, or hand levels. That blindness is the
  point: as the fixed ablation baseline, the gap between
  shop-agent-with-greedy and shop-agent-with-h0 isolates how much shop
  value depends on hand-play skill.

Intentionally NOT solver-derived: ``scripts/hand_solver.py`` is not
importable from the package (scripts/ sits above jackdaw/ in the dependency
order), and a smarter baseline would only blur the ablation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from jackdaw.engine.actions import Action, Discard, PlayHand
from jackdaw.engine.data.hands import HAND_ORDER
from jackdaw.engine.hand_eval import get_best_hand, get_hand_eval_flags

# Hand types weak enough to spend a discard improving, when one is available.
_WEAK_HANDS = frozenset({"High Card", "Pair"})

# Engine cap on cards per discard.
_DISCARD_LIMIT = 5

# Lower index = stronger hand (HAND_ORDER is the engine's priority walk).
_HAND_PRIORITY: dict[str, int] = {ht.value: i for i, ht in enumerate(HAND_ORDER)}


def _best_selection(hand: list, flags: dict[str, bool]) -> tuple[str, set[int]]:
    """Best (hand_name, scoring card ids) over all 5-card selections.

    ``get_best_hand`` mirrors Lua's played-selection semantics: flush and
    straight detection only trigger within a <=5-card selection, so calling
    it once on a full 8-card hand misses them entirely. C(8,5)=56 subset
    evaluations, a few ms — fine for a scripted baseline.
    """
    n = len(hand)
    subsets = [tuple(range(n))] if n <= 5 else list(combinations(range(n), 5))

    best_key: tuple[int, int] | None = None
    best_name = "High Card"
    best_ids: set[int] = {id(max(hand, key=lambda c: getattr(c.base, "id", 0)))}
    for subset in subsets:
        cards = [hand[i] for i in subset]
        name, scoring, _ = get_best_hand(
            cards,
            four_fingers=flags["four_fingers"],
            shortcut=flags["shortcut"],
            smeared=flags["smeared"],
        )
        if name == "NULL" or not scoring:
            continue
        rank_total = sum(getattr(c.base, "id", 0) for c in scoring)
        key = (_HAND_PRIORITY.get(name, len(HAND_ORDER)), -rank_total)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
            best_ids = {id(c) for c in scoring}
    return best_name, best_ids


def estimate_best_hand_type(hand: list, jokers: list) -> str:
    """Cheap best-detectable hand-type name for ``hand`` (e.g. ``"Flush"``).

    Public wrapper around ``_best_selection`` for callers that need a fast
    "what's this hand's best line" estimate without a full policy decision
    -- e.g. ``HandPlayAdapter``'s boss round-history sampling, which needs
    a hand-type name in the same format ``Blind.debuff_hand`` compares
    against (``hands_used`` keys / ``only_hand``). Uses the same
    ``get_best_hand`` call ``debuff_hand`` does internally, so name
    formats always agree by construction.
    """
    flags = get_hand_eval_flags(jokers)
    name, _ = _best_selection(hand, flags)
    return name


class GreedyHandPolicy:
    """Play the engine-detected best hand; discard chaff first if it's weak.

    Decision rule (all state read from the engine's game-state dict):

    1. Find the strongest hand over all 5-card selections (respecting Four
       Fingers / Shortcut / Smeared, via the real engine evaluator per
       selection — see ``_best_selection`` for why per-selection matters).
    2. If the best hand is High Card or Pair AND discards remain AND there
       is at least one non-contributing card: discard up to 5
       non-contributing cards (lowest ranks first).
    3. Otherwise play the best hand's scoring cards (1-5 of them).
    """

    def __call__(self, game_state: dict[str, Any]) -> Action:
        hand: list = game_state.get("hand", [])
        if not hand:
            raise ValueError("GreedyHandPolicy called with an empty hand")

        jokers: list = game_state.get("jokers", [])
        cr = game_state.get("current_round", {})
        flags = get_hand_eval_flags(jokers)

        best_name, scoring_ids = _best_selection(hand, flags)
        chaff = [i for i, c in enumerate(hand) if id(c) not in scoring_ids]

        if best_name in _WEAK_HANDS and cr.get("discards_left", 0) > 0 and chaff:
            # Lowest-rank chaff first, capped at the engine's discard limit.
            chaff.sort(key=lambda i: getattr(hand[i].base, "id", 0))
            return Discard(card_indices=tuple(sorted(chaff[:_DISCARD_LIMIT])))

        play = sorted(i for i, c in enumerate(hand) if id(c) in scoring_ids)
        if not play or len(play) > 5:
            # Defensive fallback: play the 5 highest-rank cards. Reached only
            # if detection returns nothing usable (shouldn't happen).
            by_rank = sorted(
                range(len(hand)),
                key=lambda i: getattr(hand[i].base, "id", 0),
                reverse=True,
            )
            play = sorted(by_rank[:5])
        return PlayHand(card_indices=tuple(play))

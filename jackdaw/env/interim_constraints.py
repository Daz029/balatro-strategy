"""Interim env-side adapters between today's env actions and the vanilla engine.

Engine Phase 6 (2026-10-12) made the engine fully vanilla; the environments
hold the constraints, and their masks are rebuilt later (CLAUDE.md TODO 4).
Until that rebuild, the env masks and hand policies do not know about
everything the vanilla engine now enforces. Each function here bridges ONE
such gap explicitly, at the point where an env action enters the engine, and
goes away when the masks are rebuilt from :mod:`jackdaw.engine.legality`.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import Action, Discard, PlayHand
from jackdaw.engine.legality import with_forced_card


def with_cerulean_bell(gs: dict[str, Any], action: Action) -> Action:
    """Make a hand play/discard include Cerulean Bell's forced card.

    The vanilla engine rejects a selection without it (audit D27); today's
    hand masks (``hand_play_gym.hand_action_mask``, the pointer head) and
    hand policies (greedy, h0.5) are unaware of it. The selection is completed
    the way a Lua player's would be (the forced card is pre-highlighted and
    counts toward the 5): see :func:`jackdaw.engine.legality.with_forced_card`.
    A no-op for every other action and on every non-Cerulean state.
    """
    if isinstance(action, PlayHand) and action.card_indices:
        return PlayHand(card_indices=with_forced_card(gs, action.card_indices))
    if isinstance(action, Discard) and action.card_indices:
        return Discard(card_indices=with_forced_card(gs, action.card_indices))
    return action

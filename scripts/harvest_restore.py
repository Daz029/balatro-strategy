"""Restore a harvested engine blob into a live ``gs``, repairing capture skew.

The counterpart to ``harvest_s0_rollouts.py``'s capture. Phase 1 pickled live
engine states; phase 2 (C2) restores them and labels them with the CURRENT
solver against the CURRENT engine. This module is the seam between those two,
and it exists because of one asymmetry:

    An engine fix that changes COMPUTATION is inherited by the harvest for
    free. An engine fix that changes STORED STATE is not.

Anything the engine derives at scoring time (Blueprint/Brainstorm compat, joker
ordering, the prescreen, the discard ranking) is recomputed from the restored
state, so a blob simply flows through the fixed code. But some values are
derived ONCE at round start, CACHED into ``current_round``, and later read back
rather than recomputed. For those, the blob's stale copy wins — a perfectly
faithful, RNG-exact snapshot of a state that an older engine created. Fidelity
to the capture is precisely what preserves the old bug.

KNOWN SKEW — The Idol (``j_idol``)
----------------------------------
The corpus in ``data/harvest_s0`` was captured at sha 57f1088 (on the 9600X;
that commit is not in this repo, so the C2 sha check can only ever report
"differs", never diff it). The B2 Idol fix landed after capture. It changed
``round_lifecycle.reset_round_targets`` to cache an ``"id"`` alongside the
drawn card's rank and suit::

    before:  cr["idol_card"] = {"suit": "Hearts", "rank": "King"}
    after:   cr["idol_card"] = {"suit": "Hearts", "rank": "King", "id": 13}

The handler matches on that numeric id, not the rank string
(``jokers.py``: ``ctx.other_card.get_id() == ctx.game.idol_card.get("id")``).
With no ``"id"`` key the comparison is against ``None``, never fires, and the
solver labels the state as if the joker slot were empty. Silent, and it reaches
~0.53% of harvested hand records.

WHY THE REPAIR IS EXACT, not a patch: ``id`` is ``idol.base.id`` and ``rank``
is ``idol.base.rank`` — both were read off the SAME drawn card, and rank
determines id (``card.py::CardBase.from_card_key`` populates ``id`` from
``_RANK_ID[value]``). The blob already stores ``rank``, so the id was never
lost, only uncached. Reconstructing it reproduces byte-for-byte what the fixed
engine would have written for that same card. We reuse the engine's own
``_RANK_ID`` table rather than restating the mapping here — a second copy of
engine knowledge in feature code is the drift class this project keeps getting
bitten by (see the copy-resolution pitfall).

The repair is idempotent and version-agnostic: a blob captured AFTER the fix
already carries ``"id"`` and passes through untouched, so a future re-harvest
needs no changes here.

SHAPE GUARD: everything else about a restored state was checked against a fresh
one and matches (``current_round``/``round_resets`` key sets, Card fields;
``mail_card`` already carried its id; ancient/castle are suit-only by design;
an absent ``consumable_usage_total`` genuinely means no consumable was used).
But note that a key-set diff STRUCTURALLY CANNOT catch this bug class — the
missing field lived inside a cached dict value, not at a key boundary. So the
guard below hard-fails on any unexpected ``idol_card`` shape rather than
defaulting, which is what keeps the NEXT skew of this class from passing as
quietly as this one did.
"""

from __future__ import annotations

import pickle
from typing import Any

# The engine's own rank-string -> numeric-id table: the exact mapping
# CardBase.from_card_key uses to populate base.id, which is what
# reset_round_targets caches. Imported (private though it is) rather than
# duplicated, so the repair cannot drift from the engine; pinned against the
# real construction path in tests.
from jackdaw.engine.card import _RANK_ID

# A repaired/fresh idol_card carries exactly these; {"rank", "suit"} is the
# pre-fix capture shape. Anything else is unknown drift.
_IDOL_REQUIRED_FIELDS = frozenset({"rank", "suit"})
_IDOL_KNOWN_FIELDS = frozenset({"rank", "suit", "id"})


class CaptureSkewError(Exception):
    """A restored state's shape is not one this module knows how to repair.

    Deliberately fatal per-record rather than defaulted: a silently wrong
    cached field is exactly how the Idol bug survived capture.
    """


def repair_idol_card(gs: dict[str, Any]) -> bool:
    """Backfill ``current_round["idol_card"]["id"]`` on a pre-fix blob.

    Returns True if a repair was applied, False if the state already carried
    the field (post-fix capture — idempotent, so re-restoring is safe).
    Raises :class:`CaptureSkewError` on any shape this module doesn't know.
    """
    cr = gs.get("current_round")
    if not isinstance(cr, dict):
        raise CaptureSkewError(f"restored state has no current_round dict (got {type(cr)!r})")

    idol = cr.get("idol_card")
    if idol is None:
        # reset_round_targets writes idol_card unconditionally at round start,
        # and every harvested record is a mid-round decision state, so absence
        # means the state is not what we think it is.
        raise CaptureSkewError(
            "restored state has no current_round['idol_card']; "
            "reset_round_targets writes it unconditionally at round start"
        )
    if not isinstance(idol, dict):
        raise CaptureSkewError(f"idol_card is not a dict (got {type(idol)!r})")

    fields = set(idol)
    if unknown := fields - _IDOL_KNOWN_FIELDS:
        raise CaptureSkewError(
            f"idol_card carries unknown field(s) {sorted(unknown)} — the engine's "
            "round-target cache has changed shape since this module was written; "
            "re-check the capture skew before labeling"
        )
    if missing := _IDOL_REQUIRED_FIELDS - fields:
        raise CaptureSkewError(f"idol_card is missing required field(s) {sorted(missing)}")

    if "id" in idol:
        return False  # captured post-fix (or already repaired)

    rank = idol["rank"]
    if rank not in _RANK_ID:
        raise CaptureSkewError(
            f"idol_card rank {rank!r} is not a rank the engine knows; cannot "
            f"reconstruct base.id (known: {sorted(_RANK_ID)})"
        )
    idol["id"] = _RANK_ID[rank]
    return True


def repair_capture_skew(gs: dict[str, Any]) -> dict[str, bool]:
    """Apply every known capture-skew repair to a restored state.

    Returns a ``{repair_name: applied}`` report so callers can surface how much
    of a corpus needed repair instead of it happening invisibly.
    """
    return {"idol_card_id": repair_idol_card(gs)}


def restore_state(blob: bytes) -> dict[str, Any]:
    """Unpickle a harvested blob and bring it up to current-engine semantics.

    This is the ONLY supported way to turn a harvested blob into a ``gs``:
    unpickling directly would restore the stale cache described in the module
    docstring and label the state under an engine bug that no longer exists.
    """
    gs = pickle.loads(blob)
    if not isinstance(gs, dict):
        raise CaptureSkewError(f"harvested blob did not restore to a game state (got {type(gs)!r})")
    repair_capture_skew(gs)
    return gs

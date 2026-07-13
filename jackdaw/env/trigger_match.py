"""Per-card x per-joker trigger-match bits (h1 schema bump, B2 slice 2).

Pooling destroys which-card-matches-which-joker structure, so the h1 obs
carries an explicit boolean match matrix
``trigger_match[card, joker_slot, {scored, held}]`` consumed at BC time as
fixed-weight cross-attention (see CLAUDE.md "Feature spec v4.1"). This
module owns the trigger taxonomy and the matrix builder; wiring into
``build_observation`` / the demo writer lands with the schema_version bump
(B2 slice 4).

Taxonomy — every joker key in the frozen center-key vocabulary is
classified into exactly one class, and the module HARD-FAILS at import if
any key is unclassified (a new joker must be classified deliberately, never
silently zeroed):

* **Class 1 — static per-card**: the trigger condition is a property of
  the card itself (suit / rank / face / parity / enhancement). The
  predicate may still consult the hand-eval flags (a Smeared Joker widens
  suit matches; Pareidolia widens face matches) because the ENGINE's own
  ``is_suit`` / ``is_face`` calls do.
* **Class 2 — state-dependent per-card**: the condition reads live game
  state (Ancient Joker's rotating suit, The Idol's card, Castle's suit,
  Mail-In Rebate's rank, Dusk's last-hand gate, Raised Fist's
  lowest-held-rank). A static config table would silently mismark these
  (pitfall: Ancient rotates every round).
* **Class 3 — set-level**: the trigger reads the played/discarded/held SET
  (hand-type conditionals, Flower Pot, Seeing Double, Blackboard, ...).
  All-zero rows BY DESIGN — no honest per-card bit exists, and fabricating
  one (e.g. "mark the scarcest suit") was explicitly rejected. Their
  signal is the GC set-structure features.
* **Class 4 — non-card**: economy / meta / deck-composition jokers. The
  cards in hand are irrelevant to their trigger; all-zero rows.

Bit semantics (pitfall 10): a set bit means the card is a CANDIDATE for
that joker's card-linked effect — class membership, not will-fire.
Photograph marks every face card though only the first scored face gets
the x2; Hanging Chad marks every card though only the first scored one
retriggers. Whether a candidate actually fires depends on the chosen
subset/order, which is the policy's job to learn. Two engine-exact
exceptions narrow candidacy:

* A DEBUFFED card is never a candidate for anything: the scoring loops
  skip debuffed cards before any joker sees them (``scoring.py`` phases
  7/8), and the discard-context handlers check ``other_card.debuff``.
* A DEBUFFED joker matches nothing: every joker loop in scoring skips it.

Channel semantics: ``scored`` = the joker reacts to this card being played
and scored (individual/repetition play contexts, plus scoring-adjacent
mutators like Midas Mask and Vampire). ``held`` = the joker reacts to this
card while it stays in hand — including DISCARD-triggered jokers (Mail,
Castle, Hit the Road, Faceless): discarding happens from the hand, so
"this card has an in-hand interaction with joker j" is the honest reading.

Face-down / base-less cards get all-zero rows (hidden information — same
rule as every other per-card encoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from jackdaw.engine.card import Card
from jackdaw.engine.hand_eval import get_hand_eval_flags
from jackdaw.engine.jokers import (
    JokerContext,
    _find_leftmost,
    _find_right_neighbor,
    blueprint_compatible,
)
from jackdaw.env.observation import center_key_id, center_key_vocabulary

_COPY_JOKERS = frozenset({"j_blueprint", "j_brainstorm"})

# Predicate: (card, joker_card, gs, flags) -> (scored, held).
# `joker_card` is the live joker (Castle stores its suit on the joker's
# ability); `gs` is the raw engine game_state; `flags` is
# get_hand_eval_flags(jokers) computed once per matrix build.
Predicate = Callable[[Card, Card, dict[str, Any], dict[str, bool]], tuple[bool, bool]]

# ---------------------------------------------------------------------------
# Class-1 predicate builders (static per-card conditions)
# ---------------------------------------------------------------------------


def _scored_suit(suit: str) -> Predicate:
    """Suit condition via the engine's own Card.is_suit — smeared-aware,
    Wild Cards match every suit, exactly like the handlers' _is_suit."""

    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        return card.is_suit(suit, smeared=flags["smeared"]), False

    return pred


def _scored_ranks(*ids: int) -> Predicate:
    rank_set = frozenset(ids)

    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        return card.get_id() in rank_set, False

    return pred


def _held_ranks(*ids: int) -> Predicate:
    rank_set = frozenset(ids)

    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        return False, card.get_id() in rank_set

    return pred


def _scored_face(pareidolia_aware: bool = True) -> Predicate:
    """Face condition. Almost every face handler passes ctx.pareidolia to
    is_face; Ride the Bus calls is_face() bare (engine-verified), hence
    the flag."""

    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        p = flags["pareidolia"] if pareidolia_aware else False
        return card.is_face(pareidolia=p), False

    return pred


def _held_face() -> Predicate:
    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        return False, card.is_face(pareidolia=flags["pareidolia"])

    return pred


def _scored_always(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    return True, False


def _held_always(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    return False, True


def _scored_parity(even: bool) -> Predicate:
    """Even Steven / Odd Todd, mirroring the handlers exactly:
    even = id in 2..10 and id % 2 == 0; odd = (id <= 10 and id % 2 == 1)
    or Ace (14)."""

    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        oid = card.get_id()
        if even:
            return 0 <= oid <= 10 and oid % 2 == 0, False
        return (0 <= oid <= 10 and oid % 2 == 1) or oid == 14, False

    return pred


def _scored_ability_name(name: str) -> Predicate:
    """Enhancement identity by ability name (Golden Ticket checks
    'Gold Card', Lucky Cat accumulates on Lucky Card triggers)."""

    def pred(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
        return card.ability.get("name") == name, False

    return pred


def _scored_any_enhancement(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    """Vampire's own eligibility test: any non-default enhancement."""
    scored = (
        card.ability.get("effect", "") not in ("", "Default Base")
        and card.ability.get("name", "") != "Default Base"
    )
    return scored, False


# ---------------------------------------------------------------------------
# Class-2 predicates (live game state)
# ---------------------------------------------------------------------------


def _pred_ancient(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    suit = gs.get("current_round", {}).get("ancient_card", {}).get("suit")
    return bool(suit) and card.is_suit(suit, smeared=flags["smeared"]), False


def _pred_idol(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    idol = gs.get("current_round", {}).get("idol_card") or {}
    matched = (
        idol.get("id") is not None
        and card.get_id() == idol.get("id")
        and card.is_suit(idol.get("suit", ""), smeared=flags["smeared"])
    )
    return matched, False


def _pred_mail(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    mail_id = gs.get("current_round", {}).get("mail_card", {}).get("id")
    return False, mail_id is not None and card.get_id() == mail_id


def _pred_castle(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    # Castle's target suit lives on the JOKER's ability, not current_round
    # (the handler reads card.ability["castle_card_suit"]).
    castle_suit = joker.ability.get("castle_card_suit")
    return False, bool(castle_suit) and card.is_suit(castle_suit, smeared=flags["smeared"])


def _pred_dusk(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    # Dusk retriggers all scored cards on the round's LAST hand. The engine
    # decrements hands_left before scoring, so a hand decided at
    # hands_left == 1 is the one Dusk fires on.
    return gs.get("current_round", {}).get("hands_left", 0) == 1, False


def _pred_raised_fist(card: Card, joker: Card, gs: dict, flags: dict) -> tuple[bool, bool]:
    # Handler: lowest get_id() among non-Stone held cards (first occurrence
    # wins). Candidate semantics: mark every card tied at the minimum —
    # which specific one is "first" changes as cards leave the hand.
    if card.ability.get("effect") == "Stone Card":
        return False, False
    lowest = 15
    for c in gs.get("hand", []):
        if c.ability.get("effect") != "Stone Card" and c.get_id() < lowest:
            lowest = c.get_id()
    return False, card.get_id() == lowest


# ---------------------------------------------------------------------------
# The taxonomy — every vocabulary joker key, exactly once
# ---------------------------------------------------------------------------

# Classes 1 and 2 carry predicates; the split below is documentation of
# WHY each key is where it is (class 2 = reads live state).

_CLASS1_PREDICATES: dict[str, Predicate] = {
    # -- suit-conditional, scored (smeared/Wild via engine is_suit) --
    "j_greedy_joker": _scored_suit("Diamonds"),
    "j_lusty_joker": _scored_suit("Hearts"),
    "j_wrathful_joker": _scored_suit("Spades"),
    "j_gluttenous_joker": _scored_suit("Clubs"),
    "j_arrowhead": _scored_suit("Spades"),
    "j_onyx_agate": _scored_suit("Clubs"),
    "j_rough_gem": _scored_suit("Diamonds"),
    "j_bloodstone": _scored_suit("Hearts"),  # probabilistic fire; candidacy is the suit
    # -- rank-conditional, scored --
    "j_fibonacci": _scored_ranks(2, 3, 5, 8, 14),
    "j_scholar": _scored_ranks(14),
    "j_walkie_talkie": _scored_ranks(4, 10),
    "j_triboulet": _scored_ranks(12, 13),
    "j_wee": _scored_ranks(2),
    "j_hack": _scored_ranks(2, 3, 4, 5),  # retrigger
    "j_8_ball": _scored_ranks(8),  # probabilistic Tarot creation
    "j_sixth_sense": _scored_ranks(6),  # fires only on a lone first-hand 6; candidacy is the rank
    "j_even_steven": _scored_parity(even=True),
    "j_odd_todd": _scored_parity(even=False),
    # -- face-conditional, scored (pareidolia-aware like the handlers) --
    "j_scary_face": _scored_face(),
    "j_smiley": _scored_face(),
    "j_photograph": _scored_face(),  # only the FIRST scored face fires; all faces are candidates
    "j_business": _scored_face(),  # probabilistic $
    "j_sock_and_buskin": _scored_face(),  # retrigger
    "j_midas_mask": _scored_face(),  # scored faces turn Gold (before-context mutator)
    "j_ride_the_bus": _scored_face(pareidolia_aware=False),  # handler calls is_face() bare
    # -- enhancement-conditional, scored --
    "j_ticket": _scored_ability_name("Gold Card"),
    "j_lucky_cat": _scored_ability_name("Lucky Card"),
    "j_glass": _scored_ability_name("Glass Card"),  # grows when scored glass breaks
    "j_vampire": _scored_any_enhancement,  # strips enhancements from scored cards
    # -- unconditional per scored card --
    "j_hiker": _scored_always,  # permanent +chips on every scored card
    "j_selzer": _scored_always,  # retrigger all scored (while charges remain)
    "j_hanging_chad": _scored_always,  # fires on FIRST scored card; position is set-dependent
    # -- held-in-hand --
    "j_baron": _held_ranks(13),
    "j_shoot_the_moon": _held_ranks(12),
    "j_reserved_parking": _held_face(),  # probabilistic $
    "j_mime": _held_always,  # retrigger all held-card effects
    # -- discard-triggered (held channel: the interaction lives in hand) --
    "j_hit_the_road": _held_ranks(11),  # xmult per discarded Jack
    "j_faceless": _held_face(),  # $ when 3+ faces discarded together
}

_CLASS2_PREDICATES: dict[str, Predicate] = {
    "j_ancient": _pred_ancient,  # suit rotates every round
    "j_idol": _pred_idol,  # card changes every round
    "j_castle": _pred_castle,  # suit changes every round (held: discard-triggered)
    "j_mail": _pred_mail,  # rank changes every round (held: discard-triggered)
    "j_dusk": _pred_dusk,  # all scored cards, last hand only
    "j_raised_fist": _pred_raised_fist,  # lowest held rank
}

# Set-level triggers: the condition is a property of the played/discarded/
# held SET, so no honest per-card bit exists. All-zero rows BY DESIGN — the
# GC set-structure features (hand-type indicators, suit/rank counts, window
# occupancy) are their signal. Includes the detection-modifier passives
# (Four Fingers etc.): they change how the SET evaluates, and they are
# already surfaced as flag bits in the obs.
_CLASS3_SET_LEVEL: frozenset[str] = frozenset(
    {
        # hand-type conditionals
        "j_jolly", "j_zany", "j_mad", "j_crazy", "j_droll",
        "j_sly", "j_wily", "j_clever", "j_devious", "j_crafty",
        "j_duo", "j_trio", "j_family", "j_order", "j_tribe",
        "j_supernova", "j_card_sharp", "j_todo_list", "j_obelisk",
        "j_seance", "j_superposition", "j_runner", "j_trousers",
        # played/held-set structure
        "j_blackboard", "j_flower_pot", "j_seeing_double",
        "j_half", "j_square", "j_dna", "j_trading", "j_burnt",
        # detection modifiers (set-evaluation passives)
        "j_four_fingers", "j_shortcut", "j_smeared", "j_splash", "j_pareidolia",
    }
)

# Non-card triggers: economy, shop, per-round, deck-composition, meta.
# The hand's cards are irrelevant to whether these fire.
_CLASS4_NON_CARD: frozenset[str] = frozenset(
    {
        "j_joker", "j_misprint", "j_stuntman", "j_abstract", "j_acrobat",
        "j_mystic_summit", "j_banner", "j_blue_joker", "j_erosion",
        "j_stone", "j_steel_joker", "j_bull", "j_drivers_license",
        "j_stencil", "j_bootstraps", "j_fortune_teller", "j_loyalty_card",
        "j_matador", "j_blueprint", "j_brainstorm", "j_green_joker",
        "j_ice_cream", "j_popcorn", "j_flash", "j_red_card", "j_campfire",
        "j_hologram", "j_constellation", "j_caino", "j_madness",
        "j_throwback", "j_yorick", "j_ceremonial", "j_baseball",
        "j_swashbuckler", "j_certificate", "j_marble", "j_riff_raff",
        "j_cartomancer", "j_vagabond", "j_hallucination", "j_gros_michel",
        "j_cavendish", "j_chicot", "j_luchador", "j_burglar", "j_rocket",
        "j_egg", "j_gift", "j_invisible", "j_diet_cola", "j_space",
        "j_to_the_moon", "j_golden", "j_delayed_grat", "j_satellite",
        "j_ramen", "j_mr_bones", "j_turtle_bean", "j_perkeo",
        "j_cloud_9",  # $ per 9 in the full deck at end of round — deck census, not hand

        # passive/config-only (no handler)
        "j_astronomer", "j_chaos", "j_credit_card", "j_drunkard",
        "j_juggler", "j_merry_andy", "j_oops", "j_ring_master",
        "j_troubadour",
    }
)

_PREDICATES: dict[str, Predicate] = {**_CLASS1_PREDICATES, **_CLASS2_PREDICATES}


def _check_taxonomy() -> None:
    """Build-time coverage check: every joker key in the frozen vocabulary
    is classified exactly once. Runs at import — an unclassified joker is
    a hard error, never a silent all-zero row."""
    vocab = {k for k in center_key_vocabulary() if k.startswith("j_")}
    classified: dict[str, str] = {}
    for name, keys in (
        ("class1", _CLASS1_PREDICATES),
        ("class2", _CLASS2_PREDICATES),
        ("class3", _CLASS3_SET_LEVEL),
        ("class4", _CLASS4_NON_CARD),
    ):
        for key in keys:
            if key in classified:
                raise RuntimeError(
                    f"trigger taxonomy: {key} classified twice ({classified[key]} and {name})"
                )
            classified[key] = name

    unclassified = vocab - classified.keys()
    if unclassified:
        raise RuntimeError(
            f"trigger taxonomy: {len(unclassified)} vocabulary jokers unclassified: "
            f"{sorted(unclassified)} — classify every new joker deliberately"
        )
    unknown = classified.keys() - vocab
    if unknown:
        raise RuntimeError(
            f"trigger taxonomy: {sorted(unknown)} not in the center-key vocabulary "
            f"(typo, or centers.json changed — which would corrupt embedding ids)"
        )


_check_taxonomy()


# ---------------------------------------------------------------------------
# Copy resolution (Blueprint / Brainstorm) — B2 slice 3
# ---------------------------------------------------------------------------
#
# Pooling destroys adjacency, so "has a copy effect" alone is structurally
# uninterpretable in the obs — each copy joker's row must say WHAT it
# currently copies. Resolution reuses the ENGINE's own path (pitfall 11:
# a reimplementation of compatibility/termination rules WILL drift):
# `_find_right_neighbor` / `_find_leftmost` are the handlers' target
# selectors, `blueprint_compatible` is their compat guard, and the walk
# mirrors the handlers' delegation loop including the
# `blueprint > len(jokers) + 1` counter cap.


@dataclass(frozen=True)
class CopyResolution:
    """Resolved copy target for one joker slot.

    ``active`` is False for non-copy jokers AND for copies pointing at
    nothing / a debuffed target / an incompatible target / a loop — in
    which case both target fields are zeroed (the spec's "inactive →
    zeroed target fields").
    """

    active: bool
    target_index: int  # index into gs["jokers"]; -1 when inactive
    target_key_id: int  # frozen-vocab center-key id; 0 when inactive

    @classmethod
    def inactive(cls) -> CopyResolution:
        return cls(active=False, target_index=-1, target_key_id=0)


def resolve_copy_targets(gs: dict[str, Any]) -> list[CopyResolution]:
    """Per-joker resolved copy targets, one entry per gs["jokers"] slot.

    Walks Blueprint→right-neighbor / Brainstorm→leftmost chains exactly
    like the handlers' recursive delegation: each hop applies the debuff
    and blueprint_compat guards, and the hop counter cap
    (``> len(jokers) + 1``) turns loops into inactive resolutions. A chain
    is active only if it terminates on a NON-copy joker that passed every
    guard. A debuffed copy joker is itself inactive (the scoring loops
    never call it).
    """
    jokers: list[Card] = gs.get("jokers", [])
    ctx = JokerContext(jokers=jokers)
    index_by_id = {id(j): i for i, j in enumerate(jokers)}

    out: list[CopyResolution] = []
    for joker in jokers:
        if joker.center_key not in _COPY_JOKERS or joker.debuff:
            out.append(CopyResolution.inactive())
            continue

        current = joker
        hops = 0
        resolution = CopyResolution.inactive()
        while current.center_key in _COPY_JOKERS:
            hops += 1
            if hops > len(jokers) + 1:  # the handlers' loop cap
                break
            target = (
                _find_right_neighbor(current, ctx)
                if current.center_key == "j_blueprint"
                else _find_leftmost(current, ctx)
            )
            if target is None or target.debuff or not blueprint_compatible(target):
                break
            current = target
        else:
            resolution = CopyResolution(
                active=True,
                target_index=index_by_id[id(current)],
                target_key_id=center_key_id(current.center_key),
            )
        out.append(resolution)
    return out


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------


def trigger_match_matrix(gs: dict[str, Any]) -> np.ndarray:
    """``(len(hand), len(jokers), 2)`` bool — [..., 0] = scored, [..., 1] = held.

    Rows for face-down / base-less cards are all-zero (hidden information),
    as are rows for DEBUFFED cards and columns for DEBUFFED jokers — the
    engine's scoring loops skip both before any handler runs, so those
    zeros are engine-exact, not a candidacy judgment.

    A Blueprint/Brainstorm column INHERITS the resolved copy target's
    match row: the target's predicate is evaluated with the TARGET joker
    card (Castle's suit lives on the target's ability). An inactive copy
    stays all-zero. Known simplification: mutation-guarded scaling
    triggers (``not ctx.blueprint`` in the handlers) inherit candidacy
    anyway — the bit is class membership, not will-fire.
    """
    hand: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    if not hand or not jokers:
        return np.zeros((len(hand), len(jokers), 2), dtype=bool)

    flags = get_hand_eval_flags(jokers)
    out = np.zeros((len(hand), len(jokers), 2), dtype=bool)

    resolutions: list[CopyResolution] | None = None
    if any(j.center_key in _COPY_JOKERS for j in jokers):
        resolutions = resolve_copy_targets(gs)

    active: list[tuple[int, Card, Predicate]] = []
    for j, joker in enumerate(jokers):
        if joker.debuff:
            continue
        pred_owner = joker
        if joker.center_key in _COPY_JOKERS:
            assert resolutions is not None
            res = resolutions[j]
            if not res.active:
                continue
            pred_owner = jokers[res.target_index]
        pred = _PREDICATES.get(pred_owner.center_key)
        if pred is not None:
            active.append((j, pred_owner, pred))
    if not active:
        return out

    for i, card in enumerate(hand):
        if card.base is None or card.facing == "back" or card.debuff:
            continue
        for j, joker, pred in active:
            scored, held = pred(card, joker, gs, flags)
            out[i, j, 0] = scored
            out[i, j, 1] = held
    return out


def joker_center_key_ids(gs: dict[str, Any]) -> np.ndarray:
    """``(len(jokers),)`` int64 of frozen-vocabulary center-key ids —
    stored alongside the match matrix so the BC-time cross-attention can
    look up each matched joker's embedding + descriptor."""
    return np.array(
        [center_key_id(j.center_key) for j in gs.get("jokers", [])],
        dtype=np.int64,
    )

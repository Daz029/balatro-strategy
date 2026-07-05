"""Scoring-order sensitivity analysis and optimal play-order search.

Balatro scores played cards in *selection order* (see game.py
``_handle_play_hand``: the first index in ``card_indices`` is the leftmost
scored card), and a small set of effects make the total depend on that
order: position-identity jokers (Photograph reads the first scored face
card, Hanging Chad retriggers the first scored card), per-card xmult
sources interleaving with additive mult (Glass cards, Polychrome editions,
xmult jokers), and shared-RNG-stream effects (Lucky Card).

This module owns the "does order matter, and which order is best" logic so
that both the offline solver (``scripts/hand_solver.py``) and the RL
hand-play environment (``jackdaw/env/hand_play_gym.py``) share one tested
implementation. The RL design deliberately keeps card *ordering* out of the
agent's action space (the agent picks a subset; ordering is a mechanical
optimization the engine can do exactly) -- see the /grilling session
recorded in CLAUDE.md's ante-play track.

Originally written and tested inside ``scripts/hand_solver.py``; the tests
in ``tests/scripts/test_hand_solver_order_sensitivity.py`` and
``tests/scripts/test_hand_solver_permutation_coverage.py`` still exercise
these functions through ``hand_solver``'s re-imports, and
``tests/engine/test_play_ordering.py`` covers the env-facing
``best_play_order`` entry point.
"""

from __future__ import annotations

import itertools
from typing import Any

from jackdaw.engine.card import Card
from jackdaw.engine.hand_levels import HandLevels, HandState
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand

MAX_PERMUTATIONS = 24  # cap on order-permutations tried per value eval

# ---------------------------------------------------------------------------
# Fast cloning for hypothetical-evaluation hot paths
# ---------------------------------------------------------------------------
#
# ``score_hand`` mutates state in place (hand_levels.record_play/level_up;
# Vampire strips card enhancements; Wee Joker/Lucky Cat accumulate onto
# ability["extra"]/["x_mult"]; Lucky Card sets a "lucky_trigger" flag), so a
# hypothetical evaluation must never touch the caller's real objects. Each
# type's mutable surface is small and known, so purpose-built clones give
# the same isolation as copy.deepcopy for a fraction of the cost (generic
# deepcopy was ~73% of solver runtime before this existed). Mutation safety
# is pinned by tests/scripts/test_hand_solver_mutation.py.
#
#   - Card.base (CardBase): never mutated by scoring (`times_played` is only
#     incremented in game.py's post-scoring bookkeeping) -- shared by
#     reference.
#   - Card.edition: never mutated in place by scoring/joker code (only
#     reassigned via Card.set_edition, which scoring never calls) -- shared
#     by reference.
#   - Card.ability: mutated extensively -- copied one level deep (covers
#     every observed case, including nested dicts like
#     ability["extra"]["chips"]).
#   - HandLevels._hands: mutated via record_play/level_up -- each HandState
#     (a flat dataclass of primitives) is reconstructed fresh.
#   - PseudoRandom._state: mutated via .seed() (Lucky Card et al.) -- all
#     values are floats/strings, so a shallow dict copy fully suffices.


def fast_clone_ability(ability: dict) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in ability.items()}


def fast_clone_card(c: Card) -> Card:
    new = Card.__new__(Card)
    new.__dict__.update(c.__dict__)
    new.ability = fast_clone_ability(c.ability)
    return new


def fast_clone_hand_levels(hl: HandLevels) -> HandLevels:
    clone = HandLevels.__new__(HandLevels)
    clone._hands = {
        ht: HandState(hs.level, hs.chips, hs.mult, hs.played, hs.played_this_round, hs.visible)
        for ht, hs in hl._hands.items()
    }
    return clone


def fast_clone_rng(rng: PseudoRandom) -> PseudoRandom:
    clone = PseudoRandom.__new__(PseudoRandom)
    clone._state = dict(rng._state)
    return clone


# ---------------------------------------------------------------------------
# Order-sensitivity detection
# ---------------------------------------------------------------------------
#
# ``score_hand``'s per-card loop (scoring.py: `_apply_individual_joker_effects`)
# accumulates `mult` *sequentially* across scored cards: `mult += additive`
# then `mult *= multiplicative`, once per scored card in order. Pure-additive
# scoring is always commutative (order can never change a sum), so scoring
# order can only matter when at least one per-card MULTIPLICATIVE (xmult)
# source is present -- interleaving that with additive mult elsewhere in the
# sequence is what makes `(m + a) * x != (m * x) + a` in general. Joker
# *list* order (as opposed to card order) has the same "only matters with
# xmult" shape in Phase 9's `joker_main` loop, but nothing here permutes
# joker order (it's fixed by the real joker board). Position-copying edge
# cases (Blueprint/Brainstorm) are out of scope for now.

XMULT_JOKER_KEYS = frozenset(
    {"j_photograph", "j_bloodstone", "j_ancient", "j_triboulet", "j_idol"}
)

# Hanging Chad gives extra retriggers to whichever card is scored *first* --
# an identity effect, not an xmult one, so it stays order-sensitive even in
# an all-additive hand (a differently-valued card becoming "first" changes
# how much its (possibly purely additive) effect gets amplified).
IDENTITY_ORDER_SENSITIVE_JOKER_KEYS = frozenset({"j_hanging_chad"})


def card_has_xmult(c: Card) -> bool:
    """Whether *c*'s own enhancement/edition multiplies `mult` when scored.

    Glass enhancement (`ability["x_mult"]`) and Polychrome edition
    (`edition["x_mult"]`) are the two card-level sources; see
    `Card.get_chip_x_mult`/`Card.get_edition`.
    """
    if c.ability.get("x_mult", 1) > 1:
        return True
    edition = c.edition or {}
    return bool(edition.get("x_mult"))


def count_order_sensitive_sources(played_cards: list[Card], jokers: list[Card]) -> int:
    """Count of interior-order-sensitive contributors among `played_cards`.

    Used to decide whether the (first, last)-covering permutation set (see
    `first_last_covering_permutations`) is sufficient, or whether interior
    positions can interact and a full permutation search is required.

    Identity-only effects (`IDENTITY_ORDER_SENSITIVE_JOKER_KEYS`, i.e.
    Hanging Chad) are deliberately excluded from this count -- they depend
    solely on which card is scored first, a dimension the covering set
    already explores exhaustively regardless of how many such jokers are
    present.
    """
    joker_keys = {getattr(j, "center_key", None) for j in jokers}
    count = sum(1 for c in played_cards if card_has_xmult(c))
    count += sum(1 for c in played_cards if c.ability.get("effect") == "Lucky Card")
    if joker_keys & XMULT_JOKER_KEYS:
        count += 1
    return count


def first_last_covering_permutations(cards: list[Card]) -> list[tuple[Card, ...]]:
    """Deterministic permutation set covering every (first, last) ordered
    pair of `cards` exactly once, in `len(cards) * (len(cards) - 1)`
    permutations rather than `len(cards)!`.

    For each of the `n` choices of first card (via whole-sequence rotation),
    the remaining `n - 1` cards are cycled through their `n - 1` rotations,
    which places each of them in the last slot exactly once. Interior
    positions only ever see this single rotation pattern, not all `(n-2)!`
    arrangements of the interior -- exact when there is at most one
    order-sensitive contributor (see `count_order_sensitive_sources`), not
    guaranteed exact with two or more.
    """
    n = len(cards)
    perms: list[tuple[Card, ...]] = []
    for outer in range(n):
        rotated = cards[-outer:] + cards[:-outer] if outer else list(cards)
        first, rest = rotated[0], rotated[1:]
        for inner in range(len(rest)):
            rest_rotated = rest[-inner:] + rest[:-inner] if inner else list(rest)
            perms.append(tuple([first, *rest_rotated]))
    return perms


def needs_permutation_search(played_cards: list[Card], jokers: list[Card]) -> bool:
    """Whether permuting `played_cards`' scoring order can change the total.

    True when:
      - any played card carries the Lucky Card enhancement -- rolls a shared
        RNG stream (`rng.random("lucky_mult")`) once per qualifying scored
        card, so which card's roll succeeds depends on scoring order; or
      - Hanging Chad is active (identity-based, see above); or
      - any per-card xmult source is present at all (a joker in
        `XMULT_JOKER_KEYS`, a Glass-enhanced card, or a Polychrome-edition
        card) -- xmult interleaving with additive mult elsewhere in the
        sequence is order-sensitive, and checking for xmult presence alone
        is a small, easy-to-verify surface compared to also having to
        enumerate every additive-mult source.
    """
    joker_keys = {getattr(j, "center_key", None) for j in jokers}

    if any(c.ability.get("effect") == "Lucky Card" for c in played_cards):
        return True
    if joker_keys & IDENTITY_ORDER_SENSITIVE_JOKER_KEYS:
        return True
    if joker_keys & XMULT_JOKER_KEYS:
        return True
    return any(card_has_xmult(c) for c in played_cards)


def candidate_orderings(
    played_cards: list[Card], jokers: list[Card]
) -> list[tuple[Card, ...]]:
    """The orderings worth scoring for `played_cards` on this joker board.

    Returns a single ordering (the given one) when order provably can't
    matter; the (first, last) covering set when exactly one order-sensitive
    contributor is present; and full enumeration when contributors can
    interact at interior positions (or the hand is small enough that full
    enumeration is within ``MAX_PERMUTATIONS`` anyway).
    """
    n = len(played_cards)
    if n <= 1 or not needs_permutation_search(played_cards, jokers):
        return [tuple(played_cards)]
    all_perms = list(itertools.permutations(played_cards))
    if len(all_perms) > MAX_PERMUTATIONS:
        if count_order_sensitive_sources(played_cards, jokers) > 1:
            # Two+ order-sensitive contributors can interact at interior
            # positions, not just at the first/last slots -- fall back
            # to full enumeration rather than risk missing the true max.
            return all_perms
        # Exactly one order-sensitive contributor (or none beyond an
        # identity effect): its optimum is always achievable by some
        # choice of first and/or last slot, so the (first, last)
        # covering set is exact here without needing all n!.
        return first_last_covering_permutations(played_cards)
    return all_perms


def best_play_order(
    played_cards: list[Card],
    held_cards: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind: Any,
    rng: PseudoRandom,
    game_state: dict[str, Any] | None = None,
    blind_chips: int = 0,
) -> tuple[Card, ...]:
    """Return `played_cards` reordered to maximize the scored total.

    Scores each candidate ordering against CLONED mutable state (the live
    game is never touched), so callers can submit the returned order to the
    real engine afterwards. When order can't matter (the overwhelmingly
    common case), returns the input order without scoring anything -- the
    fast path costs one ``needs_permutation_search`` check.
    """
    orderings = candidate_orderings(played_cards, jokers)
    if len(orderings) == 1:
        return orderings[0]

    best_order: tuple[Card, ...] | None = None
    best_total = float("-inf")
    for order in orderings:
        hl_copy = fast_clone_hand_levels(hand_levels)
        rng_copy = fast_clone_rng(rng)
        played_copy = [fast_clone_card(c) for c in order]
        held_copy = [fast_clone_card(c) for c in held_cards]
        jokers_copy = [fast_clone_card(j) for j in jokers]
        result = score_hand(
            played_copy,
            held_copy,
            jokers_copy,
            hl_copy,
            blind,
            rng_copy,
            game_state=game_state,
            blind_chips=blind_chips,
        )
        if result.total > best_total:
            best_total = result.total
            best_order = order
    assert best_order is not None
    return best_order

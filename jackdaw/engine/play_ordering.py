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

from jackdaw.engine.blind import Blind
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
#   - Blind.hands_used / .only_hand / .triggered: mutated by debuff_hand on
#     EVERY call, hypothetical or not (The Eye marks the hand type used, The
#     Mouth locks the first type it sees) -- score_hand has no "preview,
#     don't mutate" mode, so every hypothetical evaluation must run against
#     a throwaway clone or successive hypothetical evaluations corrupt each
#     other (confirmed: two independent hypothetical evals of the same hand
#     type under a shared Blind -- neither ever actually played -- would
#     otherwise see the second one incorrectly blocked). ``hands_used`` is a
#     dict, so it's the one field needing an explicit copy; everything else
#     is an immutable scalar (str/float/bool/None), safe by value via
#     ``__dict__.update``.


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


def fast_clone_blind(blind: Blind) -> Blind:
    clone = Blind.__new__(Blind)
    clone.__dict__.update(blind.__dict__)
    clone.hands_used = dict(blind.hands_used)
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


# ---------------------------------------------------------------------------
# Joker-list ordering (B3)
# ---------------------------------------------------------------------------
#
# Phase 9 of ``score_hand`` walks the joker list left to right, accumulating
# `mult += mult_mod` then `mult *= Xmult_mod` per joker, so joker LIST order
# changes the total whenever an x-mult contributor interleaves with additive
# mult: (m0 + m) * x >= m0 * x + m for x >= 1, with equality only when m == 0
# or x == 1. Vanilla exposes joker reordering as a free unrestricted action,
# so auto-ordering is vanilla-faithful; the RL design keeps reorder actions
# out of the agent's action space for the same reason card ordering is
# env-side (see module docstring). The per-card phase (Phase 7) runs jokers
# in the same list order inside each scored card's loop, so the same
# additive-before-xmult key benefits it too -- but the closed form is only
# PROVEN for the independent Phase-9 chain; the per-card interleaving is
# pinned empirically by the brute-force tests.
#
# Known approximations (accepted, documented):
#   - Baseball Card contributes x-mult at every OTHER uncommon joker's
#     position (the 9c joker-on-joker loop), so uncommon additive jokers
#     carry an x-mult rider when it's owned -- the binary sort ignores this
#     (second-order; the engine still scores whatever order we submit, so
#     labels stay honest).
#   - A joker mixing additive and x-mult at one position (e.g. a Foil/Holo
#     edition on an x-mult joker: the additive edition applies at 9a before
#     the joker's own effect) is classified by its x-mult component alone.
#     The exact closed form for mixed (add-then-multiply) blocks is a sort
#     by a*x/(x-1) descending, which needs per-joker magnitudes we don't
#     have without evaluation -- the binary partition is the locked design.
#   - Copy-joker editions couple placement to target adjacency: a copy
#     fires its OWN edition at its own slot, but its slot must sit
#     adjacent-left of its target, and `best_joker_order` only inserts
#     copies into the FIXED sorted base -- an order that serves a copy's
#     x-mult edition by moving its TARGET into the late block (Polychrome
#     Blueprint + early-additive target) is never proposed. Second-order
#     for the same reason as above; the in-blind-merge compile layer
#     closes it with exhaustive ordering search on Holo/Polychrome boards
#     (Foil is chips-only, Negative non-scoring -- order-insensitive;
#     exact algorithm deliberately TBD at the merge, edge cases abound --
#     see the ordering_objective v2 record in
#     docs/post-regen-training-plan.md).

# Center keys whose handler can return Xmult_mod in the joker_main context
# (Phase 9b) -- i.e. jokers that MULTIPLY at their own list position. Kept
# hand-written for import cheapness; regenerated-and-compared from the
# jokers.py handler source by tests/engine/test_play_ordering.py, so drift
# against the engine dies in CI rather than silently mis-sorting.
MAIN_PHASE_XMULT_JOKER_KEYS = frozenset(
    {
        "j_acrobat",
        "j_blackboard",
        "j_caino",
        "j_campfire",
        "j_card_sharp",
        "j_cavendish",
        "j_constellation",
        "j_drivers_license",
        "j_duo",
        "j_family",
        "j_flower_pot",
        "j_glass",
        "j_hit_the_road",
        "j_hologram",
        "j_loyalty_card",
        "j_lucky_cat",
        "j_madness",
        "j_obelisk",
        "j_order",
        "j_ramen",
        "j_seeing_double",
        "j_steel_joker",
        "j_stencil",
        "j_throwback",
        "j_tribe",
        "j_trio",
        "j_vampire",
        "j_yorick",
    }
)

COPY_JOKER_KEYS = frozenset({"j_blueprint", "j_brainstorm"})

# With multiple copy jokers, placements are brute-forced as a full
# cross-product only up to this many total jokers (10 slots: 2 copies among
# 8 others = 90 candidate orderings, each one cheap score_hand); wider
# boards fall back to sequential-greedy placement.
MAX_JOKERS_FOR_COPY_BRUTE_FORCE = 10


def joker_multiplies_at_position(j: Card) -> bool:
    """Whether *j* contributes an x-mult AT ITS OWN list position: its main
    handler can return ``Xmult_mod``, or it carries a Polychrome edition
    (``x_mult_mod`` applies at Phase 9d, after the joker's own effect, at
    the joker's position)."""
    if getattr(j, "center_key", None) in MAIN_PHASE_XMULT_JOKER_KEYS:
        return True
    edition = j.get_edition() or {}
    return "x_mult_mod" in edition


def joker_order_matters(jokers: list[Card]) -> bool:
    """Fast gate: can reordering `jokers` change a score at all?

    True when a copy joker is owned (its target is adjacency-defined), or
    when an x-mult-position joker coexists with any other joker (the
    additive-vs-x interleaving case). Chips-only boards and single jokers
    are order-free."""
    if len(jokers) < 2:
        return False
    keys = {getattr(j, "center_key", None) for j in jokers}
    if keys & COPY_JOKER_KEYS:
        return True
    n_x = sum(1 for j in jokers if joker_multiplies_at_position(j))
    return 0 < n_x < len(jokers)


def sorted_joker_order(jokers: list[Card]) -> list[Card]:
    """Closed-form base ordering: stable partition with additive/neutral
    jokers before x-mult-position jokers, copy jokers placed by adjacency
    heuristic (no evaluation -- the context-free tier):

      - Brainstorm copies the LEFTMOST joker regardless of its own slot; its
        copied effect fires at its own position. After the partition the
        leftmost is additive whenever any additive joker exists, so an early
        slot (index 1, right after its target) is the safe placement.
      - Blueprint copies its RIGHT neighbor, so it goes immediately before
        the LAST non-copy joker -- an x-mult joker whenever one exists
        (doubling an x-mult dominates doubling an additive of comparable
        tier). Which x-mult is best needs evaluation; that precision lives
        in `best_joker_order`'s argmax, not here.
    """
    non_copy = [j for j in jokers if getattr(j, "center_key", None) not in COPY_JOKER_KEYS]
    ordered = sorted(non_copy, key=joker_multiplies_at_position)  # stable
    brainstorms = [j for j in jokers if getattr(j, "center_key", None) == "j_brainstorm"]
    blueprints = [j for j in jokers if getattr(j, "center_key", None) == "j_blueprint"]
    for b in brainstorms:
        ordered.insert(min(1, len(ordered)), b)
    for b in blueprints:
        ordered.insert(max(0, len(ordered) - 1), b)
    return ordered


def _cheap_play_value(
    played_cards: list[Card],
    held_cards: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind: Any,
    rng: PseudoRandom,
    game_state: dict[str, Any] | None,
    blind_chips: int,
    objective: Any = None,
) -> Any:
    """One fixed-order `score_hand` on full clones (including the joker
    list -- the fast_clone discipline extends to it: scaling jokers
    accumulate onto ability state on every hypothetical call).

    `objective` maps the ScoreResult to the value being argmaxed; None =
    raw score total. The h1-seam upgrade path (user call 2026-07-15): the
    double-agent env passes a money-aware objective so copy-joker
    placement can weigh score against dollars. The copyable money channel
    flows THROUGH scoring (Business Card's per-scored-card $, lucky-money
    rolls amplified by copied retrigger jokers), so it lands in
    ``ScoreResult.dollars_earned`` and the objective reads it off this
    eval directly; end-of-round payers (Golden Joker, Rocket, Egg...) are
    blueprint-INCOMPATIBLE (the engine's 29-joker blueprint_compat list)
    and never copyable, so their invisibility here costs nothing. Copy
    compatibility itself is engine-resolved: this eval runs the real
    Blueprint/Brainstorm handlers, compat guard included. Solver labels
    stay score-only either way -- loose label/env convergence accepted
    (PPO against the real game is the standing corrector)."""
    result = score_hand(
        [fast_clone_card(c) for c in played_cards],
        [fast_clone_card(c) for c in held_cards],
        [fast_clone_card(j) for j in jokers],
        fast_clone_hand_levels(hand_levels),
        fast_clone_blind(blind) if blind is not None else None,
        fast_clone_rng(rng),
        game_state=game_state,
        blind_chips=blind_chips,
    )
    # Any comparable value (floats, tie-break tuples) -- the argmax only
    # needs ordering, not arithmetic.
    return float(result.total) if objective is None else objective(result)


def best_joker_order(
    jokers: list[Card],
    played_cards: list[Card] | None = None,
    held_cards: list[Card] | None = None,
    hand_levels: HandLevels | None = None,
    blind: Any = None,
    rng: PseudoRandom | None = None,
    game_state: dict[str, Any] | None = None,
    blind_chips: int = 0,
    *,
    objective: Any = None,
) -> list[Card]:
    """Best scoring order for the joker LIST (a new list of the same Card
    objects; callers decide whether to write it back into the live state).

    Two tiers, matching the play-order design:

      - Context-free (no `played_cards`): the closed-form
        `sorted_joker_order` -- additive before x-mult, adjacency-heuristic
        copy placement. Used once per solver hand-turn and anywhere no
        candidate play exists yet (the MC future-hand tier).
      - With a candidate play AND copy joker(s) owned: argmax over ALL
        copy-joker placements into the sorted base order, each candidate
        scored by one cheap fixed-order evaluation. One copy joker = every
        insertion slot (Blueprint's target = right neighbor, so each slot IS
        a target choice; board size doesn't cap this -- 6+ jokers via
        negative editions just means more slots). Multiple copy jokers =
        FULL cross-product of ordered placements (user call 2026-07-15),
        up to ``MAX_JOKERS_FOR_COPY_BRUTE_FORCE`` total jokers -- e.g. two
        copies among 8 others is 90 candidates; beyond the cap it falls
        back to sequential-greedy placement (each copy argmaxed in turn).

    Without a copy joker the closed form needs no evaluation, so the play
    context is ignored. Never mutates its inputs; evaluation runs on clones.

    `objective` (keyword-only) re-targets the placement argmax away from
    raw score -- see `_cheap_play_value` for the money-aware upgrade path.
    It only affects the evaluated (copy-joker) tier; the closed-form sort
    is objective-free by construction.
    """
    base = sorted_joker_order(jokers)
    copy_jokers = [j for j in jokers if getattr(j, "center_key", None) in COPY_JOKER_KEYS]
    if (
        not copy_jokers
        or played_cards is None
        or hand_levels is None
        or rng is None
        or len(jokers) < 2
    ):
        return base

    held = held_cards or []
    # id()-based exclusion: duplicate jokers exist (two Blueprints via
    # negative editions), and value-equality would drop both when one was
    # meant -- the Erratic-deck bug class from best_immediate_play.
    copy_ids = {id(j) for j in copy_jokers}
    non_copy_order = [j for j in base if id(j) not in copy_ids]

    def evaluate(candidate: list[Card]) -> float:
        return _cheap_play_value(
            played_cards,
            held,
            candidate,
            hand_levels,
            blind,
            rng,
            game_state,
            blind_chips,
            objective=objective,
        )

    if len(jokers) <= MAX_JOKERS_FOR_COPY_BRUTE_FORCE:
        candidates = [non_copy_order]
        for copy_joker in copy_jokers:
            candidates = [
                seq[:slot] + [copy_joker] + seq[slot:]
                for seq in candidates
                for slot in range(len(seq) + 1)
            ]
        return max(candidates, key=evaluate)

    # Board too wide for the cross-product: place each copy joker greedily.
    order = non_copy_order
    for copy_joker in copy_jokers:
        order = max(
            (order[:slot] + [copy_joker] + order[slot:] for slot in range(len(order) + 1)),
            key=evaluate,
        )
    return order


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
        # blind clone (not just hand_levels/rng/cards): this loop scores
        # several candidate orderings before picking one, and score_hand
        # mutates history-dependent boss state (The Eye/The Mouth) on
        # EVERY call. Without this, evaluating a discarded ordering could
        # corrupt the real, live Blind (e.g. mark a hand type "used" that
        # was never actually played) before the caller re-executes the
        # chosen order through the real engine.
        hl_copy = fast_clone_hand_levels(hand_levels)
        rng_copy = fast_clone_rng(rng)
        blind_copy = fast_clone_blind(blind)
        played_copy = [fast_clone_card(c) for c in order]
        held_copy = [fast_clone_card(c) for c in held_cards]
        jokers_copy = [fast_clone_card(j) for j in jokers]
        result = score_hand(
            played_copy,
            held_copy,
            jokers_copy,
            hl_copy,
            blind_copy,
            rng_copy,
            game_state=game_state,
            blind_chips=blind_chips,
        )
        if result.total > best_total:
            best_total = result.total
            best_order = order
    assert best_order is not None
    return best_order

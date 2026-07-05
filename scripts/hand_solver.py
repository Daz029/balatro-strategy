"""In-blind hand/discard solver for jackdaw-balatro.

Design (see conversation for full rationale):

  1. TEMPLATES — a small, fixed, joker-agnostic set of target patterns
     (flush-by-suit, straight-by-window, rank-count groups). Joker
     awareness is delegated entirely to the real scoring engine
     (`score_hand`), never hardcoded here.

  2. HOLD CONSTRUCTION — for a given template, the optimal hold set is a
     direct filter (O(hand size)), not a search over 2^n subsets.

  3. REACHABILITY — probability of completing a template after discarding,
     computed exactly via the hypergeometric distribution over the known
     remaining-deck composition. No sampling.

  4. VALUE — for a *fixed* set of <=5 cards, the true value (given current
     jokers) is obtained by calling the engine's real `score_hand`. The
     permutation search over card order is skipped unless a genuinely
     order-sensitive effect is present (see `_needs_permutation_search`) --
     position-identity jokers (Photograph, Hanging Chad) or shared-RNG-
     stream effects (Bloodstone, Lucky Card). When it IS needed and the hand
     is a full 5 cards (120 orderings), a single order-sensitive contributor
     is handled exactly via `_first_last_covering_permutations` (20
     orderings covering every first/last pair) rather than all 120; two or
     more contributors fall back to full enumeration, since interior
     positions can then interact in ways the covering set doesn't explore
     (see `_count_order_sensitive_sources`).

  5. DISCARD DECISION — backward induction over the (small) number of
     discards left: compare "play now" vs "discard toward each template",
     weighting each template's representative value by its exact
     reachability probability.

KNOWN APPROXIMATIONS (flagged explicitly, not hidden):

  - Step 4's "value" for a *not-yet-drawn* completion uses a representative
    best-case completion of the template, not a full expectation over every
    possible draw. This is optimistic, not exact EV.
  - RNG-dependent joker effects (e.g. Lucky Card) are evaluated with a
    *cloned* RNG per hypothetical, so each evaluation is one sample of
    that randomness, not its true expectation.
  - Cross-hand state (jokers whose effect depends on what was played
    earlier *in the same blind*, e.g. Ride the Bus) is NOT handled here.
    This solver only reasons about a single hand-in-isolation. It is meant
    to be called as the leaf evaluator inside an outer per-blind DP that
    tracks that state; see NOTES at the bottom of the file.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from math import comb

from jackdaw.engine.card import Card
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.play_ordering import (
    MAX_PERMUTATIONS,  # noqa: F401 -- re-export for existing importers
    candidate_orderings,
)
from jackdaw.engine.play_ordering import (
    count_order_sensitive_sources as _count_order_sensitive_sources,  # noqa: F401
)
from jackdaw.engine.play_ordering import (
    fast_clone_blind as _fast_clone_blind,
)
from jackdaw.engine.play_ordering import (
    fast_clone_card as _fast_clone_card,
)
from jackdaw.engine.play_ordering import (
    fast_clone_hand_levels as _fast_clone_hand_levels,
)
from jackdaw.engine.play_ordering import (
    fast_clone_rng as _fast_clone_rng,
)
from jackdaw.engine.play_ordering import (
    first_last_covering_permutations as _first_last_covering_permutations,  # noqa: F401
)
from jackdaw.engine.play_ordering import (
    needs_permutation_search as _needs_permutation_search,  # noqa: F401
)
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import ScoreResult, score_hand

# NOTE: The fast-clone helpers, order-sensitivity detection, and covering-
# permutation construction that used to live here moved to
# `jackdaw/engine/play_ordering.py` so the RL hand-play environment can
# reuse them for env-side optimal ordering (the agent picks a subset;
# ordering is delegated to the engine -- see CLAUDE.md). They are
# re-imported above under their historical underscore names for the
# existing tests and any external callers.

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
RANK_ID = {r: i + 2 for i, r in enumerate(RANKS)}  # 2..14, Ace=14
SUITS = ["Spades", "Hearts", "Clubs", "Diamonds"]


# ---------------------------------------------------------------------------
# Deck accounting
# ---------------------------------------------------------------------------


@dataclass
class DeckComposition:
    """Counts of unseen cards by rank id and suit (from the draw pile only —
    the current hand is already seen, so it is excluded)."""

    by_rank: dict[int, int] = field(default_factory=dict)
    by_suit: dict[str, int] = field(default_factory=dict)
    by_rank_suit: dict[tuple[int, str], int] = field(default_factory=dict)
    total: int = 0

    @classmethod
    def from_deck(cls, deck_cards: list[Card]) -> DeckComposition:
        dc = cls()
        for c in deck_cards:
            if c.base is None:
                continue  # skip non playing-cards defensively
            rid = c.base.id
            suit = c.base.suit
            dc.by_rank[rid] = dc.by_rank.get(rid, 0) + 1
            dc.by_suit[suit] = dc.by_suit.get(suit, 0) + 1
            dc.by_rank_suit[(rid, suit)] = dc.by_rank_suit.get((rid, suit), 0) + 1
            dc.total += 1
        return dc

    def count_matching(self, predicate: Callable[[int, str], bool]) -> int:
        """Count unseen cards matching a (rank_id, suit) -> bool predicate."""
        return sum(
            n for (rid, suit), n in self.by_rank_suit.items() if predicate(rid, suit)
        )

    def without(self, cards: list[Card]) -> DeckComposition:
        """Return a new DeckComposition with the given cards' rank/suit
        counts removed -- used when recursing into a hypothetical future
        state where those cards have been drawn out of the deck."""
        dc = DeckComposition(
            by_rank=dict(self.by_rank),
            by_suit=dict(self.by_suit),
            by_rank_suit=dict(self.by_rank_suit),
            total=self.total,
        )
        for c in cards:
            if c.base is None:
                continue
            rid, suit = c.base.id, c.base.suit
            if dc.by_rank_suit.get((rid, suit), 0) <= 0:
                continue  # defensive: card wasn't in this composition
            dc.by_rank[rid] -= 1
            dc.by_suit[suit] -= 1
            dc.by_rank_suit[(rid, suit)] -= 1
            dc.total -= 1
        return dc


def multivariate_cover_probability(
    population: int, bucket_sizes: list[int], draws: int
) -> float:
    """P(every bucket gets >= 1 card in the draw), drawing `draws` cards
    without replacement from a population of `population` cards partitioned
    into len(bucket_sizes) *distinct required* categories (plus an implicit
    remainder category of everything else, which is unconstrained).

    This is the correct model for straights: you need one card of EACH of
    the still-missing ranks, not just "some count of window-matching
    cards" -- those are genuinely different categories, not interchangeable
    successes, so a flat hypergeometric threshold overcounts whenever a
    single rank is heavily duplicated (e.g. an Erratic-style deck with 10
    Kings) -- extra Kings can't stand in for a missing Queen.

    Computed exactly via inclusion-exclusion (cheap: at most 2^m subsets,
    m = number of missing ranks, <= 4 for a 5-card straight):

        P(cover all) = sum_{S subset of buckets} (-1)^|S| *
                        C(population - sum_{i in S} bucket_sizes[i], draws)
                        / C(population, draws)
    """
    total_ways = comb(population, draws)
    if total_ways == 0:
        return 0.0
    m = len(bucket_sizes)
    result = 0.0
    for r in range(m + 1):
        for subset in itertools.combinations(range(m), r):
            excluded = sum(bucket_sizes[i] for i in subset)
            remaining_pop = population - excluded
            term = comb(remaining_pop, draws) if draws <= remaining_pop else 0
            result += ((-1) ** r) * term
    return max(0.0, min(1.0, result / total_ways))


def hypergeometric_at_least_k(
    population: int, successes_in_pop: int, draws: int, k: int
) -> float:
    """P(>= k successes) drawing `draws` cards without replacement from a
    population of `population` cards containing `successes_in_pop` successes.

    NOTE: this treats successes as one homogeneous, interchangeable bucket.
    That's correct for flush templates (any matching-suit card helps
    equally) and rank-count templates (any card of the target rank helps
    equally), but it is NOT correct for straights, where the "successes"
    are actually several distinct required ranks that can't substitute for
    each other -- use `multivariate_cover_probability` for those instead.
    """
    if k <= 0:
        return 1.0
    if draws <= 0 or successes_in_pop <= 0:
        return 0.0
    if k > draws or k > successes_in_pop:
        # still possibly nonzero unless k > draws or k > successes_in_pop
        if k > successes_in_pop:
            return 0.0
    total_ways = comb(population, draws)
    if total_ways == 0:
        return 0.0
    p_fail = 0.0
    max_bad_draws = min(draws, k - 1)
    for i in range(0, max_bad_draws + 1):
        ways = comb(successes_in_pop, i) * comb(population - successes_in_pop, draws - i)
        p_fail += ways
    p_at_least_k = 1.0 - (p_fail / total_ways)
    return max(0.0, min(1.0, p_at_least_k))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@dataclass
class Template:
    name: str
    predicate: Callable[[Card], bool]  # does a held/drawn card match?
    needed: int  # cards required to "complete" this template (post 4-fingers etc.)


def _card_rank_id(c: Card) -> int | None:
    return c.base.id if c.base is not None else None


def _card_suit(c: Card) -> str | None:
    return c.base.suit if c.base is not None else None


def build_templates(
    hand: list[Card],
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
) -> list[Template]:
    """Fixed, joker-agnostic template set. `four_fingers` / `shortcut` should
    be derived from the live joker list before calling (both loosen size/gap
    requirements for straights & flushes) -- pass them in rather than
    re-deriving joker knowledge here.
    """
    templates: list[Template] = []
    flush_need = 4 if four_fingers else 5
    straight_need = 4 if four_fingers else 5

    # --- flush-by-suit (4 templates) ---
    for suit in SUITS:
        templates.append(
            Template(
                name=f"flush_{suit}",
                predicate=lambda c, s=suit: _card_suit(c) == s,
                needed=flush_need,
            )
        )

    # --- straight-by-window ---
    # ranks 2..14 (Ace high); Ace can also play low (id 14 treated as 1).
    windows = []
    lo, hi = 2, 14
    for start in range(lo, hi - straight_need + 2):
        windows.append(list(range(start, start + straight_need)))
    # ace-low window, e.g. [14(as1),2,3,4] for four_fingers, or [..,5] normal
    ace_low_window = [14] + list(range(2, 2 + straight_need - 1))
    windows.append(ace_low_window)

    gap = 1 if shortcut else 0
    for w in windows:
        wset = set(w)
        if gap:
            # widen predicate: allow ranks within the window OR adjacent gap-fillers
            # (approximation: shortcut's true gap logic lives in hand_eval;
            # here we just loosen membership by +/-1 rank of window bounds)
            lo_w, hi_w = min(w), max(w)
            wset = set(range(lo_w - gap, hi_w + gap + 1))
        templates.append(
            Template(
                name=f"straight_{min(w)}-{max(w)}",
                predicate=lambda c, ws=wset: _card_rank_id(c) in ws,
                needed=straight_need,
            )
        )

    # --- rank-count groups (pairs/trips/quads/five-of-a-kind) ---
    # Only worth proposing for ranks actually present in hand (grouping around
    # a rank with zero current copies is never better than grouping around
    # one you already hold).
    ranks_in_hand = {_card_rank_id(c) for c in hand if c.base is not None}
    for rid in ranks_in_hand:
        for needed, label in [(2, "pair"), (3, "trips"), (4, "quads")]:
            templates.append(
                Template(
                    name=f"{label}_{rid}",
                    predicate=lambda c, r=rid: _card_rank_id(c) == r,
                    needed=needed,
                )
            )

    return templates


def construct_hold(hand: list[Card], template: Template) -> tuple[list[Card], list[Card]]:
    """Direct filter construction: O(hand size), no search.
    Returns (hold, discard).

    For straight templates specifically, a duplicate rank within the target
    window is dead weight -- a straight uses at most one card per rank, so
    keeping a second copy wastes a hold slot and lowers redraw odds for no
    benefit. Rank-count templates (pair/trips/quads) are the opposite case
    -- duplicates ARE the point there -- so dedup only applies to straights.
    Flush templates never have this problem (same-suit cards are never
    "duplicates" of each other since they differ in rank).
    """
    matches = [c for c in hand if template.predicate(c)]
    non_matches = [c for c in hand if not template.predicate(c)]

    if template.name.startswith("straight_"):
        best_per_rank: dict[int, Card] = {}
        redundant: list[Card] = []
        for c in matches:
            rid = _card_rank_id(c)
            if rid is None:
                redundant.append(c)
                continue
            current = best_per_rank.get(rid)
            if current is None:
                best_per_rank[rid] = c
            else:
                # keep the higher base chip value as the representative;
                # the other copy is redundant for this straight and should
                # be freed up for discard/redraw instead.
                cur_val = current.base.nominal if current.base else 0
                new_val = c.base.nominal if c.base else 0
                if new_val > cur_val:
                    redundant.append(current)
                    best_per_rank[rid] = c
                else:
                    redundant.append(c)
        hold = list(best_per_rank.values())
        discard = non_matches + redundant
        return hold, discard

    return matches, non_matches


# The engine (and real Balatro) cap a discard at 5 selected cards
# (game.py::_handle_discard raises above 5). Template construction can
# produce more non-matches than that (e.g. 2 flush cards held in an 8-card
# hand -> 6 non-matches), and labeling those as one discard action produced
# unexecutable demonstrations AND optimistic reachability math (6-8
# replacement draws where the game allows at most 5) -- found when BC
# training hit real stage-1 data, 8.3% of labels affected.
DISCARD_LIMIT = 5


def _keep_priority(c: Card) -> tuple[int, int]:
    """Sort key for which excess non-matches to KEEP in hand when a
    template's discard set exceeds DISCARD_LIMIT (higher = keep). Enhanced
    cards (steel/glass/gold/bonus/...) carry value beyond their rank;
    otherwise prefer keeping higher-nominal cards as fallback material.
    A heuristic, flagged as such -- the exact answer would require
    searching over which subset to keep."""
    enhanced = c.center_key not in (None, "", "c_base")
    nominal = c.base.nominal if c.base else 0
    return (1 if enhanced else 0, nominal)


def cap_discard(discard: list[Card]) -> tuple[list[Card], list[Card]]:
    """Split a template's non-matches into (to_discard, kept_in_hand) with
    ``len(to_discard) <= DISCARD_LIMIT``. Reachability must use
    ``len(to_discard)`` as the draw count, and hand reconstruction must
    carry ``kept_in_hand`` forward -- those cards stay in the hand."""
    if len(discard) <= DISCARD_LIMIT:
        return list(discard), []
    ranked = sorted(discard, key=_keep_priority)  # discard-first ordering
    return ranked[:DISCARD_LIMIT], ranked[DISCARD_LIMIT:]


# ---------------------------------------------------------------------------
# Value evaluation (calls the real engine)
# ---------------------------------------------------------------------------


def evaluate_value(
    played_cards: list[Card],
    held_cards: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    game_state: dict | None = None,
    blind_chips: int = 0,
    *,
    search_orderings: bool = True,
) -> ScoreResult:
    """Max score over card-order permutations of `played_cards`, using
    CLONED mutable state so the live run is never touched by a hypothetical
    evaluation. This is the only place that talks to the real scoring
    engine -- all joker-specific knowledge lives here implicitly.

    ``search_orderings=False`` scores the given order only. Reserved for
    consumers whose output is already a coarse approximation (the future-
    hand MC sampler): paying a 6-20x permutation search to find the exact
    best ordering of a *hypothetical* hand is precision the estimate can't
    use. Exact consumers (labels, current-turn decisions) must leave it on.
    """
    if search_orderings:
        orderings = candidate_orderings(played_cards, jokers)
    else:
        orderings = [tuple(played_cards)]
    best: ScoreResult | None = None
    for order in orderings:
        # Fast clones, not copy.deepcopy -- see the module-level comment
        # above _fast_clone_card for why this is safe (score_hand mutates
        # card/joker/hand_levels/rng/blind state in place, so isolation
        # between trials is required either way; the fast clones just do it
        # without generic deepcopy's reflection overhead). blind ALSO needs
        # a clone here (not just hand_levels/rng/cards): under a
        # history-dependent boss (The Eye/The Mouth), score_hand mutates
        # blind.hands_used/only_hand on every call regardless of whether
        # this is the chosen play -- without cloning, two purely
        # hypothetical evaluations of the same hand type would corrupt each
        # other (the second incorrectly reads as already-used/mismatched).
        hl_copy = _fast_clone_hand_levels(hand_levels)
        rng_copy = _fast_clone_rng(rng)
        blind_copy = _fast_clone_blind(blind)
        played_copy = [_fast_clone_card(c) for c in order]
        held_copy = [_fast_clone_card(c) for c in held_cards]
        jokers_copy = [_fast_clone_card(j) for j in jokers]
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
        if best is None or result.total > best.total:
            best = result
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------


@dataclass
class DiscardChoice:
    action: str  # "play" or "discard"
    template_name: str | None
    hold: list[Card]
    discard: list[Card]
    expected_value: float
    reach_probability: float | None  # None for "play now"


def best_immediate_play(
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    game_state: dict | None = None,
    blind_chips: int = 0,
    *,
    search_orderings: bool = True,
) -> tuple[list[Card], ScoreResult]:
    """No discards being considered -- brute force over all non-empty
    subsets of size <=5 (cheap: C(8,5)=56 worst case).

    `search_orderings` -- see `evaluate_value`; forwarded per subset."""
    best_subset: list[Card] | None = None
    best_result: ScoreResult | None = None
    n = len(hand)
    for size in range(1, min(5, n) + 1):
        for combo in itertools.combinations(hand, size):
            # Identity-based, not `c not in combo` (value-equality): Card is
            # a plain @dataclass, so two genuinely distinct cards that
            # happen to have identical field values (e.g. an Erratic deck's
            # duplicate rank/suit cards) would otherwise BOTH match a single
            # `combo` membership check and get excluded from `held` --
            # undercounting it by one, which corrupts held-card-count-based
            # joker scoring.
            combo_ids = {id(c) for c in combo}
            held = [c for c in hand if id(c) not in combo_ids]
            result = evaluate_value(
                list(combo), held, jokers, hand_levels, blind, rng, game_state, blind_chips,
                search_orderings=search_orderings,
            )
            if best_result is None or result.total > best_result.total:
                best_result = result
                best_subset = list(combo)
    assert best_subset is not None and best_result is not None
    return best_subset, best_result


def solve_discard_decision(
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    deck: DeckComposition,
    discards_left: int,
    game_state: dict | None = None,
    blind_chips: int = 0,
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
) -> DiscardChoice:
    """One-step decision: compare playing now vs. discarding toward each
    template. Does NOT recurse across multiple discards left (see
    solve_blind_dp below for that) -- this is the single-decision building
    block.
    """
    # Option A: play immediately.
    play_subset, play_result = best_immediate_play(
        hand, jokers, hand_levels, blind, rng, game_state, blind_chips
    )
    best_choice = DiscardChoice(
        action="play",
        template_name=None,
        hold=play_subset,
        discard=[],
        expected_value=play_result.total,
        reach_probability=None,
    )

    if discards_left <= 0:
        return best_choice

    templates = build_templates(hand, four_fingers=four_fingers, shortcut=shortcut)

    for template in templates:
        hold, discard = construct_hold(hand, template)
        already_have = len(hold)
        still_needed = max(0, template.needed - already_have)
        discard, kept = cap_discard(discard)  # engine caps a discard at 5 cards
        draws = len(discard)  # cards redrawn = cards discarded (hand size held constant)

        if still_needed == 0:
            # already completed -- this degenerates to "play now" on this
            # subset, evaluate directly (no probability discount needed)
            result = evaluate_value(
                hold[:5], [], jokers, hand_levels, blind, rng, game_state, blind_chips
            )
            prob = 1.0
        else:
            if template.name.startswith("straight_"):
                # Straights need one card of EACH missing rank -- these are
                # distinct categories, not one interchangeable pool, so
                # duplicates of a rank you already hold (or of each other)
                # can't substitute for a different missing rank. Build one
                # bucket per missing rank and use exact multivariate
                # coverage, not the flat threshold formula.
                held_ranks = {_card_rank_id(c) for c in hold}
                window_ranks = {
                    rid
                    for rid in RANK_ID.values()
                    if template.predicate(_FakeCard(rid, SUITS[0]))
                }
                missing_ranks = sorted(window_ranks - held_ranks)
                # cap at still_needed in case widened (shortcut) windows
                # produced more "missing" ranks than are actually required
                missing_ranks = missing_ranks[:still_needed]
                bucket_sizes = [
                    deck.by_rank.get(rid, 0) for rid in missing_ranks
                ]
                if any(b == 0 for b in bucket_sizes):
                    continue  # a required rank is fully exhausted -- unreachable
                prob = multivariate_cover_probability(deck.total, bucket_sizes, draws)
            else:
                # Flush / rank-count templates: successes are genuinely
                # interchangeable (any matching-suit card, or any card of
                # the target rank, helps equally), so the flat threshold
                # formula is exact here.
                successes_in_pop = deck.count_matching(
                    lambda rid, suit, t=template: t.predicate(_FakeCard(rid, suit))
                )
                prob = hypergeometric_at_least_k(
                    deck.total, successes_in_pop, draws, still_needed
                )
            if prob <= 0.0:
                continue
            # Representative "best completion": take the held cards plus the
            # `still_needed` highest-value unseen matching cards as a stand-in
            # for the actual (unknown) draw. This is the approximation
            # flagged at the top of the file -- optimistic, not exact EV.
            completion_cards = _best_completion_cards(deck, template, still_needed)
            hypothetical_played = (hold + completion_cards)[:5]
            result = evaluate_value(
                hypothetical_played, [], jokers, hand_levels, blind, rng, game_state, blind_chips
            )

        ev = prob * result.total
        if ev > best_choice.expected_value:
            best_choice = DiscardChoice(
                action="discard",
                template_name=template.name,
                hold=hold + kept,
                discard=discard,
                expected_value=ev,
                reach_probability=prob,
            )

    return best_choice


# ---------------------------------------------------------------------------
# Helpers for the "representative completion" approximation
# ---------------------------------------------------------------------------


class _FakeCard:
    """Minimal stand-in so Template.predicate (which expects a Card with
    .base.id / .base.suit) can be evaluated against raw (rank_id, suit)
    pairs pulled from DeckComposition, without constructing real Card
    objects just to check membership."""

    class _B:
        def __init__(self, rid, suit):
            self.id = rid
            self.suit = suit

    def __init__(self, rid, suit):
        self.base = self._B(rid, suit)


def _best_completion_cards(
    deck: DeckComposition, template: Template, n_needed: int
) -> list[Card]:
    """Construct `n_needed` real Card objects representing a plausible best
    completion of `template` from the unseen deck, for use as the value
    evaluator's input. Picks the highest-chip matching rank(s) available.
    NOTE: this does not attempt to model the *actual* draw RNG -- it's a
    representative best-case stand-in per the documented approximation.
    """
    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank, Suit

    matches = [
        (rid, suit)
        for (rid, suit), n in deck.by_rank_suit.items()
        if n > 0 and template.predicate(_FakeCard(rid, suit))
    ]
    # prefer higher rank ids (higher chip nominal) as the representative pick
    matches.sort(key=lambda rs: rs[0], reverse=True)
    id_to_rank = {v: k for k, v in RANK_ID.items()}
    picked: list[Card] = []
    for rid, suit in matches[:n_needed]:
        rank_name = id_to_rank.get(rid, "Ace")
        picked.append(create_playing_card(Suit(suit), Rank(rank_name)))
    return picked


def _representative_miss_cards(
    deck: DeckComposition, template: Template, n_needed: int
) -> list[Card]:
    """Representative cards for a FAILED completion draw: unseen cards that
    do NOT match the template, lowest-chip first as a pessimistic floor
    (consistent with the conservative-bias framing used elsewhere in this
    file). Like `_best_completion_cards`, this is a stand-in for an unknown
    real draw, not an attempt to model the true miss distribution exactly.
    """
    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank, Suit

    non_matches = [
        (rid, suit)
        for (rid, suit), n in deck.by_rank_suit.items()
        if n > 0 and not template.predicate(_FakeCard(rid, suit))
    ]
    non_matches.sort(key=lambda rs: rs[0])  # lowest rank first
    id_to_rank = {v: k for k, v in RANK_ID.items()}
    picked: list[Card] = []
    for rid, suit in non_matches[:n_needed]:
        rank_name = id_to_rank.get(rid, "Ace")
        picked.append(create_playing_card(Suit(suit), Rank(rank_name)))
    return picked


def _fill_hand_to_size(
    deck: DeckComposition, hold: list[Card], priority_cards: list[Card], target_size: int
) -> tuple[list[Card], list[Card]]:
    """A discard refills the WHOLE hand back to its original size, not just
    to 5 cards -- the template only dictates `priority_cards` (the specific
    completion or miss cards), the remaining redraw slots get generic
    filler so the recursive call receives a correctly-sized hand to make
    its next decision from. Filler card identity is itself an
    approximation (highest-remaining-rank, arbitrary but deterministic) --
    flagged rather than hidden, same spirit as the other representative-
    card helpers in this file.

    Returns (full_hand, all_newly_drawn_cards) -- the latter is what should
    be removed from the deck via `DeckComposition.without()` for the
    recursive call (discarded hand cards are NOT removed from the deck --
    they never came from it).
    """
    from collections import Counter

    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank, Suit

    current = hold + priority_cards
    needed = target_size - len(current)
    filler: list[Card] = []
    if needed > 0:
        used = Counter((c.base.id, c.base.suit) for c in priority_cards if c.base)
        candidates: list[tuple[int, str]] = []
        for (rid, suit), n in deck.by_rank_suit.items():
            avail = n - used.get((rid, suit), 0)
            candidates.extend([(rid, suit)] * max(0, avail))
        candidates.sort(key=lambda rs: rs[0], reverse=True)
        id_to_rank = {v: k for k, v in RANK_ID.items()}
        for rid, suit in candidates[:needed]:
            filler.append(create_playing_card(Suit(suit), Rank(id_to_rank.get(rid, "Ace"))))
    return current + filler, priority_cards + filler



# ---------------------------------------------------------------------------
# Outer per-blind DP -- objective is P(clear the blind), not raw EV
# ---------------------------------------------------------------------------
#
# Two genuinely different kinds of transition happen in a blind, and they
# get different treatment:
#
#   DISCARD (within a hand-turn): conditioned. You choose a target, so the
#   outcome is a clean hit/miss relative to that target -- computed EXACTLY
#   via the reachability math above (hypergeometric / multivariate
#   coverage). This recurses cheaply because discards_left is small
#   (<=3ish), giving a full small decision tree, not an approximation.
#
#   PLAY -> next hand-turn: unconditioned. There is no target yet for a
#   hand you haven't seen, so there's no hit/miss structure to exploit --
#   it's a genuinely fresh draw from the deck. This is approximated via
#   Monte Carlo sampling (`estimate_future_hand_distribution` /
#   `prob_clear_given_future`), which is the ONLY place in this file that's
#   not doing exact reachability math, and it's approximate for a
#   structural reason (a real unknown), not for convenience.
#
# Recursing on ~40 templates at every discard-chain node would be
# expensive (see conversation), so template SELECTION is split into a
# cheap pass and an expensive pass:
#   - cheap pass: rank all templates using `score_hand_base` (jokerless,
#     so also order-invariant -- no permutation search needed) purely to
#     shortlist the top-K candidates worth exploring further.
#   - expensive pass: only the shortlisted candidates get real,
#     joker-aware `evaluate_value` calls and get recursed into.
# This can misrank when a joker specifically inverts which hand-type is
# good (e.g. rewards High Card heavily) -- a jokerless ranking pass would
# never surface that as a candidate. Mitigate by keeping top-K generous
# (default 4) rather than top-1.


def rank_templates_cheaply(
    hand: list[Card],
    deck: DeckComposition,
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    top_k: int = 4,
) -> list[tuple[Template, list[Card], list[Card], list[Card], float, float, int]]:
    """Returns up to `top_k` (template, hold, kept, discard, p_reach,
    cheap_value, still_needed) tuples, ranked by p_reach * cheap_value
    using the jokerless `score_hand_base` (no engine joker loop, no
    permutation search -- safe because base scoring is order-invariant
    without jokers).

    `hold` contains only template-MATCHING cards (still_needed/eval math
    depends on that); `kept` is excess non-matches retained in hand because
    the discard is capped at DISCARD_LIMIT -- callers reconstructing the
    post-discard hand must include them.
    """
    from jackdaw.engine.scoring import score_hand_base

    templates = build_templates(hand, four_fingers=four_fingers, shortcut=shortcut)
    scored: list[tuple[Template, list[Card], list[Card], list[Card], float, float, int]] = []

    for template in templates:
        hold, discard = construct_hold(hand, template)
        still_needed = max(0, template.needed - len(hold))
        discard, kept = cap_discard(discard)
        draws = len(discard)

        if still_needed == 0:
            p_reach = 1.0
            eval_cards = hold[:5]
        elif template.name.startswith("straight_"):
            held_ranks = {_card_rank_id(c) for c in hold}
            window_ranks = {
                rid for rid in RANK_ID.values() if template.predicate(_FakeCard(rid, SUITS[0]))
            }
            missing_ranks = sorted(window_ranks - held_ranks)[:still_needed]
            bucket_sizes = [deck.by_rank.get(rid, 0) for rid in missing_ranks]
            if any(b == 0 for b in bucket_sizes):
                continue
            p_reach = multivariate_cover_probability(deck.total, bucket_sizes, draws)
            if p_reach <= 0.0:
                continue
            eval_cards = (hold + _best_completion_cards(deck, template, still_needed))[:5]
        else:
            successes = deck.count_matching(
                lambda rid, suit, t=template: t.predicate(_FakeCard(rid, suit))
            )
            p_reach = hypergeometric_at_least_k(deck.total, successes, draws, still_needed)
            if p_reach <= 0.0:
                continue
            eval_cards = (hold + _best_completion_cards(deck, template, still_needed))[:5]

        # blind clone: see the note in evaluate_value -- this loop tries
        # many candidate templates per decision, and score_hand_base
        # mutates history-dependent boss state (The Eye/The Mouth) on every
        # hypothetical call, not just the eventually-chosen one.
        cheap_result = score_hand_base(
            eval_cards,
            [],
            _fast_clone_hand_levels(hand_levels),
            _fast_clone_blind(blind),
            _fast_clone_rng(rng),
        )
        scored.append((template, hold, kept, discard, p_reach, cheap_result.total, still_needed))

    scored.sort(key=lambda t: t[4] * t[5], reverse=True)
    return scored[:top_k]


def estimate_future_hand_distribution(
    deck: DeckComposition,
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    hand_size: int = 8,
    n_samples: int = 16,
    game_state: dict | None = None,
    blind_chips: int = 0,
    mc_seed: str | None = None,
) -> list[float]:
    """Monte Carlo stand-in for "what is a typical NEXT hand-turn worth,"
    drawn fresh from the given deck composition -- used only at the
    play -> next-hand-turn boundary, where there's no target yet to
    condition on (see module-level note above). Returns a list of sampled
    best-immediate-play values.

    `mc_seed` seeds a LOCAL Random for the hand draws; None falls back to
    an unseeded instance (the pre-seeding behavior). Callers that label
    training data must pass the example's seed so p_clear values are
    reproducible -- historically this used the unseeded global `random`,
    so labels carried run-to-run MC noise.

    Everything here stays inside the approximation boundary, deliberately
    (profiled 2026-07: this function was 75-97% of per-example solve time
    at 40 samples with full ordering search):
      - `search_orderings=False`: the exact best card ORDERING of a
        hypothetical hand is precision a 16-point empirical distribution
        can't use, and it cost a 6-20x multiplier on order-sensitive
        boards. Slightly understates sample values -- same direction as
        bias (a) below.
      - 16 samples (down from 40) describes the distribution nearly as
        well for a third of the cost.

    Known biases (all understate future value, so this errs pessimistic):
      (a) sampled hands are scored via `best_immediate_play` only -- they
          don't get to use their own discards. NOTE the strategic tilt
          this implies for in-blind play: discards spent THIS turn are
          fully credited by the exact recursion, discards BANKED for
          future turns are credited at zero, so labels lean toward
          spending discards early. Accepted because PPO fine-tunes against
          the real game where banking pays off (see CLAUDE.md); if eval
          shows PPO gains concentrated in high-discards/multi-hand states,
          consider a one-discard lookahead per sample here.
      (b) all samples are drawn from THIS deck snapshot; a real hand 3 or 4
          turns later draws from a further-depleted deck.
    """
    import random as _random

    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank
    from jackdaw.engine.data.enums import Suit as SuitEnum

    sampler = _random.Random(mc_seed) if mc_seed is not None else _random.Random()

    id_to_rank = {v: k for k, v in RANK_ID.items()}
    pool: list[tuple[int, str]] = []
    for (rid, suit), n in deck.by_rank_suit.items():
        pool.extend([(rid, suit)] * n)

    if len(pool) < hand_size:
        return [0.0]  # not enough cards left to even sample -- degenerate

    samples: list[float] = []
    for _ in range(n_samples):
        drawn = sampler.sample(pool, hand_size)
        hand_cards = [
            create_playing_card(SuitEnum(suit), Rank(id_to_rank[rid])) for rid, suit in drawn
        ]
        # evaluate_value/best_immediate_play deep-copy rng internally, so
        # passing the live rng here is safe -- no mutation of real state.
        _, result = best_immediate_play(
            hand_cards, jokers, hand_levels, blind, rng, game_state, blind_chips,
            search_orderings=False,
        )
        samples.append(result.total)
    return samples


def prob_clear_given_future(
    chips_gap: float, hands_remaining: int, future_samples: list[float], n_mc: int = 4000
) -> float:
    """P(sum of `hands_remaining` i.i.d. draws from future_samples >=
    chips_gap), estimated by Monte Carlo resampling. i.i.d. is itself an
    approximation (bias (b) above -- real hands aren't independent, since
    they share a depleting deck), acceptable over the small hands_remaining
    counts (<=4ish) typical within one blind.
    """
    if chips_gap <= 0:
        return 1.0
    if hands_remaining <= 0:
        return 0.0
    if not future_samples:
        return 0.0
    hits = 0
    n = len(future_samples)
    for _ in range(n_mc):
        total = 0.0
        for _ in range(hands_remaining):
            total += future_samples[_fast_rand_index(n)]
            if total >= chips_gap:
                hits += 1
                break
    return hits / n_mc


_rand_state = [12345]  # tiny local PRNG so this stays dependency-free & fast


def _fast_rand_index(n: int) -> int:
    _rand_state[0] = (1103515245 * _rand_state[0] + 12345) & 0x7FFFFFFF
    return _rand_state[0] % n


@dataclass
class AnteClearChoice:
    action: str  # "play" or "discard"
    template_name: str | None
    hold: list[Card]
    discard: list[Card]
    p_clear: float  # P(cumulative chips reach the blind requirement)
    immediate_value: float  # exact value if this specific outcome hits


def solve_hand_turn(
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    deck: DeckComposition,
    chips_needed: float,
    hands_left: int,
    discards_left: int,
    future_samples: list[float],
    game_state: dict | None = None,
    blind_chips: int = 0,
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    top_k: int = 4,
    hand_size: int | None = None,
) -> AnteClearChoice:
    """Recursive discard-chain solver for ONE hand-turn. At every node:
    compare playing now (deterministic, exact) against discarding toward
    each of the top-K cheaply-ranked templates (exact hit/miss
    probabilities, each branch recursed into with discards_left - 1).
    Bottoms out either at discards_left == 0 or by playing.

    `future_samples` must be precomputed ONCE per hand-turn (via
    `estimate_future_hand_distribution`) and passed through unchanged
    across the whole recursion -- it depends only on `hands_left`
    remaining after THIS hand-turn ends, not on where we are inside the
    discard chain, so recomputing it per node would be wasted work (see
    conversation). It is intentionally NOT recomputed against the
    shrinking deck as the discard chain progresses -- a documented
    simplification, since within one hand-turn the deck only changes by a
    handful of cards.
    """
    if hand_size is None:
        hand_size = len(hand)

    if chips_needed <= 0:
        subset, result = best_immediate_play(
            hand, jokers, hand_levels, blind, rng, game_state, blind_chips
        )
        return AnteClearChoice("play", None, subset, [], 1.0, result.total)

    def clear_prob(gap: float, remaining_hands: int) -> float:
        if gap <= 0:
            return 1.0
        if remaining_hands <= 0:
            return 0.0
        return prob_clear_given_future(gap, remaining_hands, future_samples)

    # --- play now: deterministic, exact ---
    play_subset, play_result = best_immediate_play(
        hand, jokers, hand_levels, blind, rng, game_state, blind_chips
    )
    best = AnteClearChoice(
        action="play",
        template_name=None,
        hold=play_subset,
        discard=[],
        p_clear=clear_prob(chips_needed - play_result.total, hands_left - 1),
        immediate_value=play_result.total,
    )

    if discards_left <= 0:
        return best

    candidates = rank_templates_cheaply(
        hand, deck, hand_levels, blind, rng,
        four_fingers=four_fingers, shortcut=shortcut, top_k=top_k,
    )

    for template, hold, kept, discard, p_reach, _cheap_val, still_needed in candidates:
        # Post-discard hand = template matches + capped-out non-matches
        # (kept in hand, see cap_discard) + replacement draws.
        base_hold = hold + kept
        if still_needed == 0:
            # already satisfied -- degenerate to a deterministic recursive
            # call on the (unchanged, already-complete) hold, refilled.
            hit_hand, hit_drawn = _fill_hand_to_size(deck, base_hold, [], hand_size)
            hit_deck = deck.without(hit_drawn)
            hit_choice = solve_hand_turn(
                hit_hand, jokers, hand_levels, blind, rng, hit_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, top_k=top_k, hand_size=hand_size,
            )
            p_clear = hit_choice.p_clear
        else:
            hit_priority = _best_completion_cards(deck, template, still_needed)
            hit_hand, hit_drawn = _fill_hand_to_size(deck, base_hold, hit_priority, hand_size)
            hit_deck = deck.without(hit_drawn)
            hit_choice = solve_hand_turn(
                hit_hand, jokers, hand_levels, blind, rng, hit_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, top_k=top_k, hand_size=hand_size,
            )

            miss_priority = _representative_miss_cards(deck, template, still_needed)
            miss_hand, miss_drawn = _fill_hand_to_size(deck, base_hold, miss_priority, hand_size)
            miss_deck = deck.without(miss_drawn)
            miss_choice = solve_hand_turn(
                miss_hand, jokers, hand_levels, blind, rng, miss_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, top_k=top_k, hand_size=hand_size,
            )
            p_clear = p_reach * hit_choice.p_clear + (1 - p_reach) * miss_choice.p_clear

        if p_clear > best.p_clear:
            best = AnteClearChoice(
                action="discard",
                template_name=template.name,
                hold=base_hold,
                discard=discard,
                p_clear=p_clear,
                immediate_value=hit_choice.immediate_value,
            )

    return best


def solve_hand_for_ante_clear(
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    deck: DeckComposition,
    chips_needed: float,
    hands_left: int,
    discards_left: int,
    game_state: dict | None = None,
    blind_chips: int = 0,
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    top_k: int = 4,
    mc_seed: str | None = None,
) -> AnteClearChoice:
    """Top-level entry point: computes the future-hand distribution once
    (for `hands_left - 1` downstream hand-turns) and kicks off the
    recursive discard-chain solve. This is what to call from a live
    decision point -- `solve_hand_turn` is the internal recursive worker.

    Pass `mc_seed` (e.g. the episode seed) whenever the output becomes a
    training label -- it makes the future-hand MC draws, and therefore
    p_clear, reproducible.
    """
    if mc_seed is not None:
        # `prob_clear_given_future`'s tiny LCG carries state across calls
        # within a process; reset it per solve or p_clear would depend on
        # how many solves this worker ran before this one.
        import zlib

        _rand_state[0] = zlib.crc32(mc_seed.encode()) & 0x7FFFFFFF or 12345
    future_samples = estimate_future_hand_distribution(
        deck, jokers, hand_levels, blind, rng, len(hand),
        game_state=game_state, blind_chips=blind_chips, mc_seed=mc_seed,
    )
    return solve_hand_turn(
        hand, jokers, hand_levels, blind, rng, deck, chips_needed, hands_left, discards_left,
        future_samples, game_state, blind_chips,
        four_fingers=four_fingers, shortcut=shortcut, top_k=top_k,
    )


"""
NOTES on wiring this into a full per-blind DP
------------------------------------------------
This module solves a SINGLE hand-in-isolation decision. Two things it does
NOT do, by design, because they belong one layer up:

1. Multi-hand-in-blind sequencing with joker state (Ride the Bus, Green
   Joker, etc.): wrap calls to `solve_discard_decision` /
   `best_immediate_play` inside an outer loop that carries whatever mutable
   joker ability-state exists across hands within the blind, updating it
   after each committed play using the REAL (non-cloned) hand_levels/rng/
   jokers -- only the search-time evaluations inside this module should use
   clones.

2. The score/cash exchange rate discussed for money-earning jokers: this
   module always maximizes raw `result.total` (score). If you want it to
   trade off against `result.dollars_earned` using a marginal-value-of-a-
   dollar curve from an outer shop-level critic, that weighting should be
   injected as a modified comparison key wherever `.total` is currently
   used to rank candidates (both in `best_immediate_play` and in the `ev >
   best_choice.expected_value` comparison above), e.g.:

       score_key = result.total + mv_dollar(current_money) * result.dollars_earned
"""
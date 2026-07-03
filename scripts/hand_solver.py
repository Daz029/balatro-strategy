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

import copy
import itertools
from dataclasses import dataclass, field
from math import comb
from typing import Callable

from jackdaw.engine.card import Card
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import ScoreResult, score_hand

MAX_PERMUTATIONS = 24  # cap on order-permutations tried per value eval

# `score_hand`'s per-card loop (scoring.py: `_apply_individual_joker_effects`)
# accumulates `mult` *sequentially* across scored cards: `mult += additive`
# then `mult *= multiplicative`, once per scored card in order. Pure-additive
# scoring is always commutative (order can never change a sum), so scoring
# order can only matter when at least one per-card MULTIPLICATIVE (xmult)
# source is present -- interleaving that with additive mult elsewhere in the
# sequence is what makes `(m + a) * x != (m * x) + a` in general. Joker
# *list* order (as opposed to card order) has the same "only matters with
# xmult" shape in Phase 9's `joker_main` loop, but the solver never permutes
# joker order (it's fixed by the real joker board), so that's not handled
# here. Position-copying edge cases (Blueprint/Brainstorm) are out of scope
# for now.
_XMULT_JOKER_KEYS = frozenset(
    {"j_photograph", "j_bloodstone", "j_ancient", "j_triboulet", "j_idol"}
)

# Hanging Chad gives extra retriggers to whichever card is scored *first* --
# an identity effect, not an xmult one, so it stays order-sensitive even in
# an all-additive hand (a differently-valued card becoming "first" changes
# how much its (possibly purely additive) effect gets amplified).
_IDENTITY_ORDER_SENSITIVE_JOKER_KEYS = frozenset({"j_hanging_chad"})


def _card_has_xmult(c: Card) -> bool:
    """Whether *c*'s own enhancement/edition multiplies `mult` when scored.

    Glass enhancement (`ability["x_mult"]`) and Polychrome edition
    (`edition["x_mult"]`) are the two card-level sources; see
    `Card.get_chip_x_mult`/`Card.get_edition`.
    """
    if c.ability.get("x_mult", 1) > 1:
        return True
    edition = c.edition or {}
    return bool(edition.get("x_mult"))


def _count_order_sensitive_sources(played_cards: list[Card], jokers: list[Card]) -> int:
    """Count of interior-order-sensitive contributors among `played_cards`.

    Used to decide whether the (first, last)-covering permutation set (see
    `_first_last_covering_permutations`) is sufficient, or whether interior
    positions can interact and a full permutation search is required.

    Identity-only effects (`_IDENTITY_ORDER_SENSITIVE_JOKER_KEYS`, i.e.
    Hanging Chad) are deliberately excluded from this count -- they depend
    solely on which card is scored first, a dimension the covering set
    already explores exhaustively regardless of how many such jokers are
    present.
    """
    joker_keys = {getattr(j, "center_key", None) for j in jokers}
    count = sum(1 for c in played_cards if _card_has_xmult(c))
    count += sum(1 for c in played_cards if c.ability.get("effect") == "Lucky Card")
    if joker_keys & _XMULT_JOKER_KEYS:
        count += 1
    return count


def _first_last_covering_permutations(cards: list[Card]) -> list[tuple[Card, ...]]:
    """Deterministic permutation set covering every (first, last) ordered
    pair of `cards` exactly once, in `len(cards) * (len(cards) - 1)`
    permutations rather than `len(cards)!`.

    For each of the `n` choices of first card (via whole-sequence rotation),
    the remaining `n - 1` cards are cycled through their `n - 1` rotations,
    which places each of them in the last slot exactly once. Interior
    positions only ever see this single rotation pattern, not all `(n-2)!`
    arrangements of the interior -- exact when there is at most one
    order-sensitive contributor (see `_count_order_sensitive_sources`), not
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


def _needs_permutation_search(played_cards: list[Card], jokers: list[Card]) -> bool:
    """Whether permuting `played_cards`' scoring order can change the total.

    True when:
      - any played card carries the Lucky Card enhancement -- rolls a shared
        RNG stream (`rng.random("lucky_mult")`) once per qualifying scored
        card, so which card's roll succeeds depends on scoring order; or
      - Hanging Chad is active (identity-based, see above); or
      - any per-card xmult source is present at all (a joker in
        `_XMULT_JOKER_KEYS`, a Glass-enhanced card, or a Polychrome-edition
        card) -- xmult interleaving with additive mult elsewhere in the
        sequence is order-sensitive, and checking for xmult presence alone
        is a small, easy-to-verify surface compared to also having to
        enumerate every additive-mult source.
    """
    joker_keys = {getattr(j, "center_key", None) for j in jokers}

    if any(c.ability.get("effect") == "Lucky Card" for c in played_cards):
        return True
    if joker_keys & _IDENTITY_ORDER_SENSITIVE_JOKER_KEYS:
        return True
    if joker_keys & _XMULT_JOKER_KEYS:
        return True
    return any(_card_has_xmult(c) for c in played_cards)


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
    def from_deck(cls, deck_cards: list[Card]) -> "DeckComposition":
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

    def without(self, cards: list[Card]) -> "DeckComposition":
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
    span = straight_need if not shortcut else straight_need  # gap handled below
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
) -> ScoreResult:
    """Max score over card-order permutations of `played_cards`, using
    CLONED mutable state so the live run is never touched by a hypothetical
    evaluation. This is the only place that talks to the real scoring
    engine -- all joker-specific knowledge lives here implicitly.
    """
    best: ScoreResult | None = None
    n = len(played_cards)
    perms: list[tuple[Card, ...]]
    if n <= 1 or not _needs_permutation_search(played_cards, jokers):
        perms = [tuple(played_cards)]
    else:
        all_perms = list(itertools.permutations(played_cards))
        if len(all_perms) > MAX_PERMUTATIONS:
            if _count_order_sensitive_sources(played_cards, jokers) > 1:
                # Two+ order-sensitive contributors can interact at interior
                # positions, not just at the first/last slots -- fall back
                # to full enumeration rather than risk missing the true max.
                perms = all_perms
            else:
                # Exactly one order-sensitive contributor (or none beyond an
                # identity effect): its optimum is always achievable by some
                # choice of first and/or last slot, so the (first, last)
                # covering set is exact here without needing all n!.
                perms = _first_last_covering_permutations(played_cards)
        else:
            perms = all_perms

    for order in perms:
        hl_copy = copy.deepcopy(hand_levels)
        rng_copy = copy.deepcopy(rng)
        # Deep-copy cards and jokers too -- score_hand mutates card/joker
        # state in place for several effects (Vampire strips enhancements
        # off scored cards, Wee Joker/Lucky Cat accumulate onto
        # ability["extra"]/["x_mult"], Lucky Card sets a "lucky_trigger"
        # flag). Without this, a hypothetical evaluation here would
        # permanently corrupt the caller's real hand/joker objects, and
        # repeated trials within this same permutation loop would
        # contaminate each other.
        played_copy = [copy.deepcopy(c) for c in order]
        held_copy = [copy.deepcopy(c) for c in held_cards]
        jokers_copy = [copy.deepcopy(j) for j in jokers]
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
) -> tuple[list[Card], ScoreResult]:
    """No discards being considered -- brute force over all non-empty
    subsets of size <=5 (cheap: C(8,5)=56 worst case)."""
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
                list(combo), held, jokers, hand_levels, blind, rng, game_state, blind_chips
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
                hold=hold,
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
) -> list[tuple[Template, list[Card], list[Card], float, float, int]]:
    """Returns up to `top_k` (template, hold, discard, p_reach, cheap_value,
    still_needed) tuples, ranked by p_reach * cheap_value using the
    jokerless `score_hand_base` (no engine joker loop, no permutation
    search -- safe because base scoring is order-invariant without
    jokers).
    """
    from jackdaw.engine.scoring import score_hand_base

    templates = build_templates(hand, four_fingers=four_fingers, shortcut=shortcut)
    scored: list[tuple[Template, list[Card], list[Card], float, float, int]] = []

    for template in templates:
        hold, discard = construct_hold(hand, template)
        still_needed = max(0, template.needed - len(hold))
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

        cheap_result = score_hand_base(
            eval_cards, [], copy.deepcopy(hand_levels), blind, copy.deepcopy(rng)
        )
        scored.append((template, hold, discard, p_reach, cheap_result.total, still_needed))

    scored.sort(key=lambda t: t[3] * t[4], reverse=True)
    return scored[:top_k]


def estimate_future_hand_distribution(
    deck: DeckComposition,
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    hand_size: int = 8,
    n_samples: int = 40,
    game_state: dict | None = None,
    blind_chips: int = 0,
) -> list[float]:
    """Monte Carlo stand-in for "what is a typical NEXT hand-turn worth,"
    drawn fresh from the given deck composition -- used only at the
    play -> next-hand-turn boundary, where there's no target yet to
    condition on (see module-level note above). Returns a list of sampled
    best-immediate-play values.

    Known biases (both understate future value, so this errs pessimistic):
      (a) sampled hands are scored via `best_immediate_play` only -- they
          don't get to use their own discards.
      (b) all samples are drawn from THIS deck snapshot; a real hand 3 or 4
          turns later draws from a further-depleted deck.
    """
    import random as _random

    from jackdaw.engine.card_factory import create_playing_card
    from jackdaw.engine.data.enums import Rank, Suit as SuitEnum

    id_to_rank = {v: k for k, v in RANK_ID.items()}
    pool: list[tuple[int, str]] = []
    for (rid, suit), n in deck.by_rank_suit.items():
        pool.extend([(rid, suit)] * n)

    if len(pool) < hand_size:
        return [0.0]  # not enough cards left to even sample -- degenerate

    samples: list[float] = []
    for _ in range(n_samples):
        drawn = _random.sample(pool, hand_size)
        hand_cards = [
            create_playing_card(SuitEnum(suit), Rank(id_to_rank[rid])) for rid, suit in drawn
        ]
        # evaluate_value/best_immediate_play deep-copy rng internally, so
        # passing the live rng here is safe -- no mutation of real state.
        _, result = best_immediate_play(
            hand_cards, jokers, hand_levels, blind, rng, game_state, blind_chips
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

    for template, hold, discard, p_reach, _cheap_val, still_needed in candidates:
        if still_needed == 0:
            # already satisfied -- degenerate to a deterministic recursive
            # call on the (unchanged, already-complete) hold, refilled.
            hit_hand, hit_drawn = _fill_hand_to_size(deck, hold, [], hand_size)
            hit_deck = deck.without(hit_drawn)
            hit_choice = solve_hand_turn(
                hit_hand, jokers, hand_levels, blind, rng, hit_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, top_k=top_k, hand_size=hand_size,
            )
            p_clear = hit_choice.p_clear
        else:
            hit_priority = _best_completion_cards(deck, template, still_needed)
            hit_hand, hit_drawn = _fill_hand_to_size(deck, hold, hit_priority, hand_size)
            hit_deck = deck.without(hit_drawn)
            hit_choice = solve_hand_turn(
                hit_hand, jokers, hand_levels, blind, rng, hit_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, top_k=top_k, hand_size=hand_size,
            )

            miss_priority = _representative_miss_cards(deck, template, still_needed)
            miss_hand, miss_drawn = _fill_hand_to_size(deck, hold, miss_priority, hand_size)
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
                hold=hold,
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
) -> AnteClearChoice:
    """Top-level entry point: computes the future-hand distribution once
    (for `hands_left - 1` downstream hand-turns) and kicks off the
    recursive discard-chain solve. This is what to call from a live
    decision point -- `solve_hand_turn` is the internal recursive worker.
    """
    future_samples = estimate_future_hand_distribution(
        deck, jokers, hand_levels, blind, rng, len(hand), 40, game_state, blind_chips
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
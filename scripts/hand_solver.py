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
from jackdaw.engine.data.hands import HAND_ORDER
from jackdaw.engine.hand_eval import get_best_hand, get_hand_eval_flags
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.play_ordering import (
    MAX_PERMUTATIONS,  # noqa: F401 -- re-export for existing importers
    COPY_JOKER_KEYS,
    best_joker_order,
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
from jackdaw.env.trigger_match import resolve_copy_targets, trigger_predicate

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


def _suit_is_red(suit) -> bool:
    return (suit.value if hasattr(suit, "value") else suit) in ("Hearts", "Diamonds")


def _flush_match(c, suit: str, smeared: bool) -> bool:
    """Does `c` count toward a `suit` flush?

    Delegates to the ENGINE's own `Card.is_suit(flush_calc=True)`, which
    owns three rules this module must not re-derive: Stone cards never
    count, WILD cards count as EVERY suit, and Smeared merges
    Hearts=Diamonds / Spades=Clubs. The old predicate here was a bare
    `_card_suit(c) == s`, which got all three wrong -- wild and smeared
    flushes were structurally unproposable (measured: stage3_full_00003496,
    truth `QC 9S 8S 3C 2C` is a Flush under Smeared and the solver could
    only offer two pair, 200 chips of regret).

    `_FakeCard` (the DeckComposition membership probe) has no `is_suit`, so
    it falls back to suit identity. That is the honest answer there and not
    a shortcut: deck composition tracks (rank, suit) ONLY, so it cannot
    represent enhancements at all -- a wild card sitting in the deck is
    invisible to it either way. The fallback therefore UNDERCOUNTS wild
    draws, which is the safe direction (a real wild in the deck can only
    raise p_reach, never lower it).
    """
    engine_is_suit = getattr(c, "is_suit", None)
    if engine_is_suit is not None:
        return engine_is_suit(suit, flush_calc=True, smeared=smeared)
    cs = _card_suit(c)
    if cs is None:
        return False
    if smeared:
        return _suit_is_red(cs) == _suit_is_red(suit)
    return cs == suit


def build_templates(
    hand: list[Card],
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
) -> list[Template]:
    """Fixed, joker-agnostic template set. `four_fingers` / `shortcut` /
    `smeared` should be derived from the live joker list before calling (all
    three loosen size/gap/suit requirements for straights & flushes) -- pass
    them in rather than re-deriving joker knowledge here.
    """
    templates: list[Template] = []
    flush_need = 4 if four_fingers else 5

    # --- flush-by-suit ---
    # Smeared collapses the four suits into TWO colour groups, so emitting
    # all four would just be the same two templates twice: identical
    # predicates, identical holds, doubling the box for nothing (and
    # crowding `family_best`, pitfall #13). One representative per group.
    flush_suits = ["Spades", "Hearts"] if smeared else SUITS
    for suit in flush_suits:
        name = f"flush_{'black' if suit == 'Spades' else 'red'}" if smeared else f"flush_{suit}"
        templates.append(
            Template(
                name=name,
                predicate=lambda c, s=suit, sm=smeared: _flush_match(c, s, sm),
                needed=flush_need,
            )
        )

    # --- straight-by-window ---
    # ranks 2..14 (Ace high); Ace can also play low (id 14 treated as 1).
    #
    # Four Fingers ADDS the 4-length windows, it does not REPLACE the
    # 5-length ones: the joker is permissive (4 consecutive cards now also
    # count) and a natural 5-card straight stays legal and scores MORE.
    # Emitting only 4-windows under it -- the original behaviour -- made a
    # 5-card straight structurally unproposable, because a window is an
    # explicit rank set and the 5th rank fails its predicate. Measured on
    # `stage2_curated_00002468` (Four Fingers + Crazy): the true line
    # JS-10C-9D-8D-7C (1480) could not be constructed and the best
    # candidate was the 4-card JS-10C-9D-8D (1340).
    #
    # Both needs are honest, so both are kept: 4-of-4 completes a straight
    # under Four Fingers, 5-of-5 completes the natural straight. (A
    # 5-window with needed=4 would be WRONG -- "4 of these 5 ranks" does
    # not imply 4 CONSECUTIVE ranks: {7,8,9,J} is not a straight.)
    # The flush side needs no such fix: its predicate is "same suit", so
    # `construct_hold` already gathers all 5 same-suit cards and `needed`
    # only feeds the reachability math.
    straight_needs = (4, 5) if four_fingers else (5,)
    windows: list[tuple[list[int], int]] = []
    lo, hi = 2, 14
    for need in straight_needs:
        for start in range(lo, hi - need + 2):
            windows.append((list(range(start, start + need)), need))
        # ace-low window, e.g. [14(as1),2,3,4] at need=4, [..,5] at need=5
        windows.append(([14] + list(range(2, 2 + need - 1)), need))

    gap = 1 if shortcut else 0
    for w, need in windows:
        wset = set(w)
        if gap:
            # widen predicate: allow ranks within the window OR adjacent gap-fillers
            # (approximation: shortcut's true gap logic lives in hand_eval;
            # here we just loosen membership by +/-1 rank of window bounds)
            lo_w, hi_w = min(w), max(w)
            wset = set(range(lo_w - gap, hi_w + gap + 1))
        # The `_n{need}` suffix disambiguates the two ace-low windows, which
        # both render as `straight_2-14` (min 2, max 14) regardless of
        # length. Applied only under four_fingers, so every non-four-fingers
        # template name is byte-identical to before.
        name = f"straight_{min(w)}-{max(w)}"
        if four_fingers:
            name = f"{name}_n{need}"
        templates.append(
            Template(
                name=name,
                predicate=lambda c, ws=wset: _card_rank_id(c) in ws,
                needed=need,
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

    B3 joker auto-ordering: on the exact path, the joker LIST is also
    re-ordered per candidate play via ``best_joker_order`` -- the closed-
    form additive-before-x-mult sort always (idempotent when the caller
    pre-sorted at hand-turn entry), plus the copy-target placement argmax
    when Blueprint/Brainstorm is owned. The MC future-hand path
    (``search_orderings=False``) keeps the caller's fixed hand-turn-entry
    order -- same approximation tier as its fixed card order. Labels
    valued this way assume the env commits plays under the same ordering,
    which ``action_to_engine_action`` does (consistency pinned in
    tests/engine/test_play_ordering.py).
    """
    if search_orderings:
        jokers = best_joker_order(
            jokers,
            played_cards,
            held_cards,
            hand_levels,
            blind,
            rng,
            game_state=game_state,
            blind_chips=blind_chips,
        )
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


# --- Big-hand play prescreen ------------------------------------------------
#
# `best_immediate_play` is C(n,5) exact evaluations per call: 56 subsets at
# n=8, ~4.4k at n=16 -- and it runs at every recursion node of
# `solve_hand_turn` plus 16x per MC future-hand estimate. Beyond
# PRESCREEN_HAND_LIMIT cards, only the top-k template-derived candidate
# subsets are evaluated exactly; the label becomes "exact among prescreened
# candidates" (PPO against the real game is the documented corrector for the
# residual bias -- see CLAUDE.md "Solver big-hand cost").
#
# PRESCREEN_TOP_K validated 2026-07-14 (`scripts/validate_prescreen.py`, 48
# hands flat over sizes 9-12, stage3_full + hand-size tail) and REVALIDATED
# 2026-07-15 after the rank-combination generator widening (all 17/48
# generation holes were cross-rank-group lines -- two pair / full house --
# that single-rank templates + nominal-priority kicker padding could never
# propose; see the combination pass in prescreen_play_candidates):
# best-in-cut rate 0.646 -> 0.958, regret 0.0 at every tested k on 44/48
# MC-active states vs noise floor 0.022, boundary-stress regret 0.12 ->
# 0.02 at f=1.0 (the knife-edge placement) and -> 0.0 at f=1.1. Minimal
# passing k is 3 both times; set to 5 as a user-called margin (2026-07-15,
# preemptive headroom after the widening). KNOWN ACCEPTED RESIDUAL (user
# call 2026-07-15): 2/48 kicker-CHOICE misses remain (right line, wrong
# kicker -- keep-priority pads a nominal-best card where a joker values a
# specific suit/rank; ratios 0.92/0.71, measured regret 0.0). The named
# lever if it ever matters: kicker VARIANTS per combination (per-suit /
# per-enhancement alternatives), not k. Full report:
# data/prescreen_validation.json.
#
# PRESCREEN_HAND_LIMIT DELETED (K3, 2026-07-17). Every hand size is now
# screened uniformly. The old `n > 8` seam existed because the prescreen was
# only trusted on big hands, but it was also where B5's residual hid: n=8
# kept brute-forcing, so the box was never measured there and the kicker bug
# went unseen until the n=8 harness was pointed at it (0.845 capture, misses
# up to 90% of a play's value). One path, measured everywhere.
#
# Licensed by K3's gate, both arms: root arm capture-by-value 0.980 at n=8
# (stage2 brute, k-invariant), 0.980 on the stage3 copy-joker sample, 0.950
# on the 9-12 tail; full-solve arm 0.9808 node-level at true depth (d0 0.981,
# d1 0.997, d2/d3 1.000), root-action agreement 1.0. Bar was >=0.95.
#
# ACCEPTED RESIDUALS (user call 2026-07-17), all PPO-correctable in the
# documented sense -- the real reward is P(clear) and a mis-ranked play is a
# legal single-step action (the A3 "training problem, not labels" precedent):
#   - ~5% of states, mean regret ~22 chips on the tail, one coherent family:
#     class-3 SET-LEVEL jokers (Jolly/Droll/Blackboard/Square), where no
#     honest per-card bit exists by taxonomy design, plus Four Fingers.
#   - The tail's aggregate 0.950 is carried by n=9/n=12; n=10 (0.939) and
#     n=11 (0.935) are individually below the bar.
#   - The full-solve arm's `mc` stratum sits at 0.925 (160 nodes, mean regret
#     44) -- future-hand samples valued at the coarse search_orderings=False
#     tier. It feeds p_clear VALUES (the critic's warm start), not just
#     actions; PPO retrains the critic on real returns.
#   - "PPO corrects it" is a claim about the END of fine-tuning: BC teaches
#     the prior and the KL leash holds the policy near it early, so a line BC
#     never proposes may go unsampled for a while. The leash provably decays
#     to zero.
PRESCREEN_TOP_K = 5

# --- B7 discard-shortlist depth gating (locked 2026-07-16) ------------------
# The joker/held-aware discard ranker (B7) ranks by `p_reach * cheap_value`,
# where `cheap_value` scores ONE idealized completion -- an EV of the peak hit,
# not P(clear). It is therefore threshold-blind and occasionally over-ranks a
# high-ceiling completion, dropping a higher-CONVERSION discard past the top-k
# cut. The faithful-MC top_k sweep (data/discard_ranking_sweep.json) showed
# every such regression lives exactly at that cut: widening the box to 6
# re-includes the dropped discard and heals the hard cases. But solve cost
# scales ~(k/4)^discards_left, so a flat 6 roughly doubles the whole regen
# wall (flat 8 ~4x) with multi-minute stragglers on deep big-hand states. The
# regressors all live at SHALLOW depth (discards_left 1-2), where the wide box
# is cheap, so we widen there and keep 4 on deep chains to cap the (k/4)^3
# tail. See CLAUDE.md "B7 discard-shortlist DEPTH-GATED WIDENING" for the full
# rationale + worked example (seed DISCARD_RANK_VAL_00000247, Shoot the Moon).
DISCARD_TOPK_WIDE = 6
DISCARD_TOPK_NARROW = 4
DISCARD_TOPK_WIDE_MAX_DISCARDS = 2


def _discard_shortlist_k(discards_left: int) -> int:
    """Production discard-shortlist width for a node with `discards_left`
    discards remaining: `DISCARD_TOPK_WIDE` on shallow chains
    (`discards_left <= DISCARD_TOPK_WIDE_MAX_DISCARDS`), else
    `DISCARD_TOPK_NARROW`. Callers that pass an explicit `top_k` bypass this
    gate entirely (fixed box) -- that path is for the B7 validation harness,
    the top_k sweep, and the existence-proof tests, which need a k held
    constant across old-vs-new comparisons; production label generation leaves
    `top_k=None` and gets the gate."""
    if discards_left <= DISCARD_TOPK_WIDE_MAX_DISCARDS:
        return DISCARD_TOPK_WIDE
    return DISCARD_TOPK_NARROW


# Realized hand types that count as an in-hand rank line for the pair pin
# (see prescreen_play_candidates): complete now, zero draw risk.
_RANK_LINE_TYPES = frozenset(
    {"Pair", "Two Pair", "Three of a Kind", "Full House", "Four of a Kind", "Five of a Kind"}
)


def _ranking_score(
    cards: list[Card],
    held: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    game_state: dict | None,
    blind_chips: int,
) -> tuple[ScoreResult, dict[int, Card]]:
    """Shared RANKING-tier scorer for candidate selection (the play
    prescreen and the discard-template ranking): one `score_hand` call,
    fixed order, no permutation search, joker- and held-aware, fast-clones
    throughout (score_hand mutates card/joker/hand_levels/rng/blind state
    in place on every hypothetical call -- The Eye/The Mouth history,
    scaling-joker accumulation). Returns the result plus a clone-id ->
    original-card map so callers can key on original card identity
    (score_hand reports scoring_cards as the clones it was given)."""
    played_clones = [_fast_clone_card(c) for c in cards]
    held_clones = [_fast_clone_card(c) for c in held]
    result = score_hand(
        played_clones,
        held_clones,
        [_fast_clone_card(j) for j in jokers],
        _fast_clone_hand_levels(hand_levels),
        _fast_clone_blind(blind),
        _fast_clone_rng(rng),
        game_state=game_state,
        blind_chips=blind_chips,
    )
    clone_to_orig = {
        id(clone): orig for clone, orig in zip(played_clones + held_clones, cards + held)
    }
    return result, clone_to_orig


# --- K1 kicker variants (locked 2026-07-16) ---------------------------------
#
# The prescreen proposes the right scoring LINE and the right size and then
# picks the wrong KICKERS: the old `_kicker_pad` padded every line by keep-priority
# nominal-best, which assumes kickers are inert filler. That assumption is
# false on exactly the boards stage2 concentrates -- under Splash the
# kickers SCORE (so their suit/rank feeds Lusty/Greedy/Even Steven), and
# under Raised Fist the cards you RETAIN set the mult. Measured at n=8 on
# stage2 density: 0.845 score-capture, misses up to 90% of the play's value,
# and INVARIANT in k (27/27 misses had true_size 5 and the right line) --
# the candidate GENERATOR was starved, so no k rescues it. Full measurement
# record: docs/bruteforce_speedup_and_kicker_design.md.
#
# The fix is GENERATION-ONLY. `_ranking_score` is deliberately untouched: it
# is a real joker/held-aware `score_hand` call that already values kickers
# correctly when it is GIVEN them -- ranking judged what it was handed, and
# changing it would force a B7 revalidation for no benefit.
#
# A variant is the GREEDY argmax completion of a line under ONE hypothesis
# about where kicker value lives. There are NO magnitudes anywhere below: a
# hypothesis's only job is to decide which cards are PRESENT in the set, and
# `best_immediate_play`'s exact pass (full ordering search, real score_hand)
# arbitrates between the variants. That is what makes raw derivation safe
# here -- a wrong hypothesis costs one wasted candidate, never a wrong label.
# Variants RIDE their line: `top_k` counts LINES, and every surviving line
# carries all of its variants into the exact pass.

def _has_held_enhancement(card: Card) -> bool:
    """Does this card's ENHANCEMENT pay off while it sits in hand -- i.e.
    does playing it away as a kicker forfeit something?

    Read from the engine's own held-channel config rather than a name set
    ({"Steel Card", "Gold Card"} would be correct on today's content and
    would rot the moment an enhancement is added or rebalanced -- the same
    hand-written-list failure the trigger taxonomy exists to avoid):

      * `get_chip_h_x_mult` -- Steel's x1.5 (card.lua:1011)
      * `get_chip_h_mult`   -- no standard enhancement uses it today, but
                               it is the engine's held additive-mult hook
      * `ability["h_dollars"]` -- Gold's end-of-round $3 (card.py:251; it
                               has no accessor, hence the raw read, so the
                               debuff guard is ours to apply)

    The two accessors already return 0 for a debuffed card, which is the
    behaviour we want: a debuffed Steel card has no held value and is fair
    game as a kicker.

    ACCEPTED RESIDUAL: SEALS are not consulted. A Blue seal is genuinely
    held-value (a Planet at end of round if the card stays in hand), so a
    held-value variant can pad one away. Same class as the spec's
    documented mixed-hypothesis residuals -- it earns a term only if the
    K3 rescan shows it.
    """
    if card.debuff:
        return False
    return (
        card.get_chip_h_mult() > 0
        or card.get_chip_h_x_mult() > 0
        or card.ability.get("h_dollars", 0) > 0
    )


def _resolved_joker_views(jokers: list[Card]) -> list[tuple[str, Card]]:
    """Active joker effects as (effective center key, the joker card whose
    ability to read), one per slot, after COPY RESOLUTION.

    Blueprint/Brainstorm resolve through the ENGINE's own path
    (`resolve_copy_targets`) rather than a reimplementation, so the gates
    below read what actually fires: a Blueprint copying Castle presents as
    Castle AND reads the suit off the Castle joker (Castle stores it on its
    own ability). `blueprint_compat` comes free from that path -- which is
    load-bearing for the scored-value gate, since Splash is on the
    29-joker incompat list and therefore can never be copied.

    A copy joker that resolves to nothing keeps its own key, which carries
    no predicate (class 4) and so gates nothing. Debuffed jokers are
    dropped: the scoring loops skip them before any handler runs.
    """
    resolutions = resolve_copy_targets({"jokers": jokers})
    views: list[tuple[str, Card]] = []
    for joker, resolution in zip(jokers, resolutions):
        if joker.debuff:
            continue
        if resolution.active:
            target = jokers[resolution.target_index]
            views.append((target.center_key, target))
        else:
            views.append((joker.center_key, joker))
    return views


def _card_channel_counts(
    hand: list[Card],
    views: list[tuple[str, Card]],
    game_state: dict | None,
    flags: dict[str, bool],
) -> dict[int, tuple[int, int]]:
    """id(card) -> (scored candidacies, held candidacies) over the B2
    trigger taxonomy's OWN predicates (`trigger_predicate`) -- never a
    second hand-written joker table, which would drift out of sync the
    moment a joker is reclassified.

    These are PRESENCE tallies -- how many active joker effects list this
    card as a candidate -- not an estimate of what those effects are worth.
    Class-3 (set-level) and class-4 (non-card) jokers contribute nothing by
    construction, which is correct here: no honest per-card bit exists for
    them.

    Raised Fist is excluded from the tally entirely. Its bit marks the
    CURRENT lowest held card, which is the wrong rule for a counterfactual
    question about which cards to keep -- once a card leaves the hand the
    minimum moves. Its min term gets an exact hypothesis of its own below.
    """
    counts = {id(c): [0, 0] for c in hand}
    gs = game_state or {}
    for key, joker in views:
        if key == "j_raised_fist":
            continue
        predicate = trigger_predicate(key)
        if predicate is None:
            continue
        for card in hand:
            scored, held = predicate(card, joker, gs, flags)
            if scored:
                counts[id(card)][0] += 1
            if held:
                counts[id(card)][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


@dataclass(frozen=True)
class _KickerGates:
    """Which kicker hypotheses are live on this board. Each gate reads
    RESOLVED joker identities (never raw keys), so a copied effect gates
    exactly like the original. Ungated hypotheses are pure waste -- without
    Splash a kicker never scores, so the scored-value variant would only
    burn an exact evaluation -- which is why the budget stays adaptive
    rather than 3x flat."""

    scored_value: bool
    held_value: bool
    play_away_lowest: bool


def _kicker_gates(
    hand: list[Card],
    views: list[tuple[str, Card]],
    flags: dict[str, bool],
    counts: dict[int, tuple[int, int]],
) -> _KickerGates:
    keys = {key for key, _ in views}
    return _KickerGates(
        # Splash is the ONLY way a non-line card scores, so it alone makes
        # kicker suit/rank matter. The flag comes from `get_hand_eval_flags`
        # (the engine's own detection-modifier read) because Splash is
        # class-3 all-zero in the trigger matrix BY DESIGN -- the matrix
        # cannot answer this question.
        scored_value=bool(flags.get("splash", False)),
        held_value=any(held for _, held in counts.values())
        or any(_has_held_enhancement(c) for c in hand),
        play_away_lowest="j_raised_fist" in keys,
    )


def _scored_kicker_key(
    counts: dict[int, tuple[int, int]],
) -> Callable[[Card], tuple]:
    """Descending kicker order under Splash: scored-channel candidacies
    first, then EDITION presence, then `_keep_priority`'s (enhancement,
    nominal).

    The edition term is not in the taxonomy and cannot be: `trigger_match`
    is a card x JOKER matrix, and an edition is a property of the card
    itself -- `grep edition jackdaw/env/trigger_match.py` is empty by
    design. But editions fire on the SCORED channel (`scoring.py`'s
    per-scoring-card `card.get_edition()`), so under Splash a Polychrome
    kicker is x1.5 mult and a Foil kicker is +50 chips. Without it this key
    would rank a Polychrome 2 below a plain King -- precisely the
    right-line/wrong-kicker miss K1 exists to kill. (The locked spec's
    "chips + enhancement + scored-channel candidacy bits" omitted this;
    caught in review 2026-07-16.)

    Presence only, per the section header: which cards are in the set, not
    what they are worth. Editions are scored-channel exclusively, so they
    earn no term in the held-value key.
    """

    def key(card: Card) -> tuple:
        return (
            counts[id(card)][0],
            1 if card.edition else 0,
        ) + _keep_priority(card)

    return key


def _held_kicker_key(
    counts: dict[int, tuple[int, int]],
) -> Callable[[Card], tuple]:
    """Held-value order: held-channel candidacies, then held-enhancement
    presence, then `_keep_priority`. Sorted ASCENDING at the call site --
    hypothesis 3 pads with the cards LEAST valuable held, so that the ones
    held effects want (Baron's Kings, Shoot the Moon's Queens, Mime's
    retriggers, Steel, Gold) stay in hand.

    Editions earn no term here: they fire on the scored channel only, so a
    Polychrome card is worth nothing held and is fair game as a kicker.
    """

    def key(card: Card) -> tuple:
        return (
            counts[id(card)][1],
            1 if _has_held_enhancement(card) else 0,
        ) + _keep_priority(card)

    return key


def _raised_fist_key(card: Card) -> tuple[int, int]:
    """Ascending play-away order for Raised Fist's min term: lowest RANK ID
    first (the handler's own comparison -- `nominal` would tie J/Q/K, which
    the id distinguishes), Stone Cards last. Stones are excluded from the
    handler's minimum, so playing one away cannot raise it -- it is never
    the card this hypothesis wants to spend."""
    stone = card.ability.get("effect") == "Stone Card"
    return (1 if stone else 0, card.get_id())


_HAND_TYPE_RANK: dict[str, int] = {
    (ht.value if hasattr(ht, "value") else ht): len(HAND_ORDER) - i
    for i, ht in enumerate(HAND_ORDER)
}


def _hand_type_rank(name) -> int:
    """Higher = better. Unknown / "NULL" ranks below every real type."""
    return _HAND_TYPE_RANK.get(name.value if hasattr(name, "value") else name, 0)


def _type_upgrade_completion(
    line: list[Card],
    pool: list[Card],
    need: int,
    flags: dict[str, bool],
) -> list[Card]:
    """Greedy completion maximizing the DETECTED HAND TYPE, ties broken by
    `_keep_priority`.

    Hypothesis 5, and the only one that is not about what a kicker is worth
    AS A CARD: the pad can change what the hand IS. The other four sort the
    pool by a per-card key, so a pad that upgrades the hand type is
    invisible to all of them.

    The motivating miss (stage3_full_00001545, Four Fingers): the line is
    the 4-card heart flush `QH 7H 5H 4H`, and padding with `6C` -- a CLUB,
    which nominal-best ranks below `KC` and every other hypothesis ignores
    -- makes it a STRAIGHT FLUSH, because under Four Fingers the flush
    needs 4 hearts and the straight needs 4 of 7-6-5-4 and vanilla lets
    those be different cards. 892 chips of regret, the largest single miss
    in the arm.

    NOT Four-Fingers-specific, so it is ungated: padding trips with a pair
    makes a Full House, and nominal-best takes the two highest cards
    instead. Templates cannot cover these because the winning set is the
    UNION of two templates' cards -- no single predicate proposes it.

    Presence only, per the section header: this ranks hand TYPES, never
    magnitudes, and the exact evaluator still arbitrates. Detection is the
    engine's own `get_best_hand` scan (as `_line_family` uses), so Splash
    is ignored here and copy/compat rules come free.
    """
    cur = list(line)
    remaining = list(pool)
    for _ in range(need):
        best_key: tuple | None = None
        best_card: Card | None = None
        for candidate in remaining:
            hand_type, _, _ = get_best_hand(
                cur + [candidate],
                four_fingers=flags.get("four_fingers", False),
                shortcut=flags.get("shortcut", False),
                smeared=flags.get("smeared", False),
            )
            key = (_hand_type_rank(hand_type),) + _keep_priority(candidate)
            if best_key is None or key > best_key:
                best_key, best_card = key, candidate
        if best_card is None:
            break
        cur.append(best_card)
        # id() compared with `!=`, never `is not` (two large ints are
        # separate objects, so `is not` is always True and the pick is
        # never removed -- the greedy loop then re-picks it and emits the
        # same card 5 times), and never list.remove (Card equality is
        # value-based, so an Erratic deck's duplicate would drop the wrong
        # copy -- the bug class of best_immediate_play's `held` filter).
        remaining = [c for c in remaining if id(c) != id(best_card)]
    return cur


def _kicker_variants(
    line: list[Card],
    hand: list[Card],
    gates: _KickerGates,
    counts: dict[int, tuple[int, int]],
    flags: dict[str, bool],
) -> list[list[Card]]:
    """Completions of `line` to 5 cards, one per live hypothesis about
    where kicker value lives (see the section header). Deduped by card-
    identity set, so on a plain board every hypothesis collapses onto
    nominal-best and the line costs exactly one candidate.

    Hypotheses:
      1. inert / nominal-best -- the pre-K1 behaviour, always emitted.
      2. scored-value (Splash only) -- kickers score, so prefer the ones
         the most scored-channel effects list as candidates, then chips.
      3. held-value -- retain the cards held effects want (Baron/Shoot the
         Moon/Mime, Steel/Gold), i.e. pad with the cards LEAST valuable
         held.
      4. play-away-lowest -- exact for Raised Fist's min term.
      5. type-upgrade -- the pad card can change the detected hand type
         (FF straight flush, trips + pair -> full house); ungated, since
         no board makes it structurally impossible.
    """
    if len(line) >= 5:
        return [line]
    need = 5 - len(line)
    chosen = {id(c) for c in line}
    pool = [c for c in hand if id(c) not in chosen]
    if not pool:
        return [line]

    def _complete(key: Callable[[Card], tuple], *, best_first: bool) -> list[Card]:
        return line + sorted(pool, key=key, reverse=best_first)[:need]

    variants = [_complete(_keep_priority, best_first=True)]
    if gates.scored_value:
        variants.append(_complete(_scored_kicker_key(counts), best_first=True))
    if gates.held_value:
        variants.append(_complete(_held_kicker_key(counts), best_first=False))
    if gates.play_away_lowest:
        variants.append(_complete(_raised_fist_key, best_first=False))
    variants.append(_type_upgrade_completion(line, pool, need, flags))

    seen: set[frozenset[int]] = set()
    unique: list[list[Card]] = []
    for variant in variants:
        key = frozenset(id(c) for c in variant)
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def _line_family(
    cards: list[Card], *, four_fingers: bool, shortcut: bool, smeared: bool
) -> tuple[str, frozenset[int]]:
    """(realized hand type, identity set of the cards FORMING the line),
    from a SPLASH-AGNOSTIC scan of the played cards.

    Deliberately NOT `score_hand`'s `scoring_cards`: under Splash every
    played card scores, so each kicker variant of one line would report a
    different scoring set -- a different family -- and the variants would
    crowd `family_best` out of the genuinely distinct lines. That is
    handoff pitfall #13 recreated by the fix, on exactly the boards the fix
    targets. The engine's own detection scan ignores Splash, so variants of
    a line collide into one family as intended, which is also what lets
    them ride their line into the exact pass.

    The card SET stays in the key (hand type alone would collapse a pair of
    Kings and a pair of 3s into one family). `get_best_hand` is the
    engine's own path and tolerates base-less Stone cards -- they group but
    never emit, so a stone-only candidate falls back to a bare hand type.
    """
    hand_type, line_cards, _ = get_best_hand(
        cards, four_fingers=four_fingers, shortcut=shortcut, smeared=smeared
    )
    return (hand_type, frozenset(id(c) for c in line_cards))


SEATING_MAX_POOL = 9
SEATING_BUDGET = 50


def _seat_variants(
    hold: list[Card],
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    game_state: dict | None,
    blind_chips: int,
) -> list[list[Card]]:
    """Which 5 of a template's matching cards to SEAT, when it matches more
    than 5. Returns the seatings worth carrying, best cheap-score first.

    THE K3 TAIL BUG. A straight window normally matches exactly 5 cards and
    `[:5]` is a no-op -- but Shortcut widens the window predicate and Four
    Fingers adds shorter windows, so the hold OVERFLOWS and five seats must
    be chosen from more candidates. That choice was
    `sorted(hold, key=_keep_priority)[:5]` = (enhanced, nominal): JOKER-
    BLIND. Measured (stage2_curated_00002127, Shortcut + Wee Joker): hold
    `7S 6S 5S 4S 3C 2D`, seated `7-6-5-4-3`, truth `7-6-5-3-2` -- Wee pays
    per 2 SCORED and the 2 was dropped for being the lowest card. It is also
    the tail arm's hand-size gradient (capture 0.950 at n=9 decaying to
    0.896 at n=12 while n=8 sits at 0.980): a bigger hand puts more cards
    inside a widened window, so there is more to get wrong.

    K1's hypotheses cannot reach this. They choose KICKERS, and they only
    run when `len(playable) < 5` -- these lines are OVER five. A kicker sits
    outside the line and swaps freely; a line member is LOAD-BEARING.

    NO hypothesis, NO type filter, NO family dedupe -- all three were tried
    in review and all three are the same mistake in different hats: they
    discard a candidate on a PROXY for value (nominal rank / hand type /
    line identity) before the scorer ever sees it, which is precisely the
    bug being fixed. Instead: enumerate and let `_ranking_score` -- a real
    joker- and held-aware `score_hand` -- rank them. The K spec's own point
    is that ranking was never wrong, generation was starving it. Validity
    needs no check either: an illegal seating (`7-6-5-4-2`, whose 4->2 gap
    exceeds Shortcut's one) simply scores as High Card and loses on merit.

    COST (measured, tail states): `_ranking_score` 0.26ms, exact
    `evaluate_value` 2.85ms, mean 8.6 seatings/state, and 80% of tail states
    have NO overflowing hold at all. Ranking every seating costs ~2ms/node;
    the riders cost ~24ms/node. A node goes ~76ms -> ~102ms (+35%), tail
    labels 0.7-10s -> ~0.9-13.5s, and the tail is 10% of the regen => ~+3.5%
    overall. The cost is the RIDING, not the enumeration.

    `SEATING_BUDGET` therefore is NOT an average-case lever (the mean, 8.6,
    never reaches it): it bounds the worst node, where 62 seatings would
    take that node to ~3x. Cutting by cheap score is a real risk -- the
    cheap tier is fixed-order, so an order-sensitive board (Photograph,
    Hanging Chad) can mis-rank -- but at 50 the cheap tier only excludes
    what it ranked below fifty others, which is a budget, not the
    "cheap-arbitration-is-final" pattern the spec rejects (that was
    top-N-by-cheap-score deciding the answer). Everything surviving RIDES
    into the exact pass, which arbitrates with the full ordering search.
    `SEATING_MAX_POOL` is a guard, not a mechanism: measured max hold is 9,
    so it does not bind today.
    """
    if len(hold) <= 5:
        return [list(hold)]

    pool = sorted(hold, key=_keep_priority, reverse=True)[:SEATING_MAX_POOL]
    scored: list[tuple[float, list[Card]]] = []
    for combo in itertools.combinations(pool, 5):
        cards = list(combo)
        ids = {id(c) for c in cards}
        held = [c for c in hand if id(c) not in ids]
        result, _ = _ranking_score(
            cards, held, jokers, hand_levels, blind, rng, game_state, blind_chips
        )
        scored.append((result.total, cards))
    scored.sort(key=lambda t: -t[0])
    return [cards for _, cards in scored[:SEATING_BUDGET]]


def prescreen_play_candidates(
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind,
    rng: PseudoRandom,
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
    top_k: int = PRESCREEN_TOP_K,
    game_state: dict | None = None,
    blind_chips: int = 0,
    eval_flags: dict[str, bool] | None = None,
) -> list[list[Card]]:
    """Template-derived candidate play subsets for big hands, ranked
    cheaply and selected FAMILY-DIVERSE: the best candidate of each family
    first, then remaining slots filled by cheap rank. Naive top-k would
    return k variants of one dominant line and starve e.g. straight
    candidates whose cheap rank is systematically lower than their exact
    value (handoff pitfall #13).

    The ranking is JOKER-AWARE (user call, 2026-07-14): one `score_hand`
    call per candidate, fixed order, no permutation search. Jokerless
    ranking (`score_hand_base`, the `rank_templates_cheaply` precedent) is
    wrong HERE because the ranking decides which lines ever reach exact
    evaluation -- a joker-favored line (Greedy turning a weak diamond
    flush into the true best play) would be filtered out before the
    joker-aware exact pass could see it. Held cards are passed too (Baron/
    steel-class held effects shift line values). The single fixed ordering
    is the same approximation tier `estimate_future_hand_distribution`
    accepts: ranking precision, not label precision -- the top-k get the
    full ordering search afterwards.

    `top_k` counts LINES (families), not returned candidates: every
    surviving line carries ALL of its kicker variants into the exact pass,
    which is what arbitrates between them (K1). So the returned list is
    generally LONGER than `top_k` -- on a plain board the variants dedupe
    to one apiece and it approaches `top_k`, on a Splash/Raised-Fist board
    it fans out. Budget math: ~5 lines x ~3 variants ~= 15 candidates
    against brute force's 218.

    A family is the candidate's REALIZED scoring line -- (detected hand
    type, line-card identity set) from a SPLASH-AGNOSTIC scan of the played
    cards (`_line_family`) -- not the template that spawned it, and not
    `score_hand`'s `scoring_cards`. Template identity is the wrong key:
    kicker padding lets every weak template piggyback a dominant line (a
    lone Queen padded with four held Kings scores as the SAME four-of-a-kind
    as the quads template's own candidate), so template-keyed diversity
    would fill every slot with re-labeled copies of one line -- the exact
    crowding the family pass exists to prevent. `scoring_cards` is the
    wrong key for the mirror-image reason: under Splash every played card
    scores, so each kicker variant would report a different scoring set and
    the variants would crowd out the distinct lines instead.

    Ordering is PREFIX-STABLE in `top_k` at LINE granularity: the
    `top_k=j` call's output is a list PREFIX of the `top_k=k` call's for
    j < k. It is no longer indexable by k (entry j is not line j), so a
    caller wanting a j-line cut must pass `top_k=j` rather than slice.

    PAIR PIN: the best already-realized rank line (pair or better) is
    promoted to index 1 of the ordering, so every k>=2 cut evaluates it.
    A pair's cheap rank is weak but its value is CONSISTENT -- complete
    now, no draw, no luck -- and cheap (jokerless) ranking systematically
    underprices it against speculative draw lines.

    Candidates per template: the template's matching cards capped at 5
    (highest keep-priority first), plus -- when that leaves room -- the
    same cards padded to 5 under each live kicker hypothesis (see the K1
    section header). The fallback candidate (top 5 of the whole hand by
    keep-priority) covers hands where no template matches anything useful
    (e.g. base-less stone cards, which no rank/suit predicate can see).

    `eval_flags` is `get_hand_eval_flags(jokers)` from the caller (which
    has already computed it); it sources the Splash gate and the family
    scan's smeared handling. None = compute it here.
    """
    flags = eval_flags if eval_flags is not None else get_hand_eval_flags(jokers)
    # This function takes the detection flags TWICE -- as three booleans
    # (they reach `build_templates`) and inside `eval_flags` (it reaches the
    # kicker gates and the family scan). Nothing made them agree, so a
    # caller could pass `eval_flags` with smeared=True and leave the boolean
    # at its default, and the box would silently be built from raw-suit
    # templates: exactly the K3 arm-C miss that survived the Bug C fix,
    # because BOTH validation harnesses did precisely this. Same shape as
    # the `jokers=None` bug one layer down -- a call site quietly not
    # forwarding what it holds. Loud, per the Riff-raff precedent.
    for _name, _passed in (
        ("four_fingers", four_fingers),
        ("shortcut", shortcut),
        ("smeared", smeared),
    ):
        if bool(flags.get(_name, False)) != bool(_passed):
            raise ValueError(
                f"prescreen_play_candidates: {_name}={_passed} contradicts "
                f"eval_flags[{_name!r}]={flags.get(_name)}. The booleans build the "
                "templates and eval_flags gates the kickers; disagreeing means the "
                "box is built for a different board than it is scored against."
            )
    templates = build_templates(
        hand, four_fingers=four_fingers, shortcut=shortcut, smeared=smeared
    )
    views = _resolved_joker_views(jokers)
    counts = _card_channel_counts(hand, views, game_state, flags)
    gates = _kicker_gates(hand, views, flags, counts)

    def _variants(cards: list[Card]) -> list[list[Card]]:
        return _kicker_variants(cards, hand, gates, counts, flags)

    raw: list[list[Card]] = []
    for template in templates:
        hold, _ = construct_hold(hand, template)
        if not hold:
            continue
        playable = sorted(hold, key=_keep_priority, reverse=True)[:5]
        raw.append(playable)
        if len(hold) > 5:
            # Overflowing hold: `playable` above is only ONE of the possible
            # seatings, and it is the joker-blind one. Emit the rest ranked
            # by the real scorer (K3 tail fix; see `_seat_variants`). They
            # are distinct FAMILIES, not kicker variants of one line, so
            # they compete for top_k slots on merit rather than riding for
            # free -- which is what top_k is for, and capture-vs-k is the
            # readout that says whether k needs raising.
            raw.extend(
                _seat_variants(
                    hold, hand, jokers, hand_levels, blind, rng, game_state, blind_chips
                )
            )
        if len(playable) < 5:
            # Padded variants: line + kickers, one per live hypothesis (K1).
            # (Historically claimed to also cover full house / two pair
            # "emerging" from a rank hold padded with another rank's cards --
            # FALSE in practice: padding picks kickers by nominal priority,
            # so the second rank group is only chosen when it happens to
            # outrank every loose high card. All 17/48 generation holes in
            # the B5 validation were exactly the missing combinations; see
            # the rank-combination pass below.)
            raw.extend(_variants(playable))
    raw.extend(_variants([]))  # template-free fallback

    # Rank-line COMBINATIONS (B5 widening, 2026-07-15): two pair and full
    # house are cross-GROUP lines no single-rank template can propose.
    # Enumerate ordered pairs of multi-card rank groups: 2+2 (two pair,
    # bare and kicker-padded -- the bare variant matters when the 5th card
    # is better held: Baron/Blackboard-class held effects) and 3+2 (full
    # house, from either group's trips side). Within-group card choice by
    # keep-priority (enhancement-aware); dedup + the family pass absorb
    # the overlap.
    rank_groups: dict[object, list[Card]] = {}
    for c in hand:
        if c.base is not None:
            rank_groups.setdefault(c.base.id, []).append(c)
    multi = [
        sorted(g, key=_keep_priority, reverse=True)
        for g in rank_groups.values()
        if len(g) >= 2
    ]
    for g1, g2 in itertools.permutations(multi, 2):
        two_pair = g1[:2] + g2[:2]
        raw.append(two_pair)
        raw.extend(_variants(list(two_pair)))
        if len(g1) >= 3:
            raw.append(g1[:3] + g2[:2])  # full house

    # Dedup by card-identity set (padded variants of different templates
    # frequently coincide), cheap-score survivors on CLONES -- score_hand
    # mutates card/joker/hand_levels/rng/blind state in place on every
    # hypothetical call (The Eye/The Mouth history, scaling-joker
    # accumulation; see evaluate_value's clone note). Family = the realized
    # scoring line (see docstring), mapped back from the scored CLONES to
    # original card identity so identical lines actually collide.
    # MAX OVER EMITTED ORDERS, not first-wins (K3, 2026-07-17).
    # `_ranking_score` is FIXED-ORDER by design (searching orderings is the
    # exact pass's job), so it scores whatever order the generator emitted.
    # Deduping by card-identity SET and keeping the FIRST therefore made a
    # family's cheap rank a lottery decided by `itertools` ordering.
    # Measured (stage2_curated_00002797; Photograph x2 on the first face
    # scored + Hanging Chad retriggering the first card): the
    # `permutations(multi, 2)` pass yields (threes, kings) BEFORE
    # (kings, threes), so the full house was first emitted as
    # `3S 3C 13H 13C 13D` -> 1408, and the kings-first emission of the SAME
    # FIVE CARDS -> 4080 arrived later and was skipped. At 1408 it ranked
    # ~7th and was cut at top_k=5, so the exact pass -- whose ordering
    # search would have found 4080 -- never saw a FULL HOUSE at all; it lost
    # to a 3008 straight.
    # The generator already emits such sets in several orders, so taking the
    # MAX recovers the good one for free. A canonical sort was rejected: any
    # fixed order is a guess (keep-priority-descending suits Photograph, and
    # would bury a joker that wants a low card first), so it trades one
    # arbitrary order for another. This invents nothing and only uses orders
    # the generator produced. It is NOT an ordering search and must not
    # become one -- that stays the exact pass's job.
    # GATED on `_needs_permutation_search`, and that gate is not an
    # optimization but the difference between viable and not. `raw` is dense
    # with duplicate emissions (kicker variants of different templates
    # coincide constantly -- see the rank-combination note above), so
    # re-scoring every one of them costs 7x at n=8 (16.9 -> 117.4 ms/state,
    # measured), which would leave the prescreen barely beating the 56-subset
    # brute force it exists to replace -- and n=8 takes this path the moment
    # PRESCREEN_HAND_LIMIT is deleted. But when NO order-sensitive
    # contributor is present, every order of a set scores IDENTICALLY, so
    # first-wins is already exact and re-scoring buys nothing. Re-score only
    # when order can actually matter (the same predicate that decides whether
    # the exact pass bothers with a permutation search).
    # `_ranking_score` is untouched, so B7's discard-side sweep does not move.
    best_by_set: dict[frozenset[int], tuple[float, list[Card], tuple]] = {}
    for cards in raw:
        if not cards:
            continue
        key = frozenset(id(c) for c in cards)
        if key in best_by_set and not _needs_permutation_search(cards, jokers):
            continue
        held = [c for c in hand if id(c) not in key]
        cheap, _ = _ranking_score(
            cards, held, jokers, hand_levels, blind, rng, game_state, blind_chips
        )
        prev = best_by_set.get(key)
        if prev is None:
            # The family is a property of the SET (a splash-agnostic scan),
            # so it is order-invariant -- compute it once per set, not once
            # per emission.
            family = _line_family(
                cards,
                four_fingers=four_fingers,
                shortcut=shortcut,
                smeared=bool(flags.get("smeared", False)),
            )
            best_by_set[key] = (cheap.total, cards, family)
        elif cheap.total > prev[0]:
            best_by_set[key] = (cheap.total, cards, prev[2])
    scored = [(family, cards, val) for val, cards, family in best_by_set.values()]
    scored.sort(key=lambda t: t[2], reverse=True)

    # Group variants under their line. `scored` is best-first, and dicts
    # preserve insertion order, so a family's rank IS its best variant's
    # cheap rank, and each family's variants stay best-first within it.
    by_family: dict[tuple, list[list[Card]]] = {}
    for family, cards, _val in scored:
        by_family.setdefault(family, []).append(cards)
    families = list(by_family)

    # Pair pin (user call, 2026-07-14): the best ALREADY-REALIZED rank line
    # (pair or better) is guaranteed a slot in every k>=2 cut. Its cheap
    # rank is weak -- base score of a low pair loses to any flashy draw
    # line -- but it is the one line that needs no draw and no luck, and
    # rank-triggered jokers can make its exact value far exceed its cheap
    # rank. Promoted to index 1 (never displacing the overall best).
    pair_idx = next(
        (i for i, f in enumerate(families) if f[0] in _RANK_LINE_TYPES), None
    )
    if pair_idx is not None and pair_idx > 1:
        families.insert(1, families.pop(pair_idx))

    out: list[list[Card]] = []
    for family in families[:top_k]:
        out.extend(by_family[family])
    return out


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
    prescreen_top_k: int | None = None,
) -> tuple[list[Card], ScoreResult]:
    """No discards being considered. Exact evaluation of the prescreened
    template-derived candidates (see `prescreen_play_candidates`);
    `prescreen_top_k` overrides the module default k there
    (None = PRESCREEN_TOP_K).

    The prescreen runs at EVERY hand size (K3, 2026-07-17): the old
    `n > PRESCREEN_HAND_LIMIT` seam is deleted, so the label is uniformly
    "exact among prescreened candidates" rather than "brute at n<=8, screened
    above". That seam was where B5's residual hid -- n=8 brute-forcing meant
    the box was never measured there. Gate + accepted residuals: see the
    PRESCREEN_TOP_K block.

    `search_orderings` -- see `evaluate_value`; forwarded per subset."""
    best_subset: list[Card] | None = None
    best_result: ScoreResult | None = None

    flags = get_hand_eval_flags(jokers)
    candidates = prescreen_play_candidates(
        hand,
        jokers,
        hand_levels,
        blind,
        rng,
        four_fingers=flags["four_fingers"],
        shortcut=flags["shortcut"],
        smeared=flags["smeared"],
        top_k=prescreen_top_k if prescreen_top_k is not None else PRESCREEN_TOP_K,
        game_state=game_state,
        blind_chips=blind_chips,
        eval_flags=flags,
    )
    subset_pools: list = [candidates]

    for pool in subset_pools:
        for combo in pool:
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
    smeared: bool = False,
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

    templates = build_templates(
        hand, four_fingers=four_fingers, shortcut=shortcut, smeared=smeared
    )

    for template in templates:
        hold, discard = construct_hold(hand, template)
        already_have = len(hold)
        still_needed = max(0, template.needed - already_have)
        discard, kept = cap_discard(discard)  # engine caps a discard at 5 cards
        draws = len(discard)  # cards redrawn = cards discarded (hand size held constant)

        # A legal discard is 1-5 cards. `cap_discard` clamps the UPPER bound;
        # this clamps the LOWER. The discard here is a template COMPLEMENT
        # (hand minus the matching `hold`), not a move chosen from the legal
        # action space, so a template matching the WHOLE hand yields discard=[]
        # -- "discard nothing", which the engine cannot execute. That is the
        # same "solver proposes an action the engine can't run" class as the
        # 6-8 card discard `cap_discard` was added for, just the other bound.
        # See `rank_templates_cheaply` for the full root-cause note.
        if not discard:
            continue

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
    smeared: bool = False,
    top_k: int = 4,
    jokers: list[Card] | None = None,
    game_state: dict | None = None,
    blind_chips: int = 0,
    joker_aware: bool = True,
) -> list[tuple[Template, list[Card], list[Card], list[Card], float, float, int]]:
    """Returns up to `top_k` (template, hold, kept, discard, p_reach,
    cheap_value, still_needed) tuples, ranked by p_reach * cheap_value.

    RANKING FIDELITY (B7, user-locked 2026-07-14): when `jokers` is given
    (and `joker_aware` is left on), cheap_value comes from one fixed-order
    `score_hand` with the branch's honest held cards -- the same ranking
    tier as the play prescreen. The historical jokerless/held-empty
    `score_hand_base` path had the failure shape the prescreen fixed: a
    joker- or held-favored line (Greedy's suit, Baron's held Kings) could
    rank below its true value and never reach the exact hit/miss
    recursion. Per-branch held = `kept` plus any (hold+completion)
    overflow beyond the played 5; unknown replacement draws contribute
    nothing (same representative-completion approximation tier). Only the
    RANKING scorer changes -- reachability math and the exact recursion
    valuation are untouched, so labels shift only where a different
    template set gets explored.

    `jokers=None` or `joker_aware=False` selects the old jokerless scorer
    (kept as the comparison arm for `scripts/validate_discard_ranking.py`
    and for legacy callers).

    `hold` contains only template-MATCHING cards (still_needed/eval math
    depends on that); `kept` is excess non-matches retained in hand because
    the discard is capped at DISCARD_LIMIT -- callers reconstructing the
    post-discard hand must include them.
    """
    from jackdaw.engine.scoring import score_hand_base

    templates = build_templates(
        hand, four_fingers=four_fingers, shortcut=shortcut, smeared=smeared
    )
    scored: list[tuple[Template, list[Card], list[Card], list[Card], float, float, int]] = []

    for template in templates:
        hold, discard = construct_hold(hand, template)
        still_needed = max(0, template.needed - len(hold))
        discard, kept = cap_discard(discard)
        draws = len(discard)

        # LOWER-BOUND CLAMP on the discard (K3, 2026-07-18). A legal discard
        # is 1-5 cards; `cap_discard` enforces <=5, this enforces >=1. The
        # discard is a template COMPLEMENT (hand minus `hold`), not a move
        # picked from the legal action space, so a template that matches the
        # WHOLE hand yields discard=[] -- an illegal "discard nothing". This
        # is the mirror of the 6-8 card discard bug `cap_discard` was added
        # for: the solver derives discards as `hand - hold` with no size
        # bound, so it can leave BOTH ends of the engine's 1..5 range.
        #
        # It fires on a Shortcut-widened straight window (or FF-shortened,
        # or an 8-card single-suit flush) that swallows every card:
        # still_needed=0, nothing left to throw. Two facts make this a prune,
        # not a value call:
        #   1. "Discard nothing" is ILLEGAL -- reached the executability
        #      check as GenerationError on 2 stage2 relabel seeds
        #      (00001300 FF+Shortcut, 00002997 Jolly+Shortcut).
        #   2. It is a PHANTOM of a real discard: it keeps all cards and
        #      recurses at discards_left-1, and a real discard toward the
        #      same completion reaches the identical continuation one discard
        #      cheaper, so it TIES the phantom's p_clear (measured: 00001300
        #      empty branch 0.4442 vs executable straight_6-9 0.4445;
        #      00002997 both 1.0). Pruning routes the label to that real
        #      discard -- the chain here genuinely favors discarding and the
        #      fix surfaces the LEGAL spelling, it does not suppress it.
        # Placed before scoring so the freed shortlist slot goes to a real
        # candidate; without that, the phantom -- cheap-rank 0, since a
        # complete hand scores highest -- both starves a real discard AND
        # wins the p_clear tie by being first under the strict `>` selection.
        #
        # LATENT UNTIL K3: score_hand got jokers=None (03e288d), so Shortcut
        # never DETECTED -> the completed straight scored as High Card ->
        # low cheap rank AND low p_clear, so the phantom neither floated to
        # the front nor won. The engine fix simultaneously floated it and
        # inflated its p_clear; that is why the 2026-07-16 brute run had 0
        # failures across 4000.
        if not discard:
            continue

        if still_needed == 0:
            p_reach = 1.0
            # KNOWN GAP, deliberately unfixed (K3, 2026-07-17): when a hold
            # OVERFLOWS (Shortcut widens a straight window, Four Fingers adds
            # shorter ones, a 6+ card flush), `[:5]` seats the first five in
            # HAND ORDER -- not even `_keep_priority`, let alone joker-aware.
            # Same class as the play-side seat blindness that failed K3's
            # tail arm (capture 0.915; hold `7S 6S 5S 4S 3C 2D` seated
            # `7-6-5-4-3` when Wee Joker wanted the 2), and the play side is
            # being fixed by scoring the seatings with `_ranking_score`.
            #
            # NOT fixed here because the K spec says the discard side is
            # MEASURE-FIRST, no code change: its completions are IDEALIZED
            # DRAWS, so a seating hypothesis here is a hypothesis about
            # hypothetical cards. It is also a label-semantics change at
            # every hand size, so it would drag B7's whole validated sweep
            # with it (`validate_discard_ranking_sweep.py`, the depth-gate
            # rationale) for a benefit nobody has measured.
            # The tripwire is the spec's: rerun that sweep on stage2-config
            # density and look for the directional signature (dropping
            # discards whose value lives in kickers/held cards).
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
        # many candidate templates per decision, and the scorer mutates
        # history-dependent boss state (The Eye/The Mouth) on every
        # hypothetical call, not just the eventually-chosen one.
        if joker_aware and jokers is not None:
            eval_ids = {id(c) for c in eval_cards}
            branch_held = [c for c in list(hold) + kept if id(c) not in eval_ids]
            cheap_result, _ = _ranking_score(
                eval_cards, branch_held, jokers, hand_levels, blind, rng,
                game_state, blind_chips,
            )
        else:
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
    smeared: bool = False,
    top_k: int | None = None,
    hand_size: int | None = None,
    joker_aware: bool = True,
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

    `top_k` sizes the discard shortlist. `None` (production) uses the
    depth-gate `_discard_shortlist_k`: width 6 while `discards_left <= 2`,
    width 4 deeper -- widening only the shallow chains where B7's
    boundary regressions live, keeping the narrow box on deep chains to cap
    the `(k/4)^discards_left` cost tail. An explicit int forces a fixed box
    at every node (validation harness / sweep / existence-proof tests).

    `joker_aware` selects the discard-shortlist ranker (B7): the default
    True is production (joker/held-aware `rank_templates_cheaply`); False
    forces the legacy jokerless/held-empty scorer at EVERY node and exists
    only as the comparison arm for B7 validation and existence-proof tests.
    It is threaded through the recursion so an old-vs-new full-solver
    comparison differs at every shortlist cut, not just the root.
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

    # top_k=None (production) -> depth-gated width; an explicit int is a fixed
    # box (validation/tests). Re-evaluated per node off THIS node's
    # discards_left, so a chain widens as it approaches its leaf.
    effective_top_k = top_k if top_k is not None else _discard_shortlist_k(discards_left)
    candidates = rank_templates_cheaply(
        hand, deck, hand_levels, blind, rng,
        four_fingers=four_fingers, shortcut=shortcut, smeared=smeared, top_k=effective_top_k,
        jokers=jokers, game_state=game_state, blind_chips=blind_chips,
        joker_aware=joker_aware,
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
                four_fingers=four_fingers, shortcut=shortcut, smeared=smeared, top_k=top_k,
                hand_size=hand_size,
                joker_aware=joker_aware,
            )
            p_clear = hit_choice.p_clear
        else:
            hit_priority = _best_completion_cards(deck, template, still_needed)
            hit_hand, hit_drawn = _fill_hand_to_size(deck, base_hold, hit_priority, hand_size)
            hit_deck = deck.without(hit_drawn)
            hit_choice = solve_hand_turn(
                hit_hand, jokers, hand_levels, blind, rng, hit_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, smeared=smeared, top_k=top_k,
                hand_size=hand_size,
                joker_aware=joker_aware,
            )

            miss_priority = _representative_miss_cards(deck, template, still_needed)
            miss_hand, miss_drawn = _fill_hand_to_size(deck, base_hold, miss_priority, hand_size)
            miss_deck = deck.without(miss_drawn)
            miss_choice = solve_hand_turn(
                miss_hand, jokers, hand_levels, blind, rng, miss_deck, chips_needed,
                hands_left, discards_left - 1, future_samples, game_state, blind_chips,
                four_fingers=four_fingers, shortcut=shortcut, smeared=smeared, top_k=top_k,
                hand_size=hand_size,
                joker_aware=joker_aware,
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
    smeared: bool = False,
    top_k: int | None = None,
    mc_seed: str | None = None,
) -> AnteClearChoice:
    """Top-level entry point: computes the future-hand distribution once
    (for `hands_left - 1` downstream hand-turns) and kicks off the
    recursive discard-chain solve. This is what to call from a live
    decision point -- `solve_hand_turn` is the internal recursive worker.

    Pass `mc_seed` (e.g. the episode seed) whenever the output becomes a
    training label -- it makes the future-hand MC draws, and therefore
    p_clear, reproducible.

    `top_k` defaults to `None` = the production discard-shortlist depth gate
    (6 at `discards_left <= 2`, 4 deeper; see `_discard_shortlist_k` and
    `solve_hand_turn`). Pass an explicit int only to force a fixed box.
    """
    if mc_seed is not None:
        # `prob_clear_given_future`'s tiny LCG carries state across calls
        # within a process; reset it per solve or p_clear would depend on
        # how many solves this worker ran before this one.
        import zlib

        _rand_state[0] = zlib.crc32(mc_seed.encode()) & 0x7FFFFFFF or 12345
    # B3: closed-form joker sort ONCE per hand-turn (subset-independent) --
    # the whole solve (ranking tier, MC sampler, recursion) then runs under
    # this order; the exact path additionally re-runs the copy-target
    # argmax per candidate inside `evaluate_value`. The env commits plays
    # under the same ordering (action_to_engine_action), so labels and
    # execution agree.
    jokers = best_joker_order(jokers)
    future_samples = estimate_future_hand_distribution(
        deck, jokers, hand_levels, blind, rng, len(hand),
        game_state=game_state, blind_chips=blind_chips, mc_seed=mc_seed,
    )
    return solve_hand_turn(
        hand, jokers, hand_levels, blind, rng, deck, chips_needed, hands_left, discards_left,
        future_samples, game_state, blind_chips,
        four_fingers=four_fingers, shortcut=shortcut, smeared=smeared, top_k=top_k,
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
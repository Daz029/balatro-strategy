"""Tests for the K1 kicker-variant emitter in the big-hand play prescreen.

The prescreen proposed the right scoring LINE and the wrong KICKERS.
Measured at n=8 on stage2's joker-dense pool: 0.845 score-capture, misses
up to 90% of the play's value, and INVARIANT in k (27/27 misses had the
right line at true_size 5) -- so the candidate GENERATOR was starved and no
k rescues it. `_kicker_pad`'s keep-priority nominal-best padding assumed
kickers are inert filler, which is false under Splash (kickers score) and
Raised Fist (the cards you RETAIN set the mult).

K1 turns the pad into a hypothesis-gated variant emitter. A variant is the
GREEDY argmax completion of a line under ONE hypothesis about where kicker
value lives; there are no magnitudes anywhere, because the exact pass
arbitrates. These tests pin the gates, the four hypotheses, the
config-derived held-enhancement test, and the redefined family key.

Full measurement record: docs/bruteforce_speedup_and_kicker_design.md.
Decision record: CLAUDE.md, "Kicker variants + prescreen-at-n=8".
"""

from __future__ import annotations

import itertools

import hand_solver
from hand_solver import best_immediate_play, evaluate_value, prescreen_play_candidates

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_eval import get_hand_eval_flags
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _fixtures() -> tuple[HandLevels, Blind, PseudoRandom]:
    return HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("kicker")


# Hypothesis 5 (type-upgrade) reads the detection flags. These per-hypothesis
# tests isolate hypotheses 1-4, whose keys are flag-independent, so they pass
# the all-false dict. `_kicker_variants` takes it as a REQUIRED argument on
# purpose: an empty default would let a real call site silently claim "no
# Four Fingers / Shortcut / Smeared" and mislabel the board.
_NO_FLAGS: dict[str, bool] = {"four_fingers": False, "shortcut": False, "smeared": False}


def _enhanced(suit: Suit, rank: Rank, enhancement: str) -> Card:
    """Enhancement via the engine's own `set_ability`, so held-channel
    config (h_x_mult / h_dollars) actually populates. Poking `center_key`
    directly -- as the sibling prescreen suite's `_card` helper does --
    leaves `ability` untouched, and every held test here would pass
    vacuously."""
    c = create_playing_card(suit, rank)
    c.set_ability(enhancement)
    return c


def _flush_plus_junk_hand() -> list[Card]:
    """10 cards: a complete heart flush plus low offsuit junk."""
    return [
        create_playing_card(Suit.HEARTS, Rank.ACE),
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.HEARTS, Rank.NINE),
        create_playing_card(Suit.HEARTS, Rank.SEVEN),
        create_playing_card(Suit.HEARTS, Rank.FOUR),
        create_playing_card(Suit.SPADES, Rank.TWO),
        create_playing_card(Suit.CLUBS, Rank.THREE),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.SPADES, Rank.SIX),
        create_playing_card(Suit.CLUBS, Rank.EIGHT),
    ]


def _quads_and_flush_hand() -> list[Card]:
    """11 cards: four Kings AND five non-King hearts -- two strong families."""
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.KING),
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.HEARTS, Rank.QUEEN),
        create_playing_card(Suit.HEARTS, Rank.TEN),
        create_playing_card(Suit.HEARTS, Rank.EIGHT),
        create_playing_card(Suit.HEARTS, Rank.SIX),
        create_playing_card(Suit.HEARTS, Rank.THREE),
        create_playing_card(Suit.SPADES, Rank.FOUR),
        create_playing_card(Suit.CLUBS, Rank.SEVEN),
    ]


def _brute_force_best(hand: list[Card], jokers: list[Card]) -> float:
    """The pre-prescreen behaviour: exact evaluation of every subset."""
    hand_levels, blind, rng = _fixtures()
    best = None
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(hand, size):
            combo_ids = {id(c) for c in combo}
            held = [c for c in hand if id(c) not in combo_ids]
            result = evaluate_value(list(combo), held, jokers, hand_levels, blind, rng)
            if best is None or result.total > best.total:
                best = result
    assert best is not None
    return best.total


def _gates_for(hand: list[Card], jokers: list[Card]) -> hand_solver._KickerGates:
    flags = get_hand_eval_flags(jokers)
    views = hand_solver._resolved_joker_views(jokers)
    counts = hand_solver._card_channel_counts(hand, views, {}, flags)
    return hand_solver._kicker_gates(hand, views, flags, counts)


class TestGates:
    def test_plain_board_gates_all_off(self):
        """No Splash, no held-channel joker, no held enhancement -> every
        hypothesis collapses onto nominal-best, so the fix costs a plain
        board nothing. This is what keeps the budget adaptive, not 3x flat."""
        gates = _gates_for(_flush_plus_junk_hand(), [])
        assert not gates.scored_value
        assert not gates.held_value
        assert not gates.play_away_lowest

    def test_splash_opens_scored_value_gate(self):
        """Splash is the only way a non-line card scores. It is class-3
        (all-zero) in the trigger matrix BY DESIGN, so the gate reads
        `get_hand_eval_flags` -- the matrix cannot answer this."""
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_splash")])
        assert gates.scored_value

    def test_raised_fist_opens_play_away_gate(self):
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_raised_fist")])
        assert gates.play_away_lowest

    def test_held_channel_joker_opens_held_gate(self):
        """Baron reads HELD Kings -- the taxonomy's held channel, which is
        what makes hypothesis 3 principled rather than a joker wishlist."""
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_baron")])
        assert gates.held_value

    def test_steel_and_gold_open_held_gate_without_any_joker(self):
        for enhancement in ("m_steel", "m_gold"):
            hand = _flush_plus_junk_hand()[:-1] + [
                _enhanced(Suit.CLUBS, Rank.EIGHT, enhancement)
            ]
            assert _gates_for(hand, []).held_value, enhancement

    def test_debuffed_joker_gates_nothing(self):
        fist = create_joker("j_raised_fist")
        fist.debuff = True
        assert not _gates_for(_flush_plus_junk_hand(), [fist]).play_away_lowest


class TestHeldEnhancementDetection:
    """The held test is read from the engine's own config
    (h_x_mult / h_mult / h_dollars), never a {"Steel Card", "Gold Card"}
    name set -- which would be correct on today's content and rot the
    moment an enhancement is added or rebalanced."""

    def test_steel_detected_via_h_x_mult(self):
        steel = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_steel")
        assert steel.get_chip_h_x_mult() > 0
        assert hand_solver._has_held_enhancement(steel)

    def test_gold_detected_via_h_dollars(self):
        """Gold has no accessor -- its held payout is a raw ability read."""
        gold = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_gold")
        assert gold.ability.get("h_dollars", 0) > 0
        assert hand_solver._has_held_enhancement(gold)

    def test_scored_channel_enhancement_has_no_held_value(self):
        """Glass is x_mult when PLAYED -- worth nothing held, so it must
        stay fair game as a kicker."""
        glass = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_glass")
        assert not hand_solver._has_held_enhancement(glass)

    def test_plain_card_has_no_held_value(self):
        assert not hand_solver._has_held_enhancement(
            create_playing_card(Suit.CLUBS, Rank.EIGHT)
        )

    def test_debuffed_steel_has_no_held_value(self):
        """The accessors return 0 for a debuffed card, so a debuffed Steel
        card is correctly spendable as a kicker."""
        steel = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_steel")
        steel.debuff = True
        assert not hand_solver._has_held_enhancement(steel)


class TestCopyResolution:
    """Gates read RESOLVED joker identities through the ENGINE's own copy
    path, never raw center keys."""

    def test_blueprint_gates_as_the_joker_it_copies(self):
        jokers = [create_joker("j_blueprint"), create_joker("j_lusty_joker")]
        views = hand_solver._resolved_joker_views(jokers)
        assert [k for k, _ in views] == ["j_lusty_joker", "j_lusty_joker"]

    def test_blueprint_cannot_manufacture_a_splash_gate(self):
        """Splash is on the 29-joker blueprint_compat incompat list, so a
        copy can never produce one. That comes free from resolving through
        the engine; a hand-rolled resolver would have to remember it."""
        jokers = [create_joker("j_blueprint"), create_joker("j_splash")]
        views = hand_solver._resolved_joker_views(jokers)
        assert views[0][0] != "j_splash", "Blueprint must not resolve to Splash"

    def test_unresolved_copy_joker_gates_nothing(self):
        """A lone Blueprint has no right-neighbour: it keeps its own key,
        which carries no predicate (class 4), so it gates nothing."""
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_blueprint")])
        assert not gates.scored_value
        assert not gates.play_away_lowest


class TestHypotheses:
    def test_raised_fist_variant_spends_the_lowest_cards(self):
        """Hypothesis 4 is exact for Raised Fist's min term: to raise the
        lowest HELD rank you spend the lowest cards as kickers."""
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:3]
        gates = hand_solver._KickerGates(False, False, True)
        variants = hand_solver._kicker_variants(line, hand, gates, counts, _NO_FLAGS)
        line_ids = {id(c) for c in line}
        kickers = [c for c in variants[-1] if id(c) not in line_ids]
        junk_ids = sorted(
            c.base.id for c in hand if c.base.suit != Suit.HEARTS.value
        )
        assert sorted(c.base.id for c in kickers) == junk_ids[: len(kickers)]

    def test_raised_fist_key_ranks_stone_cards_last(self):
        """A Stone card is excluded from Raised Fist's minimum, so playing
        one away cannot raise it -- it must never be the spent kicker."""
        stone = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_stone")
        two = create_playing_card(Suit.SPADES, Rank.TWO)
        assert hand_solver._raised_fist_key(two) < hand_solver._raised_fist_key(stone)

    def test_raised_fist_key_uses_rank_id_not_nominal(self):
        """`nominal` ties J/Q/K at 10; the handler compares `get_id()`,
        which distinguishes them. Using nominal would make the min term
        wrong on face-heavy hands."""
        jack = create_playing_card(Suit.SPADES, Rank.JACK)
        king = create_playing_card(Suit.SPADES, Rank.KING)
        assert jack.base.nominal == king.base.nominal
        assert hand_solver._raised_fist_key(jack) < hand_solver._raised_fist_key(king)

    def test_scored_variant_prefers_edition_kickers(self):
        """Editions fire on the SCORED channel, so under Splash a
        Polychrome 3 beats plain junk as a kicker. They are absent from
        trigger_match by design (it is a card x JOKER matrix), so the key
        carries the term itself -- caught in review 2026-07-16, and missing
        from the locked spec's "chips + enhancement + candidacy bits"."""
        hand = _flush_plus_junk_hand()
        junk = [c for c in hand if c.base.suit != Suit.HEARTS.value]
        poly = min(junk, key=lambda c: c.base.nominal)  # the WORST junk card
        poly.set_edition({"polychrome": True})
        counts = {id(c): (0, 0) for c in hand}
        key = hand_solver._scored_kicker_key(counts)
        assert all(key(poly) > key(o) for o in junk if o is not poly), (
            "a Polychrome kicker must outrank higher plain junk under Splash"
        )

    def test_held_variant_keeps_baron_kings_in_hand(self):
        """Hypothesis 3 pads with the cards LEAST valuable held, so Baron's
        Kings stay in hand rather than being spent as kickers."""
        hand = _quads_and_flush_hand()
        jokers = [create_joker("j_baron")]
        flags = get_hand_eval_flags(jokers)
        views = hand_solver._resolved_joker_views(jokers)
        counts = hand_solver._card_channel_counts(hand, views, {}, flags)
        line = [
            c
            for c in hand
            if c.base.suit == Suit.HEARTS.value and c.base.id != 13
        ][:3]
        gates = hand_solver._KickerGates(False, True, False)
        variants = hand_solver._kicker_variants(line, hand, gates, counts, _NO_FLAGS)
        line_ids = {id(c) for c in line}
        # variants == [nominal-best, held-value, type-upgrade] under these
        # gates. Index 1, NOT [-1]: hypothesis 5 is ungated and appends
        # after every gated one, so [-1] silently stopped being the
        # held-value variant this test is about.
        kickers = [c for c in variants[1] if id(c) not in line_ids]
        assert all(c.base.id != 13 for c in kickers), (
            "held-value variant must not spend Baron's Kings as kickers"
        )

    def test_nominal_best_variant_is_always_first(self):
        """The pre-K1 behaviour is kept as variant 1 on every board."""
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:3]
        all_on = hand_solver._KickerGates(True, True, True)
        off = hand_solver._KickerGates(False, False, False)
        assert (
            hand_solver._kicker_variants(line, hand, all_on, counts, _NO_FLAGS)[0]
            == hand_solver._kicker_variants(line, hand, off, counts, _NO_FLAGS)[0]
        )

    def test_variants_dedupe_by_card_identity(self):
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:3]
        all_on = hand_solver._KickerGates(True, True, True)
        variants = hand_solver._kicker_variants(line, hand, all_on, counts, _NO_FLAGS)
        keys = [frozenset(id(c) for c in v) for v in variants]
        assert len(keys) == len(set(keys))

    def test_full_line_emits_no_variants(self):
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:5]
        all_on = hand_solver._KickerGates(True, True, True)
        assert hand_solver._kicker_variants(line, hand, all_on, counts, _NO_FLAGS) == [line]


class TestFamilyKey:
    def test_family_key_is_splash_agnostic(self):
        """THE pitfall-#13 guard. Under Splash `score_hand`'s scoring_cards
        is all 5 played cards, so keying on it would give every kicker
        variant of one line a DIFFERENT family -- the variants would crowd
        out genuinely distinct lines, recreating the exact bug the family
        pass exists to prevent, on precisely the boards K1 targets. The
        engine's detection scan ignores Splash, so variants collide."""
        hand = _quads_and_flush_hand()
        kings = [c for c in hand if c.base.id == 13]
        junk = [c for c in hand if c.base.suit != Suit.HEARTS.value and c.base.id != 13]
        hearts = [c for c in hand if c.base.suit == Suit.HEARTS.value and c.base.id != 13]

        family_a = hand_solver._line_family(
            kings + junk[:1], four_fingers=False, shortcut=False, smeared=False
        )
        family_b = hand_solver._line_family(
            kings + hearts[:1], four_fingers=False, shortcut=False, smeared=False
        )
        assert family_a == family_b, "same line, different kicker -> same family"
        assert family_a[0] == "Four of a Kind"

    def test_family_key_separates_distinct_rank_lines(self):
        """The card SET stays in the key: hand type alone would collapse a
        pair of Kings and a pair of 3s into one family."""
        hand = _quads_and_flush_hand()
        kings = [c for c in hand if c.base.id == 13][:2]
        hearts = [c for c in hand if c.base.suit == Suit.HEARTS.value and c.base.id != 13]
        pair_of_kings = hand_solver._line_family(
            kings, four_fingers=False, shortcut=False, smeared=False
        )
        high_card = hand_solver._line_family(
            hearts[:1], four_fingers=False, shortcut=False, smeared=False
        )
        assert pair_of_kings != high_card

    def test_family_key_tolerates_baseless_stone_cards(self):
        """Stone cards have no base; `get_best_hand` groups but never emits
        them, so the scan must not raise (the existing fallback path)."""
        stone = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_stone")
        stone.base = None
        family = hand_solver._line_family(
            [stone], four_fingers=False, shortcut=False, smeared=False
        )
        assert isinstance(family[0], str)


class TestVariantsRideTheirLine:
    def test_top_k_counts_lines_not_candidates(self):
        """`top_k` caps FAMILIES; every surviving line carries all of its
        variants into the exact pass, which arbitrates between them."""
        hand = _quads_and_flush_hand()
        jokers = [create_joker("j_splash")]
        hand_levels, blind, rng = _fixtures()
        for k in (1, 3, 5):
            candidates = prescreen_play_candidates(
                hand, jokers, hand_levels, blind, rng, top_k=k, game_state={}
            )
            families = {
                hand_solver._line_family(
                    cards, four_fingers=False, shortcut=False, smeared=False
                )
                for cards in candidates
            }
            assert len(families) <= k

    def test_budget_stays_far_under_brute_force(self):
        """The whole point: ~15 candidates against brute force's 218."""
        hand = _quads_and_flush_hand()
        hand_levels, blind, rng = _fixtures()
        jokers = [create_joker("j_splash"), create_joker("j_raised_fist")]
        candidates = prescreen_play_candidates(
            hand, jokers, hand_levels, blind, rng, top_k=5, game_state={}
        )
        assert len(candidates) < 60, (
            f"variant fan-out must stay well under 218; got {len(candidates)}"
        )


def _trips_plus_kicker_bait_hand() -> list[Card]:
    """9 cards engineered to reproduce the K1 miss.

    Trip Kings are the LINE and are never in doubt -- what varies is the two
    KICKERS. Nominal-best padding grabs the Ace and Queen (highest
    keep-priority); the low hearts are what a suit joker under Splash
    actually wants, and the low cards are what Raised Fist wants played
    away. So a generator that pads by nominal rank proposes the right line
    at the right size and still loses badly, which is exactly the measured
    signature (27/27 misses had `true_size = 5`).

    Note the sibling suite's `_flush_plus_junk_hand` CANNOT show this: its
    flush is complete and dominant, so no kicker choice exists to get wrong
    -- those tests pass on pre-K1 code and prove nothing about the fix.
    """
    return [
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.KING),
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.ACE),
        create_playing_card(Suit.CLUBS, Rank.QUEEN),
        create_playing_card(Suit.SPADES, Rank.JACK),
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.HEARTS, Rank.THREE),
        create_playing_card(Suit.DIAMONDS, Rank.FOUR),
    ]


def _ff_straight_flush_bait_hand() -> list[Card]:
    """9 cards (so the prescreen path fires) reproducing K3 arm-C miss
    stage3_full_00001545. Four hearts Q-7-5-4 are a flush under Four
    Fingers; 7-6-5-4 is a straight under it too, and the 6 is a CLUB. So the
    winning 5 are the union of a flush template's cards and a straight
    template's -- no single predicate proposes that set, and no per-card
    kicker key ranks the club 6 above the King.
    """
    return [
        create_playing_card(Suit.CLUBS, Rank.KING),
        create_playing_card(Suit.HEARTS, Rank.QUEEN),
        create_playing_card(Suit.HEARTS, Rank.SEVEN),
        create_playing_card(Suit.CLUBS, Rank.SIX),
        create_playing_card(Suit.DIAMONDS, Rank.SIX),
        create_playing_card(Suit.HEARTS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.FOUR),
        create_playing_card(Suit.CLUBS, Rank.THREE),
        create_playing_card(Suit.SPADES, Rank.TWO),
    ]


def _smeared_flush_bait_hand() -> list[Card]:
    """9 cards reproducing K3 arm-C miss stage3_full_00003496. Under Smeared
    the five black cards Q(C) 9(S) 8(S) 3(C) 2(C) are a Flush; the raw-suit
    templates see only 3 clubs and 2 spades and offer two pair instead.
    """
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.QUEEN),
        create_playing_card(Suit.DIAMONDS, Rank.QUEEN),
        create_playing_card(Suit.SPADES, Rank.NINE),
        create_playing_card(Suit.SPADES, Rank.EIGHT),
        create_playing_card(Suit.DIAMONDS, Rank.EIGHT),
        create_playing_card(Suit.CLUBS, Rank.THREE),
        create_playing_card(Suit.CLUBS, Rank.TWO),
        create_playing_card(Suit.HEARTS, Rank.SEVEN),
    ]


def _wild_flush_bait_hand() -> list[Card]:
    """9 cards where the only 5-card flush needs the WILD card to stand in
    as a spade. No joker required -- Wild is a card enhancement, so this
    board was mislabeled long before the detection-flag bug existed.

    Making this board DISCRIMINATE took two goes, and the reason is worth
    keeping. The line is 4 natural spades, so exactly one pad slot decides
    the flush -- and `_keep_priority` is `(enhanced, nominal)`, which ranks
    ANY enhanced card top. A wild that is the pool's best keep-priority card
    therefore gets padded in by nominal-best for the wrong reason, and the
    test passes on broken code (draft 1: a wild 9H was the highest card;
    draft 2: a wild 2H still won on the `enhanced` bit). The Glass ACE
    outranks the wild 2 on the same key, so nominal-best takes the Ace and
    only a suit predicate that actually asks the engine finds the flush.
    """
    wild = create_playing_card(Suit.HEARTS, Rank.TWO)
    wild.set_ability("m_wild")
    decoy = create_playing_card(Suit.DIAMONDS, Rank.ACE)
    decoy.set_ability("m_glass")
    return [
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.QUEEN),
        create_playing_card(Suit.SPADES, Rank.JACK),
        create_playing_card(Suit.SPADES, Rank.FOUR),
        wild,
        decoy,
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.SEVEN),
        create_playing_card(Suit.HEARTS, Rank.SIX),
    ]


def _order_lottery_hand() -> list[Card]:
    """12 cards reproducing K3 tail miss stage2_curated_00002797.

    Three Kings and two 3s (so the rank-combination pass emits the full
    house from BOTH permutation directions -- that is the point), plus a
    J-10-8-6-5 shortcut straight to be the rival family that wins when the
    full house is buried, and enough spare cards to reach a tail hand size.
    """
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.KING),
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.HEARTS, Rank.JACK),
        create_playing_card(Suit.CLUBS, Rank.TEN),
        create_playing_card(Suit.CLUBS, Rank.EIGHT),
        create_playing_card(Suit.SPADES, Rank.SIX),
        create_playing_card(Suit.SPADES, Rank.FIVE),
        create_playing_card(Suit.DIAMONDS, Rank.FOUR),
        create_playing_card(Suit.SPADES, Rank.THREE),
        create_playing_card(Suit.CLUBS, Rank.THREE),
        create_playing_card(Suit.SPADES, Rank.TWO),
    ]


class TestRegressionAgainstBruteForce:
    """The tests that actually catch the bug: each one FAILS on pre-K1 code
    with the regret recorded in its docstring, measured on this branch.

    A prescreened label is "exact among prescreened candidates", so
    agreement with full brute force on these boards is the property K1
    exists to restore.
    """

    def test_splash_plus_suit_joker(self):
        """Lusty under Splash -- kicker SUIT feeds the mult. Pre-K1 pads the
        trip Kings with the Ace by nominal rank and scores 444 against brute
        force's 585: REGRET 141, 24.1% of the play's value. The same
        interaction produced the doc's seed 879 (regret 279).
        """
        hand = _trips_plus_kicker_bait_hand()
        jokers = [create_joker("j_splash"), create_joker("j_lusty_joker")]
        hand_levels, blind, rng = _fixtures()
        subset, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)
        # The fix is specifically that BOTH hearts get played as kickers.
        assert sum(1 for c in subset if c.base.suit == Suit.HEARTS.value) == 2

    def test_raised_fist(self):
        """Raised Fist reads the lowest HELD card, so the cards you DON'T
        play set the mult -- it was 16 of the 27 measured misses. Pre-K1
        strands the 2 in hand and scores 540 against brute force's 660:
        REGRET 120, 18.2% of the play's value.
        """
        hand = _trips_plus_kicker_bait_hand()
        jokers = [create_joker("j_raised_fist")]
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)

    def test_type_upgrading_kicker_four_fingers_straight_flush(self):
        """Hypothesis 5. Under Four Fingers the flush needs 4 hearts and the
        straight needs 4 of 7-6-5-4, and vanilla lets those be DIFFERENT
        cards -- so padding the 4-card heart flush with the 6 of CLUBS makes
        a Straight Flush.

        Every other hypothesis sorts the pool by a per-card key and so ranks
        the club 6 below the King; only a hypothesis that asks "what does
        this pad make the HAND" can see it. From the K3 arm-C miss
        stage3_full_00001545: regret 892, 73.4% of the play's value, the
        largest single miss in the arm.
        """
        hand = _ff_straight_flush_bait_hand()
        jokers = [create_joker("j_four_fingers")]
        hand_levels, blind, rng = _fixtures()
        subset, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)
        # The point is specifically that an off-suit card is played as the pad.
        assert any(c.base.suit != Suit.HEARTS.value for c in subset)

    def test_smeared_flush_is_proposable(self):
        """Bug C. Smeared merges Hearts=Diamonds / Spades=Clubs for flush
        purposes, so `QC 9S 8S 3C 2C` is a Flush. The flush templates
        compared raw suit identity, so no template could ever gather that
        set and the solver offered two pair instead.

        From the K3 arm-C miss stage3_full_00003496: regret 200, 37.3%.
        Invisible before the engine fix (03e288d) because Smeared was inert.
        """
        hand = _smeared_flush_bait_hand()
        jokers = [create_joker("j_smeared")]
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)

    def test_wild_card_counts_toward_every_flush(self):
        """The third rule the old suit-identity predicate got wrong, and the
        one that was NEVER flag-gated -- a Wild Card counts as every suit for
        flush purposes, so it completes a 4-spade flush. Independent of the
        engine fix; it was simply always broken.
        """
        hand = _wild_flush_bait_hand()
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, [], hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, [])

    def test_type_upgrade_variant_never_repeats_a_card(self):
        """The greedy loop removes its pick by `id(c) != id(best)`. Written
        with `is not`, it compares two large ints by OBJECT identity -- never
        equal, so nothing is removed and the same card is emitted five times
        (caught in review; the box contained ['8S','8D','8D','8D','8D']).
        A duplicated card is not a legal play, so this must stay pinned.
        """
        hand = _ff_straight_flush_bait_hand()
        jokers = [create_joker("j_four_fingers")]
        flags = get_hand_eval_flags(jokers)
        views = hand_solver._resolved_joker_views(jokers)
        counts = hand_solver._card_channel_counts(hand, views, {}, flags)
        gates = hand_solver._kicker_gates(hand, views, flags, counts)
        line = hand[:2]
        for variant in hand_solver._kicker_variants(line, hand, gates, counts, flags):
            assert len({id(c) for c in variant}) == len(variant), variant

    def test_emission_order_does_not_bury_a_family(self):
        """The cheap tier is FIXED-ORDER, so it scores whatever order the
        generator emitted; deduping by card-identity SET and keeping the
        first therefore made a family's cheap rank a lottery.

        From K3 tail miss stage2_curated_00002797 (Photograph x2 on the first
        face scored + Hanging Chad retriggering the first card):
        `permutations(multi, 2)` yields (threes, kings) BEFORE (kings,
        threes), so the full house was first emitted threes-first and scored
        1408, and the kings-first emission of the SAME FIVE CARDS -- worth
        4080 -- was skipped as a duplicate. At 1408 it ranked ~7th, was cut
        at top_k=5, and the exact pass never saw a full house at all: it lost
        to a 3008 straight.

        Taking the MAX over emitted orders recovers it. Asserted through
        `best_immediate_play` (the exact pass), so it fails if the family is
        cut before reaching it.
        """
        hand = _order_lottery_hand()
        jokers = [
            create_joker("j_shortcut"),
            create_joker("j_photograph"),
            create_joker("j_hack"),
            create_joker("j_hanging_chad"),
            create_joker("j_jolly"),
        ]
        hand_levels, blind, rng = _fixtures()
        subset, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)
        # Specifically: the full house must be found, not a straight.
        assert result.hand_type == "Full House", result.hand_type
        assert sorted(c.base.id for c in subset) == [3, 3, 13, 13, 13]

    def test_plain_board_unchanged(self):
        """The other side of the contract: with every gate shut, K1 must not
        move a plain board's label at all."""
        hand = _trips_plus_kicker_bait_hand()
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, [], hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, [])

    def test_dominant_flush_board_unchanged(self):
        """A complete dominant flush has no kicker choice to get wrong, so
        this passes pre-K1 too -- kept as a don't-break-the-easy-case guard,
        NOT as evidence for the fix."""
        hand = _flush_plus_junk_hand()
        jokers = [create_joker("j_splash"), create_joker("j_lusty_joker")]
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)

"""Tests for the per-card x per-joker trigger-match matrix
(``jackdaw/env/trigger_match.py``), the h1 schema bump's B2 slice 2.

The named pitfall cases from docs/pre-regen-handoff.md are all pinned
here: taxonomy coverage (every vocab joker classified), class-2
predicates tracking LIVE state (Ancient's suit rotates), Photograph
marking ALL faces (candidate semantics, not will-fire), and Flower Pot
staying all-zero by design.
"""

from __future__ import annotations

import numpy as np

from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.env.observation import center_key_id, center_key_vocabulary
from jackdaw.env.trigger_match import (
    _CLASS1_PREDICATES,
    _CLASS2_PREDICATES,
    _CLASS3_SET_LEVEL,
    _CLASS4_NON_CARD,
    _check_taxonomy,
    joker_center_key_ids,
    resolve_copy_targets,
    trigger_match_matrix,
)

SCORED, HELD = 0, 1


def _gs(hand, jokers, **extra):
    gs = {"hand": hand, "jokers": jokers, "current_round": {}}
    gs.update(extra)
    return gs


def _cards(*specs):
    return [create_playing_card(s, r) for s, r in specs]


class TestTaxonomyCoverage:
    def test_every_vocab_joker_classified_exactly_once(self):
        # Import already ran the check; run it explicitly too so a broken
        # table fails HERE with the offending keys, not as an ImportError.
        _check_taxonomy()
        vocab = {k for k in center_key_vocabulary() if k.startswith("j_")}
        classified = (
            set(_CLASS1_PREDICATES)
            | set(_CLASS2_PREDICATES)
            | set(_CLASS3_SET_LEVEL)
            | set(_CLASS4_NON_CARD)
        )
        assert classified == vocab
        total = (
            len(_CLASS1_PREDICATES)
            + len(_CLASS2_PREDICATES)
            + len(_CLASS3_SET_LEVEL)
            + len(_CLASS4_NON_CARD)
        )
        assert total == len(vocab)  # disjoint

    def test_shapes_and_empty(self):
        hand = _cards((Suit.HEARTS, Rank.TWO))
        jokers = [create_joker("j_joker")]
        m = trigger_match_matrix(_gs(hand, jokers))
        assert m.shape == (1, 1, 2)
        assert m.dtype == bool
        assert trigger_match_matrix(_gs([], jokers)).shape == (0, 1, 2)
        assert trigger_match_matrix(_gs(hand, [])).shape == (1, 0, 2)

    def test_joker_center_key_ids(self):
        jokers = [create_joker("j_lusty_joker"), create_joker("j_baron")]
        ids = joker_center_key_ids(_gs([], jokers))
        assert ids.dtype == np.int64
        assert ids.tolist() == [
            center_key_id("j_lusty_joker"),
            center_key_id("j_baron"),
        ]


class TestClass1Scored:
    def test_suit_joker_marks_only_its_suit(self):
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.SPADES, Rank.KING),
            (Suit.HEARTS, Rank.NINE),
        )
        jokers = [create_joker("j_lusty_joker")]
        m = trigger_match_matrix(_gs(hand, jokers))
        assert m[:, 0, SCORED].tolist() == [True, False, True]
        assert not m[:, 0, HELD].any()

    def test_wild_card_matches_every_suit_joker(self):
        wild = create_playing_card(Suit.SPADES, Rank.FIVE, "m_wild")
        jokers = [create_joker("j_lusty_joker"), create_joker("j_greedy_joker")]
        m = trigger_match_matrix(_gs([wild], jokers))
        assert m[0, 0, SCORED] and m[0, 1, SCORED]

    def test_smeared_widens_suit_match(self):
        # Smeared Joker: Hearts=Diamonds — a diamond triggers Lusty
        diamond = create_playing_card(Suit.DIAMONDS, Rank.FIVE)
        lusty = create_joker("j_lusty_joker")
        m_plain = trigger_match_matrix(_gs([diamond], [lusty]))
        assert not m_plain[0, 0, SCORED]
        m_smeared = trigger_match_matrix(
            _gs([diamond], [lusty, create_joker("j_smeared")])
        )
        assert m_smeared[0, 0, SCORED]

    def test_rank_joker(self):
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.SPADES, Rank.EIGHT),
            (Suit.CLUBS, Rank.NINE),
        )
        m = trigger_match_matrix(_gs(hand, [create_joker("j_fibonacci")]))
        # Fibonacci: A, 2, 3, 5, 8
        assert m[:, 0, SCORED].tolist() == [True, True, False]

    def test_odd_todd_counts_ace_not_face(self):
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.SPADES, Rank.NINE),
            (Suit.CLUBS, Rank.KING),  # id 13 > 10: not "odd" for Todd
            (Suit.DIAMONDS, Rank.FOUR),
        )
        m = trigger_match_matrix(_gs(hand, [create_joker("j_odd_todd")]))
        assert m[:, 0, SCORED].tolist() == [True, True, False, False]

    def test_photograph_marks_all_faces(self):
        # Pitfall 10: only the FIRST scored face fires, but the bit is
        # candidacy — every face must be marked.
        hand = _cards(
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.QUEEN),
            (Suit.CLUBS, Rank.FOUR),
            (Suit.DIAMONDS, Rank.JACK),
        )
        m = trigger_match_matrix(_gs(hand, [create_joker("j_photograph")]))
        assert m[:, 0, SCORED].tolist() == [True, True, False, True]

    def test_pareidolia_widens_faces_but_not_ride_the_bus(self):
        hand = _cards((Suit.HEARTS, Rank.FOUR))
        photograph = create_joker("j_photograph")
        bus = create_joker("j_ride_the_bus")
        pareidolia = create_joker("j_pareidolia")
        m = trigger_match_matrix(_gs(hand, [photograph, bus, pareidolia]))
        # Photograph's handler passes ctx.pareidolia to is_face; Ride the
        # Bus calls is_face() bare (engine-verified) — mirror exactly.
        assert m[0, 0, SCORED]
        assert not m[0, 1, SCORED]

    def test_enhancement_jokers(self):
        gold = create_playing_card(Suit.HEARTS, Rank.TEN, "m_gold")
        glass = create_playing_card(Suit.SPADES, Rank.NINE, "m_glass")
        plain = create_playing_card(Suit.CLUBS, Rank.TWO)
        jokers = [
            create_joker("j_ticket"),
            create_joker("j_glass"),
            create_joker("j_vampire"),
        ]
        m = trigger_match_matrix(_gs([gold, glass, plain], jokers))
        assert m[:, 0, SCORED].tolist() == [True, False, False]  # Golden Ticket
        assert m[:, 1, SCORED].tolist() == [False, True, False]  # Glass Joker
        assert m[:, 2, SCORED].tolist() == [True, True, False]  # Vampire: any enhancement

    def test_unconditional_scored(self):
        hand = _cards((Suit.HEARTS, Rank.TWO), (Suit.SPADES, Rank.KING))
        m = trigger_match_matrix(_gs(hand, [create_joker("j_hiker")]))
        assert m[:, 0, SCORED].all()
        assert not m[:, 0, HELD].any()


class TestClass1Held:
    def test_baron_marks_kings_held_only(self):
        hand = _cards((Suit.HEARTS, Rank.KING), (Suit.SPADES, Rank.QUEEN))
        m = trigger_match_matrix(_gs(hand, [create_joker("j_baron")]))
        assert m[:, 0, HELD].tolist() == [True, False]
        assert not m[:, 0, SCORED].any()

    def test_mime_marks_all_held(self):
        hand = _cards((Suit.HEARTS, Rank.TWO), (Suit.SPADES, Rank.KING))
        m = trigger_match_matrix(_gs(hand, [create_joker("j_mime")]))
        assert m[:, 0, HELD].all()
        assert not m[:, 0, SCORED].any()

    def test_discard_triggered_jokers_use_held_channel(self):
        hand = _cards((Suit.HEARTS, Rank.JACK), (Suit.SPADES, Rank.TWO))
        m = trigger_match_matrix(_gs(hand, [create_joker("j_hit_the_road")]))
        assert m[:, 0, HELD].tolist() == [True, False]
        assert not m[:, 0, SCORED].any()


class TestClass2LiveState:
    def test_ancient_tracks_rotating_suit(self):
        # Pitfall 8: a static table would mismark Ancient every round.
        hand = _cards((Suit.HEARTS, Rank.FIVE), (Suit.SPADES, Rank.FIVE))
        jokers = [create_joker("j_ancient")]
        gs = _gs(hand, jokers)
        gs["current_round"]["ancient_card"] = {"suit": "Hearts"}
        m1 = trigger_match_matrix(gs)
        assert m1[:, 0, SCORED].tolist() == [True, False]
        gs["current_round"]["ancient_card"] = {"suit": "Spades"}  # rotated
        m2 = trigger_match_matrix(gs)
        assert m2[:, 0, SCORED].tolist() == [False, True]

    def test_ancient_without_state_marks_nothing(self):
        hand = _cards((Suit.HEARTS, Rank.FIVE))
        m = trigger_match_matrix(_gs(hand, [create_joker("j_ancient")]))
        assert not m.any()

    def test_idol_requires_rank_and_suit(self):
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.SPADES, Rank.ACE),
            (Suit.HEARTS, Rank.KING),
        )
        gs = _gs(hand, [create_joker("j_idol")])
        gs["current_round"]["idol_card"] = {"suit": "Hearts", "rank": "Ace", "id": 14}
        m = trigger_match_matrix(gs)
        assert m[:, 0, SCORED].tolist() == [True, False, False]

    def test_mail_marks_rank_on_held_channel(self):
        hand = _cards((Suit.HEARTS, Rank.SEVEN), (Suit.SPADES, Rank.TWO))
        gs = _gs(hand, [create_joker("j_mail")])
        gs["current_round"]["mail_card"] = {"rank": "7", "id": 7}
        m = trigger_match_matrix(gs)
        assert m[:, 0, HELD].tolist() == [True, False]
        assert not m[:, 0, SCORED].any()

    def test_castle_reads_suit_from_joker_ability(self):
        hand = _cards((Suit.CLUBS, Rank.SEVEN), (Suit.SPADES, Rank.TWO))
        castle = create_joker("j_castle")
        castle.ability["castle_card_suit"] = "Clubs"
        m = trigger_match_matrix(_gs(hand, [castle]))
        assert m[:, 0, HELD].tolist() == [True, False]

    def test_dusk_only_on_last_hand(self):
        hand = _cards((Suit.HEARTS, Rank.TWO))
        jokers = [create_joker("j_dusk")]
        gs = _gs(hand, jokers)
        gs["current_round"]["hands_left"] = 2
        assert not trigger_match_matrix(gs).any()
        gs["current_round"]["hands_left"] = 1
        m = trigger_match_matrix(gs)
        assert m[0, 0, SCORED]

    def test_raised_fist_marks_lowest_held_rank(self):
        hand = _cards(
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.TWO),
            (Suit.CLUBS, Rank.TWO),  # tied lowest — both are candidates
            (Suit.DIAMONDS, Rank.NINE),
        )
        m = trigger_match_matrix(_gs(hand, [create_joker("j_raised_fist")]))
        assert m[:, 0, HELD].tolist() == [False, True, True, False]

    def test_raised_fist_ignores_stone_cards(self):
        stone = create_playing_card(Suit.HEARTS, Rank.TWO, "m_stone")
        hand = [stone] + _cards((Suit.SPADES, Rank.FIVE), (Suit.CLUBS, Rank.KING))
        m = trigger_match_matrix(_gs(hand, [create_joker("j_raised_fist")]))
        # The stone 2 is excluded (handler's own filter); lowest is the 5
        assert m[:, 0, HELD].tolist() == [False, True, False]


class TestClass3And4AllZero:
    def test_flower_pot_all_zero_despite_four_suits(self):
        # Pitfall 9: "which card triggers Flower Pot?" has no honest
        # answer — rows stay zero even when the hand satisfies it.
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.DIAMONDS, Rank.FIVE),
            (Suit.CLUBS, Rank.NINE),
            (Suit.SPADES, Rank.KING),
        )
        m = trigger_match_matrix(_gs(hand, [create_joker("j_flower_pot")]))
        assert not m.any()

    def test_hand_type_and_economy_jokers_all_zero(self):
        hand = _cards((Suit.HEARTS, Rank.ACE), (Suit.SPADES, Rank.ACE))  # a pair
        jokers = [create_joker("j_jolly"), create_joker("j_golden")]
        m = trigger_match_matrix(_gs(hand, jokers))
        assert not m.any()


class TestCopyResolution:
    def test_blueprint_resolves_right_neighbor(self):
        bp = create_joker("j_blueprint")
        photo = create_joker("j_photograph")
        res = resolve_copy_targets(_gs([], [bp, photo]))
        assert res[0].active
        assert res[0].target_index == 1
        assert res[0].target_key_id == center_key_id("j_photograph")
        # Non-copy jokers get inactive entries with zeroed fields
        assert not res[1].active
        assert res[1].target_index == -1
        assert res[1].target_key_id == 0

    def test_blueprint_rightmost_inactive(self):
        res = resolve_copy_targets(
            _gs([], [create_joker("j_photograph"), create_joker("j_blueprint")])
        )
        assert not res[1].active

    def test_brainstorm_resolves_leftmost_skipping_self(self):
        brain = create_joker("j_brainstorm")
        photo = create_joker("j_photograph")
        res = resolve_copy_targets(_gs([], [brain, photo]))
        assert res[0].active
        assert res[0].target_index == 1

    def test_chain_resolves_through_copies(self):
        # [lusty, blueprint, brainstorm]: Blueprint copies Brainstorm,
        # Brainstorm copies leftmost Lusty — both resolve to index 0.
        lusty = create_joker("j_lusty_joker")
        bp = create_joker("j_blueprint")
        brain = create_joker("j_brainstorm")
        res = resolve_copy_targets(_gs([], [lusty, bp, brain]))
        assert res[1].active and res[1].target_index == 0
        assert res[2].active and res[2].target_index == 0

    def test_copy_loop_inactive(self):
        # [blueprint, brainstorm]: Blueprint→Brainstorm→Blueprint→... —
        # the handlers' counter cap turns this into no effect.
        res = resolve_copy_targets(
            _gs([], [create_joker("j_blueprint"), create_joker("j_brainstorm")])
        )
        assert not res[0].active
        assert not res[1].active

    def test_debuffed_target_inactive(self):
        photo = create_joker("j_photograph")
        photo.debuff = True
        res = resolve_copy_targets(_gs([], [create_joker("j_blueprint"), photo]))
        assert not res[0].active

    def test_incompatible_target_inactive(self):
        # Egg is blueprint_compat=False in centers.json — the engine's own
        # compat guard must flow through resolution.
        res = resolve_copy_targets(
            _gs([], [create_joker("j_blueprint"), create_joker("j_egg")])
        )
        assert not res[0].active

    def test_debuffed_copy_joker_inactive(self):
        bp = create_joker("j_blueprint")
        bp.debuff = True
        res = resolve_copy_targets(_gs([], [bp, create_joker("j_photograph")]))
        assert not res[0].active


class TestCopyInheritance:
    def test_blueprint_inherits_photograph_faces(self):
        hand = _cards(
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.FOUR),
        )
        bp = create_joker("j_blueprint")
        photo = create_joker("j_photograph")
        m = trigger_match_matrix(_gs(hand, [bp, photo]))
        # Blueprint's column carries the resolved target's matches
        assert m[:, 0, SCORED].tolist() == [True, False]
        assert m[:, 1, SCORED].tolist() == [True, False]

    def test_inactive_blueprint_all_zero(self):
        hand = _cards((Suit.HEARTS, Rank.KING))
        # Rightmost Blueprint has nothing to copy
        m = trigger_match_matrix(
            _gs(hand, [create_joker("j_photograph"), create_joker("j_blueprint")])
        )
        assert m[0, 0, SCORED]
        assert not m[:, 1].any()

    def test_inheritance_uses_target_ability_state(self):
        # Brainstorm copying Castle must read the TARGET's castle suit
        hand = _cards((Suit.CLUBS, Rank.SEVEN), (Suit.SPADES, Rank.TWO))
        castle = create_joker("j_castle")
        castle.ability["castle_card_suit"] = "Clubs"
        brain = create_joker("j_brainstorm")
        m = trigger_match_matrix(_gs(hand, [castle, brain]))
        assert m[:, 1, HELD].tolist() == [True, False]

    def test_copy_of_set_level_joker_stays_zero(self):
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.DIAMONDS, Rank.FIVE),
            (Suit.CLUBS, Rank.NINE),
            (Suit.SPADES, Rank.KING),
        )
        m = trigger_match_matrix(
            _gs(hand, [create_joker("j_blueprint"), create_joker("j_flower_pot")])
        )
        # Flower Pot has no honest per-card bit; neither does a copy of it
        assert not m.any()

    def test_ids_array_keeps_own_key(self):
        # The ids array is the joker's OWN identity; the resolved-target id
        # is a separate obs field (wired at the schema bump).
        jokers = [create_joker("j_blueprint"), create_joker("j_photograph")]
        ids = joker_center_key_ids(_gs([], jokers))
        assert ids[0] == center_key_id("j_blueprint")


class TestEngineExactExclusions:
    def test_debuffed_card_row_all_zero(self):
        heart = create_playing_card(Suit.HEARTS, Rank.FIVE)
        debuffed_heart = create_playing_card(Suit.HEARTS, Rank.NINE)
        debuffed_heart.debuff = True
        m = trigger_match_matrix(
            _gs([heart, debuffed_heart], [create_joker("j_lusty_joker")])
        )
        # scoring.py phases 7/8 skip debuffed cards before any joker runs
        assert m[0, 0, SCORED]
        assert not m[1].any()

    def test_debuffed_joker_column_all_zero(self):
        heart = create_playing_card(Suit.HEARTS, Rank.FIVE)
        lusty = create_joker("j_lusty_joker")
        lusty.debuff = True
        m = trigger_match_matrix(_gs([heart], [lusty]))
        assert not m.any()

    def test_face_down_card_row_all_zero(self):
        hidden = create_playing_card(Suit.HEARTS, Rank.KING)
        hidden.facing = "back"
        m = trigger_match_matrix(_gs([hidden], [create_joker("j_photograph")]))
        assert not m.any()

    def test_debuffed_smeared_does_not_widen(self):
        # get_hand_eval_flags excludes debuffed modifier jokers
        diamond = create_playing_card(Suit.DIAMONDS, Rank.FIVE)
        smeared = create_joker("j_smeared")
        smeared.debuff = True
        m = trigger_match_matrix(
            _gs([diamond], [create_joker("j_lusty_joker"), smeared])
        )
        assert not m[0, 0, SCORED]

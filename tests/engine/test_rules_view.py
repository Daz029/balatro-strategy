"""Regressions for globally active Pareidolia/Smeared Joker rules."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import Discard, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.read import rules_for
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand


def _score(cards, jokers):
    return score_hand(
        cards,
        [],
        jokers,
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("RULES_VIEW"),
    )


def _card(suit: Suit, rank: Rank, enhancement: str = "c_base"):
    return create_playing_card(suit, rank, enhancement=enhancement)


def test_ride_the_bus_resets_on_non_face_with_pareidolia():
    """Lua asks global Pareidolia, so even a Two resets Ride the Bus."""
    bus = create_joker("j_ride_the_bus")
    bus.ability["mult"] = 4

    _score(
        [_card(Suit.HEARTS, Rank.TWO)],
        [bus, create_joker("j_pareidolia")],
    )

    assert bus.ability["mult"] == 0


def test_flower_pot_uses_smeared_for_all_four_suits():
    """Two red and two black cards can cover all suits under Smeared."""
    cards = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.HEARTS, Rank.THREE),
        _card(Suit.SPADES, Rank.FOUR),
        _card(Suit.SPADES, Rank.FIVE),
        _card(Suit.HEARTS, Rank.SIX),
    ]

    result = _score(cards, [create_joker("j_flower_pot"), create_joker("j_smeared")])

    assert result.hand_type == "Straight"
    assert result.mult == 12


def test_seeing_double_uses_smeared_for_club_and_other_suit():
    """With Smeared, a Spade is both black suits for Seeing Double."""
    result = _score(
        [_card(Suit.SPADES, Rank.TWO)],
        [create_joker("j_seeing_double"), create_joker("j_smeared")],
    )

    assert result.mult == 2


def test_goad_debuffs_clubs_with_smeared():
    club = _card(Suit.CLUBS, Rank.TWO)
    smeared = create_joker("j_smeared")

    Blind.create("bl_goad", ante=1).debuff_card(club, rules_for([smeared]), {})

    assert club.debuff is True


def test_faceless_joker_discard_uses_pareidolia():
    """Faceless Joker is the discard-context face check in the source."""
    gs = initialize_run("b_red", 1, "RULES_DISCARD")
    step(gs, SelectBlind())
    gs["jokers"] = [create_joker("j_faceless"), create_joker("j_pareidolia")]
    gs["hand"] = [
        _card(Suit.HEARTS, Rank.TWO),
        _card(Suit.CLUBS, Rank.THREE),
        _card(Suit.SPADES, Rank.FOUR),
    ]
    gs["deck"] = []
    dollars_before = gs["dollars"]

    step(gs, Discard(card_indices=(0, 1, 2)))

    assert gs["dollars"] == dollars_before + 5


def test_stone_king_is_not_a_face_without_pareidolia():
    stone_king = _card(Suit.HEARTS, Rank.KING, enhancement="m_stone")

    result = _score([stone_king], [create_joker("j_scary_face")])

    assert result.chips == 55


def test_face_and_suit_require_rules():
    card = _card(Suit.HEARTS, Rank.KING)

    with pytest.raises(TypeError):
        card.is_face()
    with pytest.raises(TypeError):
        card.is_suit("Hearts")

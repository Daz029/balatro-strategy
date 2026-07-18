"""Regression tests for the end-of-round engine money fixes."""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import CashOut, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def _init_gs(seed: str = "ENGINEFIXES") -> dict[str, Any]:
    return initialize_run("b_red", 1, seed)


def _beat_blind_at(gs: dict[str, Any], *, dollars: int) -> None:
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    gs["dollars"] = dollars
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))


class TestJokerEndOfRoundPaysAfterInterest:
    def test_golden_joker_pays_once_and_never_bumps_interest(self):
        gs = _init_gs()
        gs["jokers"].append(create_joker("j_golden"))
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        gs["dollars"] = 1
        dollars_before_round = gs["dollars"]

        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

        earnings = gs["round_earnings"]
        assert gs["dollars"] == dollars_before_round
        assert earnings.joker_dollars == 4
        assert earnings.interest == 0

        step(gs, CashOut())
        assert gs["dollars"] == dollars_before_round + earnings.total


class TestRentalChargedOnceBeforeInterest:
    def test_single_charge_interest_on_post_rental_balance(self):
        gs = _init_gs()
        joker = create_joker("j_joker")
        joker.set_rental(True)
        gs["jokers"].append(joker)
        _beat_blind_at(gs, dollars=20)

        assert gs["dollars"] == 20
        earnings = gs["round_earnings"]
        assert earnings.rental_cost == 3
        assert earnings.interest == (20 - 3) // 5

        step(gs, CashOut())
        assert gs["dollars"] == 20 + earnings.total

    def test_debuffed_rental_still_charges(self):
        gs = _init_gs()
        joker = create_joker("j_joker")
        joker.set_rental(True)
        joker.set_debuff(True)
        gs["jokers"].append(joker)
        _beat_blind_at(gs, dollars=0)
        assert gs["round_earnings"].rental_cost == 3


class TestHeldGoldCardMoney:
    def _win_with_held_card(self, card, dollars: int):
        gs = _init_gs()
        step(gs, SelectBlind())
        gs["hand"][5] = card
        gs["blind"].chips = 1
        gs["dollars"] = dollars
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        return gs

    def test_held_gold_card_pays_and_crosses_interest_bracket(self):
        gold = create_playing_card(Suit.HEARTS, Rank.KING)
        gold.set_ability("m_gold")
        gs = self._win_with_held_card(gold, dollars=2)
        assert gs["dollars"] == 2 + 3
        assert gs["round_earnings"].interest == 1

    def test_held_gold_seal_pays_nothing(self):
        sealed = create_playing_card(Suit.HEARTS, Rank.KING)
        sealed.set_seal("Gold")
        gs = self._win_with_held_card(sealed, dollars=2)
        assert gs["dollars"] == 2
        assert gs["round_earnings"].interest == 0

    def _play_high_card_with_seal_on(self, seal_index: int):
        gs = _init_gs()
        step(gs, SelectBlind())
        played = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.TWO),
            create_playing_card(Suit.CLUBS, Rank.THREE),
            create_playing_card(Suit.DIAMONDS, Rank.FOUR),
            create_playing_card(Suit.SPADES, Rank.SIX),
        ]
        played[seal_index].set_seal("Gold")
        gs["hand"][0:5] = played
        gs["blind"].chips = 1
        gs["dollars"] = 2
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        return gs

    def test_scored_gold_seal_pays_in_blind(self):
        gs = self._play_high_card_with_seal_on(0)
        assert gs["dollars"] == 2 + 3
        assert gs["round_earnings"].interest == 1

    def test_played_but_unscored_gold_seal_pays_nothing(self):
        gs = self._play_high_card_with_seal_on(1)
        assert gs["dollars"] == 2
        assert gs["round_earnings"].interest == 0

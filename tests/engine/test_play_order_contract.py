"""NEW-P5-1-05, DECIDED 2026-10-12: the bot's pick order IS the play order.

Lua scores played cards in physical left-to-right order
(``state_events.lua:459-483``). Vanilla lets a player rearrange the hand for
free before playing, so the engine reads ``PlayHand.card_indices`` as that
left-to-right arrangement: index order in the action = scoring order. Which
order to submit is the caller's choice (``hand_play_gym`` submits
``best_play_order``'s order); the solver's internal ordering never reaches
the engine.
"""

from __future__ import annotations

from jackdaw.engine import lifecycle
from jackdaw.engine.actions import PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def _pair_of_kings_total(order: tuple[int, int]) -> int:
    """Score a Pair of Kings under Photograph with the given index order.

    Hand index 0 = plain King, 1 = Mult King (+4 mult). Photograph (X2)
    fires on the FIRST scoring face card, so its position relative to the
    +4 decides the result:
      Mult King first: (2 + 4) * 2 = 12 mult -> 30 chips * 12 = 360
      plain King first: 2 * 2 + 4  =  8 mult -> 30 chips *  8 = 240
    (Pair: 10 chips / 2 mult at level 1; each King adds 10 chips.)
    """
    gs = initialize_run("b_red", 1, "NEW_P5_1_05_ORDER")
    step(gs, SelectBlind())
    gs["blind"].chips = 10**9
    lifecycle.emplace(gs, create_joker("j_photograph", game_state=gs), "jokers")
    plain = create_playing_card(Suit.SPADES, Rank.KING, game_state=gs)
    mult = create_playing_card(Suit.HEARTS, Rank.KING, enhancement="m_mult", game_state=gs)
    gs["hand"] = [plain, mult]

    step(gs, PlayHand(card_indices=order))

    return gs["last_score_result"].total


def test_index_order_is_left_to_right_scoring_order() -> None:
    assert _pair_of_kings_total((1, 0)) == 360
    assert _pair_of_kings_total((0, 1)) == 240

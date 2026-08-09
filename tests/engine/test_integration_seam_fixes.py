"""Integration-seam fixes from engine PR-2.

Both bugs below are the project's recurring "dead feature" class (Throwback /
The Idol / blueprint_compat / Marble / Riff-raff): the handler is correct and
its unit tests pass, but the caller never supplies the state the handler reads,
so the feature is inert in the real pipeline.  Assertions therefore go through
the integration path, never a hand-built context.
"""

from __future__ import annotations

from jackdaw.engine.card import Card
from jackdaw.engine.consumables import can_use_consumable


def _card(key: str) -> Card:
    c = Card()
    c.set_ability(key)
    return c


class TestFoolNeedsGameState:
    """The Fool copies the last Tarot/Planet used, so it needs run state.

    ``_handle_use_consumable`` called ``can_use_consumable`` without
    ``game_state``, so the ``last_tarot_planet`` lookup always saw an empty
    dict and The Fool was permanently unusable.
    """

    @staticmethod
    def _gs(last_tarot_planet: str | None) -> dict:
        """Minimal SHOP-phase run state holding one Fool."""
        from jackdaw.engine.actions import GamePhase
        from jackdaw.engine.rng import PseudoRandom

        return {
            "phase": GamePhase.SHOP,
            "consumables": [_card("c_fool")],
            "consumable_slots": 2,
            "jokers": [],
            "joker_slots": 5,
            "hand": [],
            "deck": [],
            "current_round": {},
            "round_resets": {"ante": 1},
            "rng": PseudoRandom("FOOL"),
            "last_tarot_planet": last_tarot_planet,
        }

    def test_usable_when_a_prior_tarot_was_used(self):
        """Must go through _handle_use_consumable: can_use_consumable itself was
        always correct, so calling it directly cannot detect this bug."""
        from jackdaw.engine.game import _handle_use_consumable

        gs = self._gs("c_magician")
        _handle_use_consumable(gs, 0, None)  # must not raise
        # The Fool was consumed and replaced by a copy of the last tarot.
        assert [c.center_key for c in gs["consumables"]] == ["c_magician"]

    def test_unusable_with_no_prior_tarot(self):
        import pytest

        from jackdaw.engine.game import IllegalActionError, _handle_use_consumable

        with pytest.raises(IllegalActionError):
            _handle_use_consumable(self._gs(None), 0, None)

    def test_fool_cannot_copy_itself(self):
        """Vanilla forbids The Fool producing another Fool."""
        import pytest

        from jackdaw.engine.game import IllegalActionError, _handle_use_consumable

        with pytest.raises(IllegalActionError):
            _handle_use_consumable(self._gs("c_fool"), 0, None)

    def test_handler_still_rejects_without_run_state(self):
        """Direct call with no game_state degrades to 'unusable', never raises."""
        assert can_use_consumable(_card("c_fool"), consumables=[], consumable_limit=2) is False


class TestCastleSuitReachesSnapshot:
    """Castle's suit lives on current_round.castle_card (card.lua:2857).

    The handler read ``card.ability['castle_card_suit']`` — a field the joker
    never carries — so Castle could not fire.  The discard snapshot now
    forwards the real per-round suit.
    """

    def test_discard_snapshot_carries_castle_suit(self):
        from jackdaw.engine.game import _build_discard_snapshot

        gs = {"current_round": {"castle_card": {"suit": "Hearts"}}}
        assert _build_discard_snapshot(gs, []).castle_card_suit == "Hearts"

    def test_absent_castle_card_is_none(self):
        from jackdaw.engine.game import _build_discard_snapshot

        assert _build_discard_snapshot({"current_round": {}}, []).castle_card_suit is None


class TestRoundTargetsSeeEveryZone:
    """Vanilla iterates G.playing_cards — every card in the run, any zone.

    Restricting the draw to the draw pile made cards in hand or the discard
    pile ineligible to be the idol/mail/ancient/castle card.
    """

    def test_hand_and_discard_cards_are_eligible(self):
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.rng import PseudoRandom
        from jackdaw.engine.round_lifecycle import reset_round_targets

        # Everything is out of the draw pile: deck empty, cards in hand/discard.
        # Both are Hearts so the result is unambiguous — when no card is
        # eligible the draw falls through to its hardcoded "Spades" placeholder,
        # which is exactly what the deck-only version produced here.
        gs = {
            "current_round": {},
            "deck": [],
            "hand": [create_playing_card(Suit.HEARTS, Rank.ACE)],
            "discard_pile": [create_playing_card(Suit.HEARTS, Rank.KING)],
        }
        reset_round_targets(PseudoRandom("ZONES"), 1, gs)

        cr = gs["current_round"]
        assert cr["idol_card"]["suit"] == "Hearts"
        assert cr["castle_card"]["suit"] == "Hearts"
        assert cr["idol_card"]["id"] in {14, 13}

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


class TestEndOfRoundContextCarriesTheBlind:
    """``on_end_of_round`` built its context without the blind the caller had.

    Campfire's boss reset and Rocket's boss increment both gate on
    ``ctx.blind.boss``, so both branches were unreachable: Campfire never
    lost its accumulated xMult and Rocket paid a flat $1 for the whole run.
    """

    @staticmethod
    def _run_to_boss(joker_key: str, prime=None):
        """Play three rounds (Small, Big, Boss) holding one joker."""
        from jackdaw.engine.actions import CashOut, NextRound, PlayHand, SelectBlind
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.game import step
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "EOR_BLIND")
        gs["jokers"] = [create_joker(joker_key)]
        saw_boss = False
        for _ in range(3):
            step(gs, SelectBlind())
            saw_boss = saw_boss or getattr(gs["blind"], "boss", False)
            if prime is not None:
                prime(gs["jokers"][0])
            gs["blind"].chips = 1
            step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
            step(gs, CashOut())
            step(gs, NextRound())
        assert saw_boss, "test never reached a boss blind"
        return gs["jokers"][0]

    def test_campfire_resets_xmult_after_the_boss(self):
        def prime(joker):
            joker.ability["x_mult"] = 2.0

        assert self._run_to_boss("j_campfire", prime).ability["x_mult"] == 1

    def test_rocket_payout_grows_after_the_boss(self):
        rocket = self._run_to_boss("j_rocket")
        # Base $1 in centers.json, +$2 for the one boss beaten.
        assert rocket.ability["extra"]["dollars"] == 3


class TestEndOfRoundMutationsAreApplied:
    """``on_end_of_round`` returned ``mutations`` that nothing consumed.

    Gros Michel's extinction emitted ``pool_flag: gros_michel_extinct`` and
    it was dropped, so ``pool_flags`` stayed empty forever: Cavendish
    (``yes_pool_flag``) could never be offered by any shop or pack, and Gros
    Michel (``no_pool_flag``) kept re-appearing after going extinct.
    """

    def test_gros_michel_extinction_sets_the_pool_flag(self):
        from jackdaw.engine.actions import CashOut, NextRound, PlayHand, SelectBlind
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.game import step
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "GM_FLAG")
        gs["jokers"] = [create_joker("j_gros_michel")]
        rounds = 0
        while gs["jokers"] and rounds < 80:
            step(gs, SelectBlind())
            gs["blind"].chips = 1
            step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
            step(gs, CashOut())
            step(gs, NextRound())
            rounds += 1

        assert not gs["jokers"], "Gros Michel never went extinct in 80 rounds"
        assert gs["pool_flags"].get("gros_michel_extinct") is True

    def test_the_flag_swaps_which_banana_the_pool_offers(self):
        from jackdaw.engine.pools import get_current_pool
        from jackdaw.engine.rng import PseudoRandom

        def keys(flags):
            pool = get_current_pool("Joker", PseudoRandom("BANANA"), 2, pool_flags=flags)
            return pool[0] if isinstance(pool, tuple) else pool

        before, after = keys({}), keys({"gros_michel_extinct": True})
        assert "j_gros_michel" in before and "j_cavendish" not in before
        assert "j_cavendish" in after and "j_gros_michel" not in after

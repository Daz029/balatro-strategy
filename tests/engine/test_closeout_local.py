"""Integration regressions for engine close-out ticket CO-1."""

from __future__ import annotations

from jackdaw.engine import lifecycle, read
from jackdaw.engine.actions import (
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
    UseConsumable,
)
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_consumable, create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.hands import HAND_ORDER, HandType
from jackdaw.engine.game import _sort_hand_desc, step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand
from jackdaw.env.observation import encode_global_context
from jackdaw.env.trigger_match import resolve_copy_targets


def _owned_joker(gs: dict, key: str):
    joker = create_joker(key, game_state=gs)
    lifecycle.emplace(gs, joker, "jokers")
    joker.set_cost(gs)
    return joker


def _owned_consumable(gs: dict, key: str):
    card = create_consumable(key, game_state=gs)
    lifecycle.emplace(gs, card, "consumables")
    card.set_cost(gs)
    return card


def _card(gs: dict, suit: Suit, rank: Rank, *, enhancement: str = "c_base"):
    return create_playing_card(suit, rank, enhancement=enhancement, game_state=gs)


def _score(gs: dict, played: list, held: list, jokers: list):
    blind = Blind.create("bl_small", ante=1)
    gs.update(
        {
            "blind": blind,
            "hand_levels": HandLevels(),
            "jokers": jokers,
            "hand": held,
            "played_cards_area": played,
        }
    )
    return score_hand(
        played,
        held,
        jokers,
        gs["hand_levels"],
        blind,
        gs["rng"],
        game_state=gs,
    )


class TestD01Nominal:
    """Lua ``Card:get_nominal`` at ``card.lua:950-955`` is the single ordering key."""

    def test_score_hand_and_area_sort_use_unique_val_tiebreak(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D01")
        earlier = _card(gs, Suit.HEARTS, Rank.FIVE)
        later = _card(gs, Suit.HEARTS, Rank.FIVE)
        assert later.base is not None
        later.base.id = 6

        result = _score(gs, [later, earlier], [], [])

        assert result.hand_type == "High Card"
        assert len(result.scoring_cards) == 1
        assert result.scoring_cards[0] is earlier
        hand = [later, earlier]
        _sort_hand_desc(hand)  # the production hand sort (CardArea:sort 'desc')
        assert hand[0] is earlier
        assert hand[1] is later

    def test_stone_nominal_scales_suit_terms(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D01_STONE")
        spade = _card(gs, Suit.SPADES, Rank.ACE, enhancement="m_stone")
        diamond = _card(gs, Suit.DIAMONDS, Rank.ACE, enhancement="m_stone")

        _score(gs, [spade, diamond], [], [])

        assert diamond.get_nominal() > spade.get_nominal()


class TestD05Brainstorm:
    """Lua ``card.lua:2321-2327`` targets slot one and no-ops when that is itself."""

    def test_leftmost_brainstorm_is_inactive_everywhere(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D05")
        brainstorm = create_joker("j_brainstorm", game_state=gs)
        joker = create_joker("j_joker", game_state=gs)
        played = _card(gs, Suit.HEARTS, Rank.TWO)

        result = _score(gs, [played], [], [brainstorm, joker])
        gs["jokers"] = [brainstorm, joker]
        resolutions = resolve_copy_targets(gs)

        assert result.total == 35
        assert resolutions[0].active is False


class TestD06BlueprintGuards:
    """Lua copyable branches omit guards at ``card.lua:2413,2749,3067,3501``."""

    def test_blueprint_copies_hiker_permanent_upgrade(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D06_HIKER")
        blueprint = create_joker("j_blueprint", game_state=gs)
        hiker = create_joker("j_hiker", game_state=gs)
        played = _card(gs, Suit.HEARTS, Rank.TWO)

        _score(gs, [played], [], [blueprint, hiker])

        assert played.ability["perma_bonus"] == 10

    def test_blueprint_copies_dna_first_hand_branch(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D06_DNA")
        step(gs, SelectBlind())
        blueprint = _owned_joker(gs, "j_blueprint")
        dna = _owned_joker(gs, "j_dna")
        del blueprint, dna
        played = _card(gs, Suit.HEARTS, Rank.TWO)
        gs["deck"].extend(gs["hand"])
        gs["hand"] = [played]
        gs["blind"].chips = 10**9

        step(gs, PlayHand(card_indices=(0,)))

        copies = [
            card
            for card in gs["hand"]
            if card.base is not None
            and card.base.rank is Rank.TWO
            and card.base.suit is Suit.HEARTS
        ]
        assert len(copies) == 2

    def test_blueprint_copies_burnt_joker_level_up(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D06_BURNT")
        step(gs, SelectBlind())
        blueprint = _owned_joker(gs, "j_blueprint")
        burnt = _owned_joker(gs, "j_burnt")
        del blueprint, burnt
        gs["hand"] = [
            _card(gs, Suit.HEARTS, Rank.TWO),
            _card(gs, Suit.SPADES, Rank.TWO),
        ]
        gs["deck"] = []

        step(gs, Discard(card_indices=(0, 1)))

        assert gs["hand_levels"][HandType.PAIR].level == 3

    def test_blueprint_copies_perkeo_at_shop_end(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D06_PERKEO")
        gs["phase"] = GamePhase.SHOP
        blueprint = _owned_joker(gs, "j_blueprint")
        perkeo = _owned_joker(gs, "j_perkeo")
        hermit = _owned_consumable(gs, "c_hermit")
        del blueprint, perkeo, hermit

        step(gs, NextRound())

        assert len(gs["consumables"]) == 3
        assert all(card.center_key == "c_hermit" for card in gs["consumables"])


class TestD08RaisedFistTie:
    """Lua ``card.lua:3320-3326`` uses ``>=`` and selects the last tied low card."""

    def test_last_tied_low_card_gets_red_seal_retrigger(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D08")
        played = _card(gs, Suit.HEARTS, Rank.TWO)
        first = _card(gs, Suit.CLUBS, Rank.THREE)
        last = _card(gs, Suit.DIAMONDS, Rank.THREE)
        last.set_seal(gs, "Red")
        raised_fist = create_joker("j_raised_fist", game_state=gs)

        result = _score(gs, [played], [first, last], [raised_fist])

        assert result.mult == 13
        assert result.total == 91


class TestD09BlackboardEmptyHand:
    """Lua ``card.lua:3951-3963`` passes when black-card count equals zero cards."""

    def test_playing_the_whole_hand_triggers_blackboard(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D09")
        step(gs, SelectBlind())
        blackboard = _owned_joker(gs, "j_blackboard")
        del blackboard
        played = _card(gs, Suit.HEARTS, Rank.TWO)
        gs["deck"].extend(gs["hand"])
        gs["hand"] = [played]
        gs["blind"].chips = 10**9

        step(gs, PlayHand(card_indices=(0,)))

        assert gs["last_score_result"].total == 21


class TestD10JokerOnJokerDebuffedTarget:
    """Lua ``state_events.lua:916-927`` runs other_joker for every ``_card``.

    The joker-on-joker loop is outside ``_card``'s own debuff check, so Baseball
    Card (``card.lua:3397``) still gives X1.5 for a DEBUFFED Uncommon.
    """

    def test_baseball_counts_a_debuffed_uncommon(self) -> None:
        gs = initialize_run("b_red", 1, "CO_D10_J2J")
        played = [_card(gs, Suit.HEARTS, Rank.FIVE), _card(gs, Suit.SPADES, Rank.FIVE)]
        baseball = create_joker("j_baseball", game_state=gs)
        alone = _score(gs, played, [], [baseball]).total

        uncommon = create_joker("j_steel_joker", game_state=gs)
        uncommon.debuff = True  # test fixture: no passive to toggle on Steel Joker
        with_debuffed = _score(gs, played, [], [baseball, uncommon]).total

        assert with_debuffed == int(alone * 1.5)


class TestC13bObservatory:
    """Lua ``card.lua:2293-2300``: with Observatory, each held Planet whose
    hand type is the scoring hand gives X1.5 in the joker_main pass, which
    walks jokers THEN consumables (``state_events.lua:877-878``)."""

    def test_held_matching_planet_multiplies(self) -> None:
        gs = initialize_run("b_red", 1, "CO_C13B")
        _owned_consumable(gs, "c_mercury")  # Pair
        _owned_consumable(gs, "c_pluto")  # High Card: must not fire
        played = [_card(gs, Suit.HEARTS, Rank.FIVE), _card(gs, Suit.SPADES, Rank.FIVE)]
        # Pair L1: (10 + 5 + 5) chips * 2 mult = 40.
        assert _score(gs, played, [], []).total == 40

        gs["used_vouchers"]["v_observatory"] = True
        # X1.5 from Mercury only: 20 * 3 = 60.
        assert _score(gs, played, [], []).total == 60


class TestD13SecretHandVisibility:
    """Lua reveals at ``state_events.lua:578``; ``common_events.lua:464`` does not."""

    def test_black_hole_levels_without_revealing_consumers(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D13_LEVEL")
        gs["phase"] = GamePhase.SHOP
        black_hole = _owned_consumable(gs, "c_black_hole")
        del black_hole

        step(gs, UseConsumable(card_index=0))

        secret = gs["hand_levels"][HandType.FLUSH_FIVE]
        assert secret.level == 2
        assert secret.visible is False
        assert read.telescope_hand(gs) is None
        flush_five_index = list(HandType).index(HandType.FLUSH_FIVE)
        visible_feature = 30 + flush_five_index * 5 + 4
        assert encode_global_context(gs)[visible_feature] == 0

    def test_playing_secret_hand_reveals_it(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D13_PLAY")
        step(gs, SelectBlind())
        gs["deck"].extend(gs["hand"])
        gs["hand"] = [
            _card(gs, Suit.HEARTS, Rank.TWO),
            _card(gs, Suit.SPADES, Rank.TWO),
            _card(gs, Suit.CLUBS, Rank.TWO),
            _card(gs, Suit.DIAMONDS, Rank.TWO),
            _card(gs, Suit.HEARTS, Rank.TWO),
        ]
        gs["blind"].chips = 10**9

        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

        assert gs["hand_levels"][HandType.FIVE_OF_A_KIND].visible is True

    def test_to_do_list_rerolls_only_to_a_visible_hand(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D13_TODO")
        gs["phase"] = GamePhase.SHOP
        black_hole = _owned_consumable(gs, "c_black_hole")
        del black_hole
        step(gs, UseConsumable(card_index=0))
        todo = _owned_joker(gs, "j_todo_list")
        step(gs, NextRound())
        step(gs, SelectBlind())
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        target = todo.ability["extra"]["poker_hand"]
        assert gs["hand_levels"][target].visible is True

    def test_to_do_list_reroll_excludes_current_target(self) -> None:
        """NEW-CO1-01, Lua ``card.lua:2975-2980``: ``k ~= to_do_poker_hand``.

        With only Pair and High Card visible and Pair as the target, the only
        legal re-roll is High Card. Without the exclusion each seed keeps Pair
        with probability 1/2, so 20 seeds cannot all pass by luck.
        """
        for i in range(20):
            gs = initialize_run("b_red", 1, f"CO1_TODO_EXCLUDE_{i}")
            gs["phase"] = GamePhase.SHOP
            todo = _owned_joker(gs, "j_todo_list")
            step(gs, NextRound())
            step(gs, SelectBlind())
            for ht in HAND_ORDER:
                gs["hand_levels"][ht].visible = str(ht) in ("Pair", "High Card")
            todo.ability["extra"]["poker_hand"] = "Pair"
            gs["blind"].chips = 1

            step(gs, PlayHand(card_indices=(0,)))

            assert todo.ability["extra"]["poker_hand"] == "High Card", i


class TestD21GiftCardConsumables:
    """Lua ``card.lua:2993-3005`` reprices both Joker and consumable areas."""

    def test_end_round_increases_consumable_sell_value(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D21")
        step(gs, SelectBlind())
        gift = _owned_joker(gs, "j_gift")
        hermit = _owned_consumable(gs, "c_hermit")
        gift_before = gift.sell_cost
        hermit_before = hermit.sell_cost
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        assert gift.sell_cost == gift_before + 1
        assert hermit.sell_cost == hermit_before + 1


class TestD34EctoplasmPenalty:
    """Lua ``card.lua:1494-1498`` spends and increments run-wide ``ecto_minus``."""

    def test_successive_uses_cost_one_then_two_hand_size(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D34")
        gs["phase"] = GamePhase.SHOP
        joker = _owned_joker(gs, "j_joker")
        mime = _owned_joker(gs, "j_mime")
        first = _owned_consumable(gs, "c_ectoplasm")
        second = _owned_consumable(gs, "c_ectoplasm")
        del joker, mime, first, second
        original_size = gs["hand_size"]

        step(gs, UseConsumable(card_index=0))
        step(gs, UseConsumable(card_index=0))

        assert gs["hand_size"] == original_size - 3
        assert gs["ecto_minus"] == 3


class TestD38GrimRng:
    """Lua ``card.lua:1319-1327`` fixes Ace and draws only ``grim_create`` suit."""

    def test_fixed_seed_exact_created_suits(self) -> None:
        gs = initialize_run("b_red", 1, "D38_GRIM")
        gs["phase"] = GamePhase.SHOP
        grim = _owned_consumable(gs, "c_grim")
        del grim
        gs["hand"] = [
            _card(gs, Suit.CLUBS, Rank.TWO),
            _card(gs, Suit.HEARTS, Rank.THREE),
        ]
        gs["deck"] = []

        step(gs, UseConsumable(card_index=0))

        # Lua's loop fixes rank='A', then advances grim_create once per suit.
        # The first two D38_GRIM stream selections over S,H,D,C are S then D;
        # spe_card advances a separate enhancement stream between those draws.
        created = [card for card in gs["hand"] if card.base.rank is Rank.ACE]
        assert [(card.base.rank, card.base.suit) for card in created] == [
            (Rank.ACE, Suit.SPADES),
            (Rank.ACE, Suit.DIAMONDS),
        ]


class TestD40ConsumableUsage:
    """Lua ``misc_functions.lua:1184-1225`` records usage by the center's set."""

    def test_tarot_black_hole_and_fool_bookkeeping(self) -> None:
        gs = initialize_run("b_red", 1, "CO1_D40")
        gs["phase"] = GamePhase.SHOP
        gs["consumable_slots"] = 3
        hermit = _owned_consumable(gs, "c_hermit")
        black_hole = _owned_consumable(gs, "c_black_hole")
        fool = _owned_consumable(gs, "c_fool")
        del hermit, black_hole, fool

        step(gs, UseConsumable(card_index=0))
        step(gs, UseConsumable(card_index=0))
        assert gs["last_tarot_planet"] == "c_hermit"
        step(gs, UseConsumable(card_index=0))

        totals = gs["consumeable_usage_total"]
        assert totals["tarot"] == 2
        assert totals["spectral"] == 1
        assert totals["planet"] == 0
        assert [card.center_key for card in gs["consumables"]] == ["c_hermit"]
        fortune = create_joker("j_fortune_teller", game_state=gs)
        played = _card(gs, Suit.HEARTS, Rank.TWO)
        result = _score(gs, [played], [], [fortune])
        assert result.mult == 3

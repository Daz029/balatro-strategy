"""Tests for pure engine state getters and the lazy scoring view."""

from __future__ import annotations

from dataclasses import fields

import pytest

from jackdaw.engine.card import Card
from jackdaw.engine.data.hands import HAND_ORDER, HandType
from jackdaw.engine.hand_eval import get_hand_eval_flags
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import GameSnapshot
from jackdaw.engine.read import (
    Rules,
    StateView,
    ancient_suit,
    castle_suit,
    count_enhancement,
    deck_cards_remaining,
    deck_enhancements,
    discards_left,
    discards_used,
    editionless_jokers,
    enhanced_count,
    find_joker,
    hands_left,
    hands_played_this_round,
    has_voucher,
    idol_card,
    joker_count,
    last_blind,
    mail_card_id,
    money_committed,
    money_with_buffer,
    playing_card_count,
    playing_cards,
    probability,
    rank_count,
    rules,
    rules_for,
    stencil_xmult,
    swashbuckler_mult,
    tarot_usage,
    telescope_hand,
    temperance_money,
)


def _joker(
    name: str,
    *,
    center_key: str = "j_joker",
    debuff: bool = False,
    sell_cost: int = 0,
    edition: dict[str, bool] | None = None,
    set_name: str = "Joker",
) -> Card:
    return Card(
        center_key=center_key,
        ability={"name": name, "set": set_name},
        debuff=debuff,
        sell_cost=sell_cost,
        edition=edition,
    )


def _playing_card(rank: str, *, enhancement: str = "c_base") -> Card:
    rank_letter = {"9": "9", "King": "K"}[rank]
    card = Card()
    card.set_base(f"S_{rank_letter}", "Spades", rank)
    card.center_key = enhancement
    card.ability = {"effect": "Stone Card" if enhancement == "m_stone" else "Base"}
    return card


class TestMoneyProbabilityAndVouchers:
    def test_money_committed_ignores_pending_buffer(self):
        assert money_committed({"dollars": 7, "dollar_buffer": 4}) == 7

    def test_money_with_buffer_defaults_and_adds(self):
        assert money_with_buffer({"dollars": 7}) == 7
        assert money_with_buffer({"dollars": 7, "dollar_buffer": -2}) == 5

    def test_probability_defaults_to_one(self):
        assert probability({}) == 1.0
        assert probability({"probabilities": {"normal": 3}}) == 3.0

    def test_has_voucher_uses_truthiness(self):
        gs = {"used_vouchers": {"v_telescope": True, "v_blank": 0}}
        assert has_voucher(gs, "v_telescope")
        assert not has_voucher(gs, "v_blank")
        assert not has_voucher({}, "v_missing")


class TestJokerGetters:
    def test_find_joker_order_debuff_filter_and_consumables(self):
        first = _joker("Showman")
        debuffed = _joker("Showman", debuff=True)
        consumable = _joker("Showman", set_name="Tarot")
        gs = {"jokers": [first, debuffed], "consumables": [consumable]}

        assert find_joker(gs, "Showman") == [first, consumable]
        assert find_joker(gs, "Showman", include_debuffed=True) == [
            first,
            debuffed,
            consumable,
        ]

    def test_joker_count_counts_every_slot(self):
        assert joker_count([_joker("Joker"), _joker("Joker", debuff=True)]) == 2

    def test_stencil_xmult_includes_debuffed_stencils(self):
        jokers = [
            _joker("Joker Stencil"),
            _joker("Joker Stencil", debuff=True),
            _joker("Joker"),
        ]
        assert stencil_xmult(jokers, joker_slots=5) == 4

    def test_swashbuckler_uses_identity_and_includes_debuffed(self):
        swashbuckler = _joker("Swashbuckler", sell_cost=100)
        same_value = _joker("Swashbuckler", sell_cost=100)
        debuffed = _joker("Joker", sell_cost=4, debuff=True)
        assert swashbuckler_mult([swashbuckler, same_value, debuffed], swashbuckler) == 104

    def test_temperance_filters_set_but_includes_debuffed_and_caps(self):
        jokers = [
            _joker("Joker", sell_cost=4),
            _joker("Joker", sell_cost=5, debuff=True),
            _joker("Tarot", sell_cost=100, set_name="Tarot"),
        ]
        assert temperance_money(jokers, cap=7) == 7

    def test_editionless_jokers_filter_set_and_not_debuff(self):
        eligible = _joker("Joker")
        debuffed = _joker("Joker", debuff=True)
        negative = _joker("Joker", edition={"negative": True})
        tarot = _joker("Tarot", set_name="Tarot")
        assert editionless_jokers([eligible, debuffed, negative, tarot]) == [eligible, debuffed]


class TestPlayingCardGetters:
    def test_playing_cards_area_order_and_identity_deduplication(self):
        deck = _playing_card("9")
        shared = _playing_card("King")
        discard = _playing_card("9", enhancement="m_steel")
        played = _playing_card("9", enhancement="m_stone")
        pack_only = _playing_card("King")
        gs = {
            "deck": [deck],
            "hand": [shared],
            "discard_pile": [discard],
            "played_cards_area": [played],
            "pack_hand": [shared, pack_only],
        }
        assert playing_cards(gs) == [deck, shared, discard, played, pack_only]

    def test_playing_card_count_uses_deduplicated_cards(self):
        shared = _playing_card("9")
        assert playing_card_count({"hand": [shared], "pack_hand": [shared]}) == 1

    def test_count_enhancement_counts_exact_center_key(self):
        gs = {
            "deck": [
                _playing_card("9", enhancement="m_steel"),
                _playing_card("9", enhancement="m_stone"),
                _playing_card("King", enhancement="m_steel"),
            ]
        }
        assert count_enhancement(gs, "m_steel") == 2

    def test_enhanced_count_counts_every_non_base_center(self):
        gs = {
            "deck": [
                _playing_card("9"),
                _playing_card("9", enhancement="m_steel"),
                _playing_card("King", enhancement="m_stone"),
            ]
        }
        assert enhanced_count(gs) == 2

    def test_rank_count_ignores_stone_card(self):
        nine = _playing_card("9")
        stone_nine = _playing_card("9", enhancement="m_stone")
        assert nine.get_id() == 9
        assert stone_nine.get_id() < 0
        assert rank_count({"deck": [nine, stone_nine]}, 9) == 1

    def test_deck_enhancements_returns_live_center_key_set(self):
        gs = {
            "deck": [_playing_card("9")],
            "hand": [_playing_card("King", enhancement="m_steel")],
        }
        assert deck_enhancements(gs) == {"c_base", "m_steel"}

    def test_deck_cards_remaining_only_counts_draw_pile(self):
        gs = {"deck": [_playing_card("9")], "hand": [_playing_card("King")]}
        assert deck_cards_remaining(gs) == 1


class TestRoundBlindAndHistoryGetters:
    def test_round_counters(self):
        gs = {
            "current_round": {
                "hands_left": 3,
                "hands_played": 2,
                "discards_left": 1,
                "discards_used": 4,
            }
        }
        assert hands_left(gs) == 3
        assert hands_played_this_round(gs) == 2
        assert discards_left(gs) == 1
        assert discards_used(gs) == 4

    def test_last_blind_is_current_blind(self):
        blind = object()
        assert last_blind({"blind": blind}) is blind
        assert last_blind({}) is None

    def test_telescope_ignores_invisible_and_breaks_ties_by_hand_order(self):
        levels = HandLevels()
        levels[HandType.FLUSH_FIVE].played = 50
        levels[HandType.FLUSH_FIVE].visible = False
        levels[HandType.STRAIGHT_FLUSH].played = 4
        levels[HandType.FOUR_OF_A_KIND].played = 4

        assert list(HAND_ORDER).index(HandType.STRAIGHT_FLUSH) < list(HAND_ORDER).index(
            HandType.FOUR_OF_A_KIND
        )
        assert telescope_hand({"hand_levels": levels}) == "Straight Flush"

    def test_telescope_returns_none_when_no_visible_hand_was_played(self):
        assert telescope_hand({"hand_levels": HandLevels()}) is None

    def test_tarot_usage_defaults_to_zero(self):
        assert tarot_usage({}) == 0
        assert tarot_usage({"consumeable_usage_total": {"tarot": 6}}) == 6

    def test_idol_card_returns_whole_target(self):
        target = {"id": 14, "rank": "Ace", "suit": "Hearts"}
        assert idol_card({"current_round": {"idol_card": target}}) is target

    def test_mail_card_id_returns_numeric_rank(self):
        assert mail_card_id({"current_round": {"mail_card": {"rank": "9", "id": 9}}}) == 9

    def test_ancient_suit_returns_target_suit(self):
        assert ancient_suit({"current_round": {"ancient_card": {"suit": "Clubs"}}}) == "Clubs"

    def test_castle_suit_returns_target_suit(self):
        assert castle_suit({"current_round": {"castle_card": {"suit": "Diamonds"}}}) == "Diamonds"


_RULE_JOKERS = {
    "pareidolia": ("Pareidolia", "j_pareidolia"),
    "smeared": ("Smeared Joker", "j_smeared"),
    "four_fingers": ("Four Fingers", "j_four_fingers"),
    "shortcut": ("Shortcut", "j_shortcut"),
    "splash": ("Splash", "j_splash"),
    "showman": ("Showman", "j_showman"),
}


class TestRules:
    def test_rules_for_sets_named_active_jokers_only(self):
        jokers = [
            _joker(name, center_key=center_key, debuff=flag in {"shortcut", "showman"})
            for flag, (name, center_key) in _RULE_JOKERS.items()
        ]
        assert rules_for(jokers) == Rules(
            pareidolia=True,
            smeared=True,
            four_fingers=True,
            shortcut=False,
            splash=True,
            showman=False,
        )

    def test_rules_reads_state_jokers(self):
        assert rules({"jokers": [_joker("Showman", center_key="j_showman")]}) == Rules(
            pareidolia=False,
            smeared=False,
            four_fingers=False,
            shortcut=False,
            splash=False,
            showman=True,
        )

    @pytest.mark.parametrize(
        "jokers",
        [
            [],
            [_joker("Four Fingers", center_key="j_four_fingers")],
            [
                _joker("Smeared Joker", center_key="j_smeared", debuff=True),
                _joker("Shortcut", center_key="j_shortcut"),
                _joker("Splash", center_key="j_splash"),
            ],
        ],
    )
    def test_rules_agree_with_hand_eval_shared_flags(self, jokers):
        result = rules_for(jokers)
        hand_eval_flags = get_hand_eval_flags(jokers)
        for flag in ("smeared", "four_fingers", "shortcut", "splash"):
            assert getattr(result, flag) is hand_eval_flags[flag]


def _state_view_fixture() -> tuple[dict, list[Card], GameSnapshot]:
    base = _playing_card("9")
    stone = _playing_card("9", enhancement="m_stone")
    steel = _playing_card("King", enhancement="m_steel")
    state_joker = _joker("Joker")
    scored_jokers = [
        _joker("Four Fingers", center_key="j_four_fingers"),
        _joker("Joker", debuff=True),
    ]
    idol = {"id": 9, "rank": "9", "suit": "Hearts"}
    gs = {
        "dollars": 13,
        "dollar_buffer": 2,
        "probabilities": {"normal": 2},
        "jokers": [state_joker],
        "joker_slots": 6,
        "deck": [base],
        "hand": [stone],
        "discard_pile": [steel],
        "pack_hand": [stone],
        "starting_deck_size": 52,
        "current_round": {
            "hands_left": 3,
            "hands_played": 1,
            "discards_left": 2,
            "discards_used": 1,
            "mail_card": {"id": 13},
            "idol_card": idol,
            "ancient_card": {"suit": "Clubs"},
            "castle_card": {"suit": "Hearts"},
        },
        "consumeable_usage_total": {"tarot": 4},
        "skips": 2,
    }
    snapshot = GameSnapshot(
        joker_count=2,
        joker_slots=6,
        money=13,
        deck_cards_remaining=1,
        starting_deck_size=52,
        playing_cards_count=3,
        stone_tally=1,
        steel_tally=1,
        nine_tally=1,
        enhanced_card_count=2,
        hands_left=3,
        hands_played=1,
        discards_left=2,
        discards_used=1,
        probabilities_normal=2.0,
        consumable_usage_tarot=4,
        mail_card_id=13,
        idol_card=idol,
        ancient_suit="Clubs",
        castle_card_suit="Hearts",
        skips=2,
        rules=rules_for(scored_jokers),
    )
    return gs, scored_jokers, snapshot


class TestStateView:
    def test_exposes_every_snapshot_field_with_equivalent_values(self):
        gs, scored_jokers, snapshot = _state_view_fixture()
        view = StateView(gs, jokers=scored_jokers)
        snapshot_fields = {field.name for field in fields(GameSnapshot)}

        assert snapshot_fields == {name for name in snapshot_fields if hasattr(view, name)}
        for field in fields(GameSnapshot):
            assert getattr(view, field.name) == getattr(snapshot, field.name)

        assert view.money_with_buffer == 15
        assert view.rules.four_fingers
        assert view.gs is gs

    def test_defaults_jokers_to_state_area(self):
        gs, _, _ = _state_view_fixture()
        assert StateView(gs).joker_count == 1

    def test_properties_are_lazy_and_cached_for_view_lifetime(self):
        gs, scored_jokers, _ = _state_view_fixture()
        view = StateView(gs, jokers=scored_jokers)
        assert "stone_tally" not in view.__dict__

        assert view.stone_tally == 1
        assert view.__dict__["stone_tally"] == 1
        gs["hand"].append(_playing_card("King", enhancement="m_stone"))
        assert view.stone_tally == 1
        assert StateView(gs, jokers=scored_jokers).stone_tally == 2

    def test_overrides_replace_existing_properties(self):
        gs, scored_jokers, _ = _state_view_fixture()
        view = StateView(gs, jokers=scored_jokers, overrides={"probabilities_normal": 7.0})
        assert view.probabilities_normal == 7.0

    def test_overrides_reject_unknown_attributes(self):
        gs, _, _ = _state_view_fixture()
        with pytest.raises(KeyError, match="not_a_state_field"):
            StateView(gs, overrides={"not_a_state_field": 1})

"""Integration tests for the 7 wired tag contexts + fixed dormant bugs.

Contexts wired (previously effect handlers existed in ``tags.py`` but had NO
call sites): ``shop_start`` (D6), ``store_joker_create`` (Rare/Uncommon),
``store_joker_modify`` (editions), ``voucher_add``, ``shop_final_pass``
(Coupon), ``eval`` (Investment), ``round_start_bonus`` (Juggle).

Dormant bugs fixed alongside and regression-tested here:
* ``calculate_reroll_cost`` (shop.py) used ``or`` on ``temp_reroll_cost`` —
  Lua's ``0 or x`` is 0, Python's falls through, so D6's $0 base was ignored.
* ``_handle_reroll`` recomputed cost from ``base_reroll_cost``, ignoring the
  Reroll Surplus/Glut voucher discount (mutates ``round_resets.reroll_cost``)
  and escalating the price even while free rerolls remained.
* ``_check_double_tag`` read ``gs["tags"]`` (never populated — Double Tag
  could never fire) and applied only the dollars field of the duplicate.

Tags are injected directly into ``awarded_tags`` (mimicking a skip award of
a deferred tag) so tests don't depend on seed-determined skip rewards.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    Reroll,
    SelectBlind,
    SkipBlind,
)
from jackdaw.engine.data.prototypes import JOKERS, TAGS
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.shop import calculate_reroll_cost

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_gs(seed: str = "TAGWIRE") -> dict[str, Any]:
    return initialize_run("b_red", 1, seed)


def _award(gs: dict[str, Any], key: str) -> dict[str, Any]:
    """Inject an (un-consumed, deferred) awarded tag, as a skip would."""
    entry: dict[str, Any] = {"key": key, "result": None, "blind": "Small"}
    gs.setdefault("awarded_tags", []).append(entry)
    return entry


def _beat_blind(gs: dict[str, Any]) -> None:
    """Select the pending blind and trivially beat it."""
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))


def _to_shop(gs: dict[str, Any]) -> None:
    _beat_blind(gs)
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP


def _rarity(card: Any) -> int:
    return JOKERS[card.center_key].rarity


# ---------------------------------------------------------------------------
# D6 Tag — shop_start
# ---------------------------------------------------------------------------


class TestD6Tag:
    def test_rerolls_start_at_zero_and_climb(self):
        gs = _init_gs()
        entry = _award(gs, "tag_d_six")
        _to_shop(gs)

        assert entry["consumed"] is True
        assert gs["round_resets"]["temp_reroll_cost"] == 0
        assert gs["current_round"]["reroll_cost"] == 0

        dollars_before = gs["dollars"]
        step(gs, Reroll())
        assert gs["dollars"] == dollars_before  # first reroll was $0
        assert gs["current_round"]["reroll_cost"] == 1  # $0 base climbs $1

    def test_cleared_at_next_round_start(self):
        gs = _init_gs()
        _award(gs, "tag_d_six")
        _to_shop(gs)
        step(gs, NextRound())
        _beat_blind(gs)
        step(gs, CashOut())  # second shop

        assert gs["round_resets"].get("temp_reroll_cost") is None
        assert gs["current_round"]["reroll_cost"] == 5

    def test_calculate_reroll_cost_treats_zero_temp_as_valid(self):
        # Regression: `0 or 5` — Lua's `or` keeps 0, Python's didn't.
        gs = {
            "current_round": {"free_rerolls": 0, "reroll_cost_increase": 0},
            "round_resets": {"temp_reroll_cost": 0, "reroll_cost": 5},
        }
        assert calculate_reroll_cost(gs) == 0


# ---------------------------------------------------------------------------
# Rare / Uncommon Tags — store_joker_create
# ---------------------------------------------------------------------------


class TestRareUncommonTags:
    def test_rare_forces_first_slot_joker(self):
        gs = _init_gs()
        entry = _award(gs, "tag_rare")
        _to_shop(gs)

        first = gs["shop_cards"][0]
        assert first.ability.get("set") == "Joker"
        assert _rarity(first) == 3
        assert entry["consumed"] is True

    def test_two_tags_force_two_slots(self):
        gs = _init_gs()
        _award(gs, "tag_uncommon")
        _award(gs, "tag_uncommon")
        _to_shop(gs)

        assert len(gs["shop_cards"]) == 2
        for card in gs["shop_cards"]:
            assert card.ability.get("set") == "Joker"
            assert _rarity(card) == 2

    def test_tag_create_does_not_consume_shop_stream(self):
        # Verified in real Balatro (2026-07-06): the tag joker is generated
        # on the spot on its OWN RNG stream ('rta'/'uta' appends), so the
        # normal shop sequence is untouched — the card that would have
        # filled the slot simply appears in the next slot instead.
        baseline = _init_gs()
        _to_shop(baseline)

        tagged = _init_gs()
        _award(tagged, "tag_rare")
        _to_shop(tagged)

        assert tagged["shop_cards"][1].center_key == baseline["shop_cards"][0].center_key
        # And the forced joker itself came from a different stream.
        assert _rarity(tagged["shop_cards"][0]) == 3

    def test_pending_tag_applies_on_reroll(self):
        gs = _init_gs()
        _to_shop(gs)
        entry = _award(gs, "tag_rare")  # acquired conceptually "late"
        gs["dollars"] = 50
        step(gs, Reroll())

        first = gs["shop_cards"][0]
        assert first.ability.get("set") == "Joker"
        assert _rarity(first) == 3
        assert entry["consumed"] is True


# ---------------------------------------------------------------------------
# Edition Tags — store_joker_modify
# ---------------------------------------------------------------------------


class TestEditionTags:
    def test_foil_applies_to_first_base_joker_and_is_free(self):
        gs = _init_gs()
        # Rare tag guarantees slot 0 is a Joker; foil tag then modifies it.
        _award(gs, "tag_rare")
        foil = _award(gs, "tag_foil")
        _to_shop(gs)

        first = gs["shop_cards"][0]
        assert first.edition is not None and first.edition.get("foil") is True
        assert first.cost == 0  # vanilla: "free and becomes Foil"
        assert foil["consumed"] is True

    def test_negative_edition(self):
        gs = _init_gs()
        _award(gs, "tag_uncommon")
        _award(gs, "tag_negative")
        _to_shop(gs)

        first = gs["shop_cards"][0]
        assert first.edition is not None and first.edition.get("negative") is True
        assert first.cost == 0


# ---------------------------------------------------------------------------
# Voucher Tag — voucher_add
# ---------------------------------------------------------------------------


class TestVoucherTag:
    def test_each_tag_adds_one_voucher(self):
        baseline = _init_gs()
        _to_shop(baseline)
        base_count = len(baseline["shop_vouchers"])

        gs = _init_gs()
        e1 = _award(gs, "tag_voucher")
        e2 = _award(gs, "tag_voucher")
        _to_shop(gs)

        assert len(gs["shop_vouchers"]) == base_count + 2
        keys = [v.center_key for v in gs["shop_vouchers"]]
        assert len(keys) == len(set(keys))  # no duplicate offerings
        assert e1["consumed"] and e2["consumed"]
        for v in gs["shop_vouchers"]:
            assert v.cost > 0  # tag vouchers are purchasable, not free


# ---------------------------------------------------------------------------
# Coupon Tag — shop_final_pass
# ---------------------------------------------------------------------------


class TestCouponTag:
    def test_initial_shop_free_vouchers_not(self):
        gs = _init_gs()
        entry = _award(gs, "tag_coupon")
        _to_shop(gs)

        assert all(c.cost == 0 for c in gs["shop_cards"])
        assert all(b.cost == 0 for b in gs["shop_boosters"])
        for v in gs["shop_vouchers"]:
            assert v.cost > 0  # vanilla: vouchers stay full price
        assert entry["consumed"] is True

    def test_rerolled_cards_are_not_free(self):
        gs = _init_gs()
        _award(gs, "tag_coupon")
        _to_shop(gs)
        gs["dollars"] = 50
        step(gs, Reroll())
        assert any(c.cost > 0 for c in gs["shop_cards"])


# ---------------------------------------------------------------------------
# Investment Tag — eval
# ---------------------------------------------------------------------------


class TestInvestmentTag:
    def test_no_payout_after_non_boss(self):
        gs = _init_gs()
        entry = _award(gs, "tag_investment")
        _to_shop(gs)  # Small blind beaten — not a boss
        assert not entry.get("consumed")

    def test_pays_after_boss(self):
        gs = _init_gs()
        entry = _award(gs, "tag_investment")
        _beat_blind(gs)
        gs["blind"].boss = True  # treat the beaten blind as a boss
        dollars_before_cashout = gs["dollars"]
        earnings = gs["round_earnings"].total
        step(gs, CashOut())

        payout = TAGS["tag_investment"].config["dollars"]
        assert gs["dollars"] == dollars_before_cashout + earnings + payout
        assert entry["consumed"] is True


# ---------------------------------------------------------------------------
# Juggle Tag — round_start_bonus
# ---------------------------------------------------------------------------


class TestJuggleTag:
    def test_hand_size_bonus_for_one_round(self):
        gs = _init_gs()
        entry = _award(gs, "tag_juggle")
        base_hand_size = gs["hand_size"]
        bonus = TAGS["tag_juggle"].config["h_size"]

        step(gs, SelectBlind())
        assert gs["hand_size"] == base_hand_size + bonus
        assert len(gs["hand"]) == base_hand_size + bonus
        assert entry["consumed"] is True

        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["hand_size"] == base_hand_size  # reverted at round end

    def test_next_round_unaffected(self):
        gs = _init_gs()
        _award(gs, "tag_juggle")
        base_hand_size = gs["hand_size"]
        _to_shop(gs)
        step(gs, NextRound())
        step(gs, SelectBlind())
        assert gs["hand_size"] == base_hand_size


# ---------------------------------------------------------------------------
# Double Tag — tag_add (fixed dormant: gs["tags"] was never populated)
# ---------------------------------------------------------------------------


class TestDoubleTag:
    def test_double_duplicates_next_skip_award(self):
        gs = _init_gs()
        # Pin the skip reward (seed TAGWIRE's Small tag happens to be
        # tag_double itself, which a held Double Tag correctly won't copy).
        gs["round_resets"]["blind_tags"]["Small"] = "tag_handy"
        double = _award(gs, "tag_double")
        step(gs, SkipBlind())

        awarded = gs["awarded_tags"]
        assert double["consumed"] is True
        assert double["consumed_context"] == "tag_add"
        skip_awards = [e for e in awarded if e is not double]
        assert len(skip_awards) == 2  # the award + its duplicate
        dup = [e for e in skip_awards if e["blind"] == "double"]
        assert len(dup) == 1
        assert dup[0]["key"] == skip_awards[0]["key"]

    def test_duplicate_applies_full_immediate_effect(self):
        # Orbital duplicate must level a hand type (old code only paid dollars)
        from jackdaw.engine.game import _check_double_tag

        gs = _init_gs()
        _award(gs, "tag_double")
        orbital = _award(gs, "tag_orbital")

        from jackdaw.engine.data.hands import HandType

        levels_before = {ht: gs["hand_levels"].get_state(ht).level for ht in HandType}
        _check_double_tag(gs, orbital)
        levels_after = {ht: gs["hand_levels"].get_state(ht).level for ht in HandType}
        leveled = [ht for ht in levels_after if levels_after[ht] > levels_before[ht]]
        assert len(leveled) == 1
        bump = TAGS["tag_orbital"].config["levels"]
        assert levels_after[leveled[0]] == levels_before[leveled[0]] + bump


# ---------------------------------------------------------------------------
# Reroll cost fixes (voucher discount + free-reroll non-escalation)
# ---------------------------------------------------------------------------


class TestRerollCostFixes:
    def test_reroll_surplus_discount_survives_rerolls(self):
        # Regression: old code used gs["base_reroll_cost"], losing the
        # voucher discount after the first reroll of each shop.
        gs = _init_gs()
        _to_shop(gs)
        gs["round_resets"]["reroll_cost"] = 3  # as Reroll Surplus sets it
        gs["current_round"]["reroll_cost"] = 3
        gs["dollars"] = 50

        step(gs, Reroll())
        assert gs["current_round"]["reroll_cost"] == 4  # 3+1, not 5+1

    def test_free_rerolls_do_not_escalate_price(self):
        gs = _init_gs()
        _to_shop(gs)
        gs["current_round"]["free_rerolls"] = 2
        gs["current_round"]["reroll_cost"] = 0

        step(gs, Reroll())  # consumes one free reroll, one still left
        assert gs["current_round"]["free_rerolls"] == 1
        assert gs["current_round"]["reroll_cost"] == 0
        assert gs["current_round"]["reroll_cost_increase"] == 0

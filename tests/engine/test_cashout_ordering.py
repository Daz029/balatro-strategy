"""Integration tests pinning the REAL cash-out money/interest ordering.

Drives the actual ``step()`` state machine (``SelectBlind`` -> ``PlayHand``
-> ``CashOut``) rather than calling ``calculate_round_earnings`` directly —
every existing unit test on that function (``tests/engine/test_economy.py``,
``TestFullRoundEarnings`` in ``tests/engine/test_consumables.py``) only
exercises the handler in isolation, and this project has repeatedly found
bugs that live only in the integration seam between a correct handler and
its real call site (CLAUDE.md "Integration-seam joker bugs": five jokers
found dead with green handler tests — assert through the real path, never
the handler).

Written for the h1 "Terminal $ term" V_curve wiring
(``docs/post-regen-training-plan.md`` section 3 / CLAUDE.md "h1 objective &
training"), which needs the engine's actual money-vs-interest ordering
verified before ``jackdaw/env/cashout_mirror.py`` can be trusted to predict
post-cash-out dollars from a snapshot.

Expected ordering (per the plan): in-blind money earnings (Business Card,
Rough Gem, gold cards) land BEFORE interest is computed and may cross a $5
bracket; end-of-round PAYOUT JOKERS (Golden Joker, Rocket) pay AFTER
interest and must not affect it — same rule class as the already-verified
"Investment pays after interest" (``TestInvestmentTag`` in
``test_tag_wiring.py``).

Both halves now hold: ``TestJokerEndOfRoundPaysAfterInterest`` guards the
payout ordering restored by the ``game.py::_round_won`` fix.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import CashOut, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import TAGS
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_gs(seed: str = "CASHOUTORDER") -> dict[str, Any]:
    return initialize_run("b_red", 1, seed)


def _beat_blind_at(gs: dict[str, Any], *, dollars: int) -> None:
    """Select the pending blind, pin the bank balance, then win trivially."""
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    gs["dollars"] = dollars
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))


# ---------------------------------------------------------------------------
# In-blind money (Rough Gem) lands BEFORE interest — matches expectation
# ---------------------------------------------------------------------------


class TestInBlindEarningsBeforeInterest:
    """Money earned DURING the hand (not at round-end) is already baked into
    ``gs["dollars"]`` by the time ``_round_won`` calls
    ``calculate_round_earnings`` — so it correctly crosses a $5 bracket.
    """

    def test_scored_diamonds_cross_five_dollar_bracket(self):
        gs = _init_gs()
        gs["jokers"].append(create_joker("j_rough_gem"))
        step(gs, SelectBlind())
        # Replace the first 5 dealt cards with 5 Diamonds so every scored
        # card triggers Rough Gem's deterministic +$1 (no RNG involved).
        gs["hand"][0:5] = [
            create_playing_card(Suit.DIAMONDS, Rank.TWO),
            create_playing_card(Suit.DIAMONDS, Rank.FOUR),
            create_playing_card(Suit.DIAMONDS, Rank.SIX),
            create_playing_card(Suit.DIAMONDS, Rank.EIGHT),
            create_playing_card(Suit.DIAMONDS, Rank.TEN),
        ]
        gs["blind"].chips = 1
        gs["dollars"] = 3  # below the threshold on its own
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

        # +$1 per diamond scored, applied in-blind (scoring.py), well before
        # _round_won/calculate_round_earnings run.
        assert gs["dollars"] == 3 + 5
        earnings = gs["round_earnings"]
        assert earnings.interest == 1  # 8 // 5 == 1 -- the in-blind $ counted

    def test_control_without_rough_gem_stays_under_bracket(self):
        """Paired control: same $3 start, no in-blind earnings -> no interest."""
        gs = _init_gs()
        _beat_blind_at(gs, dollars=3)
        assert gs["round_earnings"].interest == 0


# ---------------------------------------------------------------------------
# End-of-round joker payouts (Golden Joker, Rocket) pay once, after interest
# ---------------------------------------------------------------------------


class TestJokerEndOfRoundPaysAfterInterest:
    """Golden Joker's payout flows once, via ``earnings.total`` at cash-out,
    strictly after interest — regression for the inherited double-count /
    interest-leak bug fixed in ``game.py::_round_won``.
    """

    def test_golden_joker_pays_once_and_never_bumps_interest(self):
        gs = _init_gs()
        gs["jokers"].append(create_joker("j_golden"))
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        gs["dollars"] = 1  # +$4 payout would cross the $5 bracket if it leaked
        dollars_before_round = gs["dollars"]

        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

        golden_payout = 4  # Golden Joker: +$4/round, card.lua:1658
        # Not pre-applied: the balance is untouched until cash-out.
        assert gs["dollars"] == dollars_before_round

        earnings = gs["round_earnings"]
        assert earnings.joker_dollars == golden_payout
        # Interest is computed on the pre-payout balance ($1): no leak.
        assert earnings.interest == 0

        step(gs, CashOut())
        # Paid exactly once, through the total.
        assert gs["dollars"] == dollars_before_round + earnings.total
        assert earnings.total == (
            earnings.blind_reward
            + earnings.unused_hands_bonus
            + earnings.unused_discards_bonus
            + golden_payout
        )


# ---------------------------------------------------------------------------
# Investment tag pays strictly AFTER interest — matches expectation
# ---------------------------------------------------------------------------


class TestInvestmentPaysAfterInterest:
    """Precedent already verified in real Balatro (CLAUDE.md tag-wiring
    item): Investment's $25 is applied at cash-out strictly after
    ``earnings.total`` (which already has that round's interest baked in),
    so it can never retroactively bump that round's interest bracket.

    A more detailed version of ``TestInvestmentTag.test_pays_after_boss`` in
    ``test_tag_wiring.py``; kept here so this file stands alone as the
    interest-ordering verification suite for the h1 V_curve work.
    """

    def test_investment_payout_lands_after_earnings_total(self):
        gs = _init_gs()
        gs.setdefault("awarded_tags", []).append(
            {"key": "tag_investment", "result": None, "blind": "Small"}
        )
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        gs["blind"].boss = True  # Investment only pays after a boss blind
        gs["dollars"] = 0  # deliberately low: if $25 counted toward THIS
        # round's interest it would cross five separate $5 brackets
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

        earnings = gs["round_earnings"]
        pre_cashout_dollars = gs["dollars"]

        step(gs, CashOut())

        payout = TAGS["tag_investment"].config["dollars"]
        assert gs["dollars"] == pre_cashout_dollars + earnings.total + payout

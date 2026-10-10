"""End-of-round money calculation.

Ports the reward-row calculations in ``evaluate_round``
(``state_events.lua:1135``) as a pure function. Rental is already committed
by ``end_round`` at ``state_events.lua:108``.

**Order of operations** (matching Lua source):

1. Blind reward — ``blind.dollars`` only when committed chips beat it.
2. Unused hands bonus — ``hands_left × (money_per_hand or 1)``.
   Skipped when ``modifiers['no_extra_hand_money']`` is set.
3. Unused discards bonus — ``discards_left × money_per_discard``.
   Only present when ``modifiers['money_per_discard']`` is set (Green Deck).
4. Joker dollar bonuses — via ``round_dollar_bonus`` (``calc_dollar_bonus``).
5. Evaluation tags.
6. Interest — ``interest_amount × min(money // 5, interest_cap // 5)``.
   Skipped when ``modifiers['no_interest']`` is set or ``effective_money < 5``.

Source references
-----------------
- state_events.lua:96-110  — rental deduction, end-of-round joker evals
- state_events.lua:1135-1208 — evaluate_round (blind + hands + discards
  + joker bonuses + interest)
- state_events.lua:433 — discard cost modifier (Golden Needle challenge)
- game.lua:1909-1915 — ``interest_cap=25``, ``interest_amount=1``,
  ``rental_rate=3`` defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom

from jackdaw.engine.jokers import round_dollar_bonus
from jackdaw.engine.read import StateView

# ---------------------------------------------------------------------------
# RoundEarnings — per-round cash-out descriptor
# ---------------------------------------------------------------------------


@dataclass
class RoundEarnings:
    """Breakdown of money earned (and lost) at end of a round.

    All monetary values are in whole dollars. ``total`` is Lua's bottom-row
    value written to ``current_round.dollars``; rent has already changed the
    committed balance and is retained here only as audit metadata.

    ``total`` is the net dollar change applied by ``ease_dollars``:
    ``blind_reward + unused_hands_bonus + unused_discards_bonus
    + joker_dollars + tag_dollars + interest``
    """

    blind_reward: int = 0
    """Dollars for beating the blind (``blind.dollars``).

    Set to 0 by stake modifiers (``no_blind_reward``) or skipped blinds.
    """

    unused_hands_bonus: int = 0
    """``hands_left × money_per_hand`` (default $1/hand).

    0 when ``modifiers['no_extra_hand_money']`` is truthy.
    """

    unused_discards_bonus: int = 0
    """``discards_left × money_per_discard``.

    0 unless ``modifiers['money_per_discard']`` is set (Green Deck).
    """

    interest: int = 0
    """``interest_amount × min(effective_money // 5, interest_cap // 5)``.

    Computed on post-rental balance.  0 when ``modifiers['no_interest']``
    or ``effective_money < 5``.
    """

    joker_dollars: int = 0
    """Sum of per-round joker dollar bonuses (Golden Joker $4, Cloud 9 …)."""

    tag_dollars: int = 0
    """Evaluation-tag rows (Investment Tag in vanilla)."""

    rental_cost: int = 0
    """Rent already committed during F6; informational, not part of ``total``."""

    total: int = 0
    """Net dollars earned this round."""


# ---------------------------------------------------------------------------
# Discard cost helper (Golden Needle challenge)
# ---------------------------------------------------------------------------


def calculate_discard_cost(game_state: dict[str, Any]) -> int:
    """Return the dollar cost charged per discard action.

    Mirrors the ``G.GAME.modifiers.discard_cost`` check in
    ``state_events.lua:433`` — the Golden Needle challenge sets this to 1.

    Returns 0 if no discard cost modifier is active.
    """
    return game_state.get("modifiers", {}).get("discard_cost", 0)


# ---------------------------------------------------------------------------
# calculate_round_earnings
# ---------------------------------------------------------------------------


def calculate_round_earnings(
    blind: Blind,
    hands_left: int,
    discards_left: int,
    money: int,
    jokers: list[Card],
    game_state: dict[str, Any],
    rng: PseudoRandom | None = None,
    *,
    joker_dollars: int | None = None,
    blind_reward: int | None = None,
    tag_dollars: int = 0,
    rental_cost: int = 0,
) -> RoundEarnings:
    """Compute Lua's round-evaluation reward rows.

    Mirrors ``evaluate_round`` (``state_events.lua:1135-1208``). ``money``
    must already include F6 held-card payouts and rental charges.

    ``game_state`` keys used:

    +-----------------------+----------+--------------------------------------+
    | Key                   | Default  | Description                          |
    +=======================+==========+======================================+
    | ``interest_cap``      | 25       | ``G.GAME.interest_cap`` — maximum    |
    |                       |          | money that earns interest             |
    |                       |          | (Lua default 25 → 5 brackets → $5   |
    |                       |          | max at 1× rate; Seed Money→50,       |
    |                       |          | Money Tree→100).                     |
    +-----------------------+----------+--------------------------------------+
    | ``interest_amount``   | 1        | Dollars per $5 bracket.              |
    |                       |          | To the Moon adds +1 per copy.        |
    +-----------------------+----------+--------------------------------------+
    | ``rental_rate``       | 3        | Cost per rental joker per round.     |
    +-----------------------+----------+--------------------------------------+
    | ``modifiers``         | {}       | Dict of run modifiers:               |
    |                       |          | ``no_extra_hand_money``: bool        |
    |                       |          | ``money_per_hand``: int (default 1)  |
    |                       |          | ``money_per_discard``: int|None      |
    |                       |          | ``no_interest``: bool                |
    +-----------------------+----------+--------------------------------------+

    Args:
        blind: The :class:`~jackdaw.engine.blind.Blind` beaten this round.
        hands_left: Unused hands at round end.
        discards_left: Unused discards at round end.
        money: Bank balance *before* this round's earnings are applied.
        jokers: Active joker cards.
        game_state: Run-level state dict.
        rng: PseudoRandom instance (for end-of-round joker RNG effects).
        joker_dollars: Pre-computed joker dollar bonus from ``_round_won``.
            When provided, the fallback calculation is skipped.

    Returns:
        :class:`RoundEarnings` with all components and their net total.
    """
    modifiers: dict[str, Any] = game_state.get("modifiers", {})
    interest_amount: int = game_state.get("interest_amount", 1)
    interest_cap: int = game_state.get("interest_cap", 25)
    # ------------------------------------------------------------------
    # Step 2 — Blind reward (state_events.lua:1139)
    # ------------------------------------------------------------------
    if blind_reward is None:
        blind_reward = blind.dollars

    # ------------------------------------------------------------------
    # Step 3 — Unused hands bonus (state_events.lua:1165)
    # hands_left * (modifiers.money_per_hand or 1)
    # ------------------------------------------------------------------
    if hands_left > 0 and not modifiers.get("no_extra_hand_money"):
        money_per_hand: int = modifiers.get("money_per_hand", 1)
        unused_hands_bonus = hands_left * money_per_hand
    else:
        unused_hands_bonus = 0

    # ------------------------------------------------------------------
    # Step 4 — Unused discards bonus (state_events.lua:1170)
    # Only when modifiers.money_per_discard is set (Green Deck)
    # ------------------------------------------------------------------
    money_per_discard: int | None = modifiers.get("money_per_discard")
    if discards_left > 0 and money_per_discard:
        unused_discards_bonus = discards_left * money_per_discard
    else:
        unused_discards_bonus = 0

    # ------------------------------------------------------------------
    # Step 5 — Joker dollar bonuses (state_events.lua:1175)
    # calc_dollar_bonus per joker: Golden Joker, Cloud 9, Satellite, etc.
    #
    # When joker_dollars is pre-computed (passed from _round_won), skip
    # end_of_round handlers are deliberately not fired by this fallback.
    # ------------------------------------------------------------------
    if joker_dollars is None:
        joker_dollars = round_dollar_bonus(
            jokers,
            StateView(game_state or {}, jokers=jokers),
        )

    # ------------------------------------------------------------------
    # Step 6 — Interest (state_events.lua:1191)
    # interest_amount * min(effective_money // 5, interest_cap // 5)
    # Requires effective_money >= 5 and modifiers.no_interest not set.
    # ------------------------------------------------------------------
    if money >= 5 and not modifiers.get("no_interest"):
        interest = interest_amount * min(
            money // 5,
            interest_cap // 5,
        )
    else:
        interest = 0

    total = (
        blind_reward
        + unused_hands_bonus
        + unused_discards_bonus
        + joker_dollars
        + tag_dollars
        + interest
    )

    return RoundEarnings(
        blind_reward=blind_reward,
        unused_hands_bonus=unused_hands_bonus,
        unused_discards_bonus=unused_discards_bonus,
        interest=interest,
        joker_dollars=joker_dollars,
        tag_dollars=tag_dollars,
        rental_cost=rental_cost,
        total=total,
    )

"""Blind chip requirement scaling.

Matches ``get_blind_amount`` in ``misc_functions.lua:919``.

The actual chip target for a blind is::

    target = get_blind_amount(ante, scaling) × blind_mult × ante_scaling

Where ``blind_mult`` is 1 for Small, 1.5 for Big, 2 for Boss, and
``ante_scaling`` is 1.0 normally or 2.0 for Plasma Deck.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Hardcoded base amounts for antes 1-8 (from misc_functions.lua:922-944)
# ---------------------------------------------------------------------------

_AMOUNTS: dict[int, list[int]] = {
    1: [300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000],
    2: [300, 900, 2_600, 8_000, 20_000, 36_000, 60_000, 100_000],
    3: [300, 1_000, 3_200, 9_000, 25_000, 60_000, 110_000, 200_000],
}

# Blind type multipliers (from P_BLINDS: bl_small.mult, bl_big.mult, boss.mult)
BLIND_MULT: dict[str, float] = {
    "Small": 1.0,
    "Big": 1.5,
    "Boss": 2.0,
}


def get_blind_amount(ante: int, scaling: int = 1) -> int:
    """Return the base chip requirement for a given ante and scaling level.

    Args:
        ante: Current ante number (1-8 uses hardcoded tables, 9+ exponential).
        scaling: Difficulty scaling level from stake:
            1 = White/Red stake (default), 2 = Green-Blue, 3 = Purple-Gold.

    Returns:
        Base chip amount (before blind_mult and ante_scaling multipliers).
    """
    if ante < 1:
        return 100
    if ante > 16:
        # Audit D49: past here Lua's doubles and Python's ints diverge
        # (and overflow differently). Only endless reaches it, and endless
        # is not implemented (jackdaw.engine.scope).
        from jackdaw.engine.scope import UnsupportedGameModeError

        raise UnsupportedGameModeError(f"blind amounts past ante 16 (got {ante})")

    amounts = _AMOUNTS.get(scaling, _AMOUNTS[1])

    if ante <= 8:
        return amounts[ante - 1]  # 0-indexed

    # Exponential formula for antes 9+ (misc_functions.lua:927-929)
    k = 0.75
    a = amounts[7]  # ante 8 base amount
    b = 1.6
    c = ante - 8
    d = 1 + 0.2 * c

    amount = math.floor(a * (b + (k * c) ** d) ** c)

    # Significant-figure rounding (misc_functions.lua:929)
    # amount = amount - amount % (10 ^ floor(log10(amount) - 1))
    if amount > 0:
        digits = math.floor(math.log10(amount)) - 1
        rounding = 10**digits
        amount = amount - amount % rounding

    return amount



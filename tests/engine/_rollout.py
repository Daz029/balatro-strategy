"""Seeded full-run driver for structural invariant tests.

Random play dies in the first blind, so it barely reaches the shop. This
driver plays hands with the deterministic greedy hand policy, always selects
the blind, and spends up to ``shop_budget`` random non-NextRound actions per
shop visit with a dollar floor, so rerolls, purchases, sales, packs and
consumable use all happen many times per run.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.engine.actions import Action, GamePhase, NextRound, SelectBlind, get_legal_actions
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import random_agent


def drive(
    seed: str,
    rnd: random.Random,
    *,
    max_steps: int = 600,
    shop_budget: int = 12,
    dollar_floor: int = 60,
) -> Iterator[tuple[dict[str, Any], Action]]:
    """Yield ``(gs, action)`` after every engine step of one seeded run."""
    gs = initialize_run("b_red", 1, seed)
    greedy = GreedyHandPolicy()
    shop_steps = 0
    for _ in range(max_steps):
        legal = get_legal_actions(gs)
        if not legal:
            return
        phase = gs.get("phase")
        if phase == GamePhase.SELECTING_HAND:
            action = greedy(gs)
        elif phase == GamePhase.BLIND_SELECT:
            action = next((a for a in legal if isinstance(a, SelectBlind)), legal[0])
        elif phase == GamePhase.SHOP:
            gs["dollars"] = max(gs.get("dollars", 0), dollar_floor)
            others = [a for a in legal if not isinstance(a, NextRound)]
            if others and shop_steps < shop_budget:
                shop_steps += 1
                random.seed(rnd.randrange(2**32))
                action = random_agent(gs, others)
            else:
                shop_steps = 0
                action = next(a for a in legal if isinstance(a, NextRound))
        else:
            # random_agent materializes PlayHand/Discard marker actions.
            random.seed(rnd.randrange(2**32))
            action = random_agent(gs, legal)
        step(gs, action)
        yield gs, action

"""HandPlayAdapter — isolated hand-play episodes for hand-agent training.

Starts episodes directly at the ``SELECTING_HAND`` phase with an injected,
domain-randomized ante / joker set / hands-discards-left / money, bypassing
blind-select and the shop entirely. Built on the same ``initialize_run`` +
``SelectBlind`` pipeline :class:`~jackdaw.env.game_interface.DirectAdapter`
uses for full runs, so blind construction, joker ``setting_blind`` triggers,
deck shuffling, and hand-drawing all go through the real engine unchanged —
only the values fed into that pipeline are overridden before the engine
runs.

See ``CLAUDE.md`` ("Ante-play (hand/discard) track" -> "Training the
hand-agent in isolation from shop") for the training design this supports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from jackdaw.engine.actions import Action, GamePhase, SelectBlind
from jackdaw.env.game_interface import GameState, snapshot


@dataclass
class HandPlayConfig:
    """Domain-randomization ranges for isolated hand-play episodes.

    Sampled deterministically from the ``seed`` passed to :meth:`reset`, so
    episodes stay reproducible.  Boss blinds are intentionally excluded from
    ``blind_stages`` by default — their unique mechanics are out of scope
    for the initial curriculum stage (see CLAUDE.md curriculum: no jokers ->
    small random joker subsets -> full random coverage). Boss selection is
    ante-correct if enabled (see :meth:`reset`), so widening ``blind_stages``
    to include ``"Boss"`` later is safe.

    ``dollars_range`` is sampled flat/uniform regardless of ante — this is a
    stage-1/2 placeholder, not a claim that money is ante-independent in real
    play. It's out of scope until the shop-agent's marginal-value-of-$1 curve
    (see CLAUDE.md "Money/dollar handling") exists and gets fed in.

    Injected jokers are always base/no-edition, no-sticker (no Foil/Holo/
    Polychrome, no Eternal/Perishable/Rental) — a deliberate curriculum-stage
    simplification, not an oversight. Revisit once training targets "full
    random coverage" that should match real shop-purchase distributions.
    """

    ante_range: tuple[int, int] = (1, 8)
    joker_pool: tuple[str, ...] = ()
    joker_count_range: tuple[int, int] = (0, 0)
    hands_range: tuple[int, int] = (1, 4)
    discards_range: tuple[int, int] = (0, 3)
    dollars_range: tuple[int, int] = (0, 50)
    blind_stages: tuple[str, ...] = ("Small", "Big")


class HandPlayAdapter:
    """GameAdapter that starts episodes mid-run, directly in hand-play.

    Satisfies the same duck-typed contract as
    :class:`~jackdaw.env.game_interface.DirectAdapter` (see
    ``jackdaw/env/game_interface.py::GameAdapter``), so it plugs into
    :class:`~jackdaw.env.balatro_env.BalatroEnvironment` unchanged via
    ``adapter_factory=lambda: HandPlayAdapter(config)``.

    ``done``/``won`` fire at ``ROUND_EVAL`` (blind cleared) rather than
    waiting for ``GAME_OVER``, since isolated hand-play episodes have no
    shop phase to reach.
    """

    def __init__(self, config: HandPlayConfig | None = None) -> None:
        self._config = config or HandPlayConfig()
        self._gs: dict[str, Any] = {}

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        from jackdaw.engine.blind import get_new_boss
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.game import step as engine_step
        from jackdaw.engine.run_init import initialize_run

        cfg = self._config
        sampler = random.Random(seed)

        gs = initialize_run(back_key, stake, seed, challenge=challenge)
        rr = gs["round_resets"]

        ante = sampler.randint(*cfg.ante_range)
        rr["ante"] = ante
        rr["blind_ante"] = ante

        rr["hands"] = sampler.randint(*cfg.hands_range)
        rr["discards"] = sampler.randint(*cfg.discards_range)
        gs["dollars"] = sampler.randint(*cfg.dollars_range)
        gs["blind_on_deck"] = sampler.choice(cfg.blind_stages)

        # initialize_run() already picked a boss key for ante 1 (baked into
        # blind_choices["Boss"] via its internal assign_ante_blinds(1, ...)
        # call). That key is wrong for any other sampled ante, so redraw it
        # directly with get_new_boss() rather than reusing it — and rather
        # than re-running assign_ante_blinds(ante, ...), which would also
        # pollute gs["bosses_used"] with a phantom ante-1 usage count and
        # skew the "favor least-used boss" selection.
        if gs["blind_on_deck"] == "Boss":
            rr["blind_choices"]["Boss"] = get_new_boss(
                ante,
                gs["bosses_used"],
                gs["rng"],
                win_ante=gs.get("win_ante", 8),
                banned_keys=gs.get("banned_keys"),
            )

        lo, hi = cfg.joker_count_range
        if cfg.joker_pool and hi > 0:
            count = min(sampler.randint(lo, hi), len(cfg.joker_pool), gs["joker_slots"])
            keys = sampler.sample(list(cfg.joker_pool), count)
            gs["jokers"] = [create_joker(key) for key in keys]

        self._gs = gs
        engine_step(self._gs, SelectBlind())
        return snapshot(self._gs)

    def step(self, action: Action) -> GameState:
        from jackdaw.engine.game import step as engine_step

        engine_step(self._gs, action)
        return snapshot(self._gs)

    def get_legal_actions(self) -> list[Action]:
        from jackdaw.engine.actions import get_legal_actions as engine_legal

        return engine_legal(self._gs)

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._gs

    @property
    def done(self) -> bool:
        phase = self._gs.get("phase")
        return phase in (GamePhase.GAME_OVER, GamePhase.ROUND_EVAL)

    @property
    def won(self) -> bool:
        phase = self._gs.get("phase")
        return phase == GamePhase.ROUND_EVAL or bool(self._gs.get("won", False))

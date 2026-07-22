"""Full-run GameAdapter for the shop agent.

Episodes are entire runs (grilled decision — CLAUDE.md shop-agent design):
the shop agent only ever sees its own decision points (SHOP, PACK_OPENING);
everything else is auto-resolved inside :meth:`ShopRunAdapter.step`:

* ``BLIND_SELECT`` -> ``SelectBlind`` (SkipBlind deliberately not exposed in
  s0 — see the decision record; appended at s1 once tag economics matter).
  ``ShopRunConfig.s1_schema=True`` flips this: on a non-boss blind
  (Small/Big) the adapter STOPS here instead of auto-resolving, handing
  control back to the shop agent (``ShopGymEnv`` turns it into a genuine
  SkipBlind-vs-proceed decision — see that module). Boss blind-select has
  no real choice (SkipBlind is illegal) and always stays auto-resolved,
  s1_schema or not.
* ``SELECTING_HAND`` -> the injected ``hand_policy`` callable
  (``game_state -> engine Action``): :class:`GreedyHandPolicy` for tests
  and ablation baselines, a trained h-agent wrapper for real training,
* ``ROUND_EVAL`` -> ``CashOut`` (the only legal action — not a decision).

The episode ends on GAME_OVER (loss) or ``won`` (beating the ``win_ante``
boss). ``win_ante`` is the horizon-curriculum knob: stage A trains with
``win_ante=2``, then 4, then the full 8 — each stage's objective is a
prefix of the true one.

Snapshot/restore (:meth:`snapshot_state` / :meth:`restore_state`) serializes
the COMPLETE engine state including the RNG, so a restored state continues
byte-identically. This is the substrate for the start-state reservoir
(mixture of fresh runs and harvested mid-run snapshots); the reservoir and
sampling policy live with the gym env, not here.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jackdaw.engine.actions import Action, CashOut, GamePhase, SelectBlind
from jackdaw.env.game_interface import GameState, snapshot

HandDecisionObserver = Callable[[dict[str, Any], Action, dict[str, Any]], None]

# Phases where control returns to the shop agent.
DECISION_PHASES = frozenset({GamePhase.SHOP, GamePhase.PACK_OPENING})

# Generous ceiling on auto-resolved engine steps between two decision
# points (a full blind is ~10: select + <=4 plays + <=4 discards + cash
# out). Exceeding it means the hand policy is stuck in a loop.
_MAX_AUTO_STEPS = 64


@dataclass
class ShopRunConfig:
    """Episode configuration for :class:`ShopRunAdapter`.

    win_ante:
        Beating this ante's boss ends the episode as a win (the engine's
        own ``won`` check reads ``gs["win_ante"]``). The horizon-curriculum
        knob: 2 -> 4 -> 8.
    s1_schema:
        Opt-in flag (default False, preserving s0's exact auto-resolve
        behavior byte-for-byte) that exposes non-boss blind-select as a
        real agent decision instead of auto-resolving it. See
        :meth:`ShopRunAdapter._advance` and ``shop_gym.py``.
    """

    win_ante: int = 8
    s1_schema: bool = False


class ShopRunAdapter:
    """GameAdapter running full episodes with auto-resolved hand phases."""

    def __init__(
        self,
        hand_policy: Callable[[dict[str, Any]], Action],
        config: ShopRunConfig | None = None,
        *,
        hand_decision_observer: HandDecisionObserver | None = None,
    ) -> None:
        self._hand_policy = hand_policy
        self._config = config or ShopRunConfig()
        self._hand_decision_observer = hand_decision_observer
        self._gs: dict[str, Any] = {}

    # -- GameAdapter protocol -------------------------------------------------

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        from jackdaw.engine.run_init import initialize_run

        self._gs = initialize_run(back_key, stake, seed, challenge=challenge)
        self._gs["phase"] = GamePhase.BLIND_SELECT
        self._gs["blind_on_deck"] = "Small"
        self._gs["win_ante"] = self._config.win_ante
        self._advance()
        return snapshot(self._gs)

    def step(self, action: Action) -> GameState:
        from jackdaw.engine.game import step as engine_step

        engine_step(self._gs, action)
        self._advance()
        return snapshot(self._gs)

    def get_legal_actions(self) -> list[Action]:
        from jackdaw.engine.actions import get_legal_actions as engine_legal

        return engine_legal(self._gs)

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._gs

    @property
    def done(self) -> bool:
        return self.won or self._gs.get("phase") == GamePhase.GAME_OVER

    @property
    def won(self) -> bool:
        return bool(self._gs.get("won", False))

    # -- snapshot / restore (start-state reservoir substrate) ------------------

    def snapshot_state(self) -> bytes:
        """Serialize the complete engine state, RNG included.

        The bytes are self-contained: restoring them into any adapter
        instance (same code version) continues the run byte-identically.
        Pickle is fine here — snapshots are internal training artifacts,
        never untrusted input.
        """
        return pickle.dumps(self._gs, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_state(self, blob: bytes) -> GameState:
        """Restore a snapshot taken by :meth:`snapshot_state`.

        The restored state is always at a decision point (or terminal),
        because snapshots can only be taken when control is with the agent.
        """
        self._gs = pickle.loads(blob)
        return snapshot(self._gs)

    # -- internal ---------------------------------------------------------------

    def _advance(self) -> None:
        """Auto-resolve until the next shop-agent decision point or episode end."""
        from jackdaw.engine.game import step as engine_step

        for _ in range(_MAX_AUTO_STEPS):
            if self.done:
                return
            phase = self._gs.get("phase")
            if phase in DECISION_PHASES:
                return
            if phase == GamePhase.BLIND_SELECT:
                on_deck = self._gs.get("blind_on_deck", "Small")
                if self._config.s1_schema and on_deck in ("Small", "Big"):
                    # s1: expose skip-vs-proceed as an agent decision
                    # instead of auto-resolving (boss stays auto -- no
                    # real choice, SkipBlind is illegal there).
                    return
                engine_step(self._gs, SelectBlind())
            elif phase == GamePhase.SELECTING_HAND:
                pre_state = (
                    copy.deepcopy(self._gs) if self._hand_decision_observer is not None else None
                )
                action = self._hand_policy(self._gs)
                engine_step(self._gs, action)
                if self._hand_decision_observer is not None:
                    self._hand_decision_observer(pre_state, action, self._gs)
            elif phase == GamePhase.ROUND_EVAL:
                engine_step(self._gs, CashOut())
            else:
                raise RuntimeError(f"unexpected phase during auto-advance: {phase!r}")
        raise RuntimeError(f"auto-advance exceeded {_MAX_AUTO_STEPS} steps -- hand policy stuck?")

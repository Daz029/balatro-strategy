"""Gymnasium environment for isolated hand-play episodes.

Purpose-built for the BC -> MaskablePPO hand-agent pipeline (see CLAUDE.md
ante-play track), deliberately NOT reusing ``BalatroGymnasiumEnv``:

  - **Canonical Discrete(436) action space** (``jackdaw/agents/
    hand_action_space.py``) instead of the full-run wrapper's per-step,
    randomly-subsampled action table. BC labels map onto fixed indices and
    the policy head's outputs keep stable meanings across BC and PPO.
  - **Observation = the BC demo-shard schema** (same keys and feature
    layouts as ``scripts/generate_hand_demos.py`` writes), plus an
    always-masked consumable block reserved for the eventual shop-merge:
    with masked pooling an absent entity type contributes exactly nothing,
    so the dormant block costs nothing and freezes the observation space /
    checkpoint format across that merge. The hand block is wider than the
    shards' (``MAX_HAND_CARDS_OBS`` = 12 vs 8) for The Serpent's over-draw
    and +hand-size effects; the BC loader zero-pads shard rows up to it
    (exact under masked pooling), so shards need no regeneration.
  - **Env-side optimal ordering**: the agent picks a card *subset*; when an
    order-sensitive joker/card is present (Photograph, Hanging Chad, Glass,
    ...), the env submits the subset in the engine-optimal scoring order
    via ``jackdaw.engine.play_ordering.best_play_order``. Ordering is a
    mechanical optimization delegated to the engine, not something the
    agent must learn -- and it keeps realized rewards consistent with the
    solver's ``p_clear`` labels, which assume best-order play.
  - **Reward is exactly the solver's objective**: terminal 1.0 on clearing
    the blind, 0.0 otherwise, nothing dense. With ``gamma=1.0`` (episodes
    are <= hands+discards <= 7 real steps) the value function's target IS
    P(clear), matching the ``p_clear`` regression used to warm-start the
    critic. Deliberately no shaping (see CLAUDE.md: potential-based only,
    if ever).

    FUTURE HOOK (h1 fine-tune stage): once the shop-agent's critic exists,
    unused hands/dollars gain a terminal value term --
    ``1{clear} + f(hands_left, dollars)`` from the marginal-value-of-$1
    curve (CLAUDE.md "Money/dollar handling"). That is a deliberate
    objective change made here, at the terminal reward, not shaping.

Seeding: episode seeds are strings ``f"{seed_prefix}_{n:08d}"`` fed to
``HandPlayAdapter`` (fully deterministic per seed). ``reset(seed=n)`` jumps
the episode counter to *n*; ``reset(options={"episode_seed": s})`` pins an
exact seed string (used by the fixed eval suite -- prefix ``EVAL_`` is
reserved for it and must never be used for training).
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw.agents.hand_action_space import (
    NUM_HAND_ACTIONS,
    action_to_combo,
    legal_action_mask,
)
from jackdaw.engine.actions import Discard as EngineDiscard
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.actions import PlayHand as EnginePlayHand
from jackdaw.engine.play_ordering import best_play_order, needs_permutation_search
from jackdaw.env.action_space import ActionType
from jackdaw.env.hand_play_adapter import HandPlayAdapter, HandPlayConfig
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    encode_global_context,
    encode_jokers_batch,
    encode_playing_cards_batch,
)

# Observation width of the hand block -- deliberately WIDER than the action
# space's MAX_HAND_CARDS=8 (frozen, see hand_action_space.py): under The
# Serpent the engine draws exactly 3 cards after every play/discard with no
# hand-size cap, so the hand legitimately grows past 8 (and growth
# compounds: each 1-card action nets +2). +hand-size effects (Turtle Bean,
# Troubadour, Juggler, the Juggle tag, vouchers) push full-run hands to
# 10-13 as well. 12 covers the realistic range; cards beyond row 12 are
# TRUNCATED, not an error: the hand is engine-sorted descending and the
# action space can only address positions 0-7, so dropped rows are the
# lowest cards and unplayable anyway (the only cost is a sliver of
# held-card-effect visibility in an extreme tail). Positions 8-11 are
# visible-but-unplayable by construction -- a known systemic ceiling of the
# 8-position action space, recorded as an open item in CLAUDE.md (decision
# deferred to the h1 seam).
#
# BC demo shards keep writing 8-wide hand blocks (generation is
# single-snapshot; reset hands never exceed 8) -- train_bc.py's loader
# zero-pads them up to this width, which is semantically exact under masked
# pooling. Widening this constant is therefore NOT a demo-schema change.
MAX_HAND_CARDS_OBS = 12

MAX_JOKERS = 5
MAX_CONSUMABLES = 2  # dormant this stage; reserved seam for the shop merge

BACK_KEY = "b_red"
STAKE = 1

# Pure safety net: play/discard each consume budget, so episodes end
# naturally in <= hands+discards <= 7 steps. Truncation counts as a loss.
DEFAULT_MAX_STEPS = 32


def _pad(arr: np.ndarray, max_n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((max_n, dim), dtype=np.float32)
    mask = np.zeros(max_n, dtype=np.float32)
    n = arr.shape[0]
    if n > max_n:
        raise ValueError(f"entity count {n} exceeds max {max_n}")
    if n > 0:
        padded[:n] = arr
        mask[:n] = 1.0
    return padded, mask


def build_observation(gs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Encode a hand-play game state into the BC demo-shard schema.

    Must stay field-for-field in sync with ``scripts/generate_hand_demos.py``
    (``SCHEMA_VERSION`` there guards drift), plus the dormant consumable
    block the shards don't carry (BC's loader synthesizes zeros for it).
    """
    hand_arr = encode_playing_cards_batch(gs.get("hand", []), gs)
    joker_arr = encode_jokers_batch(gs.get("jokers", []), gs)
    # Truncate (don't raise) on hand overflow -- see MAX_HAND_CARDS_OBS.
    # Encode the FULL hand first: per-card features (is_best_hand_card, ...)
    # consider the whole hand, and rows 0..7 must stay index-aligned with
    # the positions the action space addresses regardless of overflow.
    hand_padded, hand_mask = _pad(
        hand_arr[:MAX_HAND_CARDS_OBS], MAX_HAND_CARDS_OBS, D_PLAYING_CARD
    )
    joker_padded, joker_mask = _pad(joker_arr, MAX_JOKERS, D_JOKER)
    return {
        "global_context": encode_global_context(gs).astype(np.float32),
        "hand_cards": hand_padded,
        "hand_mask": hand_mask,
        "jokers": joker_padded,
        "joker_mask": joker_mask,
        # HandPlayAdapter never injects consumables at this curriculum
        # stage; masked pooling makes the block contribute exactly nothing.
        "consumables": np.zeros((MAX_CONSUMABLES, D_CONSUMABLE), dtype=np.float32),
        "consumable_mask": np.zeros(MAX_CONSUMABLES, dtype=np.float32),
    }


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "global_context": spaces.Box(-np.inf, np.inf, shape=(D_GLOBAL,), dtype=np.float32),
            "hand_cards": spaces.Box(
                -np.inf, np.inf, shape=(MAX_HAND_CARDS_OBS, D_PLAYING_CARD), dtype=np.float32
            ),
            "hand_mask": spaces.Box(0.0, 1.0, shape=(MAX_HAND_CARDS_OBS,), dtype=np.float32),
            "jokers": spaces.Box(-np.inf, np.inf, shape=(MAX_JOKERS, D_JOKER), dtype=np.float32),
            "joker_mask": spaces.Box(0.0, 1.0, shape=(MAX_JOKERS,), dtype=np.float32),
            "consumables": spaces.Box(
                -np.inf, np.inf, shape=(MAX_CONSUMABLES, D_CONSUMABLE), dtype=np.float32
            ),
            "consumable_mask": spaces.Box(0.0, 1.0, shape=(MAX_CONSUMABLES,), dtype=np.float32),
        }
    )


class HandPlayGymEnv(gymnasium.Env):
    """Isolated hand-play episodes with the canonical Discrete(436) space.

    Parameters
    ----------
    config:
        Domain-randomization ranges forwarded to :class:`HandPlayAdapter`
        (use the stage presets from ``scripts/generate_hand_demos.py`` to
        match a BC dataset's generating distribution).
    seed_prefix:
        Prefix for auto-generated episode seed strings. ``EVAL_`` is
        reserved for the fixed evaluation suite.
    max_steps:
        Safety truncation (counted as a loss); see ``DEFAULT_MAX_STEPS``.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: HandPlayConfig | None = None,
        seed_prefix: str = "HANDPPO",
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        super().__init__()
        self._config = config or HandPlayConfig()
        self._seed_prefix = seed_prefix
        self._max_steps = max_steps
        self._adapter = HandPlayAdapter(self._config)
        self._episode_counter = 0
        self._steps = 0
        self._episode_seed = ""

        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(NUM_HAND_ACTIONS)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_counter = seed
        if options and "episode_seed" in options:
            self._episode_seed = str(options["episode_seed"])
        else:
            self._episode_seed = f"{self._seed_prefix}_{self._episode_counter:08d}"
            self._episode_counter += 1

        self._adapter.reset(BACK_KEY, STAKE, self._episode_seed)
        self._steps = 0
        gs = self._adapter.raw_state
        return build_observation(gs), {
            "episode_seed": self._episode_seed,
            "action_mask": self.action_masks(),
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        mask = self.action_masks()
        if not mask[action]:
            # A masked policy can never select these; reaching here means a
            # wiring bug (stale mask, wrong env pairing) -- fail loudly
            # rather than teach the agent that illegal actions no-op.
            raise ValueError(
                f"illegal action {action} (seed={self._episode_seed}, step={self._steps})"
            )

        engine_action = self._to_engine_action(action)
        self._adapter.step(engine_action)
        self._steps += 1

        gs = self._adapter.raw_state
        terminated = self._adapter.done
        truncated = not terminated and self._steps >= self._max_steps
        won = terminated and self._adapter.won
        reward = 1.0 if won else 0.0

        info: dict[str, Any] = {
            "episode_seed": self._episode_seed,
            "action_mask": self.action_masks(),
        }
        if terminated or truncated:
            info["balatro/cleared"] = won
            info["balatro/hands_left"] = gs.get("current_round", {}).get("hands_left", 0)

        return build_observation(gs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Legality mask for sb3-contrib's MaskablePPO."""
        gs = self._adapter.raw_state
        if gs.get("phase") != GamePhase.SELECTING_HAND:
            # Terminal phases: nothing is legal; MaskablePPO never queries
            # a done env, but a defensive all-False beats a stale mask.
            return np.zeros(NUM_HAND_ACTIONS, dtype=bool)
        cr = gs.get("current_round", {})
        return legal_action_mask(
            len(gs.get("hand", [])),
            cr.get("hands_left", 0),
            cr.get("discards_left", 0),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_engine_action(self, action: int) -> EnginePlayHand | EngineDiscard:
        action_type, combo = action_to_combo(action)
        if action_type == ActionType.Discard:
            return EngineDiscard(card_indices=combo)

        gs = self._adapter.raw_state
        hand = gs["hand"]
        played = [hand[i] for i in combo]
        jokers = gs.get("jokers", [])
        if len(played) > 1 and needs_permutation_search(played, jokers):
            held = [c for i, c in enumerate(hand) if i not in set(combo)]
            blind = gs.get("blind")
            ordered = best_play_order(
                played,
                held,
                jokers,
                gs["hand_levels"],
                blind,
                gs["rng"],
                game_state=gs,
                blind_chips=getattr(blind, "chips", 0) if blind else 0,
            )
            # Map back to hand indices by identity (duplicate-valued cards
            # exist in Erratic decks; value-equality would collide).
            id_to_index = {id(c): i for i, c in enumerate(hand)}
            combo = tuple(id_to_index[id(c)] for c in ordered)
        return EnginePlayHand(card_indices=combo)

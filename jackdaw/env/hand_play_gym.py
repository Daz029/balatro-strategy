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
    D_HAND_CARD,
    D_HAND_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    NUM_CENTER_KEYS,
    encode_consumable,
    encode_global_context,
    encode_hand_potential,
    encode_jokers_batch,
    encode_playing_cards_batch,
)
from jackdaw.env.trigger_match import (
    joker_center_key_ids,
    resolve_copy_targets,
    trigger_match_matrix,
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
MAX_CONSUMABLES = 2  # dormant this stage; reserved seam for the shop merge (v1)
# v2 consumable width: the v1 dormant block's 2 rows are too narrow for
# harvested full-run states -- Crystal Ball raises consumable_slots to 3,
# and negative-edition consumables (Perkeo copies) don't consume a slot at
# all, so no fixed width is provably safe. 8 is the "width is nearly free"
# call (no parameter scales with row count; masked padding contributes
# nothing): generous past any state the harvest can produce, with TAIL
# truncation as the safety valve (overflow states are Perkeo duplicates,
# exactly where truncation costs least). Rows are PER-INSTANCE in engine
# slot order, never stacked/deduplicated: at the h2 in-blind merge
# UseConsumable/SellConsumable address physical slot j, so row index must
# stay slot index -- the same "targetable items live in addressable rows"
# invariant as the shop obs. A stacked (type+count) view is a lossy
# projection the model can compute internally; the reverse is impossible,
# so shards store the engine-truthful per-instance state. Widening later
# is a loader up-pad, not a regen (masked-block widening is free).
MAX_CONSUMABLES_V2 = 8

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


def _check_joker_overfill(gs: dict[str, Any], jokers: list) -> None:
    # Negative-edition jokers don't consume a joker slot, and slot-expanding
    # vouchers raise joker_slots above 5, so a full-run state legitimately
    # holds more than MAX_JOKERS physical jokers. The hand-obs joker block
    # stays width-5 (the frozen BC demo schema; the shop merge widens it to 8
    # at the h1 seam per CLAUDE.md), so encode every joker then TRUNCATE to
    # MAX_JOKERS. Jokers are not positionally addressed by any hand action, so
    # this is a pure informativeness gap (the engine still scores every joker),
    # exactly like the >12 hand-card tail -- and it keeps h0.5's obs width
    # identical to what it was trained on (no checkpoint-compat break).
    # A GENUINE overfill -- more non-negative jokers than joker_slots -- is a
    # Riff-raff-class engine bug, not a legal build, and stays loud.
    if len(jokers) > MAX_JOKERS:
        negatives = sum(
            1 for j in jokers if getattr(j, "edition", None) and j.edition.get("negative")
        )
        joker_slots = gs.get("joker_slots", 5)
        if len(jokers) - negatives > joker_slots:
            raise ValueError(
                f"{len(jokers)} jokers ({negatives} negative) exceeds "
                f"joker_slots={joker_slots}: non-negative overfill (engine bug)"
            )


def build_observation(gs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Encode a hand-play game state into the v1 BC demo-shard schema.

    FROZEN as the v1 path: this is the exact observation h0.5's checkpoint
    trained on, and it stays the default for every h0.5 consumer
    (``HandCheckpointPolicy`` partner path, ``eval_hand_policy``,
    ``train_hand_ppo``) until those move to fresh h1 nets. The h1 schema
    lives in ``build_observation_v2``. Must stay field-for-field in sync
    with what schema-v1 shards stored, plus the dormant consumable block
    the shards don't carry (BC's loader synthesized zeros for it).
    """
    jokers = gs.get("jokers", [])
    _check_joker_overfill(gs, jokers)
    hand_arr = encode_playing_cards_batch(gs.get("hand", []), gs)
    joker_arr = encode_jokers_batch(jokers, gs)
    # Truncate (don't raise) on hand overflow -- see MAX_HAND_CARDS_OBS.
    # Encode the FULL hand first: per-card features (is_best_hand_card, ...)
    # consider the whole hand, and rows 0..7 must stay index-aligned with
    # the positions the action space addresses regardless of overflow.
    hand_padded, hand_mask = _pad(hand_arr[:MAX_HAND_CARDS_OBS], MAX_HAND_CARDS_OBS, D_PLAYING_CARD)
    joker_padded, joker_mask = _pad(joker_arr[:MAX_JOKERS], MAX_JOKERS, D_JOKER)
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


# ---------------------------------------------------------------------------
# Schema v2 (h1 bump, B2 slice 4)
# ---------------------------------------------------------------------------
#
# The v2 observation adds everything the pre-regen feature bump owns:
# hand-potential features (18-wide hand cards, 256-wide global context),
# the trigger-match matrix + joker center-key ids, Blueprint/Brainstorm
# copy-resolution fields, and a REAL consumable block. It is a versioned
# seam, not an in-place switch: v1 above stays byte-identical (h0.5's
# checkpoint obs) and the default flips only at h1 BC/PPO, whose nets are
# fresh anyway -- see docs/pre-regen-handoff.md, B2 slice 4 sequencing flag.
#
# Copy-resolution fields store the resolved target's center-key ID, not its
# 24-dim descriptor: the descriptor is a pure function of the frozen vocab
# id (the engine-derived 300x24 matrix is already a frozen model buffer),
# so storing vectors would duplicate it and create a drift surface -- the
# same id-not-vector pattern as joker_ids itself.


def encode_hand_state_v2(gs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Actual-width (unpadded) v2 entity blocks for a hand-play state.

    Shared by :func:`build_observation_v2` and the demo writer, which pad
    to different widths (obs width vs shard write width). Array values per
    key -- entity rows are in engine order throughout:

    - ``global_context``: ``(D_HAND_GLOBAL,)`` = v1 GC + 21 potential dims
    - ``hand_cards``: ``(n_hand, D_HAND_CARD)`` = v1 features + 3 potential
    - ``jokers``: ``(n_jokers, D_JOKER)`` (view into a shared buffer --
      consume/copy before the next encode call)
    - ``joker_ids``: ``(n_jokers,)`` int64 frozen-vocab center-key ids
    - ``copy_active``: ``(n_jokers,)`` float32 active-copy bits
    - ``copy_target_ids``: ``(n_jokers,)`` int64 resolved-target key ids
      (0 when inactive)
    - ``trigger_match``: ``(n_hand, n_jokers, 2)`` bool {scored, held}
    - ``consumables``: ``(n_consumables, D_CONSUMABLE)`` real owned
      consumables (labels stay consumable-blind; the block is input only)
    """
    hand = gs.get("hand", [])
    jokers = gs.get("jokers", [])

    per_card_potential, gc_ext = encode_hand_potential(gs)
    hand15 = encode_playing_cards_batch(hand, gs)
    if len(hand) > 0:
        hand_arr = np.concatenate([hand15, per_card_potential], axis=1)
    else:
        hand_arr = np.zeros((0, D_HAND_CARD), dtype=np.float32)
    gc = np.concatenate([encode_global_context(gs).astype(np.float32), gc_ext])

    resolutions = resolve_copy_targets(gs)
    consumables = gs.get("consumables", [])
    if consumables:
        cons_arr = np.stack([encode_consumable(c, gs) for c in consumables])
    else:
        cons_arr = np.zeros((0, D_CONSUMABLE), dtype=np.float32)

    return {
        "global_context": gc,
        "hand_cards": hand_arr,
        "jokers": encode_jokers_batch(jokers, gs),
        "joker_ids": joker_center_key_ids(gs),
        "copy_active": np.array([r.active for r in resolutions], dtype=np.float32),
        "copy_target_ids": np.array([r.target_key_id for r in resolutions], dtype=np.int64),
        "trigger_match": trigger_match_matrix(gs),
        "consumables": cons_arr,
    }


def _pad_1d(arr: np.ndarray, max_n: int, dtype: type) -> np.ndarray:
    out = np.zeros(max_n, dtype=dtype)
    n = min(len(arr), max_n)
    out[:n] = arr[:n]
    return out


def build_observation_v2(gs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Encode a hand-play game state into the v2 (h1) obs schema.

    Same truncation discipline as v1: hand rows beyond
    ``MAX_HAND_CARDS_OBS`` and joker rows beyond ``MAX_JOKERS`` truncate
    (with the genuine-overfill check staying loud), and consumable rows
    beyond ``MAX_CONSUMABLES_V2`` truncate the tail. The trigger-match
    matrix truncates both entity axes consistently with its row/column
    blocks.
    """
    jokers = gs.get("jokers", [])
    _check_joker_overfill(gs, jokers)
    ent = encode_hand_state_v2(gs)

    hand_padded, hand_mask = _pad(
        ent["hand_cards"][:MAX_HAND_CARDS_OBS], MAX_HAND_CARDS_OBS, D_HAND_CARD
    )
    joker_padded, joker_mask = _pad(ent["jokers"][:MAX_JOKERS], MAX_JOKERS, D_JOKER)
    cons_padded, cons_mask = _pad(
        ent["consumables"][:MAX_CONSUMABLES_V2], MAX_CONSUMABLES_V2, D_CONSUMABLE
    )
    trigger = np.zeros((MAX_HAND_CARDS_OBS, MAX_JOKERS, 2), dtype=np.float32)
    src = ent["trigger_match"][:MAX_HAND_CARDS_OBS, :MAX_JOKERS]
    trigger[: src.shape[0], : src.shape[1]] = src

    return {
        "global_context": ent["global_context"],
        "hand_cards": hand_padded,
        "hand_mask": hand_mask,
        "jokers": joker_padded,
        "joker_mask": joker_mask,
        "joker_ids": _pad_1d(ent["joker_ids"], MAX_JOKERS, np.int64),
        "copy_active": _pad_1d(ent["copy_active"], MAX_JOKERS, np.float32),
        "copy_target_ids": _pad_1d(ent["copy_target_ids"], MAX_JOKERS, np.int64),
        "trigger_match": trigger,
        "consumables": cons_padded,
        "consumable_mask": cons_mask,
    }


def observation_space_v2() -> spaces.Dict:
    def box(*shape: int) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def mask(*shape: int) -> spaces.Box:
        return spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

    def ids(n: int) -> spaces.Box:
        return spaces.Box(0, NUM_CENTER_KEYS, shape=(n,), dtype=np.int64)

    return spaces.Dict(
        {
            "global_context": box(D_HAND_GLOBAL),
            "hand_cards": box(MAX_HAND_CARDS_OBS, D_HAND_CARD),
            "hand_mask": mask(MAX_HAND_CARDS_OBS),
            "jokers": box(MAX_JOKERS, D_JOKER),
            "joker_mask": mask(MAX_JOKERS),
            "joker_ids": ids(MAX_JOKERS),
            "copy_active": mask(MAX_JOKERS),
            "copy_target_ids": ids(MAX_JOKERS),
            "trigger_match": mask(MAX_HAND_CARDS_OBS, MAX_JOKERS, 2),
            "consumables": box(MAX_CONSUMABLES_V2, D_CONSUMABLE),
            "consumable_mask": mask(MAX_CONSUMABLES_V2),
        }
    )


def hand_action_mask(gs: dict[str, Any]) -> np.ndarray:
    """Discrete(436) legality mask for a hand-play game state.

    All-False off the SELECTING_HAND phase. Module-level so any standalone
    hand policy (e.g. a checkpoint partner in the shop env) masks exactly
    like ``HandPlayGymEnv``. On a >8-card hand (The Serpent over-draw) every
    combo stays in-range (combos only touch positions 0-7), so the mask is
    always a legal play — the same graceful degradation as the truncating
    observation.
    """
    if gs.get("phase") != GamePhase.SELECTING_HAND:
        return np.zeros(NUM_HAND_ACTIONS, dtype=bool)
    cr = gs.get("current_round", {})
    return legal_action_mask(
        len(gs.get("hand", [])),
        cr.get("hands_left", 0),
        cr.get("discards_left", 0),
    )


def action_to_engine_action(action: int, gs: dict[str, Any]) -> EnginePlayHand | EngineDiscard:
    """Decode a canonical Discrete(436) index into an engine action.

    A play submits its card *subset* in engine-optimal scoring order via
    ``best_play_order`` when an order-sensitive joker/card is present (the
    agent picks a subset; ordering is a mechanical optimization delegated to
    the engine). Module-level so ``HandPlayGymEnv`` and standalone policies
    share one decode path.
    """
    action_type, combo = action_to_combo(action)
    if action_type == ActionType.Discard:
        return EngineDiscard(card_indices=combo)

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
        # Map back to hand indices by identity (duplicate-valued cards exist
        # in Erratic decks; value-equality would collide).
        id_to_index = {id(c): i for i, c in enumerate(hand)}
        combo = tuple(id_to_index[id(c)] for c in ordered)
    return EnginePlayHand(card_indices=combo)


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
    obs_version:
        Observation schema version. 1 (default) is the frozen h0.5 schema;
        2 is the h1 bump (``build_observation_v2``). The default stays 1
        until h1 BC/PPO flips it explicitly -- h0.5 consumers (the shop
        env's partner path, ``eval_hand_policy``, ``train_hand_ppo``) must
        keep seeing the exact obs the checkpoint trained on (see
        docs/pre-regen-handoff.md, B2 slice 4 sequencing flag).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: HandPlayConfig | None = None,
        seed_prefix: str = "HANDPPO",
        max_steps: int = DEFAULT_MAX_STEPS,
        obs_version: int = 1,
    ) -> None:
        super().__init__()
        if obs_version not in (1, 2):
            raise ValueError(f"unknown obs_version {obs_version} (expected 1 or 2)")
        self._config = config or HandPlayConfig()
        self._seed_prefix = seed_prefix
        self._max_steps = max_steps
        self._adapter = HandPlayAdapter(self._config)
        self._episode_counter = 0
        self._steps = 0
        self._episode_seed = ""
        self._build_obs = build_observation if obs_version == 1 else build_observation_v2

        self.observation_space = observation_space() if obs_version == 1 else observation_space_v2()
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
        return self._build_obs(gs), {
            "episode_seed": self._episode_seed,
            "action_mask": self.action_masks(),
        }

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
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

        return self._build_obs(gs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Legality mask for sb3-contrib's MaskablePPO.

        Terminal phases return all-False; MaskablePPO never queries a done
        env, but that beats a stale mask.
        """
        return hand_action_mask(self._adapter.raw_state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_engine_action(self, action: int) -> EnginePlayHand | EngineDiscard:
        return action_to_engine_action(action, self._adapter.raw_state)

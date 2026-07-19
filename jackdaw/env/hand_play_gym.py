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
    shards' actual width (up to ``MAX_HAND_CARDS_OBS`` = 40) for The
    Serpent's over-draw
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

    Terminal dollar term (h1 fine-tune stage): when a shop-agent critic's
    ``V_curve`` is supplied, a clear receives
    ``1.0 + V_curve(ante, dollars_after_cashout)``. The term is clear-gated
    because clearing dominates money in this isolated objective: losses and
    truncations pay 0.0. It is deliberately undecayed and unscaled; this is
    an objective change at the terminal reward, not shaping (CLAUDE.md
    "Money/dollar handling").

Seeding: episode seeds are strings ``f"{seed_prefix}_{n:08d}"`` fed to
``HandPlayAdapter`` (fully deterministic per seed). ``reset(seed=n)`` jumps
the episode counter to *n*; ``reset(options={"episode_seed": s})`` pins an
exact seed string (used by the fixed eval suite -- prefix ``EVAL_`` is
reserved for it and must never be used for training).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw.agents.hand_action_space import (
    NUM_HAND_ACTIONS,
    action_to_combo,
    legal_action_mask,
)
from jackdaw.agents.v_curve import VCurve
from jackdaw.engine.actions import Discard as EngineDiscard
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.actions import PlayHand as EnginePlayHand
from jackdaw.engine.play_ordering import (
    best_joker_order,
    best_play_order,
    joker_order_matters,
    needs_permutation_search,
)
from jackdaw.env.action_space import ActionType
from jackdaw.env.cashout_mirror import dollars_after_cashout
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
# 10-13 as well. Width 40 makes the observation ceiling deliberately
# irrelevant to normal builds; a finite tail still truncates lowest-first.
# Demo shards retain their actual (per-shard) hand width and train_bc.py
# zero-pads them here, which is semantically exact under masked pooling.
MAX_HAND_CARDS_OBS = 40

MAX_JOKERS = 5  # v1 (FROZEN -- h0.5's exact obs width; do NOT change)
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

# v2 joker width: the v1 cap of 5 is the game's default joker_slots, but a
# full-run state legitimately holds MORE physical jokers -- negative-edition
# jokers (Negative shop buys, the Negative tag) don't consume a slot, and
# Antimatter raises joker_slots. Truncating those away would blind the h1 model (and the
# harvest labeler) to exactly the negative/wide builds worth the most. So v2
# expands rather than truncates: encode every real joker up to a generous 15,
# well past any plausible negative stack, with lowest-slot-first truncation
# only as a pure safety valve (same "width is nearly free" call as the
# width-40 hand tail and MAX_CONSUMABLES_V2). The dual counter is unchanged:
# `_check_joker_overfill` still raises on a GENUINE overfill (non-negative
# jokers exceeding joker_slots -- a Riff-raff-class engine bug), while the
# model-view array simply holds the true physical count. Separate from v1's
# MAX_JOKERS by the same discipline as the consumable split: widening a live
# checkpoint's obs would break it, and v1 IS h0.5's frozen obs. INVARIANT:
# this must equal the demo writer's cap -- the BC loader does NOT up-pad the
# joker axis (nor trigger_match's joker axis), so generate_hand_demos.py
# imports THIS constant rather than defining its own.
MAX_JOKERS_V2 = 15

BACK_KEY = "b_red"
STAKE = 1

# Pure safety net: play/discard each consume budget, so episodes end
# naturally in <= hands+discards <= 7 steps. Truncation counts as a loss.
DEFAULT_MAX_STEPS = 32
POINTER_CARD_SLOTS = 40
POINTER_MAX_PICKS = 5
POINTER_STOP_INDEX = POINTER_CARD_SLOTS


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

    Truncation discipline: hand rows beyond ``MAX_HAND_CARDS_OBS`` and joker
    rows beyond ``MAX_JOKERS_V2`` (15, not v1's 5 -- v2 EXPANDS to keep
    negative-edition jokers) truncate, with the genuine-overfill check
    staying loud; consumable rows beyond ``MAX_CONSUMABLES_V2`` truncate the
    tail. The trigger-match matrix truncates both entity axes consistently
    with its row/column blocks.
    """
    jokers = gs.get("jokers", [])
    _check_joker_overfill(gs, jokers)
    ent = encode_hand_state_v2(gs)

    hand_padded, hand_mask = _pad(
        ent["hand_cards"][:MAX_HAND_CARDS_OBS], MAX_HAND_CARDS_OBS, D_HAND_CARD
    )
    joker_padded, joker_mask = _pad(ent["jokers"][:MAX_JOKERS_V2], MAX_JOKERS_V2, D_JOKER)
    cons_padded, cons_mask = _pad(
        ent["consumables"][:MAX_CONSUMABLES_V2], MAX_CONSUMABLES_V2, D_CONSUMABLE
    )
    trigger = np.zeros((MAX_HAND_CARDS_OBS, MAX_JOKERS_V2, 2), dtype=np.float32)
    src = ent["trigger_match"][:MAX_HAND_CARDS_OBS, :MAX_JOKERS_V2]
    trigger[: src.shape[0], : src.shape[1]] = src

    return {
        "global_context": ent["global_context"],
        "hand_cards": hand_padded,
        "hand_mask": hand_mask,
        "jokers": joker_padded,
        "joker_mask": joker_mask,
        "joker_ids": _pad_1d(ent["joker_ids"], MAX_JOKERS_V2, np.int64),
        "copy_active": _pad_1d(ent["copy_active"], MAX_JOKERS_V2, np.float32),
        "copy_target_ids": _pad_1d(ent["copy_target_ids"], MAX_JOKERS_V2, np.int64),
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
            "jokers": box(MAX_JOKERS_V2, D_JOKER),
            "joker_mask": mask(MAX_JOKERS_V2),
            "joker_ids": ids(MAX_JOKERS_V2),
            "copy_active": mask(MAX_JOKERS_V2),
            "copy_target_ids": ids(MAX_JOKERS_V2),
            "trigger_match": mask(MAX_HAND_CARDS_OBS, MAX_JOKERS_V2, 2),
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


def _selected_cards_to_engine_action(
    action_type: ActionType,
    combo: tuple[int, ...],
    gs: dict[str, Any],
    ordering_objective: Any = None,
) -> EnginePlayHand | EngineDiscard:
    """Route a validated hand subset through the canonical engine path."""

    if action_type == ActionType.Discard:
        return EngineDiscard(card_indices=combo)

    hand = gs["hand"]
    played = [hand[i] for i in combo]
    jokers = gs.get("jokers", [])
    if joker_order_matters(jokers):
        # B3 joker auto-ordering: once per COMMITTED play, applied as a
        # persistent vanilla-legal mutation of the live joker list (vanilla
        # exposes reorder as a free action; the agent never sees a reorder
        # action). Done BEFORE the card-order search below so both
        # optimizations see the same board.
        held = [c for i, c in enumerate(hand) if i not in set(combo)]
        blind = gs.get("blind")
        gs["jokers"][:] = best_joker_order(
            jokers,
            played,
            held,
            gs["hand_levels"],
            blind,
            gs["rng"],
            game_state=gs,
            blind_chips=getattr(blind, "chips", 0) if blind else 0,
            objective=ordering_objective,
        )
        jokers = gs["jokers"]
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


def action_to_engine_action(
    action: int,
    gs: dict[str, Any],
    ordering_objective: Any = None,
) -> EnginePlayHand | EngineDiscard:
    """Decode a canonical Discrete(436) index into an engine action.

    A play submits its card *subset* in engine-optimal scoring order via
    ``best_play_order`` when an order-sensitive joker/card is present (the
    agent picks a subset; ordering is a mechanical optimization delegated to
    the engine). Module-level so ``HandPlayGymEnv`` and standalone policies
    share one decode path.

    ``ordering_objective`` re-targets the JOKER-order copy-placement argmax
    (see ``best_joker_order``); None = raw score. The double-agent (shop
    env + hand partner) path passes a money-aware objective here at the h1
    seam so copy-joker placement isn't forced score-optimizing (user call
    2026-07-15); solver labels stay score-only -- loose convergence
    accepted.
    """
    action_type, combo = action_to_combo(action)
    return _selected_cards_to_engine_action(ActionType(action_type), combo, gs, ordering_objective)


class HandPlayGymEnv(gymnasium.Env):
    """Isolated hand-play episodes with versioned action encodings.

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
    action_version:
        Action encoding version. 1 (default) is the frozen Discrete(436)
        space. 2 requires ``obs_version=2`` and uses the policy-masked
        ``MultiDiscrete([2] + [41] * 5)`` pointer encoding.
    v_curve:
        Optional shop-critic dollar-value lookup used only on won terminal
        episodes.
    start_state_sampler:
        Optional ``() -> bytes | None`` called on every reset (unless a
        snapshot is pinned via ``options``). Returning a snapshot restores it
        as the episode start; returning ``None`` starts a config-sampled
        episode. The mixture policy lives in the training script, not here.
        Capture-skew repair is likewise training-side; this env receives
        already-repaired blobs and does not repair or re-randomize them.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: HandPlayConfig | None = None,
        seed_prefix: str = "HANDPPO",
        max_steps: int = DEFAULT_MAX_STEPS,
        obs_version: int = 1,
        action_version: int = 1,
        v_curve: VCurve | None = None,
        start_state_sampler: Callable[[], bytes | None] | None = None,
    ) -> None:
        super().__init__()
        if obs_version not in (1, 2):
            raise ValueError(f"unknown obs_version {obs_version} (expected 1 or 2)")
        if action_version not in (1, 2):
            raise ValueError(f"unknown action_version {action_version} (expected 1 or 2)")
        if action_version == 2 and obs_version != 2:
            raise ValueError("action_version=2 requires obs_version=2")
        self._config = config or HandPlayConfig()
        self._seed_prefix = seed_prefix
        self._max_steps = max_steps
        self._adapter = HandPlayAdapter(self._config)
        self._episode_counter = 0
        self._steps = 0
        self._episode_seed = ""
        self._sampler = start_state_sampler
        self.action_version = action_version
        self._v_curve = v_curve
        self._build_obs = build_observation if obs_version == 1 else build_observation_v2

        self.observation_space = observation_space() if obs_version == 1 else observation_space_v2()
        self.action_space = (
            spaces.MultiDiscrete([2] + [POINTER_CARD_SLOTS + 1] * POINTER_MAX_PICKS)
            if action_version == 2
            else spaces.Discrete(NUM_HAND_ACTIONS)
        )

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

        blob: bytes | None = None
        if options and "snapshot" in options:
            blob = options["snapshot"]
        elif self._sampler is not None:
            blob = self._sampler()

        if blob is not None:
            state = self._adapter.restore_state(blob)
            if state.phase != GamePhase.SELECTING_HAND:
                raise ValueError(
                    "snapshot is not a SELECTING_HAND state "
                    f"(phase={state.phase!r})"
                )
            self._episode_seed = "<restored>"
        elif options and "episode_seed" in options:
            self._episode_seed = str(options["episode_seed"])
        else:
            self._episode_seed = f"{self._seed_prefix}_{self._episode_counter:08d}"
            self._episode_counter += 1

        if blob is None:
            self._adapter.reset(BACK_KEY, STAKE, self._episode_seed)

        self._steps = 0
        gs = self._adapter.raw_state
        info: dict[str, Any] = {"episode_seed": self._episode_seed}
        if self.action_version == 1:
            info["action_mask"] = self.action_masks()
        return self._build_obs(gs), info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.action_version == 1:
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
        v_curve_term = 0.0
        dollars_after: int | None = None
        if won and self._v_curve is not None:
            dollars_after = dollars_after_cashout(gs)
            # Read the terminal ante after _round_won bookkeeping: a boss
            # clear has already advanced it to the shop-state convention used
            # when the V_curve artifact was extracted.
            ante = gs["round_resets"]["ante"]
            v_curve_term = self._v_curve.value(ante, dollars_after)
        reward = 1.0 + v_curve_term if won else 0.0

        info: dict[str, Any] = {"episode_seed": self._episode_seed}
        if self.action_version == 1:
            info["action_mask"] = self.action_masks()
        if terminated or truncated:
            info["balatro/cleared"] = won
            info["balatro/hands_left"] = gs.get("current_round", {}).get("hands_left", 0)
            info["balatro/v_curve_term"] = v_curve_term
            if dollars_after is not None:
                info["balatro/dollars_after_cashout"] = dollars_after

        return self._build_obs(gs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Legality mask for sb3-contrib's MaskablePPO.

        Terminal phases return all-False; MaskablePPO never queries a done
        env, but that beats a stale mask.
        """
        if self.action_version == 2:
            raise AttributeError("action_version=2 has no env-side action masks")
        return hand_action_mask(self._adapter.raw_state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_engine_action(self, action: int | np.ndarray) -> EnginePlayHand | EngineDiscard:
        if self.action_version == 1:
            return action_to_engine_action(int(action), self._adapter.raw_state)

        vector = np.asarray(action)
        if vector.shape != (1 + POINTER_MAX_PICKS,):
            raise ValueError(f"pointer action must have shape (6,), got {vector.shape}")
        if not np.issubdtype(vector.dtype, np.integer):
            raise ValueError("pointer action tokens must be integers")
        action_type = int(vector[0])
        if action_type not in (int(ActionType.PlayHand), int(ActionType.Discard)):
            raise ValueError(f"pointer action type {action_type} is illegal")

        padded = vector[1:].astype(np.int64, copy=False)
        stop_positions = np.flatnonzero(padded == POINTER_STOP_INDEX)
        length = int(stop_positions[0]) if len(stop_positions) else POINTER_MAX_PICKS
        if length < 1:
            raise ValueError("pointer action must select at least one card")
        if np.any(padded[length:] != POINTER_STOP_INDEX):
            raise ValueError("pointer action padding must trail the first STOP_INDEX")
        combo = tuple(int(index) for index in padded[:length])
        if any(
            index < 0 or index >= len(self._adapter.raw_state.get("hand", []))
            for index in combo
        ):
            raise ValueError("pointer action contains a dead hand index")
        if any(left >= right for left, right in zip(combo, combo[1:])):
            raise ValueError("pointer action card indices must be strictly ascending")

        gs = self._adapter.raw_state
        if gs.get("phase") != GamePhase.SELECTING_HAND:
            raise ValueError("pointer action is only legal during hand selection")
        current_round = gs.get("current_round", {})
        budget = (
            current_round.get("hands_left", 0)
            if action_type == int(ActionType.PlayHand)
            else current_round.get("discards_left", 0)
        )
        if budget < 1:
            raise ValueError(f"pointer action type {action_type} has no remaining budget")
        return _selected_cards_to_engine_action(ActionType(action_type), combo, gs)

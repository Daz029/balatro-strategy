"""Gymnasium environment for the shop agent (canonical Discrete(686) space).

Wraps :class:`ShopRunAdapter` (full-run episodes, hand phases auto-resolved
by the injected ``hand_policy``) and exposes only the shop decision surface:
SHOP, PACK_OPENING, and the env-side pending-target state.

**Pending-target state machine** (grilled decision — CLAUDE.md shop-agent
design): a carrier action (PickPackCard / UseConsumable) whose item needs
card targets does NOT step the engine. Instead the env enters a pending
state where ONLY legal SelectTarget combos are unmasked; the chosen combo
completes the engine action (targets index into ``gs["hand"]``, where the
dealt ``pack_hand`` lives during PACK_OPENING — the "targetable cards live
in the hand rows" invariant). Carriers with no target requirement (planets,
jokers, Standard-pack playing cards) resolve immediately. The pending state
is observable (``shop_context`` flag + selected bit on the carrier row),
not mask-only, and there is deliberately no cancel action.

**Reward**: the env's own reward is exactly ``1{run won}`` at termination,
gamma-agnostic and unshaped. The per-blind density term from the locked
design (``beta * c_ante * 1{blind cleared}``) is emitted as components in
``info["reward_components"]`` every step; blending (the ``beta`` schedule)
is a training-loop hyperparameter, applied by a wrapper in
``scripts/train_shop_ppo.py``, so the env never bakes in a shaping
coefficient. ``blind_clear_bonus`` is normalized so a full no-skip 8-ante
clear sums to exactly 1.

**Pack-row legality is gated env-side**: the engine's
``_handle_pick_pack_card`` performs no ``can_use`` validation, so an
unmasked pick could e.g. append a Buffoon-pack joker past ``joker_slots``
(the same unfaithful-state class as the fixed Riff-raff bug, TODO: fix in
the engine). ``action_masks`` therefore rebuilds the PickPackCard rows
itself: joker picks require a free slot (negative editions exempt),
consumable picks require ``can_use_consumable`` (untargeted) or enough
hand cards (targeted). This also lifts ``get_action_mask``'s blanket
Spectral-pack restriction — that was a balatrobot RPC limitation; this
env's two-step targeting handles Spectral cards natively.

**Snapshot/restore** (:meth:`snapshot` / restore via ``reset(options=
{"snapshot": blob})`` or the ``start_state_sampler`` hook) bundles the
adapter's engine snapshot (exact RNG round-trip) with the env-side pending
state — the substrate for the start-state reservoir mixture
{fresh_run, reservoir_shop, reservoir_pack_pending}. The sampler returning
``None`` means "fresh run" (the always-nonzero anchor fraction).

Known s0-scope notes (documented, not bugs):

* Targeted consumables can never be used from the owned-consumable rows at
  s0: in SHOP the hand is empty (vanilla-consistent — nothing to highlight),
  and the engine forbids ``UseConsumable`` during PACK_OPENING. The
  pending-``consumable`` path is implemented anyway; it goes live at the
  in-blind merge, where ``get_action_mask``'s carrier legality (it evaluates
  ``can_use_consumable`` with an empty highlight set) must be upgraded.

``select_target_mask`` constrains combo size only; per-card target
constraints live in :func:`legal_target` and are applied twice — filtering
the pending-state combo mask, and gating the carrier itself (a carrier must
never enter a pending state with zero legal targets: there is no cancel
action, so that would deadlock the episode). Currently one rule: Aura
requires an editionless target (vanilla disables it otherwise; the engine
handler would happily re-edition the card).
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw.agents.shop_action_space import (
    FAMILY_OFFSETS,
    MAX_PACK_CARDS,
    NUM_TOTAL_ACTIONS,
    ShopActionFamily,
    decode_shop_action,
    select_target_mask,
    shop_action_mask,
    target_combo_for_action,
)
from jackdaw.engine.actions import (
    Action,
    BuyCard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    RedeemVoucher,
    Reroll,
    SellCard,
    SkipPack,
    UseConsumable,
)
from jackdaw.engine.consumables import can_use_consumable
from jackdaw.env.action_space import get_action_mask, get_consumable_target_info
from jackdaw.env.shop_obs import PendingTarget, build_shop_observation, observation_space
from jackdaw.env.shop_run_adapter import DECISION_PHASES, ShopRunAdapter, ShopRunConfig

BACK_KEY = "b_red"
STAKE = 1

# Safety truncation (counted as a loss). A full 8-ante run has 24 shop
# visits; even a reroll-happy policy stays well under this.
DEFAULT_MAX_STEPS = 512

# Fresh-reset retry budget for seeds where the hand policy dies during the
# very first auto-resolved blind (rare; ante-1 Small is near-free).
_MAX_RESET_RETRIES = 16

# c_ante normalization: a full no-skip clear is 3 blinds at each of antes
# 1..8, so sum(3 * a) = 108 and the total bonus over a full clear is 1.
_C_ANTE_NORM = 108.0


def blind_clear_bonus(ante: int) -> float:
    """Per-blind reward component ``c_ante`` for a blind cleared at ``ante``.

    Linear in ante (a crude sketch of true-V increments — see the design
    record), normalized so a full 8-ante clear sums to exactly 1.
    """
    return ante / _C_ANTE_NORM


def _card_set(card: Any) -> str:
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return ability.get("set", "")
    return ""


def _is_negative(card: Any) -> bool:
    ed = getattr(card, "edition", None)
    return isinstance(ed, dict) and bool(ed.get("negative"))


def consumable_target_info(card: Any) -> tuple[int, int, bool]:
    """(min_cards, max_cards, needs_targets) for a consumable-like card.

    ``get_consumable_target_info`` reads ``max_highlighted`` from the
    center config; Aura's config is empty (its 1-highlighted requirement is
    special-cased in ``can_use_consumable``), so it is special-cased here
    too or it would look untargeted and be used as a no-op.
    """
    if getattr(card, "center_key", "") == "c_aura":
        return 1, 1, True
    return get_consumable_target_info(card)


# Carriers with per-card target constraints (see legal_target); mask
# filtering is skipped entirely for everything else.
_CONSTRAINED_TARGET_KEYS = frozenset({"c_aura"})


def legal_target(carrier_key: str, target: Any) -> bool:
    """Per-card target constraints the size-only combo mask can't express.

    Mirrors the per-card rules in ``can_use_consumable`` that depend on the
    highlighted card itself (which carrier legality evaluates with an empty
    highlight set, so they must be re-checked against concrete targets).
    """
    if carrier_key == "c_aura":
        return not getattr(target, "edition", None)
    return True


def _eligible_target_count(card: Any, gs: dict[str, Any]) -> int:
    key = getattr(card, "center_key", "")
    return sum(1 for c in gs.get("hand", []) if legal_target(key, c))


def pack_row_legal(card: Any, gs: dict[str, Any]) -> bool:
    """Whether picking this pack card is a legal, engine-faithful action.

    The engine applies pack picks without ``can_use`` validation, so this
    is the single place pack legality lives (see module docstring).
    """
    card_set = _card_set(card)
    if card_set == "Joker":
        # Same rule as the shop BuyCard mask: negative editions bypass slots.
        if _is_negative(card):
            return True
        return len(gs.get("jokers", [])) < gs.get("joker_slots", 5)
    if card_set in ("Tarot", "Planet", "Spectral"):
        min_cards, _, needs_targets = consumable_target_info(card)
        if needs_targets:
            return _eligible_target_count(card, gs) >= max(1, min_cards)
        return can_use_consumable(
            card,
            hand_cards=gs.get("hand", []),
            jokers=gs.get("jokers", []),
            consumables=gs.get("consumables", []),
            joker_limit=gs.get("joker_slots", 5),
            consumable_limit=gs.get("consumable_slots", 2),
            game_state=gs,
        )
    return True  # Standard-pack playing card: always addable to the deck


class ShopGymEnv(gymnasium.Env):
    """Full-run shop episodes on the canonical Discrete(686) action space.

    Parameters
    ----------
    hand_policy:
        ``game_state -> engine Action`` callable resolving hand phases
        (default: a fresh ``GreedyHandPolicy`` — the permanent scripted
        baseline; pass an h0/h1 checkpoint wrapper for real training).
    config:
        :class:`ShopRunConfig`; ``win_ante`` is the horizon-curriculum knob.
    seed_prefix:
        Prefix for auto-generated episode seed strings. ``EVAL_`` is
        reserved for the fixed evaluation suite and must never be trained on.
    start_state_sampler:
        Optional ``() -> bytes | None`` called on every reset (unless a
        snapshot is pinned via ``options``). Returning a blob from
        :meth:`snapshot` restores it as the episode start; returning
        ``None`` starts a fresh run. This is the reservoir hook — the
        mixture policy lives in the training script, not here.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        hand_policy: Callable[[dict[str, Any]], Action] | None = None,
        config: ShopRunConfig | None = None,
        seed_prefix: str = "SHOPPPO",
        max_steps: int = DEFAULT_MAX_STEPS,
        start_state_sampler: Callable[[], bytes | None] | None = None,
    ) -> None:
        super().__init__()
        if hand_policy is None:
            from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy

            hand_policy = GreedyHandPolicy()
        self._adapter = ShopRunAdapter(hand_policy, config)
        self._seed_prefix = seed_prefix
        self._max_steps = max_steps
        self._sampler = start_state_sampler
        self._episode_counter = 0
        self._episode_seed = ""
        self._steps = 0
        self._pending: PendingTarget | None = None
        self._last_round = 0

        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(NUM_TOTAL_ACTIONS)

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
            self._restore(blob)
            self._episode_seed = "<restored>"
        elif options and "episode_seed" in options:
            self._episode_seed = str(options["episode_seed"])
            self._adapter.reset(BACK_KEY, STAKE, self._episode_seed)
            self._pending = None
            if self._adapter.done:
                raise RuntimeError(
                    f"pinned episode seed {self._episode_seed!r} is terminal at reset "
                    "(hand policy lost the first blind)"
                )
        else:
            # Auto seeds: skip the rare seed where the hand policy dies in
            # the auto-resolved first blind (a fresh reset must not return
            # a terminal state).
            for _ in range(_MAX_RESET_RETRIES):
                self._episode_seed = f"{self._seed_prefix}_{self._episode_counter:08d}"
                self._episode_counter += 1
                self._adapter.reset(BACK_KEY, STAKE, self._episode_seed)
                self._pending = None
                if not self._adapter.done:
                    break
            else:
                raise RuntimeError(
                    f"{_MAX_RESET_RETRIES} consecutive seeds terminal at reset -- "
                    "hand policy broken?"
                )

        self._steps = 0
        gs = self._adapter.raw_state
        self._last_round = gs.get("round", 0)
        return build_shop_observation(gs, self._pending), {
            "episode_seed": self._episode_seed,
            "action_mask": self.action_masks(),
        }

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        mask = self.action_masks()
        if not mask[action]:
            # A masked policy can never select these; reaching here means a
            # wiring bug -- fail loudly, never teach "illegal no-ops".
            raise ValueError(
                f"illegal action {action} (seed={self._episode_seed}, "
                f"step={self._steps}, pending={self._pending})"
            )
        self._steps += 1

        gs = self._adapter.raw_state
        ante_before = gs.get("round_resets", {}).get("ante", 1)

        engine_action = self._resolve_action(action)
        if engine_action is not None:
            self._adapter.step(engine_action)
        # else: carrier -> pending transition; pure obs/mask change, no
        # engine step, no reward, never terminal.

        gs = self._adapter.raw_state
        cleared = gs.get("round", 0) - self._last_round
        self._last_round = gs.get("round", 0)

        terminated = self._adapter.done
        truncated = not terminated and self._steps >= self._max_steps
        won = terminated and self._adapter.won
        reward = 1.0 if won else 0.0

        info: dict[str, Any] = {
            "episode_seed": self._episode_seed,
            "action_mask": self.action_masks(),
            # Blending (the decaying beta schedule) happens in the training
            # loop; the env only reports honest components.
            "reward_components": {
                "win": reward,
                "blinds_cleared": cleared,
                "blind_bonus": cleared * blind_clear_bonus(ante_before),
            },
        }
        if terminated or truncated:
            info["balatro/won"] = won
            info["balatro/ante"] = gs.get("round_resets", {}).get("ante", 1)
            info["balatro/round"] = gs.get("round", 0)

        return build_shop_observation(gs, self._pending), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Legality mask over the canonical space for MaskablePPO."""
        gs = self._adapter.raw_state

        if self._pending is not None:
            hand = gs.get("hand", [])
            mask = select_target_mask(
                len(hand),
                self._pending.min_cards,
                self._pending.max_cards,
            )
            carrier_key = getattr(self._pending_carrier(), "center_key", "")
            if carrier_key in _CONSTRAINED_TARGET_KEYS:
                for a in np.flatnonzero(mask):
                    combo = target_combo_for_action(int(a))
                    if not all(legal_target(carrier_key, hand[i]) for i in combo):
                        mask[a] = False
            return mask

        if self._adapter.done or gs.get("phase") not in DECISION_PHASES:
            return np.zeros(NUM_TOTAL_ACTIONS, dtype=bool)

        mask = shop_action_mask(get_action_mask(gs))

        if gs.get("phase") == GamePhase.PACK_OPENING:
            # Rebuild the PickPackCard rows env-side (see module docstring:
            # engine picks are unvalidated, and the engine mask's Spectral
            # blanket-skip was a balatrobot limitation).
            offset = FAMILY_OFFSETS[ShopActionFamily.PickPackCard]
            rows = np.zeros(MAX_PACK_CARDS, dtype=bool)
            if gs.get("pack_choices_remaining", 0) > 0:
                pack_cards = gs.get("pack_cards", [])[:MAX_PACK_CARDS]
                for i, card in enumerate(pack_cards):
                    rows[i] = pack_row_legal(card, gs)
            mask[offset : offset + MAX_PACK_CARDS] = rows

        return mask

    # ------------------------------------------------------------------
    # Public state (training wrappers / diagnostics)
    # ------------------------------------------------------------------

    @property
    def raw_state(self) -> dict[str, Any]:
        """The live engine game-state dict (read-only by convention)."""
        return self._adapter.raw_state

    @property
    def pending(self) -> PendingTarget | None:
        """The pending-target state, if a carrier is awaiting SelectTarget."""
        return self._pending

    # ------------------------------------------------------------------
    # Snapshot / restore (start-state reservoir substrate)
    # ------------------------------------------------------------------

    def snapshot(self) -> bytes:
        """Serialize episode state: engine (RNG-exact) + pending-target."""
        return pickle.dumps(
            {
                "engine": self._adapter.snapshot_state(),
                "pending": self._pending,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _restore(self, blob: bytes) -> None:
        payload = pickle.loads(blob)
        self._adapter.restore_state(payload["engine"])
        self._pending = payload["pending"]
        if self._adapter.done:
            raise ValueError("snapshot is a terminal state -- purge it from the reservoir")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pending_carrier(self) -> Any:
        """The card whose pending action is awaiting a SelectTarget combo."""
        assert self._pending is not None
        gs = self._adapter.raw_state
        if self._pending.kind == "pack":
            return gs.get("pack_cards", [])[self._pending.slot]
        return gs.get("consumables", [])[self._pending.slot]

    def _resolve_action(self, action: int) -> Action | None:
        """Canonical index -> engine action, or ``None`` for a carrier
        transitioning into the pending-target state."""
        gs = self._adapter.raw_state

        if self._pending is not None:
            combo = target_combo_for_action(action)  # raises unless SelectTarget
            pending = self._pending
            self._pending = None
            if pending.kind == "pack":
                return PickPackCard(card_index=pending.slot, target_indices=combo)
            return UseConsumable(card_index=pending.slot, target_indices=combo)

        family, slot = decode_shop_action(action)

        if family is ShopActionFamily.BuyCard:
            return BuyCard(shop_index=slot)
        if family is ShopActionFamily.RedeemVoucher:
            return RedeemVoucher(card_index=slot)
        if family is ShopActionFamily.OpenBooster:
            return OpenBooster(card_index=slot)
        if family is ShopActionFamily.SellJoker:
            return SellCard(area="jokers", card_index=slot)
        if family is ShopActionFamily.SellConsumable:
            return SellCard(area="consumables", card_index=slot)
        if family is ShopActionFamily.UseConsumable:
            card = gs.get("consumables", [])[slot]
            min_cards, max_cards, needs_targets = consumable_target_info(card)
            if needs_targets and _eligible_target_count(card, gs) > 0:
                self._pending = PendingTarget("consumable", slot, min_cards, max_cards)
                return None
            return UseConsumable(card_index=slot, target_indices=None)
        if family is ShopActionFamily.Reroll:
            return Reroll()
        if family is ShopActionFamily.NextRound:
            return NextRound()
        if family is ShopActionFamily.PickPackCard:
            card = gs.get("pack_cards", [])[slot]
            min_cards, max_cards, needs_targets = consumable_target_info(card)
            if needs_targets and _eligible_target_count(card, gs) > 0:
                self._pending = PendingTarget("pack", slot, min_cards, max_cards)
                return None
            # Untargeted, or (degenerate) no dealt cards to target: the
            # handlers no-op gracefully on an empty highlight set.
            return PickPackCard(card_index=slot, target_indices=None)
        if family is ShopActionFamily.SkipPack:
            return SkipPack()
        raise ValueError(f"unhandled action family {family.name}")  # pragma: no cover

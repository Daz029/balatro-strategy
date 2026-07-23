"""Raw-RL training for the shop agent (MaskablePPO on ShopGymEnv).

Implements the training-loop side of the locked shop-agent design
(CLAUDE.md "Shop-agent design"), i.e. everything the env deliberately does
NOT bake in:

  - **Reward blending**: the env returns exactly ``1{run won}`` and reports
    the per-blind density term in ``info["reward_components"]``.
    ``ShopRewardWrapper`` blends ``r + beta * blind_bonus`` where ``beta``
    decays linearly to zero with training progress (via
    ``TrainingSchedules``, updated by ``ScheduleCallback``) — so the FINAL
    optimized objective is exactly P(win) regardless of how wrong the
    c_ante sketch is. s1 upgrade point: replace the c_ante term with
    potential-based shaping from the s0 critic.
  - **Count-based exploration bonus** (``CountBonus``): ``1/sqrt(N)`` on
    (a) the sorted owned-joker key-set, awarded when the set CHANGES (buy/
    sell/pack pick — not per step, which would just reward loitering in
    the shop), and (b) (carrier key, target-combo size) pairs, awarded when
    a pending target completes. Both scale by a coefficient that decays to
    zero on the same progress schedule.
  - **Start-state reservoir** (``ShopReservoir``): harvested engine+pending
    snapshots from ongoing rollouts, stratified by ante with a separate
    pack/pending stratum (oversampled — targeting events are rare and
    conditioned on dealt layouts; importance-shifting where experience is
    COLLECTED, not what's optimal). ``sample()`` keeps an always-nonzero
    fresh-run anchor. Because the reservoir keeps entries across policy
    updates, snapshots naturally mix current- and past-checkpoint states.
    Restart-distribution changes can't corrupt the objective (reward stays
    the honest run outcome); the only risk is coverage bias, handled by the
    anchor fraction.
  - **Horizon curriculum**: ``--win-ante`` (2 -> 4 -> 8), one stage per
    invocation; continue a stage from the previous one's checkpoint with
    ``--init-from`` (horizon stages are prefixes of the true objective, so
    nothing is unlearned at transitions).

Sharing note: schedules / counts / reservoir are plain Python objects
shared across envs — this requires ``DummyVecEnv`` (single process).
Env steps are dominated by the engine + greedy hand policy anyway;
subprocess IPC would cost more than it buys.

NOTE ON HYPERPARAMETERS: every default below is PROVISIONAL — set from
priors, not evidence. Run, inspect tensorboard + eval win rate, retune.

Usage::

    uv run python scripts/train_shop_ppo.py \
        --win-ante 2 --total-timesteps 500000 \
        --log-dir runs/shop_ppo/stage_a2 --seed 0

    uv run python scripts/train_shop_ppo.py \
        --win-ante 4 --init-from runs/shop_ppo/stage_a2/shop_ppo_final.zip \
        --log-dir runs/shop_ppo/stage_a4 --seed 0

    uv run python scripts/train_shop_ppo.py \
        --s1-schema --init-from <s0_a4_v4.zip> \
        --init-reservoir <s0 reservoir.pkl> \
        --phi-checkpoint <s0_a4_v4.zip> \
        --hand-policy <h1 pointer ckpt>

    ``--phi-checkpoint`` replaces the ``c_ante`` blend; the two density
    signals are alternatives, not additive terms.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.utils import FloatSchedule  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from jackdaw.agents.checkpoint_migration import widen_s0_checkpoint  # noqa: E402
from jackdaw.agents.phi_shaping import S0CriticPhi  # noqa: E402
from jackdaw.agents.shop_action_space import (  # noqa: E402
    NUM_TOTAL_ACTIONS,
    NUM_TOTAL_ACTIONS_S1,
    ShopActionFamily,
    decode_shop_action,
    joker_row_for_sell_action,
    target_combo_for_action,
)
from jackdaw.agents.shop_policy import ShopFeaturesExtractor  # noqa: E402
from jackdaw.engine.actions import GamePhase  # noqa: E402
from jackdaw.env.maskable_guard import install_stale_probs_guard  # noqa: E402
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunConfig  # noqa: E402

EVAL_SEED_PREFIX = "EVAL"  # reserved: never used for training rollouts


# ---------------------------------------------------------------------------
# Schedules (shared, mutable; updated by ScheduleCallback each step)
# ---------------------------------------------------------------------------


class TrainingSchedules:
    """Progress-decayed coefficients shared by all training-env wrappers.

    ``progress_remaining`` runs 1 -> 0 over training; both the blend beta
    and the count-bonus coefficient decay linearly to zero with it
    (project-standard: any soft signal decays to zero, so the final
    objective is exactly P(win)).
    """

    def __init__(
        self,
        blend_beta0: float = 1.0,
        count_beta0: float = 0.05,
        phi_beta0: float = 1.0,
    ) -> None:
        self.blend_beta0 = blend_beta0
        self.count_beta0 = count_beta0
        self.phi_beta0 = phi_beta0
        self.progress_remaining = 1.0

    @property
    def blend_beta(self) -> float:
        return self.blend_beta0 * self.progress_remaining

    @property
    def count_beta(self) -> float:
        return self.count_beta0 * self.progress_remaining

    @property
    def phi_beta(self) -> float:
        return self.phi_beta0 * self.progress_remaining


class ScheduleCallback(BaseCallback):
    """Feeds the model's progress into the shared TrainingSchedules."""

    def __init__(self, schedules: TrainingSchedules) -> None:
        super().__init__()
        self._schedules = schedules

    def _on_step(self) -> bool:
        self._schedules.progress_remaining = self.model._current_progress_remaining
        self.logger.record("shop/blend_beta", self._schedules.blend_beta)
        self.logger.record("shop/count_beta", self._schedules.count_beta)
        self.logger.record("shop/phi_beta", self._schedules.phi_beta)
        return True


class NormalizedEntropyCallback(BaseCallback):
    """Logs entropy normalized by the number of legal actions per state."""

    def __init__(self) -> None:
        super().__init__()
        self._last_normalized_entropy: float | None = None
        self._last_mean_legal_actions: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        if buf.buffer_size == 0 or not buf.full:
            return

        entropy_sum = torch.zeros((), device=self.model.device)
        valid_count = torch.zeros((), dtype=torch.long, device=self.model.device)
        legal_sum = torch.zeros((), device=self.model.device)
        total_states = 0
        # Exclude forced moves: they have no exploration signal and ln(1) is 0.
        with torch.no_grad():
            for batch in buf.get(batch_size=None):
                dist = self.model.policy.get_distribution(
                    batch.observations, action_masks=batch.action_masks
                )
                entropy = dist.entropy()
                legal = batch.action_masks.reshape(entropy.shape[0], -1).sum(dim=1).float()
                valid = legal > 1
                norm = entropy[valid] / torch.log(legal[valid])
                entropy_sum += norm.sum()
                valid_count += valid.sum()
                legal_sum += legal.sum()
                total_states += entropy.shape[0]

        if valid_count.item() > 0:
            normalized_entropy = (entropy_sum / valid_count).item()
            mean_legal_actions = (legal_sum / total_states).item()
            self._last_normalized_entropy = normalized_entropy
            self._last_mean_legal_actions = mean_legal_actions
            self.logger.record(
                "rollout/normalized_entropy", normalized_entropy
            )
            self.logger.record("rollout/mean_legal_actions", mean_legal_actions)

    def _on_training_end(self) -> None:
        # Keep the final diagnostics available after SB3's last logger dump.
        if self._last_normalized_entropy is not None:
            self.logger.record("rollout/normalized_entropy", self._last_normalized_entropy)
            self.logger.record("rollout/mean_legal_actions", self._last_mean_legal_actions)


class ReservoirCheckpointCallback(BaseCallback):
    """Pickles the shared reservoir on the same cadence as model checkpoints.

    Mirrors ``CheckpointCallback``'s ``save_freq`` (in per-env steps) so a
    killed run can be resumed from the latest model checkpoint AND its
    matching reservoir, not an empty one.
    """

    def __init__(self, reservoir: ShopReservoir, save_freq: int, save_path: Path) -> None:
        super().__init__()
        self._reservoir = reservoir
        self._save_freq = max(save_freq, 1)
        self._save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            self._reservoir.save(self._save_path)
        return True


# ---------------------------------------------------------------------------
# Count-based exploration bonus
# ---------------------------------------------------------------------------


class CountBonus:
    """1/sqrt(N) novelty bonuses over joker key-sets and target patterns."""

    def __init__(self) -> None:
        self.joker_set_counts: dict[tuple[str, ...], int] = {}
        self.target_counts: dict[tuple[str, int], int] = {}

    def joker_set(self, key_set: tuple[str, ...]) -> float:
        n = self.joker_set_counts.get(key_set, 0) + 1
        self.joker_set_counts[key_set] = n
        return 1.0 / math.sqrt(n)

    def target(self, key: tuple[str, int]) -> float:
        n = self.target_counts.get(key, 0) + 1
        self.target_counts[key] = n
        return 1.0 / math.sqrt(n)


# ---------------------------------------------------------------------------
# Start-state reservoir
# ---------------------------------------------------------------------------


class ShopReservoir:
    """Stratified snapshot reservoir with an always-nonzero fresh anchor.

    Strata: (ante, pack_pending). ``sample()`` returns ``None`` ("fresh
    run") with probability ``fresh_frac``; otherwise it picks the pack/
    pending stratum with probability ``pack_frac`` (when non-empty —
    the targeting-sparsity oversampler), else a uniform ante stratum.
    Each stratum is a bounded deque, so old-checkpoint snapshots age out
    slowly instead of being overwritten wholesale (snapshot diversity vs
    distribution collapse).
    """

    def __init__(
        self,
        fresh_frac: float = 0.5,
        pack_frac: float = 0.3,
        capacity_per_stratum: int = 256,
        seed: int = 0,
    ) -> None:
        if not 0.0 < fresh_frac <= 1.0:
            raise ValueError("fresh_frac must be in (0, 1] (the anchor is always nonzero)")
        self.fresh_frac = fresh_frac
        self.pack_frac = pack_frac
        self._capacity = capacity_per_stratum
        self._strata: dict[tuple[int, bool], deque[bytes]] = {}
        self._rng = np.random.default_rng(seed)

    def add(self, blob: bytes, ante: int, pack_pending: bool) -> None:
        key = (int(ante), bool(pack_pending))
        stratum = self._strata.get(key)
        if stratum is None:
            stratum = deque(maxlen=self._capacity)
            self._strata[key] = stratum
        stratum.append(blob)

    def __len__(self) -> int:
        return sum(len(s) for s in self._strata.values())

    def sample(self) -> bytes | None:
        if self._rng.random() < self.fresh_frac or not len(self):
            return None
        pack_keys = [k for k, s in self._strata.items() if k[1] and s]
        other_keys = [k for k, s in self._strata.items() if not k[1] and s]
        if pack_keys and (not other_keys or self._rng.random() < self.pack_frac):
            keys = pack_keys
        else:
            keys = other_keys or pack_keys
        key = keys[int(self._rng.integers(len(keys)))]
        stratum = self._strata[key]
        return stratum[int(self._rng.integers(len(stratum)))]

    # -- persistence -------------------------------------------------------
    # The reservoir is built fresh per training invocation; without this the
    # horizon-curriculum chain (a2 -> a4 -> a8, one invocation per stage) and
    # the later s0 -> s1 hop would each start EMPTY, discarding the "current
    # AND past checkpoints" snapshot diversity the design relies on. Pickle it
    # at save time and reload with --init-reservoir. RNG state round-trips too
    # so a resumed run's sampling stream continues rather than restarting.

    def save(self, path: str | Path) -> None:
        import pickle

        state = {
            "fresh_frac": self.fresh_frac,
            "pack_frac": self.pack_frac,
            "capacity": self._capacity,
            "strata": {key: list(dq) for key, dq in self._strata.items()},
            "rng_state": self._rng.bit_generator.state,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> ShopReservoir:
        import pickle

        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(
            fresh_frac=state["fresh_frac"],
            pack_frac=state["pack_frac"],
            capacity_per_stratum=state["capacity"],
        )
        obj._strata = {
            key: deque(blobs, maxlen=state["capacity"])
            for key, blobs in state["strata"].items()
        }
        obj._rng.bit_generator.state = state["rng_state"]
        return obj


# ---------------------------------------------------------------------------
# Reward-blend + harvest wrapper
# ---------------------------------------------------------------------------


class ShopRewardWrapper(gymnasium.Wrapper):
    """Blends reward components, adds count bonuses, harvests snapshots.

    The wrapped env's reward stays the honest ``1{win}``; the existing shaping
    terms are scheduled to zero by end of training. All extra terms are also
    reported in ``info["reward_components"]`` for logging/diagnosis. When
    configured, a joker bought on the immediately preceding step and then
    sold receives the configured reward; any intervening action clears that
    tracker. A configured skip-tag reward similarly fires on ``SkipBlind``.
    Either reward's decay can be disabled explicitly for experiments.
    """

    def __init__(
        self,
        env: ShopGymEnv,
        schedules: TrainingSchedules,
        counts: CountBonus,
        reservoir: ShopReservoir | None = None,
        harvest_prob: float = 0.02,
        seed: int = 0,
        phi: Callable[[dict[str, Any]], float] | None = None,
        immediate_joker_sell_reward: float | None = None,
        immediate_joker_sell_decay: bool = True,
        skip_tag_reward: float | None = None,
        skip_tag_decay: bool = True,
    ) -> None:
        super().__init__(env)
        self._schedules = schedules
        self._counts = counts
        self._reservoir = reservoir
        self._harvest_prob = harvest_prob
        self._rng = np.random.default_rng(seed)
        self._prev_joker_set: tuple[str, ...] = ()
        self._last_bought_joker: tuple[Any, ...] | None = None
        self._phi = phi
        self._phi_prev = 0.0
        self._immediate_joker_sell_reward = immediate_joker_sell_reward
        self._immediate_joker_sell_decay = immediate_joker_sell_decay
        self._skip_tag_reward = skip_tag_reward
        self._skip_tag_decay = skip_tag_decay

    # sb3-contrib discovers masking via this method; define it explicitly
    # rather than relying on Wrapper.__getattr__ forwarding.
    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def _joker_set(self) -> tuple[str, ...]:
        gs = self.env.raw_state
        return tuple(sorted(getattr(j, "center_key", "") for j in gs.get("jokers", [])))

    @staticmethod
    def _joker_signature(card: Any) -> tuple[Any, ...] | None:
        """Return the identity used to match an immediate joker sale.

        The list position is intentionally absent: buying a duplicate joker
        can append it in a different position than the one the policy sells.
        Editions are normalized so their dict insertion order is irrelevant,
        and the three joker stickers are part of the identity.
        """
        ability = getattr(card, "ability", None)
        if not isinstance(ability, dict) or ability.get("set") != "Joker":
            return None

        edition = getattr(card, "edition", None)
        edition_key = tuple(
            sorted((str(key), repr(value)) for key, value in (edition or {}).items())
        )
        stickers = tuple(
            bool(getattr(card, sticker, False) or ability.get(sticker, False))
            for sticker in ("eternal", "perishable", "rental")
        )
        return (getattr(card, "center_key", ""), edition_key, stickers)

    @staticmethod
    def _action_family_and_slot(action: int) -> tuple[ShopActionFamily, int] | None:
        # The wrapper is also used with small test environments whose action
        # spaces are not the canonical shop action space.
        try:
            return decode_shop_action(int(action))
        except (TypeError, ValueError):
            return None

    def _joker_transaction_state(
        self, action: int
    ) -> tuple[tuple[ShopActionFamily, int] | None, tuple[Any, ...] | None, tuple[Any, ...] | None]:
        """Capture the pre-step joker involved in a buy or sell action."""
        decoded = self._action_family_and_slot(action)
        if decoded is None:
            return None, None, None

        family, slot = decoded
        gs = self.env.raw_state
        bought_joker = None
        sold_joker = None
        if family is ShopActionFamily.BuyCard:
            shop_cards = gs.get("shop_cards", [])
            if slot < len(shop_cards):
                bought_joker = self._joker_signature(shop_cards[slot])
        elif family in (ShopActionFamily.SellJoker, ShopActionFamily.SellJokerExt):
            joker_row = joker_row_for_sell_action(action)
            jokers = gs.get("jokers", [])
            if joker_row < len(jokers):
                sold_joker = self._joker_signature(jokers[joker_row])
        return decoded, bought_joker, sold_joker

    def _buy_sell_is_suppressed(
        self,
        bought_joker: tuple[Any, ...] | None,
        sold_joker: tuple[Any, ...] | None,
    ) -> bool:
        """Whether the immediate buy/sell shaping reward is inapplicable."""
        gs = self.env.raw_state
        if any(getattr(joker, "center_key", "") == "j_campfire" for joker in gs.get("jokers", [])):
            return True

        transaction_jokers = (bought_joker, sold_joker)
        if any(
            signature is not None and signature[0] == "j_diet_cola"
            for signature in transaction_jokers
        ):
            return True
        if any(
            signature is not None and signature[2][2]
            for signature in transaction_jokers
        ):
            return True

        overstock_keys = {"v_overstock_norm", "v_overstock_plus"}
        return any(
            getattr(voucher, "center_key", "") in overstock_keys
            for voucher in gs.get("shop_vouchers", [])
        )

    @staticmethod
    def _cannot_be_last_bought_joker(signature: tuple[Any, ...] | None) -> bool:
        return signature is not None and (
            signature[0] == "j_diet_cola" or signature[2][2]
        )

    def _pending_carrier_key(self) -> str:
        pending = self.env.pending
        if pending is None:
            return ""
        gs = self.env.raw_state
        cards = gs.get("pack_cards" if pending.kind == "pack" else "consumables", [])
        if pending.slot < len(cards):
            return getattr(cards[pending.slot], "center_key", "")
        return ""

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._prev_joker_set = self._joker_set()
        self._last_bought_joker = None
        if self._phi is not None:
            self._phi_prev = self._phi(obs)
        return obs, info

    def step(self, action: int):
        carrier_key_before = self._pending_carrier_key()
        was_pending = self.env.pending is not None
        decoded, bought_joker, sold_joker = self._joker_transaction_state(action)
        skip_tag_decision = decoded is not None and decoded[0] is ShopActionFamily.SkipBlind
        buy_sell_suppressed = self._buy_sell_is_suppressed(bought_joker, sold_joker)
        immediate_joker_sell = (
            decoded is not None
            and decoded[0] in (ShopActionFamily.SellJoker, ShopActionFamily.SellJokerExt)
            and self._last_bought_joker is not None
            and sold_joker == self._last_bought_joker
            and not buy_sell_suppressed
        )

        obs, reward, terminated, truncated, info = self.env.step(action)
        rc = info["reward_components"]

        # This is deliberately a one-step tracker. A new joker purchase
        # overwrites the previous candidate; every other action clears it.
        self._last_bought_joker = (
            None
            if self._cannot_be_last_bought_joker(bought_joker)
            else bought_joker
        )

        bonus = self._schedules.blend_beta * rc["blind_bonus"]

        phi_term = 0.0
        if self._phi is not None:
            # Phi(terminal) == 0 covers BOTH terminated and truncated episode
            # ends: an episode boundary is an episode boundary for telescoping.
            # gamma=1, so F = Phi(s') - Phi(s).
            phi_next = 0.0 if (terminated or truncated) else self._phi(obs)
            phi_term = self._schedules.phi_beta * (phi_next - self._phi_prev)
            self._phi_prev = phi_next

        count_bonus = 0.0
        joker_set = self._joker_set()
        if joker_set != self._prev_joker_set:
            count_bonus += self._counts.joker_set(joker_set)
            self._prev_joker_set = joker_set
        if was_pending and self.env.pending is None:
            pattern = (carrier_key_before, len(target_combo_for_action(action)))
            count_bonus += self._counts.target(pattern)
        count_bonus *= self._schedules.count_beta

        rc["blend_beta"] = self._schedules.blend_beta
        rc["count_bonus"] = count_bonus
        if self._phi is not None:
            rc["phi_term"] = phi_term
            rc["phi_beta"] = self._schedules.phi_beta

        joker_sell_reward = 0.0
        if immediate_joker_sell and self._immediate_joker_sell_reward is not None:
            decay = self._schedules.progress_remaining if self._immediate_joker_sell_decay else 1.0
            joker_sell_reward = self._immediate_joker_sell_reward * decay
        rc["immediate_joker_sell_reward"] = joker_sell_reward

        skip_tag_reward = 0.0
        if skip_tag_decision and self._skip_tag_reward is not None:
            decay = self._schedules.progress_remaining if self._skip_tag_decay else 1.0
            skip_tag_reward = self._skip_tag_reward * decay
        rc["skip_tag_reward"] = skip_tag_reward

        if (
            self._reservoir is not None
            and not (terminated or truncated)
            and self._rng.random() < self._harvest_prob
        ):
            gs = self.env.raw_state
            pack_pending = self.env.pending is not None or gs.get("phase") == GamePhase.PACK_OPENING
            ante = gs.get("round_resets", {}).get("ante", 1)
            self._reservoir.add(self.env.snapshot(), ante, pack_pending)

        total_reward = reward + bonus + count_bonus
        total_reward += joker_sell_reward
        total_reward += skip_tag_reward
        if self._phi is not None:
            total_reward += phi_term
        return obs, total_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Model / env construction
# ---------------------------------------------------------------------------


def load_hand_policy(path: Path | None, *, money_aware_ordering: bool = False):
    """Build the shop episode's hand partner from a checkpoint path.

    ``None`` -> greedy baseline (env default). A ``.pt``/``.zip`` path ->
    :class:`HandCheckpointPolicy` (h0.5 and later bootstrap partners). One
    instance is returned and shared across all envs (see ``make_train_env``).
    """
    if path is None:
        return None
    from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy

    return HandCheckpointPolicy(str(path), money_aware_ordering=money_aware_ordering)


def make_train_env(
    win_ante: int,
    schedules: TrainingSchedules,
    counts: CountBonus,
    reservoir: ShopReservoir | None,
    *,
    n_envs: int = 4,
    seed_prefix: str = "SHOPPPO",
    harvest_prob: float = 0.02,
    hand_policy: Callable[[dict[str, Any]], Any] | None = None,
    s1_schema: bool = False,
    phi: Callable[[dict[str, Any]], float] | None = None,
    immediate_joker_sell_reward: float | None = None,
    immediate_joker_sell_decay: bool = True,
    skip_tag_reward: float | None = None,
    skip_tag_decay: bool = True,
) -> DummyVecEnv:
    # One partner instance shared across all envs: DummyVecEnv is single-process
    # and the hand policy is a deterministic, stateless argmax, so sharing is
    # both correct and avoids loading N copies of a torch checkpoint. None ->
    # ShopGymEnv falls back to a fresh GreedyHandPolicy per env (also cheap).
    def factory(rank: int):
        def _make() -> gymnasium.Env:
            env = ShopGymEnv(
                config=ShopRunConfig(win_ante=win_ante, s1_schema=s1_schema),
                hand_policy=hand_policy,
                seed_prefix=f"{seed_prefix}{rank}",
                start_state_sampler=reservoir.sample if reservoir is not None else None,
            )
            return ShopRewardWrapper(
                env,
                schedules,
                counts,
                reservoir,
                harvest_prob=harvest_prob,
                seed=rank,
                phi=phi,
                immediate_joker_sell_reward=immediate_joker_sell_reward,
                immediate_joker_sell_decay=immediate_joker_sell_decay,
                skip_tag_reward=skip_tag_reward,
                skip_tag_decay=skip_tag_decay,
            )

        return _make

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])


def _install_finite_grad_guard(model: MaskablePPO) -> None:
    """Keep NaN/inf out of the policy weights so every forward stays valid.

    A single non-finite weight poisons the network permanently: once one is
    present, every forward pass yields all-invalid logits and the next
    ``MaskableCategorical`` construction fails the simplex check (observed
    ~719k steps into an a4 run whose metrics were healthy right up to the
    crash — TWICE, the second time with the gradient hook already active).
    ``max_grad_norm`` clipping cannot catch it — the clip coefficient is
    ``max_norm / (grad_norm + eps)``, so a NaN ``grad_norm`` makes the
    coefficient NaN and *spreads* the poison to every parameter.

    The origin is the standard PPO tail: a large importance ratio on a
    negative-advantage transition drives the UNCLIPPED surrogate term (the one
    PPO's ``clip_range`` deliberately leaves unbounded on that side) to +inf,
    so the loss is +inf and its gradient is NaN.

    THREE layers, because a4 crashed IDENTICALLY (same step, byte-identical
    probs) with layers 1-2 already active — which proves the invalid softmax
    is generated IN THE FORWARD PASS from *finite* weights (a large-but-finite
    action logit overflowing, or a NaN from a degenerate pooled feature).
    Layers 1-2 only see non-finite gradients/weights, so they never fire on
    this failure mode and the run reproduces the crash exactly:

    1. A per-parameter backward hook maps NaN/±inf *gradients* to 0, so a bad
       step becomes a near-no-op before it can touch the optimizer.
    2. An optimizer step-post hook maps any NaN/±inf *weight* to 0 after every
       update — a backstop for leaks that route through the optimizer state.
    3. A forward hook on ``action_net`` sanitizes and clamps the ACTION LOGITS
       at their source, into a softmax-safe range. This is the layer that
       actually stops the observed crash: whatever the upstream numerics,
       ``MaskableCategorical``'s simplex check can no longer fail, because the
       logits feeding it are always finite and bounded. ±30 saturates softmax
       (exp(±30) is ~1e±13) so it never clips a *meaningful* policy preference;
       masked positions are set to a huge negative downstream, unaffected.

    Layer 3 did NOT hold under s1 — the crash returned at ~320k steps. The
    reason is in :func:`jackdaw.env.maskable_guard.install_stale_probs_guard`
    (layer 4, installed by that shared module):
    the simplex check fires on a STALE ``probs`` cache, not on the masked
    distribution layer 3 sanitizes, so no amount of logit clamping can prevent
    it.
    """
    install_stale_probs_guard()

    for param in model.policy.parameters():
        param.register_hook(
            lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        )

    def _sanitize_weights(optimizer, *_args, **_kwargs):
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if not torch.isfinite(param).all():
                        torch.nan_to_num_(param, nan=0.0, posinf=0.0, neginf=0.0)

    model.policy.optimizer.register_step_post_hook(_sanitize_weights)

    # Running count of forward passes that produced a non-finite action logit.
    # Stashed on the model so end-of-training can report the total; a one-shot
    # boolean would collapse "transient blip" and "diverging every step" into
    # the same single log line, defeating the diagnostic.
    model._finite_guard_logit_catches = 0

    def _bound_logits(_module, _inputs, output):
        safe = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            -30.0, 30.0
        )
        if not torch.isfinite(output).all():
            model._finite_guard_logit_catches += 1
            n = model._finite_guard_logit_catches
            # Print the first catch, then on a widening cadence, so a stream of
            # catches is visible in the log without flooding it.
            if n == 1 or n % 100 == 0:
                print(
                    f"[finite-guard] non-finite ACTION LOGITS caught + clamped "
                    f"(count={n}; forward-generated from finite weights, layers "
                    f"1-2 could not see this). Run continues.",
                    file=sys.stderr,
                    flush=True,
                )
        return safe

    model.policy.action_net.register_forward_hook(_bound_logits)


def soften_action_logits(model: MaskablePPO, temperature: float) -> None:
    """Flatten a warm-started policy's action logits by ``temperature``.

    A converged stage hands the next horizon a near-deterministic policy: the
    a2 -> a4 transition was measured entering a4 at ~0.05 nats of entropy
    within a few thousand steps, against 10-40 legal actions (uniform over 10
    is 2.3 nats).  PPO cannot learn from a policy that never samples anything
    but its argmax, and neither ``--ent-coef`` nor ``--learning-rate`` fixes
    it: the bonus claws back against an already-saturated softmax, and a
    smaller step size only holds the policy at the initialization more
    faithfully.  The collapse is INHERITED, so it has to be undone at load.

    Dividing the action head's weight AND bias by ``temperature > 1`` scales
    every logit uniformly, which flattens the softmax while preserving the
    complete preference ordering -- the argmax, and every ranking below it,
    are untouched.  That is the property worth having: the previous stage's
    learned ranking is the asset, its confidence is the pathology.

    Applied only at warm start, never to a fresh model (whose head is already
    near-uniform) and never mid-run.
    """
    if temperature == 1.0:
        return
    if temperature <= 0.0:
        raise ValueError(f"init temperature must be positive, got {temperature}")

    action_net = model.policy.action_net
    with torch.no_grad():
        action_net.weight.div_(temperature)
        if action_net.bias is not None:
            action_net.bias.div_(temperature)
    print(f"Softened warm-started action logits by temperature {temperature}.")


def _attach_widened_model(
    model: MaskablePPO,
    env: DummyVecEnv,
    *,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    ent_coef: float,
) -> None:
    """Attach the migration helper's one-env model to the real train env."""
    # widen_s0_checkpoint builds its temporary policy against one env. SB3's
    # set_env requires matching n_envs, and the rollout buffer must be resized
    # too when the training invocation asks for more than one env.
    model.n_envs = env.num_envs
    model.n_steps = n_steps
    model.batch_size = batch_size
    model.n_epochs = 4
    model.gamma = 1.0
    model.gae_lambda = 0.95
    model.ent_coef = ent_coef
    # learning_rate must track lr_schedule: a save/load of this model re-derives
    # the schedule from learning_rate, so a stale value would silently revert
    # the LR on resume.
    model.learning_rate = learning_rate
    model.lr_schedule = FloatSchedule(learning_rate)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate
    model.set_env(env)
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )


def build_model(
    win_ante: int,
    *,
    schedules: TrainingSchedules | None = None,
    counts: CountBonus | None = None,
    reservoir: ShopReservoir | None = None,
    init_from: Path | None = None,
    init_temperature: float = 1.0,
    seed: int = 0,
    n_envs: int = 4,
    n_steps: int = 256,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    log_dir: str | None = None,
    device: str = "auto",
    hand_policy: Callable[[dict[str, Any]], Any] | None = None,
    s1_schema: bool = False,
    phi: Callable[[dict[str, Any]], float] | None = None,
    immediate_joker_sell_reward: float | None = None,
    immediate_joker_sell_decay: bool = True,
    skip_tag_reward: float | None = None,
    skip_tag_decay: bool = True,
) -> tuple[MaskablePPO, TrainingSchedules]:
    """Construct (or resume) the shop MaskablePPO with its training env."""
    schedules = schedules or TrainingSchedules()
    counts = counts or CountBonus()
    checkpoint_width = None
    if init_from is not None:
        # Inspect before constructing an env so s0 checkpoints can take the
        # explicit widening path and incompatible widths fail clearly.
        checkpoint = MaskablePPO.load(str(init_from), device=device)
        checkpoint_width = checkpoint.action_space.n

    if s1_schema and init_from is not None and checkpoint_width == NUM_TOTAL_ACTIONS:
        model = widen_s0_checkpoint(init_from, seed=seed, device=device)
        env = make_train_env(
            win_ante,
            schedules,
            counts,
            reservoir,
            n_envs=n_envs,
            seed_prefix=f"SHOPPPO_S{seed}_R",
            hand_policy=hand_policy,
            s1_schema=True,
            phi=phi,
            immediate_joker_sell_reward=immediate_joker_sell_reward,
            immediate_joker_sell_decay=immediate_joker_sell_decay,
            skip_tag_reward=skip_tag_reward,
            skip_tag_decay=skip_tag_decay,
        )
        _attach_widened_model(
            model,
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
        )
        model.tensorboard_log = log_dir
        _install_finite_grad_guard(model)
        print(f"Widened s0 checkpoint {init_from} to the s1 schema (694 actions).")
        soften_action_logits(model, init_temperature)
        return model, schedules

    env = make_train_env(
        win_ante,
        schedules,
        counts,
        reservoir,
        n_envs=n_envs,
        seed_prefix=f"SHOPPPO_S{seed}_R",
        hand_policy=hand_policy,
        s1_schema=s1_schema,
        phi=phi,
        immediate_joker_sell_reward=immediate_joker_sell_reward,
        immediate_joker_sell_decay=immediate_joker_sell_decay,
        skip_tag_reward=skip_tag_reward,
        skip_tag_decay=skip_tag_decay,
    )

    if init_from is not None:
        expected_width = NUM_TOTAL_ACTIONS_S1 if s1_schema else NUM_TOTAL_ACTIONS
        if checkpoint_width != expected_width:
            if not s1_schema and checkpoint_width == NUM_TOTAL_ACTIONS_S1:
                raise ValueError(
                    f"checkpoint action-space width is {checkpoint_width}, but "
                    "--s1-schema is disabled; pass --s1-schema to load an s1 checkpoint"
                )
            raise ValueError(
                f"checkpoint action-space width is {checkpoint_width}, expected "
                f"{expected_width} for {'s1' if s1_schema else 's0'}"
            )
        # Horizon-curriculum continuation: same canonical action space and
        # obs schema, so the previous stage's weights load verbatim.
        model = MaskablePPO.load(str(init_from), env=env, device=device)
        model.tensorboard_log = log_dir
        _install_finite_grad_guard(model)
        soften_action_logits(model, init_temperature)
        return model, schedules

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        # PROVISIONAL hyperparameters — retune from checkpointed output.
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=1.0,  # design: undiscounted P(win); density terms decay to zero
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        seed=seed,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        policy_kwargs=dict(
            features_extractor_class=ShopFeaturesExtractor,
            **({"features_extractor_kwargs": {"s1_schema": True}} if s1_schema else {}),
            net_arch=[],  # trunk lives in the extractor; heads are single Linears
        ),
    )
    _install_finite_grad_guard(model)
    return model, schedules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--win-ante", type=int, default=2, help="horizon-curriculum stage")
    parser.add_argument("--init-from", type=Path, default=None, help="previous stage .zip")
    parser.add_argument(
        "--init-temperature",
        type=float,
        default=1.0,
        help="divide warm-started action logits by this (>1 restores exploration "
        "entropy without changing the loaded policy's preference ordering); "
        "requires --init-from",
    )
    parser.add_argument(
        "--hand-policy",
        type=Path,
        default=None,
        help="hand-partner checkpoint (.pt BC / .zip PPO); omit for the greedy baseline",
    )
    parser.add_argument(
        "--partner-money-ordering",
        action="store_true",
        help="use clear-gated money-aware copy-joker ordering with the hand partner",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", type=str, default="runs/shop_ppo/default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--blend-beta0",
        type=float,
        default=None,
        help="initial c_ante blend coefficient (defaults to 0 with Phi, else 1)",
    )
    parser.add_argument("--phi-checkpoint", type=Path, default=None, help="frozen s0 critic .zip")
    parser.add_argument("--phi-beta0", type=float, default=1.0)
    parser.add_argument("--s1-schema", action="store_true")
    parser.add_argument("--count-beta0", type=float, default=0.05)
    parser.add_argument(
        "--immediate-joker-sell-reward",
        type=float,
        default=None,
        help="enable and set the reward for buying a joker then selling a matching "
        "joker on the next action (for example -0.1); omit to disable",
    )
    parser.add_argument(
        "--immediate-joker-sell-no-decay",
        action="store_true",
        help="keep --immediate-joker-sell-reward constant instead of decaying to zero",
    )
    parser.add_argument(
        "--skip-tag-reward",
        type=float,
        default=None,
        help="enable and set the reward for taking a SkipBlind/tag decision "
        "(for example -0.1); omit to disable",
    )
    parser.add_argument(
        "--skip-tag-no-decay",
        action="store_true",
        help="keep --skip-tag-reward constant instead of decaying to zero",
    )
    parser.add_argument("--fresh-frac", type=float, default=0.5)
    parser.add_argument("--pack-frac", type=float, default=0.3)
    parser.add_argument("--harvest-prob", type=float, default=0.02)
    parser.add_argument("--reservoir-capacity", type=int, default=256, help="per stratum")
    parser.add_argument(
        "--init-reservoir",
        type=Path,
        default=None,
        help="load a prior stage's pickled reservoir (carries snapshot diversity "
        "across the a2->a4->a8 chain and the s0->s1 hop); omit to start empty",
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=20_000, help="in total env steps")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="in total env steps")
    parser.add_argument("--device", default="auto")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.partner_money_ordering and args.hand_policy is None:
        parser.error("--partner-money-ordering requires --hand-policy")
    if args.phi_checkpoint is not None and not args.s1_schema:
        parser.error("--phi-checkpoint requires --s1-schema")
    if (
        args.phi_checkpoint is not None
        and args.blend_beta0 is not None
        and args.blend_beta0 != 0.0
    ):
        parser.error("--phi-checkpoint replaces --blend-beta0; pass --blend-beta0 0")
    if args.init_temperature != 1.0:
        if args.init_from is None:
            parser.error("--init-temperature requires --init-from")
        if args.init_temperature <= 0.0:
            parser.error("--init-temperature must be positive")
    if args.immediate_joker_sell_no_decay and args.immediate_joker_sell_reward is None:
        parser.error("--immediate-joker-sell-no-decay requires --immediate-joker-sell-reward")
    if args.skip_tag_no_decay and args.skip_tag_reward is None:
        parser.error("--skip-tag-no-decay requires --skip-tag-reward")
    return args


def main() -> None:
    args = parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    phi = S0CriticPhi(args.phi_checkpoint) if args.phi_checkpoint is not None else None
    blend_beta0 = 0.0 if phi is not None else (
        1.0 if args.blend_beta0 is None else args.blend_beta0
    )
    schedules = TrainingSchedules(
        blend_beta0=blend_beta0,
        count_beta0=args.count_beta0,
        phi_beta0=args.phi_beta0,
    )
    if args.init_reservoir is not None:
        reservoir = ShopReservoir.load(args.init_reservoir)
        print(f"Loaded reservoir from {args.init_reservoir} (size {len(reservoir)})")
    else:
        reservoir = ShopReservoir(
            fresh_frac=args.fresh_frac,
            pack_frac=args.pack_frac,
            capacity_per_stratum=args.reservoir_capacity,
            seed=args.seed,
        )
    # Shared partner instance — same one drives training and eval so the eval
    # win rate is measured against the real hand policy, not the greedy baseline.
    hand_policy = load_hand_policy(
        args.hand_policy, money_aware_ordering=args.partner_money_ordering
    )
    model, schedules = build_model(
        args.win_ante,
        schedules=schedules,
        reservoir=reservoir,
        init_from=args.init_from,
        init_temperature=args.init_temperature,
        seed=args.seed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        log_dir=str(log_path),
        device=args.device,
        hand_policy=hand_policy,
        s1_schema=args.s1_schema,
        phi=phi,
        immediate_joker_sell_reward=args.immediate_joker_sell_reward,
        immediate_joker_sell_decay=not args.immediate_joker_sell_no_decay,
        skip_tag_reward=args.skip_tag_reward,
        skip_tag_decay=not args.skip_tag_no_decay,
    )

    # Eval on the reserved EVAL_* stream: plain env, no wrapper — mean
    # episode reward IS the win rate at this horizon.
    eval_env = DummyVecEnv(
        [
            lambda: ShopGymEnv(
                config=ShopRunConfig(win_ante=args.win_ante, s1_schema=args.s1_schema),
                hand_policy=hand_policy,
                seed_prefix=EVAL_SEED_PREFIX,
            )
        ]
    )
    callbacks = [
        ScheduleCallback(schedules),
        NormalizedEntropyCallback(),
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            deterministic=True,
            best_model_save_path=str(log_path / "best_model"),
            log_path=str(log_path / "eval"),
        ),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(log_path / "checkpoints"),
            name_prefix="shop_ppo",
        ),
        ReservoirCheckpointCallback(
            reservoir,
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=log_path / "reservoir.pkl",
        ),
    ]

    partner_desc = str(args.hand_policy) if args.hand_policy is not None else "greedy (baseline)"
    if args.partner_money_ordering:
        partner_desc += " + money-aware ordering"
    print(
        f"Training shop agent: win_ante={args.win_ante}, "
        f"{args.total_timesteps} timesteps (seed={args.seed}), partner={partner_desc}..."
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    save_path = log_path / "shop_ppo_final"
    model.save(str(save_path))
    reservoir_path = log_path / "reservoir.pkl"
    reservoir.save(reservoir_path)
    print(
        f"Model saved to {save_path}; reservoir ({len(reservoir)} snapshots) "
        f"saved to {reservoir_path}"
    )
    catches = getattr(model, "_finite_guard_logit_catches", 0)
    print(
        f"[finite-guard] total non-finite action-logit catches this run: {catches}"
    )


if __name__ == "__main__":
    main()

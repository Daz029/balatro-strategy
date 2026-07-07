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
from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from jackdaw.agents.shop_action_space import target_combo_for_action  # noqa: E402
from jackdaw.agents.shop_policy import ShopFeaturesExtractor  # noqa: E402
from jackdaw.engine.actions import GamePhase  # noqa: E402
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

    def __init__(self, blend_beta0: float = 1.0, count_beta0: float = 0.05) -> None:
        self.blend_beta0 = blend_beta0
        self.count_beta0 = count_beta0
        self.progress_remaining = 1.0

    @property
    def blend_beta(self) -> float:
        return self.blend_beta0 * self.progress_remaining

    @property
    def count_beta(self) -> float:
        return self.count_beta0 * self.progress_remaining


class ScheduleCallback(BaseCallback):
    """Feeds the model's progress into the shared TrainingSchedules."""

    def __init__(self, schedules: TrainingSchedules) -> None:
        super().__init__()
        self._schedules = schedules

    def _on_step(self) -> bool:
        self._schedules.progress_remaining = self.model._current_progress_remaining
        self.logger.record("shop/blend_beta", self._schedules.blend_beta)
        self.logger.record("shop/count_beta", self._schedules.count_beta)
        return True


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
    def load(cls, path: str | Path) -> "ShopReservoir":
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

    The wrapped env's reward stays the honest ``1{win}``; everything added
    here is scheduled to zero by end of training. All extra terms are also
    reported in ``info["reward_components"]`` for logging/diagnosis.
    """

    def __init__(
        self,
        env: ShopGymEnv,
        schedules: TrainingSchedules,
        counts: CountBonus,
        reservoir: ShopReservoir | None = None,
        harvest_prob: float = 0.02,
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        self._schedules = schedules
        self._counts = counts
        self._reservoir = reservoir
        self._harvest_prob = harvest_prob
        self._rng = np.random.default_rng(seed)
        self._prev_joker_set: tuple[str, ...] = ()

    # sb3-contrib discovers masking via this method; define it explicitly
    # rather than relying on Wrapper.__getattr__ forwarding.
    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def _joker_set(self) -> tuple[str, ...]:
        gs = self.env.raw_state
        return tuple(sorted(getattr(j, "center_key", "") for j in gs.get("jokers", [])))

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
        return obs, info

    def step(self, action: int):
        carrier_key_before = self._pending_carrier_key()
        was_pending = self.env.pending is not None

        obs, reward, terminated, truncated, info = self.env.step(action)
        rc = info["reward_components"]

        bonus = self._schedules.blend_beta * rc["blind_bonus"]

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

        if (
            self._reservoir is not None
            and not (terminated or truncated)
            and self._rng.random() < self._harvest_prob
        ):
            gs = self.env.raw_state
            pack_pending = self.env.pending is not None or gs.get("phase") == GamePhase.PACK_OPENING
            ante = gs.get("round_resets", {}).get("ante", 1)
            self._reservoir.add(self.env.snapshot(), ante, pack_pending)

        return obs, reward + bonus + count_bonus, terminated, truncated, info


# ---------------------------------------------------------------------------
# Model / env construction
# ---------------------------------------------------------------------------


def load_hand_policy(path: Path | None):
    """Build the shop episode's hand partner from a checkpoint path.

    ``None`` -> greedy baseline (env default). A ``.pt``/``.zip`` path ->
    :class:`HandCheckpointPolicy` (h0.5 and later bootstrap partners). One
    instance is returned and shared across all envs (see ``make_train_env``).
    """
    if path is None:
        return None
    from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy

    return HandCheckpointPolicy(str(path))


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
) -> DummyVecEnv:
    # One partner instance shared across all envs: DummyVecEnv is single-process
    # and the hand policy is a deterministic, stateless argmax, so sharing is
    # both correct and avoids loading N copies of a torch checkpoint. None ->
    # ShopGymEnv falls back to a fresh GreedyHandPolicy per env (also cheap).
    def factory(rank: int):
        def _make() -> gymnasium.Env:
            env = ShopGymEnv(
                config=ShopRunConfig(win_ante=win_ante),
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
            )

        return _make

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])


def build_model(
    win_ante: int,
    *,
    schedules: TrainingSchedules | None = None,
    counts: CountBonus | None = None,
    reservoir: ShopReservoir | None = None,
    init_from: Path | None = None,
    seed: int = 0,
    n_envs: int = 4,
    n_steps: int = 256,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    log_dir: str | None = None,
    device: str = "auto",
    hand_policy: Callable[[dict[str, Any]], Any] | None = None,
) -> tuple[MaskablePPO, TrainingSchedules]:
    """Construct (or resume) the shop MaskablePPO with its training env."""
    schedules = schedules or TrainingSchedules()
    counts = counts or CountBonus()
    env = make_train_env(
        win_ante,
        schedules,
        counts,
        reservoir,
        n_envs=n_envs,
        seed_prefix=f"SHOPPPO_S{seed}_R",
        hand_policy=hand_policy,
    )

    if init_from is not None:
        # Horizon-curriculum continuation: same canonical action space and
        # obs schema, so the previous stage's weights load verbatim.
        model = MaskablePPO.load(str(init_from), env=env, device=device)
        model.tensorboard_log = log_dir
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
            net_arch=[],  # trunk lives in the extractor; heads are single Linears
        ),
    )
    return model, schedules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--win-ante", type=int, default=2, help="horizon-curriculum stage")
    parser.add_argument("--init-from", type=Path, default=None, help="previous stage .zip")
    parser.add_argument(
        "--hand-policy",
        type=Path,
        default=None,
        help="hand-partner checkpoint (.pt BC / .zip PPO); omit for the greedy baseline",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", type=str, default="runs/shop_ppo/default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--blend-beta0", type=float, default=1.0)
    parser.add_argument("--count-beta0", type=float, default=0.05)
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
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    schedules = TrainingSchedules(blend_beta0=args.blend_beta0, count_beta0=args.count_beta0)
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
    hand_policy = load_hand_policy(args.hand_policy)
    model, schedules = build_model(
        args.win_ante,
        schedules=schedules,
        reservoir=reservoir,
        init_from=args.init_from,
        seed=args.seed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        log_dir=str(log_path),
        device=args.device,
        hand_policy=hand_policy,
    )

    # Eval on the reserved EVAL_* stream: plain env, no wrapper — mean
    # episode reward IS the win rate at this horizon.
    eval_env = DummyVecEnv(
        [
            lambda: ShopGymEnv(
                config=ShopRunConfig(win_ante=args.win_ante),
                hand_policy=hand_policy,
                seed_prefix=EVAL_SEED_PREFIX,
            )
        ]
    )
    callbacks = [
        ScheduleCallback(schedules),
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


if __name__ == "__main__":
    main()

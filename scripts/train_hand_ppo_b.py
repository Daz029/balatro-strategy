"""Pointer-action PPO fine-tuning with a teacher-forced per-step KL leash.

This script uses ordinary SB3 ``PPO`` rather than ``MaskablePPO`` because v2
has no env-side action masks: the pointer policy constructs type and prefix
masks from ``hand_mask`` and the GC budgets through the shared head functions.
The KL leash teacher-forces the taken sequence in both policies, computes
categorical ``KL(pi_theta_step || pi_BC_step)`` on legal tokens, and sums the
active type/pick/stop steps before averaging over the minibatch.  PPO's ratio
is likewise the ratio of sequence probabilities, implemented as
``exp(new_sequence_log_prob - rollout_sequence_log_prob)``; the entropy bonus
uses the corresponding active-step entropy sum.

``train()`` follows stable-baselines3 2.7.1's PPO train loop. If that pin
changes, re-diff this method before changing the version assertion in its
tests.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _path in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch as th  # noqa: E402
from generate_hand_demos import stage_presets  # noqa: E402
from harvest_snapshot_sampler import HarvestSnapshotSampler  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.utils import explained_variance  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from jackdaw.agents.hand_pointer_head import HandPointerBCModel  # noqa: E402
from jackdaw.agents.pointer_ppo_policy import (  # noqa: E402
    PointerPPOPolicy,
    load_bc_model,
    load_bc_weights,
)
from jackdaw.agents.v_curve import VCurve, load_v_curve  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import HandPlayGymEnv  # noqa: E402

EVAL_SEED_PREFIX = "EVAL"
SB3_TRAIN_VERSION = "2.7"
DEFAULT_FINETUNE_STAGES = (
    "stage1_no_jokers",
    "stage2_curated",
    "stage3_full",
    "stage4_boss",
)


class KLToBCPointerPPO(PPO):
    """SB3 PPO with an adaptive, decaying teacher-forced pointer KL leash."""

    def __init__(
        self,
        *args,
        bc_model: HandPointerBCModel | None = None,
        kl_beta0: float = 0.5,
        kl_target: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kl_beta0 = kl_beta0
        self.kl_target = kl_target
        self._kl_multiplier = 1.0
        self.bc_model: HandPointerBCModel | None = None
        if bc_model is not None:
            self.set_bc_model(bc_model)

    def set_bc_model(self, bc_model: HandPointerBCModel) -> None:
        bc_model = bc_model.to(self.device)
        bc_model.eval()
        bc_model.requires_grad_(False)
        self.bc_model = bc_model

    def _kl_to_bc(self, observations, actions: th.Tensor) -> th.Tensor:
        """Mean active-step reverse KL for the taken action prefixes."""

        assert self.bc_model is not None
        current_type, current_pointer, active = self.policy.teacher_forced_step_distributions(
            observations, actions
        )
        with th.no_grad():
            bc_cards, bc_pooled = self.bc_model.features_extractor(observations)
            hands_left, discards_left = self.bc_model._budgets(observations)
            bc_type, bc_pointer, bc_active = (
                self.bc_model.pointer_head.teacher_forced_step_distributions(
                bc_cards,
                bc_pooled,
                observations["hand_mask"],
                hands_left,
                discards_left,
                *self.policy._labels_from_actions(actions),
                )
            )
        if not th.equal(active, bc_active):
            raise RuntimeError("pointer policy and BC active-step masks diverged")

        def categorical_kl(current: th.Tensor, reference: th.Tensor) -> th.Tensor:
            legal = th.isfinite(current) & th.isfinite(reference)
            diff = th.where(legal, current - reference, th.zeros_like(current))
            probabilities = th.where(legal, current.exp(), th.zeros_like(current))
            return (probabilities * diff).sum(dim=-1)

        type_kl = categorical_kl(current_type, bc_type)
        pointer_kl = categorical_kl(current_pointer, bc_pointer)
        per_step = th.cat((type_kl.unsqueeze(-1), pointer_kl), dim=-1)
        return (per_step * active).sum(dim=-1).mean().clamp_min(0.0)

    @property
    def _kl_beta_eff(self) -> float:
        return self.kl_beta0 * self._current_progress_remaining * self._kl_multiplier

    def train(self) -> None:
        """stable-baselines3 2.7.1 PPO.train() plus the KL-to-BC leash."""

        assert self.bc_model is not None, "call set_bc_model() before learn()"
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_bc_divs = []  # --- KL-to-BC
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                # --- KL-to-BC leash -------------------------------------
                kl_bc = self._kl_to_bc(rollout_data.observations, actions)
                kl_bc_divs.append(kl_bc.item())
                # ---------------------------------------------------------
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self._kl_beta_eff * kl_bc  # --- KL-to-BC
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        mean_kl_bc = float(np.mean(kl_bc_divs)) if kl_bc_divs else 0.0
        if mean_kl_bc > 2.0 * self.kl_target:
            self._kl_multiplier = min(self._kl_multiplier * 1.5, 10.0)
        elif mean_kl_bc < 0.5 * self.kl_target:
            self._kl_multiplier = max(self._kl_multiplier / 1.5, 0.1)

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/kl_bc", mean_kl_bc)  # --- KL-to-BC
        self.logger.record("train/kl_beta_eff", self._kl_beta_eff)  # --- KL-to-BC
        self.logger.record("train/kl_multiplier", self._kl_multiplier)  # --- KL-to-BC
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def resolve_stage_configs(stages: list[str]) -> list[HandPlayConfig]:
    presets = stage_presets()
    return [presets[stage].config for stage in stages]


def make_vec_env(
    configs: HandPlayConfig | list[HandPlayConfig],
    seed_prefix: str,
    n_envs: int,
    *,
    v_curve: VCurve | None = None,
    start_state_sampler: Callable[[], bytes | None] | None = None,
) -> DummyVecEnv:
    if isinstance(configs, HandPlayConfig):
        configs = [configs]

    def factory(rank: int):
        config = configs[rank % len(configs)]
        return lambda: HandPlayGymEnv(
            config=config,
            seed_prefix=f"{seed_prefix}{rank}",
            obs_version=2,
            action_version=2,
            v_curve=v_curve,
            start_state_sampler=start_state_sampler,
        )

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])


def build_model(
    bc_checkpoint: Path,
    config: HandPlayConfig | list[HandPlayConfig],
    *,
    seed: int = 0,
    n_envs: int = 8,
    n_steps: int = 512,
    batch_size: int = 256,
    log_dir: str | None = None,
    kl_beta0: float = 0.5,
    kl_target: float = 0.03,
    learning_rate: float = 3e-5,
    ent_coef: float = 0.01,
    device: str = "auto",
    v_curve: VCurve | None = None,
    start_state_sampler: Callable[[], bytes | None] | None = None,
) -> KLToBCPointerPPO:
    # DummyVecEnv is single-process and resets workers sequentially, so one
    # seeded sampler stream is shared intentionally across every training env.
    env = make_vec_env(
        config,
        seed_prefix=f"HANDPPO_B_S{seed}_R",
        n_envs=n_envs,
        v_curve=v_curve,
        start_state_sampler=start_state_sampler,
    )
    bc_model = load_bc_model(bc_checkpoint, device="cpu")
    model = KLToBCPointerPPO(
        PointerPPOPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        kl_beta0=kl_beta0,
        kl_target=kl_target,
        seed=seed,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )
    load_bc_weights(model.policy, bc_model)
    model.set_bc_model(bc_model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(DEFAULT_FINETUNE_STAGES))
    parser.add_argument("--stage", default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--log-dir", type=str, default="runs/hand_ppo_b/default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--kl-beta0", type=float, default=0.5)
    parser.add_argument("--kl-target", type=float, default=0.03)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--v-curve", type=Path, default=None)
    parser.add_argument("--harvest-dir", type=Path, default=None)
    parser.add_argument("--config-anchor-frac", type=float, default=0.5)
    args = parser.parse_args()

    stages = [args.stage] if args.stage is not None else [s for s in args.stages.split(",") if s]
    configs = resolve_stage_configs(stages)
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    v_curve = load_v_curve(args.v_curve) if args.v_curve is not None else None
    sampler = None
    if args.harvest_dir is not None and args.config_anchor_frac != 1.0:
        sampler = HarvestSnapshotSampler(
            args.harvest_dir,
            config_anchor_frac=args.config_anchor_frac,
            seed=args.seed,
        )
    model = build_model(
        args.bc_checkpoint,
        configs,
        seed=args.seed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        log_dir=str(log_path),
        kl_beta0=args.kl_beta0,
        kl_target=args.kl_target,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        device=args.device,
        v_curve=v_curve,
        start_state_sampler=sampler,
    )
    # Keep EVAL clean: the fixed EVAL_ suite's clear-rate yardstick must stay
    # comparable across h0.5/h1 and across the ablation checkpoints.
    eval_env = make_vec_env(configs, seed_prefix=EVAL_SEED_PREFIX, n_envs=len(configs))
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
        best_model_save_path=str(log_path / "best_model"),
        log_path=str(log_path / "eval"),
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(log_path / "checkpoints"),
        name_prefix="hand_ppo_b",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=[eval_callback, checkpoint_callback])
    save_path = log_path / "hand_ppo_b_final"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

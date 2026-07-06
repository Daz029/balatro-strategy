"""PPO fine-tuning for the hand-play agent, starting from a BC checkpoint.

Implements the BC-to-RL design from CLAUDE.md's ante-play track:

  - **Warm start**: the whole policy (trunk + action head + value head)
    loads from the BC checkpoint. The value head was regressed on the
    solver's ``p_clear``, which with terminal 1/0 reward and ``gamma=1.0``
    is exactly the PPO critic's target -- a calibrated critic from step 0
    (uncalibrated critic -> noisy advantages -> can reinforce BC behavior
    for the wrong reasons).
  - **Adaptive KL leash to the frozen BC policy** (AlphaStar-style), via a
    ``MaskablePPO`` subclass whose ``train()`` adds
    ``beta_eff * KL(pi_theta || pi_BC)`` to the loss. Reverse KL: punishes
    probability mass outside BC's support but allows free sharpening
    within it. ``beta_eff = beta0 * progress_remaining * m`` where ``m``
    adapts multiplicatively toward a KL target; the ``progress_remaining``
    factor guarantees the leash reaches zero by end of training no matter
    what the adaptation does.
  - **Exploration-collapse watchdogs**: train/kl_bc, train/kl_beta_eff and
    entropy are all logged; run multiple ``--seed`` values and compare
    (CLAUDE.md mitigation #4 -- local minima here are seed-sensitive).

NOTE ON HYPERPARAMETERS: every default below (lr, n_steps, ent_coef,
beta0, kl_target, clip_range, ...) is PROVISIONAL -- set from priors, not
evidence. The intended workflow is: run, inspect the checkpointed output
(tensorboard curves + eval clear-rate vs the solver ceiling from
scripts/eval_hand_policy.py), then retune from that evidence.

``train()`` is copied from sb3-contrib 2.7.1 (sb3_contrib/ppo_mask/
ppo_mask.py) with the KL-leash insertions marked by ``# --- KL-to-BC``
comments; bumping sb3-contrib requires re-diffing that method (a unit test
asserts the KL path stays live, so silent drift fails loudly).

Usage::

    uv run python scripts/train_hand_ppo.py \
        --bc-checkpoint runs/bc/run1/bc_checkpoint.pt \
        --stage stage2_curated --total-timesteps 2000000 \
        --log-dir runs/hand_ppo/run1 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch as th  # noqa: E402
from generate_hand_demos import stage_presets  # noqa: E402
from gymnasium import spaces  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.utils import explained_variance  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from jackdaw.agents.hand_policy import (  # noqa: E402
    HandPlayBCModel,
    HandPlayFeaturesExtractor,
    load_bc_weights_into_policy,
)
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import HandPlayGymEnv, observation_space  # noqa: E402

EVAL_SEED_PREFIX = "EVAL"  # reserved: never used for training rollouts


class KLToBCMaskablePPO(MaskablePPO):
    """MaskablePPO plus an adaptive, decaying KL leash to a frozen BC policy.

    Extra constructor args:
        bc_model: frozen ``HandPlayBCModel`` (reference distribution).
        kl_beta0: base leash coefficient (PROVISIONAL default 0.5).
        kl_target: target KL(pi||pi_BC) per update for the adaptation
            (PROVISIONAL default 0.03).
    """

    def __init__(
        self,
        *args,
        bc_model: HandPlayBCModel | None = None,
        kl_beta0: float = 0.5,
        kl_target: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kl_beta0 = kl_beta0
        self.kl_target = kl_target
        self._kl_multiplier = 1.0
        self.bc_model: HandPlayBCModel | None = None
        if bc_model is not None:
            self.set_bc_model(bc_model)

    def set_bc_model(self, bc_model: HandPlayBCModel) -> None:
        bc_model = bc_model.to(self.device)
        bc_model.eval()
        bc_model.requires_grad_(False)
        self.bc_model = bc_model

    def _kl_to_bc(self, observations, action_masks: th.Tensor) -> th.Tensor:
        """Mean reverse KL(pi_theta || pi_BC) over a minibatch.

        Both distributions are masked with the SAME rollout masks; illegal
        entries are excluded via `where` (both sides are -inf there, and
        -inf - -inf = nan would otherwise poison the sum). Costs one extra
        pi-path forward per minibatch versus reusing evaluate_actions'
        internals -- accepted for not depending on policy privates.
        """
        assert self.bc_model is not None
        masks_bool = action_masks.bool()
        cur_dist = self.policy.get_distribution(observations, action_masks=action_masks)
        cur_log_probs = cur_dist.distribution.logits  # normalized masked log-probs
        with th.no_grad():
            bc_log_probs = self.bc_model.masked_log_probs(observations, masks_bool)
        diff = th.where(masks_bool, cur_log_probs - bc_log_probs, th.zeros_like(cur_log_probs))
        cur_probs = th.where(masks_bool, cur_log_probs.exp(), th.zeros_like(cur_log_probs))
        return (cur_probs * diff).sum(dim=-1).mean()

    @property
    def _kl_beta_eff(self) -> float:
        return self.kl_beta0 * self._current_progress_remaining * self._kl_multiplier

    def train(self) -> None:
        """sb3-contrib 2.7.1 MaskablePPO.train() + the KL-to-BC leash."""
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
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- KL-to-BC leash -------------------------------------
                kl_bc = self._kl_to_bc(rollout_data.observations, rollout_data.action_masks)
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

            if not continue_training:
                break

        # --- KL-to-BC: adapt the multiplier toward the KL target ----------
        mean_kl_bc = float(np.mean(kl_bc_divs)) if kl_bc_divs else 0.0
        if mean_kl_bc > 2.0 * self.kl_target:
            self._kl_multiplier = min(self._kl_multiplier * 1.5, 10.0)
        elif mean_kl_bc < 0.5 * self.kl_target:
            self._kl_multiplier = max(self._kl_multiplier / 1.5, 0.1)
        # -------------------------------------------------------------------

        self._n_updates += self.n_epochs
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


def load_bc_model(checkpoint_path: Path) -> HandPlayBCModel:
    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HandPlayBCModel(observation_space())
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# Default fine-tune distribution: the SAME four stages BC pooled. Boss
# blinds live ONLY in stage4 (stages 1-3 are Small/Big), so a single-stage
# fine-tune would leave the KL leash decaying over a boss-free distribution
# and the policy's boss play free to drift -- a real regression for a
# full-run partner that faces a Boss every ante. Round-robin across envs
# weights the stages equally (not by BC example count), which deliberately
# over-emphasizes the weak spots (ante-1 no-joker, bosses) we're fixing.
DEFAULT_FINETUNE_STAGES = (
    "stage1_no_jokers",
    "stage2_curated",
    "stage3_full",
    "stage4_boss",
)


def resolve_stage_config(stage: str | None) -> HandPlayConfig:
    """Single stage preset's config (or the no-joker default for ``None``)."""
    if stage is None:
        return HandPlayConfig()
    return stage_presets()[stage].config


def resolve_stage_configs(stages: list[str]) -> list[HandPlayConfig]:
    """Configs for a stage mixture; env distribution should match the BC
    dataset's generating distribution (presets from generate_hand_demos.py)."""
    presets = stage_presets()
    return [presets[s].config for s in stages]


def make_vec_env(
    configs: HandPlayConfig | list[HandPlayConfig], seed_prefix: str, n_envs: int
) -> DummyVecEnv:
    """Vectorized env over a single config or a round-robin stage mixture.

    A list assigns ``configs[rank % len]`` per env, so the rollout (and eval)
    distribution is an even mixture over the stages. Env steps are ~1 ms;
    subprocess IPC overhead isn't worth it, hence DummyVecEnv.
    """
    if isinstance(configs, HandPlayConfig):
        configs = [configs]

    def factory(rank: int):
        config = configs[rank % len(configs)]
        return lambda: HandPlayGymEnv(config=config, seed_prefix=f"{seed_prefix}{rank}")

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
) -> KLToBCMaskablePPO:
    """Construct the KL-leashed model with BC weights loaded into the policy."""
    env = make_vec_env(config, seed_prefix=f"HANDPPO_S{seed}_R", n_envs=n_envs)
    bc_model = load_bc_model(bc_checkpoint)

    model = KLToBCMaskablePPO(
        "MultiInputPolicy",
        env,
        # PROVISIONAL hyperparameters -- retune from checkpointed output.
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=1.0,  # episodes <= 7 real steps; V(s) == P(clear), matches BC value head
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
        policy_kwargs=dict(
            features_extractor_class=HandPlayFeaturesExtractor,
            net_arch=[],  # trunk lives in the extractor; heads are single Linears
        ),
    )
    load_bc_weights_into_policy(model.policy, bc_model)
    model.set_bc_model(bc_model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_FINETUNE_STAGES),
        help="Comma-separated stage presets to mix (round-robin across envs). "
        "Default = the four stages BC pooled (includes stage4_boss so boss "
        "play is fine-tuned, not left to drift).",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Single stage preset; overrides --stages when given (e.g. to "
        "fine-tune one stage in isolation).",
    )
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--log-dir", type=str, default="runs/hand_ppo/default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--kl-beta0", type=float, default=0.5)
    parser.add_argument("--kl-target", type=float, default=0.03)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=20_000, help="in total env steps")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="in total env steps")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if args.stage is not None:
        stages = [args.stage]
    else:
        stages = [s for s in args.stages.split(",") if s]
    configs = resolve_stage_configs(stages)
    print(f"Fine-tuning against stage mixture: {stages}")

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
    )

    # Eval on the reserved EVAL_* seed stream; mean episode reward IS the
    # clear rate (terminal 1/0). One env per stage so the aggregate best-
    # model metric weights the stages evenly (matches the training mixture).
    eval_env = make_vec_env(configs, seed_prefix=EVAL_SEED_PREFIX, n_envs=len(configs))
    eval_callback = MaskableEvalCallback(
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
        name_prefix="hand_ppo",
    )

    print(f"Fine-tuning for {args.total_timesteps} timesteps (seed={args.seed})...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    save_path = log_path / "hand_ppo_final"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

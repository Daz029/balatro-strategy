"""Weight-preserving migration of an s0 shop checkpoint to the s1 schema.

s0's action head is ``Discrete(686)`` over the pre-s1 obs schema (joker
rows 8, ``shop_context`` width 12). s1 widens the action space to
``Discrete(694)`` (SkipBlind + SellJoker slots 8-14) and the obs schema
(joker rows 15, offered-tag one-hot appended to ``shop_context`` --
see ``shop_action_space.py`` / ``shop_obs.py`` / ``shop_policy.py``).
:func:`widen_s0_checkpoint` rebuilds a policy for the s1 shapes and copies
the s0 weights in, so an s1 kickoff run starts from s0's learned behavior
rather than from scratch -- following the ``load_bc_weights_into_policy``
precedent (``jackdaw/agents/hand_policy.py``): copy per-module state
dicts rather than attempting a flat ``load_state_dict`` on the whole
policy, since the observation/action shapes differ between s0 and s1.

Per-module treatment (see the WHY at each site above):

* trunk (embedding, descriptors buffer, all per-entity encoders, the
  global encoder, and the ``shop_context``-base ``LayerNorm``+``Linear``)
  -- copied verbatim, EXACT. None of these parameters' shapes depend on
  joker row count (masked pooling) or on the offered-tag one-hot (the s1
  extractor's trunk only ever sees the original 12-dim ``shop_context``
  slice; the tag one-hot rides through a separate, zero-initialized
  ``tag_encoder`` summed in afterward -- see ``shop_policy.py``'s module
  docstring for why concatenating it INTO the shared LayerNorm would have
  broken this).
* ``tag_encoder`` -- new-only, zero-initialized (contributes exactly 0
  regardless of the tag one-hot's value on a freshly-migrated model).
* ``action_net`` -- rows ``[0, 686)`` copied verbatim, rows
  ``[686, 694)`` (SkipBlind + SellJoker ext) keep their fresh/cold init.
  A ``Linear`` layer's output rows are independent maps (row i depends
  only on weight row i), so copying a ROW SUBSET is exact -- unlike the
  LayerNorm case above, there is no cross-row interaction to break.
* ``value_net`` -- copied verbatim (no width change).

This module is NOT wired into ``scripts/train_shop_ppo.py``'s main flow
(the s1 kickoff run itself is future work, per docs/post-regen-training-
plan.md's wave plan) -- it is usable directly, or via the thin CLI at the
bottom of this file, once that kickoff is ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from jackdaw.agents.shop_action_space import NUM_TOTAL_ACTIONS
from jackdaw.agents.shop_policy import ShopFeaturesExtractor
from jackdaw.env.shop_gym import ShopGymEnv
from jackdaw.env.shop_run_adapter import ShopRunConfig


def _make_s1_env() -> DummyVecEnv:
    """A single-env ``DummyVecEnv`` on the s1 schema, purely to give
    ``MaskablePPO`` matching obs/action spaces to build its policy against
    -- no rollout happens here."""

    def _factory() -> ShopGymEnv:
        return ShopGymEnv(config=ShopRunConfig(s1_schema=True))

    return DummyVecEnv([_factory])


def widen_s0_checkpoint(
    old_zip_path: str | Path,
    *,
    seed: int = 0,
    device: str = "cpu",
) -> MaskablePPO:
    """Load an s0 ``MaskablePPO`` checkpoint and return a fresh s1-schema
    model with its weights copied in.

    The returned model has NOT been saved -- call ``.save(path)`` on it
    (or use the CLI below) to persist it.
    """
    old_model = MaskablePPO.load(str(old_zip_path), device=device)
    old_policy = old_model.policy

    env = _make_s1_env()
    new_model = MaskablePPO(
        "MultiInputPolicy",
        env,
        seed=seed,
        device=device,
        policy_kwargs=dict(
            features_extractor_class=ShopFeaturesExtractor,
            features_extractor_kwargs=dict(s1_schema=True),
            net_arch=[],  # matches train_shop_ppo.py's build_model
        ),
    )
    new_policy = new_model.policy

    _copy_trunk(old_policy, new_policy)
    _copy_action_head(old_policy, new_policy)
    new_policy.value_net.load_state_dict(old_policy.value_net.state_dict())

    return new_model


def _copy_trunk(old_policy, new_policy) -> None:
    """Copy every s0 trunk parameter into the (shape-identical) s1
    extractor, leaving only the new-only ``tag_encoder`` untouched by the
    copy (explicitly zero-initialized instead)."""
    old_fx = old_policy.features_extractor
    new_fx = new_policy.features_extractor

    old_state = old_fx.state_dict()
    new_state = new_fx.state_dict()
    missing = set(old_state) - set(new_state)
    if missing:
        raise ValueError(
            f"s1 extractor is missing keys the s0 extractor has: {missing} -- "
            "the trunk architectures have diverged; this migration assumes "
            "they stay parameter-identical except for tag_encoder."
        )
    for key, value in old_state.items():
        new_state[key] = value
    new_fx.load_state_dict(new_state)
    torch.nn.init.zeros_(new_fx.tag_encoder.weight)

    # SB3 aliases pi/vf feature extractors to the shared one by default
    # (share_features_extractor=True); load into them explicitly in case
    # sharing was ever disabled, mirroring load_bc_weights_into_policy.
    for attr in ("pi_features_extractor", "vf_features_extractor"):
        other = getattr(new_policy, attr, None)
        if other is not None and other is not new_fx:
            other.load_state_dict(new_fx.state_dict())


def _copy_action_head(old_policy, new_policy) -> None:
    """Copy action-net rows ``[0, NUM_TOTAL_ACTIONS)`` verbatim; rows
    ``[NUM_TOTAL_ACTIONS, NUM_TOTAL_ACTIONS_S1)`` (SkipBlind + SellJoker
    ext) keep the fresh model's cold init."""
    old_weight = old_policy.action_net.weight.data
    old_bias = old_policy.action_net.bias.data
    if old_weight.shape[0] != NUM_TOTAL_ACTIONS:
        raise ValueError(
            f"old checkpoint's action_net has {old_weight.shape[0]} rows, "
            f"expected the frozen s0 width {NUM_TOTAL_ACTIONS} -- refusing "
            "to guess a row mapping."
        )
    with torch.no_grad():
        new_policy.action_net.weight[:NUM_TOTAL_ACTIONS].copy_(old_weight)
        new_policy.action_net.bias[:NUM_TOTAL_ACTIONS].copy_(old_bias)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path, required=True, help="s0 .zip")
    parser.add_argument("--output", type=Path, required=True, help="output s1 .zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = widen_s0_checkpoint(args.old_checkpoint, seed=args.seed, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"widened checkpoint written to {args.output}")


if __name__ == "__main__":
    main()

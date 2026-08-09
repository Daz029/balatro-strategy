"""Tests for s1-to-s0 Phi shaping support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO  # noqa: E402

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy  # noqa: E402
from jackdaw.agents.phi_shaping import S0CriticPhi, truncate_s1_obs  # noqa: E402
from jackdaw.agents.shop_policy import ShopFeaturesExtractor  # noqa: E402
from jackdaw.engine.actions import GamePhase  # noqa: E402
from jackdaw.engine.card_factory import create_joker  # noqa: E402
from jackdaw.engine.run_init import initialize_run  # noqa: E402
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_obs import D_SHOP_CONTEXT, build_shop_observation  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunAdapter, ShopRunConfig  # noqa: E402


def _fresh_state() -> dict:
    adapter = ShopRunAdapter(GreedyHandPolicy())
    adapter.reset("b_red", 1, "PHI_FRESH_STATE")
    return adapter.raw_state


def _overfull_joker_state() -> dict:
    gs = _fresh_state()
    jokers = [create_joker("j_joker") for _ in range(12)]
    for joker in jokers[8:]:
        joker.set_edition({"negative": True})
    gs["jokers"] = jokers
    return gs


def _offered_tag_state() -> dict:
    gs = initialize_run("b_red", 1, "PHI_OFFERED_TAG")
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    return gs


@pytest.mark.parametrize(
    "state_factory",
    [_fresh_state, _overfull_joker_state, _offered_tag_state],
    ids=["fresh", "more-than-eight-jokers", "offered-tag"],
)
def test_truncation_is_the_pinned_s0_inverse(state_factory):
    gs = state_factory()
    s1_obs = build_shop_observation(gs, s1_schema=True)
    s0_obs = build_shop_observation(gs, s1_schema=False)
    if state_factory is _offered_tag_state:
        assert s1_obs["shop_context"][D_SHOP_CONTEXT:].any()

    truncated = truncate_s1_obs(s1_obs)
    assert set(truncated) == set(s0_obs)
    for key in s0_obs:
        assert np.array_equal(truncated[key], s0_obs[key]), key
        assert truncated[key].dtype == s0_obs[key].dtype, key


def test_truncation_rejects_s0_and_does_not_mutate_input():
    gs = _fresh_state()
    s1_obs = build_shop_observation(gs, s1_schema=True)
    before = {key: value.copy() for key, value in s1_obs.items()}

    with pytest.raises(ValueError):
        truncate_s1_obs(build_shop_observation(gs, s1_schema=False))

    truncated = truncate_s1_obs(s1_obs)
    for key, value in s1_obs.items():
        assert np.array_equal(value, before[key]), key
    assert truncated["global_context"] is s1_obs["global_context"]


def _save_model(path: Path, *, s1_schema: bool) -> Path:
    config = ShopRunConfig(s1_schema=s1_schema) if s1_schema else None
    env = ShopGymEnv(config=config)
    policy_kwargs = {"features_extractor_class": ShopFeaturesExtractor, "net_arch": []}
    if s1_schema:
        policy_kwargs["features_extractor_kwargs"] = {"s1_schema": True}
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    model.save(str(path))
    env.close()
    return path


def test_s0_critic_phi_is_frozen_and_schema_consistent(tmp_path):
    checkpoint = _save_model(tmp_path / "s0.zip", s1_schema=False)
    phi = S0CriticPhi(checkpoint)
    gs = _fresh_state()
    s0_obs = build_shop_observation(gs, s1_schema=False)
    s1_obs = build_shop_observation(gs, s1_schema=True)

    s0_value = phi(s0_obs)
    assert np.isfinite(s0_value)
    assert s0_value == phi(s1_obs)
    assert s0_value == phi(truncate_s1_obs(s1_obs))
    assert all(not parameter.requires_grad for parameter in phi._policy.parameters())
    assert phi._policy.optimizer is None


def test_s1_critic_phi_consumes_the_untruncated_observation(tmp_path):
    """An s1-width critic is accepted and fed its own schema.

    Truncating for it would drop the joker rows and offered tag its encoder
    was fitted with.  Reading an s0 observation is the error instead.
    """

    checkpoint = _save_model(tmp_path / "s1.zip", s1_schema=True)
    phi = S0CriticPhi(checkpoint)
    gs = _fresh_state()

    assert np.isfinite(phi(build_shop_observation(gs, s1_schema=True)))
    with pytest.raises(ValueError):
        phi(build_shop_observation(gs, s1_schema=False))

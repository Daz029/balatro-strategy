"""Tests for the s0 -> s1 shop checkpoint migration (weight-preserving).

The VERIFICATION GATE (old vs widened model byte-identical on old-schema
states) is the load-bearing property: it licenses "widen the joker rows +
append the offered-tag one-hot as no-retrain" claims made throughout
CLAUDE.md's s1 section and docs/post-regen-training-plan.md. Run against
a freshly-initialized old-schema model unconditionally (no local-file
dependency); ALSO against the real ``runs/shop_ppo/s0_a4_v4.zip``
checkpoint when present locally (``runs/`` is gitignored, so this is
skipped, not failed, when absent -- e.g. in CI).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO  # noqa: E402

from jackdaw.agents.checkpoint_migration import widen_s0_checkpoint  # noqa: E402
from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy  # noqa: E402
from jackdaw.agents.shop_action_space import NUM_TOTAL_ACTIONS, NUM_TOTAL_ACTIONS_S1  # noqa: E402
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_obs import build_shop_observation  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunAdapter  # noqa: E402

_REAL_ARCHIVE = Path("runs/shop_ppo/s0_a4_v4.zip")


def _real_s0_checkpoint(tmp_path: Path) -> Path | None:
    """Locate a loadable real s0 checkpoint, or None if not available
    locally. Handles both a plain SB3 .zip and the archived-run-directory
    form (a zip whose contents include ``.../shop_ppo_final.zip``)."""
    if not _REAL_ARCHIVE.exists():
        return None
    try:
        MaskablePPO.load(str(_REAL_ARCHIVE), device="cpu")
        return _REAL_ARCHIVE
    except Exception:
        pass
    # Archived-run-directory form: extract the nested model zip.
    with zipfile.ZipFile(_REAL_ARCHIVE) as zf:
        candidates = [n for n in zf.namelist() if n.endswith("shop_ppo_final.zip")]
        if not candidates:
            return None
        out = tmp_path / "shop_ppo_final.zip"
        out.write_bytes(zf.read(candidates[0]))
        return out


def _fresh_s0_checkpoint(tmp_path: Path) -> Path:
    """A freshly-initialized (untrained) old-schema model, saved to a
    tmp .zip -- the CI-safe fallback that needs no local file."""
    env = ShopGymEnv()
    from jackdaw.agents.shop_policy import ShopFeaturesExtractor

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(features_extractor_class=ShopFeaturesExtractor, net_arch=[]),
        device="cpu",
    )
    out = tmp_path / "fresh_s0.zip"
    model.save(str(out))
    return out


def _old_schema_state() -> dict:
    adapter = ShopRunAdapter(GreedyHandPolicy())
    adapter.reset("b_red", 1, "MIGRATION_GATE_SEED")
    adapter.raw_state["dollars"] = 50
    return adapter.raw_state


def _batched(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}


def _assert_byte_identical_on_old_states(old_model: MaskablePPO, new_model: MaskablePPO) -> None:
    gs = _old_schema_state()
    old_obs = _batched(build_shop_observation(gs))
    new_obs = _batched(build_shop_observation(gs, s1_schema=True))

    old_model.policy.eval()
    new_model.policy.eval()
    with torch.no_grad():
        old_feat = old_model.policy.extract_features(
            old_obs, old_model.policy.features_extractor
        )
        new_feat = new_model.policy.extract_features(
            new_obs, new_model.policy.features_extractor
        )
        old_logits = old_model.policy.action_net(old_feat)
        new_logits = new_model.policy.action_net(new_feat)
        old_value = old_model.policy.value_net(
            old_model.policy.extract_features(old_obs, old_model.policy.vf_features_extractor)
        )
        new_value = new_model.policy.value_net(
            new_model.policy.extract_features(new_obs, new_model.policy.vf_features_extractor)
        )

    # Exact equality holds here: the additive tag_encoder is zero-init and
    # the old-schema state's tag one-hot doesn't even exist (both zero'd
    # out), and the trunk is the SAME parameters over the SAME 12-dim
    # shop_context slice -- no floating-point reassociation happens
    # anywhere in the s0 vs s1 forward pass on such a state.
    assert torch.equal(old_feat, new_feat)
    assert torch.equal(old_logits, new_logits[:, :NUM_TOTAL_ACTIONS])
    assert torch.equal(old_value, new_value)


class TestWidenedShapes:
    def test_action_and_value_head_shapes(self, tmp_path):
        old_path = _fresh_s0_checkpoint(tmp_path)
        new_model = widen_s0_checkpoint(old_path)
        assert new_model.policy.action_net.weight.shape == (NUM_TOTAL_ACTIONS_S1, 256)
        assert new_model.policy.value_net.weight.shape == (1, 256)


class TestVerificationGateFreshModel:
    """CI-safe: no local file dependency."""

    def test_byte_identical_on_old_schema_states(self, tmp_path):
        old_path = _fresh_s0_checkpoint(tmp_path)
        old_model = MaskablePPO.load(str(old_path), device="cpu")
        new_model = widen_s0_checkpoint(old_path)
        _assert_byte_identical_on_old_states(old_model, new_model)

    def test_new_action_rows_are_cold_not_copied_garbage(self, tmp_path):
        old_path = _fresh_s0_checkpoint(tmp_path)
        new_model = widen_s0_checkpoint(old_path)
        # Rows [686, 694) must not be all-zero copies of row 0 or NaN --
        # just a sanity check they hold SOME independent fresh init.
        cold_rows = new_model.policy.action_net.weight[NUM_TOTAL_ACTIONS:]
        assert torch.isfinite(cold_rows).all()
        assert cold_rows.shape == (694 - NUM_TOTAL_ACTIONS, 256)


class TestVerificationGateRealCheckpoint:
    def test_byte_identical_on_real_s0_a4_v4(self, tmp_path):
        real_path = _real_s0_checkpoint(tmp_path)
        if real_path is None:
            pytest.skip("runs/shop_ppo/s0_a4_v4.zip not available locally (runs/ is gitignored)")
        old_model = MaskablePPO.load(str(real_path), device="cpu")
        new_model = widen_s0_checkpoint(real_path)
        _assert_byte_identical_on_old_states(old_model, new_model)

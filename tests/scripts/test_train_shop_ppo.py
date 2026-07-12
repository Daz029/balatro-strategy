"""Tests for the shop PPO training pipeline + eval suite.

Critical invariants:
- schedule decay: blend/count coefficients reach exactly zero at progress 0
  (the final optimized objective must be exactly P(win))
- count bonuses follow 1/sqrt(N) and fire on the right events (joker-set
  CHANGE, pending-target completion — not per step)
- reservoir: always-nonzero fresh anchor, pack-stratum oversampling,
  harvested snapshots actually restore
- end-to-end learn() smoke on the real env (MaskablePPO + the wrapper +
  ShopFeaturesExtractor), save/load round-trip through the eval suite
- eval suite determinism on the reserved EVAL_ seed stream
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import torch  # noqa: E402

from eval_shop_policy import NextRoundPolicy, eval_seeds, load_policy, run_suite  # noqa: E402
from train_shop_ppo import (  # noqa: E402
    CountBonus,
    ScheduleCallback,
    ShopReservoir,
    ShopRewardWrapper,
    TrainingSchedules,
    build_model,
)

from jackdaw.agents.shop_action_space import ShopActionFamily, shop_action  # noqa: E402
from jackdaw.env.shop_gym import ShopGymEnv, blind_clear_bonus  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunConfig  # noqa: E402


def _card_set(card) -> str:
    ability = getattr(card, "ability", None)
    return ability.get("set", "") if isinstance(ability, dict) else ""


class TestSchedules:
    def test_decay_to_exact_zero(self):
        s = TrainingSchedules(blend_beta0=1.0, count_beta0=0.05)
        assert s.blend_beta == 1.0 and s.count_beta == 0.05
        s.progress_remaining = 0.5
        assert s.blend_beta == 0.5
        s.progress_remaining = 0.0
        assert s.blend_beta == 0.0 and s.count_beta == 0.0


class TestCountBonus:
    def test_inverse_sqrt_visits(self):
        c = CountBonus()
        key = ("j_joker", "j_lusty_joker")
        assert c.joker_set(key) == pytest.approx(1.0)
        assert c.joker_set(key) == pytest.approx(1.0 / np.sqrt(2))
        assert c.joker_set(("j_joker",)) == pytest.approx(1.0)  # independent key

    def test_target_patterns_independent(self):
        c = CountBonus()
        assert c.target(("c_magician", 2)) == pytest.approx(1.0)
        assert c.target(("c_magician", 1)) == pytest.approx(1.0)
        assert c.target(("c_magician", 2)) == pytest.approx(1.0 / np.sqrt(2))


class TestReservoir:
    def test_fresh_anchor_must_be_nonzero(self):
        with pytest.raises(ValueError, match="fresh_frac"):
            ShopReservoir(fresh_frac=0.0)

    def test_empty_reservoir_samples_fresh(self):
        r = ShopReservoir(fresh_frac=1e-9)
        assert r.sample() is None

    def test_all_fresh_when_frac_is_one(self):
        r = ShopReservoir(fresh_frac=1.0)
        r.add(b"snap", ante=1, pack_pending=False)
        assert all(r.sample() is None for _ in range(20))

    def test_pack_stratum_oversampled(self):
        r = ShopReservoir(fresh_frac=1e-9, pack_frac=1.0, seed=3)
        r.add(b"shop", ante=1, pack_pending=False)
        r.add(b"pack", ante=1, pack_pending=True)
        assert all(r.sample() == b"pack" for _ in range(20))

    def test_stratified_storage_and_len(self):
        r = ShopReservoir(fresh_frac=0.5, capacity_per_stratum=2)
        for i in range(5):
            r.add(f"a{i}".encode(), ante=1, pack_pending=False)
        r.add(b"b", ante=2, pack_pending=False)
        assert len(r) == 3  # ante-1 stratum capped at 2, plus ante-2

    def test_save_load_roundtrip(self, tmp_path):
        # Persistence must survive across invocations (the a2->a4->a8 chain
        # and the s0->s1 hop) with strata, config, and the sampling stream all
        # intact — otherwise each stage restarts from an empty reservoir.
        r = ShopReservoir(fresh_frac=0.4, pack_frac=0.7, capacity_per_stratum=3, seed=11)
        r.add(b"shop_a1", ante=1, pack_pending=False)
        r.add(b"pack_a1", ante=1, pack_pending=True)
        r.add(b"shop_a2", ante=2, pack_pending=False)

        path = tmp_path / "reservoir.pkl"
        r.save(path)
        loaded = ShopReservoir.load(path)

        assert len(loaded) == len(r) == 3
        assert loaded.fresh_frac == 0.4
        assert loaded.pack_frac == 0.7
        assert loaded._capacity == 3
        assert loaded._strata.keys() == r._strata.keys()
        # Loaded strata are bounded deques honoring the restored capacity.
        for key, dq in loaded._strata.items():
            assert dq.maxlen == 3
            assert list(dq) == list(r._strata[key])
        # RNG state round-trips: the sampling stream continues identically.
        assert [loaded.sample() for _ in range(30)] == [r.sample() for _ in range(30)]

    def test_loaded_reservoir_respects_capacity_on_further_adds(self, tmp_path):
        r = ShopReservoir(fresh_frac=0.5, capacity_per_stratum=2, seed=0)
        r.add(b"x0", ante=1, pack_pending=False)
        path = tmp_path / "reservoir.pkl"
        r.save(path)
        loaded = ShopReservoir.load(path)
        for i in range(5):
            loaded.add(f"y{i}".encode(), ante=1, pack_pending=False)
        assert len(loaded) == 2  # deque maxlen enforced after reload


class TestRewardWrapper:
    def _make(self, reservoir=None, harvest_prob=0.0, seed="SHOPGYM_CONTRACT"):
        schedules = TrainingSchedules(blend_beta0=1.0, count_beta0=1.0)
        env = ShopRewardWrapper(
            ShopGymEnv(config=ShopRunConfig(win_ante=1)),
            schedules,
            CountBonus(),
            reservoir,
            harvest_prob=harvest_prob,
        )
        obs, info = env.reset(options={"episode_seed": seed})
        return env, schedules, obs, info

    def test_blends_blind_bonus(self):
        env, schedules, _, _ = self._make()
        ante = env.env.raw_state["round_resets"]["ante"]
        _, reward, terminated, _, info = env.step(shop_action(ShopActionFamily.NextRound))
        rc = info["reward_components"]
        expected = rc["win"] + schedules.blend_beta * rc["blind_bonus"] + rc["count_bonus"]
        assert reward == pytest.approx(expected)
        if rc["blinds_cleared"]:
            assert rc["blind_bonus"] == pytest.approx(blind_clear_bonus(ante))
        assert rc["blend_beta"] == 1.0

    def test_joker_set_bonus_fires_on_acquisition(self):
        # Probe seeds for a shop whose slot 0 offers a Joker, buy it, and
        # the novelty bonus for the new (single-joker) key-set must fire.
        for i in range(20):
            try:
                env, schedules, _, _ = self._make(seed=f"SHOPWRAP_{i:02d}")
            except RuntimeError:
                continue  # greedy partner died in the first blind; next seed
            gs = env.env.raw_state
            gs["dollars"] = 50
            shop_cards = gs.get("shop_cards", [])
            joker_slot = next(
                (k for k, c in enumerate(shop_cards) if _card_set(c) == "Joker"), None
            )
            if joker_slot is None:
                continue
            _, _, _, _, info = env.step(shop_action(ShopActionFamily.BuyCard, joker_slot))
            # First visit to this key-set: bonus == count_beta * 1/sqrt(1).
            assert info["reward_components"]["count_bonus"] == pytest.approx(schedules.count_beta)
            return
        raise AssertionError("no joker in shop slot across 20 probe seeds")

    def test_no_count_bonus_without_change(self):
        env, _, _, _ = self._make()
        env.env.raw_state["dollars"] = 50
        _, _, _, _, info = env.step(shop_action(ShopActionFamily.Reroll))
        assert info["reward_components"]["count_bonus"] == 0.0

    def test_harvested_snapshot_restores(self):
        reservoir = ShopReservoir(fresh_frac=1e-9, seed=0)
        env, _, _, _ = self._make(reservoir=reservoir, harvest_prob=1.0)
        env.env.raw_state["dollars"] = 50
        env.step(shop_action(ShopActionFamily.Reroll))
        assert len(reservoir) == 1

        blob = reservoir.sample()
        fresh = ShopGymEnv(config=ShopRunConfig(win_ante=1))
        obs, info = fresh.reset(options={"snapshot": blob})
        assert info["episode_seed"] == "<restored>"
        assert info["action_mask"].any()


class TestLearnSmoke:
    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        schedules = TrainingSchedules()
        model, schedules = build_model(
            win_ante=1,
            schedules=schedules,
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        # Two rollout iterations: SB3 updates progress_remaining at the
        # START of each collection, so a single iteration leaves it at 1.0.
        model.learn(total_timesteps=16, callback=ScheduleCallback(schedules))
        path = tmp_path_factory.mktemp("shop_ppo") / "model.zip"
        model.save(str(path))
        return model, schedules, path

    def test_one_update_moves_params_and_schedule(self, trained):
        model, schedules, _ = trained
        # The schedule callback ran: progress advanced off 1.0.
        assert schedules.progress_remaining < 1.0
        # The full canonical head exists (686 rows).
        assert model.policy.action_net.out_features == 686

    def test_save_load_roundtrip_through_eval(self, trained):
        _, _, path = trained
        policy = load_policy(str(path), "cpu")
        result = run_suite(policy, win_ante=1, n_episodes=2)
        assert result["n_played"] + result["n_dead_at_reset"] == 2
        if result["n_played"]:
            assert 0.0 <= result["win_rate"] <= 1.0
            assert result["mean_steps"] >= 1.0


class TestFiniteGradGuard:
    def test_nonfinite_gradient_is_sanitized(self):
        # A single NaN/inf gradient must NOT reach the parameter's .grad, or it
        # poisons the weights and every later MaskableCategorical fails the
        # simplex check (the observed a4 crash ~719k steps in).
        model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        param = next(model.policy.parameters())
        # +inf loss (the PPO unclipped-surrogate blow-up) -> inf/NaN raw grad.
        loss = (param * torch.tensor(float("inf"))).sum()
        loss.backward()
        assert torch.isfinite(param.grad).all()

    def test_nonfinite_weight_is_scrubbed_after_step(self):
        # Backstop: even if a NaN reaches the weights through a path the grad
        # hook can't see (the second a4 crash, guard already active), the
        # optimizer step-post hook must scrub it so no forward sees a NaN
        # weight and MaskableCategorical never fails the simplex check.
        model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        opt = model.policy.optimizer
        param = next(model.policy.parameters())
        param.grad = torch.zeros_like(param)
        with torch.no_grad():
            param.view(-1)[0] = float("nan")
        opt.step()
        assert torch.isfinite(param).all()

    def test_nonfinite_forward_logits_are_bounded(self):
        # The layer that stops the OBSERVED crash: even when action_net emits
        # non-finite logits from finite weights (the forward-generated failure
        # that reproduced byte-identically with layers 1-2 active), the values
        # reaching MaskableCategorical must be finite and softmax-safe.
        model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        assert model._finite_guard_logit_catches == 0
        latent = torch.zeros(4, model.policy.action_net.in_features)
        # Poison the logit layer's bias so its raw output is non-finite.
        with torch.no_grad():
            model.policy.action_net.bias.view(-1)[0] = float("inf")
            model.policy.action_net.bias.view(-1)[1] = float("nan")
        logits = model.policy.action_net(latent)
        assert torch.isfinite(logits).all()
        assert logits.abs().max() <= 30.0
        # The catch is counted (so a stream vs a one-off is distinguishable in
        # the log), and a finite forward does not increment it.
        assert model._finite_guard_logit_catches == 1
        model.policy.action_net(torch.zeros(4, model.policy.action_net.in_features))
        # (bias is still poisoned, so this forward is also non-finite)
        assert model._finite_guard_logit_catches == 2
        with torch.no_grad():
            model.policy.action_net.bias.view(-1)[0] = 0.0
            model.policy.action_net.bias.view(-1)[1] = 0.0
        model.policy.action_net(torch.zeros(4, model.policy.action_net.in_features))
        assert model._finite_guard_logit_catches == 2  # finite forward: no bump


class TestEvalSuite:
    def test_eval_seeds_reserved_prefix(self):
        assert eval_seeds(2) == ["EVAL_00000000", "EVAL_00000001"]

    def test_nextround_baseline_deterministic(self):
        a = run_suite(NextRoundPolicy(), win_ante=1, n_episodes=3)
        b = run_suite(NextRoundPolicy(), win_ante=1, n_episodes=3)
        assert a == b
        assert a["n_played"] + a["n_dead_at_reset"] == 3
        if a["n_played"]:
            assert 0.0 <= a["win_rate"] <= 1.0

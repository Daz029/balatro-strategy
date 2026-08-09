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

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import gymnasium  # noqa: E402
import torch  # noqa: E402
from eval_shop_policy import NextRoundPolicy, eval_seeds, load_policy, run_suite  # noqa: E402
from gymnasium import spaces  # noqa: E402
from train_shop_ppo import (  # noqa: E402
    CountBonus,
    NormalizedEntropyCallback,
    ScheduleCallback,
    ShopReservoir,
    ShopRewardWrapper,
    TrainingSchedules,
    build_model,
    load_hand_policy,
    make_train_env,
    parse_args,
)

from jackdaw.agents.phi_shaping import S0CriticPhi  # noqa: E402
from jackdaw.agents.shop_action_space import (  # noqa: E402
    NUM_TOTAL_ACTIONS,
    NUM_TOTAL_ACTIONS_S1,
    ShopActionFamily,
    decode_shop_action,
    shop_action,
)
from jackdaw.env.shop_gym import ShopGymEnv, blind_clear_bonus  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunConfig  # noqa: E402


def _card_set(card) -> str:
    ability = getattr(card, "ability", None)
    return ability.get("set", "") if isinstance(ability, dict) else ""


class _JokerTransactionStub(gymnasium.Env):
    observation_space = spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = spaces.Discrete(NUM_TOTAL_ACTIONS_S1)

    def __init__(self, jokers, shop_cards, shop_vouchers=None):
        self.raw_state = {
            "jokers": list(jokers),
            "shop_cards": list(shop_cards),
            "shop_vouchers": list(shop_vouchers or []),
            "round_resets": {"ante": 1},
        }
        self.pending = None

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {
            "reward_components": {"blind_bonus": 0.0, "win": 0.0}
        }

    def step(self, action):
        family, slot = decode_shop_action(action)
        if family is ShopActionFamily.BuyCard:
            card = self.raw_state["shop_cards"].pop(slot)
            if _card_set(card) == "Joker":
                self.raw_state["jokers"].append(card)
        elif family is ShopActionFamily.SellJoker:
            self.raw_state["jokers"].pop(slot)
        return (
            np.zeros(1, dtype=np.float32),
            0.0,
            False,
            False,
            {"reward_components": {"blind_bonus": 0.0, "win": 0.0}},
        )

    def action_masks(self):
        return np.ones(NUM_TOTAL_ACTIONS_S1, dtype=bool)


def _joker(key="j_zany_joker", *, edition=None, **stickers):
    return SimpleNamespace(
        center_key=key,
        ability={"set": "Joker"},
        edition=edition,
        eternal=stickers.get("eternal", False),
        perishable=stickers.get("perishable", False),
        rental=stickers.get("rental", False),
    )


def _voucher(key):
    return SimpleNamespace(center_key=key)


class TestSchedules:
    def test_decay_to_exact_zero(self):
        s = TrainingSchedules(blend_beta0=1.0, count_beta0=0.05)
        assert s.blend_beta == 1.0 and s.count_beta == 0.05
        assert s.phi_beta == 1.0
        s.progress_remaining = 0.5
        assert s.blend_beta == 0.5
        assert s.phi_beta == 0.5
        s.progress_remaining = 0.0
        assert s.blend_beta == 0.0 and s.count_beta == 0.0 and s.phi_beta == 0.0


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

    def _make_transaction_stub(
        self,
        *,
        reward=-0.1,
        decay=True,
        jokers=None,
        shop_cards=None,
        shop_vouchers=None,
        skip_tag_reward=None,
        skip_tag_decay=True,
    ):
        schedules = TrainingSchedules(blend_beta0=0.0, count_beta0=0.0)
        env = ShopRewardWrapper(
            _JokerTransactionStub(jokers or [], shop_cards or [], shop_vouchers),
            schedules,
            CountBonus(),
            immediate_joker_sell_reward=reward,
            immediate_joker_sell_decay=decay,
            skip_tag_reward=skip_tag_reward,
            skip_tag_decay=skip_tag_decay,
        )
        env.reset()
        return env, schedules

    def test_immediate_duplicate_joker_sale_ignores_position(self):
        env, _ = self._make_transaction_stub(
            jokers=[_joker() for _ in range(3)],
            shop_cards=[_joker()],
        )

        env.step(shop_action(ShopActionFamily.BuyCard, 0))
        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.SellJoker, 0))

        assert reward == pytest.approx(-0.1)
        assert info["reward_components"]["immediate_joker_sell_reward"] == pytest.approx(-0.1)

    def test_tracker_requires_matching_stickers_and_editions(self):
        env, _ = self._make_transaction_stub(
            jokers=[_joker(edition={"foil": True}, rental=True), _joker()],
            shop_cards=[_joker()],
        )

        env.step(shop_action(ShopActionFamily.BuyCard, 0))
        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.SellJoker, 0))

        assert reward == 0.0
        assert info["reward_components"]["immediate_joker_sell_reward"] == 0.0

    @pytest.mark.parametrize("bought", [_joker("j_diet_cola"), _joker(rental=True)])
    def test_diet_cola_and_rental_are_not_stored_as_last_bought(self, bought):
        env, _ = self._make_transaction_stub(shop_cards=[bought])

        env.step(shop_action(ShopActionFamily.BuyCard, 0))

        assert env._last_bought_joker is None

    @pytest.mark.parametrize(
        ("jokers", "shop_cards", "shop_vouchers"),
        [
            ([_joker("j_campfire"), _joker()], [_joker()], []),
            ([_joker()], [_joker("j_diet_cola")], []),
            ([_joker()], [_joker(rental=True)], []),
            ([_joker()], [_joker()], [_voucher("v_overstock_norm")]),
            ([_joker()], [_joker()], [_voucher("v_overstock_plus")]),
        ],
    )
    def test_buy_sell_reward_is_suppressed_by_special_cases(
        self, jokers, shop_cards, shop_vouchers
    ):
        env, _ = self._make_transaction_stub(
            jokers=jokers,
            shop_cards=shop_cards,
            shop_vouchers=shop_vouchers,
        )

        env.step(shop_action(ShopActionFamily.BuyCard, 0))
        _, reward, _, _, info = env.step(
            shop_action(ShopActionFamily.SellJoker, len(jokers))
        )

        assert reward == 0.0
        assert info["reward_components"]["immediate_joker_sell_reward"] == 0.0

    def test_any_intervening_action_clears_tracker_and_buy_overwrites_it(self):
        env, _ = self._make_transaction_stub(
            jokers=[_joker(), _joker("j_mad_joker")],
            shop_cards=[_joker(), _joker("j_mad_joker")],
        )

        env.step(shop_action(ShopActionFamily.BuyCard, 0))
        env.step(shop_action(ShopActionFamily.BuyCard, 0))
        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.SellJoker, 0))
        assert reward == 0.0
        assert info["reward_components"]["immediate_joker_sell_reward"] == 0.0

        env.step(shop_action(ShopActionFamily.Reroll))
        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.SellJoker, 0))
        assert reward == 0.0
        assert info["reward_components"]["immediate_joker_sell_reward"] == 0.0

    def test_reward_can_decay_or_remain_constant(self):
        decayed, schedules = self._make_transaction_stub(
            reward=-0.4, decay=True, jokers=[_joker()], shop_cards=[_joker()]
        )
        schedules.progress_remaining = 0.25
        decayed.step(shop_action(ShopActionFamily.BuyCard, 0))
        _, reward, _, _, _ = decayed.step(shop_action(ShopActionFamily.SellJoker, 0))
        assert reward == pytest.approx(-0.1)

        constant, schedules = self._make_transaction_stub(
            reward=-0.4, decay=False, jokers=[_joker()], shop_cards=[_joker()]
        )
        schedules.progress_remaining = 0.25
        constant.step(shop_action(ShopActionFamily.BuyCard, 0))
        _, reward, _, _, _ = constant.step(shop_action(ShopActionFamily.SellJoker, 0))
        assert reward == pytest.approx(-0.4)

    def test_reward_flag_enables_feature_and_no_decay_requires_reward(self):
        args = parse_args(
            ["--immediate-joker-sell-reward", "-0.25", "--immediate-joker-sell-no-decay"]
        )
        assert args.immediate_joker_sell_reward == -0.25
        assert args.immediate_joker_sell_no_decay is True

        with pytest.raises(SystemExit):
            parse_args(["--immediate-joker-sell-no-decay"])

        args = parse_args(["--skip-tag-reward", "-0.3", "--skip-tag-no-decay"])
        assert args.skip_tag_reward == -0.3
        assert args.skip_tag_no_decay is True

        with pytest.raises(SystemExit):
            parse_args(["--skip-tag-no-decay"])

    def test_skip_tag_reward_fires_only_on_skip_blind(self):
        env, schedules = self._make_transaction_stub(skip_tag_reward=-0.4)
        schedules.progress_remaining = 0.25

        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.SkipBlind))
        assert reward == pytest.approx(-0.1)
        assert info["reward_components"]["skip_tag_reward"] == pytest.approx(-0.1)

        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.NextRound))
        assert reward == 0.0
        assert info["reward_components"]["skip_tag_reward"] == 0.0

    def test_skip_tag_reward_can_remain_constant(self):
        env, schedules = self._make_transaction_stub(
            skip_tag_reward=-0.4, skip_tag_decay=False
        )
        schedules.progress_remaining = 0.25

        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.SkipBlind))
        assert reward == pytest.approx(-0.4)
        assert info["reward_components"]["skip_tag_reward"] == pytest.approx(-0.4)

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

    def test_phi_terms_telescope_to_negative_initial_potential(self):
        class StubEnv(gymnasium.Env):
            observation_space = spaces.Dict({"state": spaces.Box(0, 10, (1,), dtype=np.float32)})
            action_space = spaces.Discrete(1)

            def __init__(self):
                self.raw_state = {"jokers": [], "round_resets": {"ante": 1}}
                self.pending = None
                self._step = 0

            def reset(self, **kwargs):
                self._step = 0
                return {"state": np.array([2], dtype=np.float32)}, {
                    "action_mask": np.ones(1, dtype=bool),
                    "reward_components": {"blind_bonus": 0.0, "win": 0.0},
                }

            def step(self, action):
                self._step += 1
                terminated = self._step == 3
                return (
                    {"state": np.array([2 + self._step * 2], dtype=np.float32)},
                    0.0,
                    terminated,
                    False,
                    {
                        "action_mask": np.ones(1, dtype=bool),
                        "reward_components": {"blind_bonus": 0.0, "win": 0.0},
                    },
                )

            def action_masks(self):
                return np.ones(1, dtype=bool)

        schedules = TrainingSchedules(blend_beta0=0.0, count_beta0=0.0, phi_beta0=1.0)
        env = ShopRewardWrapper(
            StubEnv(), schedules, CountBonus(), phi=lambda obs: float(obs["state"][0])
        )
        env.reset()
        phi_terms = []
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(0)
            phi_terms.append(info["reward_components"]["phi_term"])
            if terminated or truncated:
                break

        assert sum(phi_terms) == pytest.approx(-2.0)
        assert phi_terms[-1] == pytest.approx(-6.0)
        assert info["reward_components"]["phi_beta"] == 1.0

    def test_phi_none_keeps_s0_reward_path(self):
        env, schedules, _, _ = self._make()
        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.NextRound))
        rc = info["reward_components"]
        expected = rc["win"] + schedules.blend_beta * rc["blind_bonus"] + rc["count_bonus"]
        assert reward == pytest.approx(expected)
        assert "phi_term" not in rc


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


class TestNormalizedEntropy:
    def test_logs_normalized_entropy(self):
        schedules = TrainingSchedules()
        model, _ = build_model(
            win_ante=1,
            schedules=schedules,
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        model.learn(
            total_timesteps=16,
            callback=[ScheduleCallback(schedules), NormalizedEntropyCallback()],
        )

        assert "rollout/normalized_entropy" in model.logger.name_to_value
        normalized_entropy = model.logger.name_to_value["rollout/normalized_entropy"]
        assert np.isfinite(normalized_entropy)
        assert 0.0 <= normalized_entropy <= 1.0
        assert "rollout/mean_legal_actions" in model.logger.name_to_value
        mean_legal_actions = model.logger.name_to_value["rollout/mean_legal_actions"]
        assert mean_legal_actions >= 1.0


class TestS1Wiring:
    def test_load_hand_policy_threads_money_ordering(self, monkeypatch, tmp_path):
        captured = {}

        class FakeHandCheckpointPolicy:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "jackdaw.agents.hand_checkpoint_policy.HandCheckpointPolicy",
            FakeHandCheckpointPolicy,
        )
        path = tmp_path / "hand.pt"
        result = load_hand_policy(path, money_aware_ordering=True)

        assert isinstance(result, FakeHandCheckpointPolicy)
        assert captured == {
            "args": (str(path),),
            "kwargs": {"money_aware_ordering": True},
        }

    def test_partner_money_ordering_requires_hand_policy(self):
        with pytest.raises(SystemExit):
            parse_args(["--partner-money-ordering"])

    def test_fresh_s1_model_has_widened_spaces(self):
        model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
            s1_schema=True,
        )
        assert model.action_space.n == NUM_TOTAL_ACTIONS_S1
        assert model.policy.observation_space["jokers"].shape[0] == 15

    def test_s0_checkpoint_is_auto_widened_weight_preserving(self, tmp_path):
        old_model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        checkpoint = tmp_path / "s0.zip"
        old_model.save(str(checkpoint))
        old_weight = old_model.policy.action_net.weight.detach().clone()
        old_bias = old_model.policy.action_net.bias.detach().clone()

        new_model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(blend_beta0=0.0),
            reservoir=ShopReservoir(seed=0),
            init_from=checkpoint,
            seed=0,
            n_envs=2,
            n_steps=8,
            batch_size=8,
            device="cpu",
            s1_schema=True,
        )
        assert new_model.action_space.n == NUM_TOTAL_ACTIONS_S1
        assert torch.equal(new_model.policy.action_net.weight[:NUM_TOTAL_ACTIONS], old_weight)
        assert torch.equal(new_model.policy.action_net.bias[:NUM_TOTAL_ACTIONS], old_bias)

    def test_s1_checkpoint_resumes_without_widening(self, tmp_path):
        model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
            s1_schema=True,
        )
        checkpoint = tmp_path / "s1.zip"
        model.save(str(checkpoint))
        resumed, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            init_from=checkpoint,
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
            s1_schema=True,
        )
        assert resumed.action_space.n == NUM_TOTAL_ACTIONS_S1

    def test_init_temperature_raises_entropy_and_keeps_the_ranking(self, tmp_path):
        """Softening must flatten the softmax without reordering preferences.

        The warm start's learned ranking is the asset worth carrying to the
        next horizon; only its confidence is pathological.
        """

        model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
            s1_schema=True,
        )
        # A fresh head is near-uniform; saturate it so the test starts from
        # the collapsed state a converged stage actually hands over.
        with torch.no_grad():
            model.policy.action_net.weight.mul_(50.0)
            model.policy.action_net.bias.mul_(50.0)
        checkpoint = tmp_path / "collapsed.zip"
        model.save(str(checkpoint))

        obs = model.env.reset()

        def _logits(m):
            obs_tensor, _ = m.policy.obs_to_tensor(obs)
            with torch.no_grad():
                return m.policy.get_distribution(obs_tensor).distribution.logits[0]

        def _entropy(logits):
            probs = torch.softmax(logits, dim=-1)
            return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())

        load_kwargs = dict(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            init_from=checkpoint,
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
            s1_schema=True,
        )
        hot, _ = build_model(**load_kwargs)
        cool, _ = build_model(**load_kwargs, init_temperature=10.0)

        hot_logits, cool_logits = _logits(hot), _logits(cool)
        assert _entropy(cool_logits) > _entropy(hot_logits)
        # Uniform scaling is order-preserving, so argsort is identical.
        assert torch.equal(torch.argsort(hot_logits), torch.argsort(cool_logits))

    def test_init_temperature_requires_init_from(self):
        with pytest.raises(SystemExit):
            parse_args(["--init-temperature", "5.0"])

    def test_phi_requires_s1_and_replaces_nonzero_blend(self):
        with pytest.raises(SystemExit):
            parse_args(["--phi-checkpoint", "critic.zip"])
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--s1-schema",
                    "--phi-checkpoint",
                    "critic.zip",
                    "--blend-beta0",
                    "0.1",
                ]
            )

    def test_real_s0_phi_runs_on_s1_env(self, tmp_path):
        s0_model, _ = build_model(
            win_ante=1,
            schedules=TrainingSchedules(),
            reservoir=ShopReservoir(seed=0),
            seed=0,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        checkpoint = tmp_path / "s0_phi.zip"
        s0_model.save(str(checkpoint))
        phi = S0CriticPhi(checkpoint)
        env = make_train_env(
            1,
            TrainingSchedules(blend_beta0=0.0),
            CountBonus(),
            None,
            n_envs=1,
            harvest_prob=0.0,
            s1_schema=True,
            phi=phi,
        )
        obs = env.reset()
        assert obs is not None
        for _ in range(3):
            _, _, dones, infos = env.step([shop_action(ShopActionFamily.NextRound)])
            if np.isfinite(infos[0]["reward_components"]["phi_term"]):
                break
            if dones[0]:
                break
        assert np.isfinite(infos[0]["reward_components"]["phi_term"])

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

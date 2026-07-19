"""Tests for HandCheckpointPolicy — the trained-checkpoint hand partner.

Covers both checkpoint kinds (BC .pt, MaskablePPO .zip), determinism, that
it produces engine-legal actions, drives real hand phases end-to-end, plugs
into ShopRunAdapter as a drop-in for GreedyHandPolicy, and degrades without
raising on a >8-card (Serpent over-draw) hand.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import jackdaw.agents.hand_checkpoint_policy as hand_checkpoint_policy  # noqa: E402
from jackdaw.agents.hand_action_space import combo_to_action, legal_action_mask  # noqa: E402
from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy  # noqa: E402
from jackdaw.agents.hand_pointer_head import HandPointerBCModel  # noqa: E402
from jackdaw.agents.hand_policy import HandPlayBCModel  # noqa: E402
from jackdaw.agents.pointer_ppo_policy import _action_vector_from_decode  # noqa: E402
from jackdaw.engine.actions import (  # noqa: E402
    Discard,
    GamePhase,
    PlayHand,
    SelectBlind,
)
from jackdaw.engine.card_factory import create_playing_card  # noqa: E402
from jackdaw.engine.data.enums import Rank, Suit  # noqa: E402
from jackdaw.engine.game import step  # noqa: E402
from jackdaw.engine.run_init import initialize_run  # noqa: E402
from jackdaw.env.action_space import ActionType  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import (  # noqa: E402
    HandPlayGymEnv,
    build_observation_v2,
    observation_space,
    observation_space_v2,
    pointer_action_to_engine_action,
)
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunAdapter, ShopRunConfig  # noqa: E402


@pytest.fixture(scope="module")
def bc_checkpoint(tmp_path_factory):
    """A BC checkpoint with untrained-but-real weights (inference plumbing
    doesn't depend on training quality)."""
    torch.manual_seed(0)
    model = HandPlayBCModel(observation_space())
    path = tmp_path_factory.mktemp("hcp") / "bc_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": {}}, path)
    return path


@pytest.fixture(scope="module")
def ppo_checkpoint(tmp_path_factory):
    from sb3_contrib import MaskablePPO

    from jackdaw.agents.hand_policy import HandPlayFeaturesExtractor

    model = MaskablePPO(
        "MultiInputPolicy",
        HandPlayGymEnv(config=HandPlayConfig()),
        n_steps=8,
        batch_size=8,
        device="cpu",
        policy_kwargs=dict(features_extractor_class=HandPlayFeaturesExtractor, net_arch=[]),
    )
    path = tmp_path_factory.mktemp("hcp") / "hand_ppo.zip"
    model.save(str(path))
    return path


@pytest.fixture(scope="module")
def pointer_bc_checkpoint(tmp_path_factory):
    torch.manual_seed(1)
    model = HandPointerBCModel(observation_space_v2())
    path = tmp_path_factory.mktemp("hcp_pointer") / "pointer_checkpoint.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": {"head": "pointer"}}, path
    )
    return path


def _selecting_hand_state(seed: str = "HCP_SMOKE") -> dict:
    gs = initialize_run("b_red", 1, seed)
    step(gs, SelectBlind())
    assert gs["phase"] == GamePhase.SELECTING_HAND
    return gs


def _assert_legal(action, gs) -> None:
    assert isinstance(action, (PlayHand, Discard))
    idx = action.card_indices
    assert 1 <= len(idx) <= 5
    assert all(0 <= i < len(gs["hand"]) for i in idx)
    cr = gs["current_round"]
    mask = legal_action_mask(len(gs["hand"]), cr["hands_left"], cr["discards_left"])
    at = ActionType.PlayHand if isinstance(action, PlayHand) else ActionType.Discard
    assert mask[combo_to_action(at, idx)]


class TestBCCheckpoint:
    def test_returns_legal_action(self, bc_checkpoint):
        policy = HandCheckpointPolicy(bc_checkpoint)
        gs = _selecting_hand_state()
        _assert_legal(policy(gs), gs)

    def test_deterministic(self, bc_checkpoint):
        policy = HandCheckpointPolicy(bc_checkpoint)
        gs = _selecting_hand_state()
        first = policy(gs)
        for _ in range(3):
            again = policy(gs)
            assert type(again) is type(first)
            assert again.card_indices == first.card_indices

    def test_drives_real_hand_phase(self, bc_checkpoint):
        policy = HandCheckpointPolicy(bc_checkpoint)
        gs = _selecting_hand_state()
        for _ in range(16):  # hands + discards budget, generous
            if gs["phase"] != GamePhase.SELECTING_HAND:
                break
            step(gs, policy(gs))
        else:
            raise AssertionError("policy loop did not terminate the hand phase")
        # An untrained policy needn't clear; it must exhaust the phase.
        assert gs["phase"] in (GamePhase.ROUND_EVAL, GamePhase.GAME_OVER)

    def test_handles_oversized_hand(self, bc_checkpoint):
        # Simulate a Serpent over-draw: a 9-card hand. Must not raise, and
        # must reference only the (representable) first-8 positions.
        policy = HandCheckpointPolicy(bc_checkpoint)
        gs = _selecting_hand_state()
        gs["hand"].append(create_playing_card(Suit.SPADES, Rank.TWO))
        assert len(gs["hand"]) == 9
        action = policy(gs)
        assert isinstance(action, (PlayHand, Discard))
        assert all(i < 8 for i in action.card_indices)


class TestPPOCheckpoint:
    def test_returns_legal_action(self, ppo_checkpoint):
        policy = HandCheckpointPolicy(ppo_checkpoint)
        gs = _selecting_hand_state()
        _assert_legal(policy(gs), gs)

    def test_deterministic(self, ppo_checkpoint):
        policy = HandCheckpointPolicy(ppo_checkpoint)
        gs = _selecting_hand_state()
        first = policy(gs)
        again = policy(gs)
        assert type(again) is type(first)
        assert again.card_indices == first.card_indices


class TestPointerBCCheckpoint:
    def test_returns_ascending_in_bounds_action_with_budget(self, pointer_bc_checkpoint):
        policy = HandCheckpointPolicy(pointer_bc_checkpoint)
        gs = _selecting_hand_state("HCP_POINTER")
        action = policy(gs)
        _assert_legal(action, gs)
        assert tuple(action.card_indices) == tuple(sorted(action.card_indices))

    def test_decode_round_trip_uses_shared_engine_path(self, pointer_bc_checkpoint):
        policy = HandCheckpointPolicy(pointer_bc_checkpoint)
        gs = _selecting_hand_state("HCP_POINTER_PARITY")
        obs = build_observation_v2(gs)
        with torch.no_grad():
            batch = {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}
            action_types, picked = policy._model.decode(batch)
            vector = _action_vector_from_decode(action_types, picked)
        expected = pointer_action_to_engine_action(
            vector.squeeze(0).numpy().astype(np.int64, copy=False), gs
        )
        assert policy(gs) == expected

    def test_pointer_dispatch_and_unknown_head(self, pointer_bc_checkpoint, tmp_path):
        assert HandCheckpointPolicy(pointer_bc_checkpoint)._kind == "pointer_bc"

        legacy = tmp_path / "legacy.pt"
        torch.save({"model_state_dict": HandPlayBCModel(observation_space()).state_dict()}, legacy)
        assert HandCheckpointPolicy(legacy)._kind == "bc"

        unknown = tmp_path / "unknown.pt"
        torch.save(
            {
                "model_state_dict": {},
                "metadata": {"head": "unknown"},
            },
            unknown,
        )
        with pytest.raises(ValueError, match="unrecognized BC checkpoint head"):
            HandCheckpointPolicy(unknown)

    def test_pointer_partner_plugs_into_shop_env(self, pointer_bc_checkpoint):
        env = ShopGymEnv(
            hand_policy=HandCheckpointPolicy(pointer_bc_checkpoint),
            config=ShopRunConfig(win_ante=1, s1_schema=True),
            seed_prefix="HCP_POINTER_SHOP",
        )
        _obs, info = env.reset()
        for _ in range(4):
            if env._adapter.done:
                break
            legal = np.flatnonzero(info["action_mask"])
            assert len(legal)
            # Keep this smoke at the shop/blind boundary; the pointer partner
            # itself is exercised against real SELECTING_HAND states above.
            if env._adapter.raw_state["phase"] != GamePhase.BLIND_SELECT:
                break
            _obs, _reward, terminated, truncated, info = env.step(int(legal[-1]))
            if terminated or truncated:
                break


class TestMoneyAwareOrdering:
    def test_pointer_decoder_receives_fresh_objective(self, pointer_bc_checkpoint, monkeypatch):
        captured = []
        marker = object()

        def decode(action, game_state, ordering_objective=None):
            captured.append(ordering_objective)
            return marker

        monkeypatch.setattr(hand_checkpoint_policy, "pointer_action_to_engine_action", decode)
        policy = HandCheckpointPolicy(pointer_bc_checkpoint, money_aware_ordering=True)
        policy._infer_pointer = lambda obs: np.array([int(ActionType.Discard), 0, 8, 8, 8, 8])

        assert policy(_selecting_hand_state("HCP_MONEY_POINTER")) is marker
        assert len(captured) == 1
        assert callable(captured[0])

    def test_v1_decoder_receives_fresh_objective(self, bc_checkpoint, monkeypatch):
        captured = []
        marker = object()

        def decode(action, game_state, ordering_objective=None):
            captured.append(ordering_objective)
            return marker

        monkeypatch.setattr(hand_checkpoint_policy, "action_to_engine_action", decode)
        policy = HandCheckpointPolicy(bc_checkpoint, money_aware_ordering=True)
        policy._infer = lambda obs, mask: 0

        assert policy(_selecting_hand_state("HCP_MONEY_V1")) is marker
        assert len(captured) == 1
        assert callable(captured[0])

    def test_default_decoder_objective_is_none(self, bc_checkpoint, monkeypatch):
        captured = []

        def decode(action, game_state, ordering_objective=None):
            captured.append(ordering_objective)
            return object()

        monkeypatch.setattr(hand_checkpoint_policy, "action_to_engine_action", decode)
        policy = HandCheckpointPolicy(bc_checkpoint)
        policy._infer = lambda obs, mask: 0

        policy(_selecting_hand_state("HCP_MONEY_DEFAULT"))
        assert captured == [None]

    def test_objective_is_rebuilt_after_chips_change(self, bc_checkpoint, monkeypatch):
        captured = []

        def decode(action, game_state, ordering_objective=None):
            captured.append(ordering_objective)
            return object()

        monkeypatch.setattr(hand_checkpoint_policy, "action_to_engine_action", decode)
        policy = HandCheckpointPolicy(bc_checkpoint, money_aware_ordering=True)
        policy._infer = lambda obs, mask: 0
        game_state = _selecting_hand_state("HCP_MONEY_FRESHNESS")
        game_state["chips"] = 0
        policy(game_state)
        game_state["chips"] = game_state["blind"].chips
        policy(game_state)

        probe = SimpleNamespace(total=0, dollars_earned=4)
        assert captured[0] is not captured[1]
        assert captured[0](probe) == (0.0, 0.0, 4.0)
        assert captured[1](probe) == (1.0, 4.0, 0.0)


class TestShopAdapterPartner:
    def test_drop_in_for_greedy(self, bc_checkpoint):
        # The whole point: a checkpoint partner slots into the shop env's
        # hand_policy hook exactly like GreedyHandPolicy.
        adapter = ShopRunAdapter(HandCheckpointPolicy(bc_checkpoint), ShopRunConfig(win_ante=1))
        adapter.reset("b_red", 1, "HCP_ADAPTER")
        gs = adapter.raw_state
        # reset auto-resolves hand phases via the checkpoint until a shop
        # decision or a terminal state -- never SELECTING_HAND, never raised.
        assert (
            gs["phase"]
            in (
                GamePhase.SHOP,
                GamePhase.PACK_OPENING,
                GamePhase.GAME_OVER,
            )
            or adapter.done
        )

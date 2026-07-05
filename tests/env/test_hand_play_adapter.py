"""Tests for HandPlayAdapter — isolated hand-play episode injection.

Covers:
- reset() lands directly in SELECTING_HAND with sampled ante/hands/discards/
  money/jokers applied, skipping BLIND_SELECT and SHOP entirely
- reset() is deterministic given the same seed
- step() reuses the real engine and can run a hand-play episode to
  completion (win via ROUND_EVAL or loss via GAME_OVER)
- GameAdapter protocol compliance
"""

from __future__ import annotations

import random

import pytest

from jackdaw.agents.greedy_hand_policy import estimate_best_hand_type
from jackdaw.engine.actions import Discard, GamePhase, PlayHand
from jackdaw.engine.data.hands import HAND_ORDER
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.hand_play_adapter import (
    _SCALING_SPECS,
    HandPlayAdapter,
    HandPlayConfig,
    JokerCountBand,
)

_ALL_HAND_TYPES = {ht.value for ht in HAND_ORDER}

SEED = "TEST_HAND_PLAY_1"
BACK = "b_red"
STAKE = 1


def _random_agent_step(adapter: HandPlayAdapter) -> None:
    legal = adapter.get_legal_actions()
    assert legal, "No legal actions available"

    hand = adapter.raw_state.get("hand", [])
    action = random.choice(legal)

    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        action = PlayHand(card_indices=indices)

    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        action = Discard(card_indices=indices)

    adapter.step(action)


def test_protocol_compliance() -> None:
    adapter = HandPlayAdapter()
    assert isinstance(adapter, GameAdapter)


def test_reset_lands_in_selecting_hand() -> None:
    adapter = HandPlayAdapter()
    state = adapter.reset(BACK, STAKE, SEED)

    assert state.phase == GamePhase.SELECTING_HAND
    assert adapter.raw_state["phase"] == GamePhase.SELECTING_HAND
    assert state.hands_left >= 1
    assert state.discards_left >= 0
    assert not adapter.done


def test_reset_applies_sampled_ante_hands_discards_money() -> None:
    cfg = HandPlayConfig(
        ante_range=(3, 3),
        hands_range=(2, 2),
        discards_range=(1, 1),
        dollars_range=(20, 20),
        blind_stages=("Small",),
    )
    adapter = HandPlayAdapter(cfg)
    state = adapter.reset(BACK, STAKE, SEED)

    assert state.ante == 3
    assert state.hands_left == 2
    assert state.discards_left == 1
    assert state.dollars == 20
    assert state.blind_on_deck == "Small"


def test_reset_injects_jokers() -> None:
    cfg = HandPlayConfig(
        joker_pool=("j_joker", "j_greedy_joker", "j_lusty_joker"),
        joker_count_range=(2, 2),
    )
    adapter = HandPlayAdapter(cfg)
    adapter.reset(BACK, STAKE, SEED)

    jokers = adapter.raw_state.get("jokers", [])
    assert len(jokers) == 2


def test_reset_is_deterministic_given_same_seed() -> None:
    adapter_a = HandPlayAdapter()
    adapter_b = HandPlayAdapter()

    state_a = adapter_a.reset(BACK, STAKE, SEED)
    state_b = adapter_b.reset(BACK, STAKE, SEED)

    assert state_a == state_b
    hand_a = [(c.base.suit, c.base.rank) for c in adapter_a.raw_state["hand"]]
    hand_b = [(c.base.suit, c.base.rank) for c in adapter_b.raw_state["hand"]]
    assert hand_a == hand_b


def test_no_blind_select_or_shop_phase_reached() -> None:
    adapter = HandPlayAdapter()
    adapter.reset(BACK, STAKE, SEED)

    seen_phases = {adapter.raw_state["phase"]}
    steps = 0
    while not adapter.done and steps < 200:
        _random_agent_step(adapter)
        seen_phases.add(adapter.raw_state["phase"])
        steps += 1

    assert GamePhase.BLIND_SELECT not in seen_phases
    assert GamePhase.SHOP not in seen_phases
    assert adapter.done


def test_runs_to_win_or_loss() -> None:
    adapter = HandPlayAdapter()
    adapter.reset(BACK, STAKE, SEED)

    steps = 0
    while not adapter.done and steps < 200:
        _random_agent_step(adapter)
        steps += 1

    assert adapter.done
    phase = adapter.raw_state["phase"]
    assert phase in (GamePhase.ROUND_EVAL, GamePhase.GAME_OVER)
    assert adapter.won == (phase == GamePhase.ROUND_EVAL)


# ---------------------------------------------------------------------------
# JokerCountBand — count-first weighted sampling
# ---------------------------------------------------------------------------

_TEST_BANDS = (
    JokerCountBand(count=0, weight=0.10, ante_range=(1, 2)),
    JokerCountBand(count=1, weight=0.10, ante_range=(1, 2)),
    JokerCountBand(count=2, weight=0.10),
    JokerCountBand(count=3, weight=0.20),
    JokerCountBand(count=4, weight=0.20),
    JokerCountBand(count=5, weight=0.30, ante_range=(3, 8)),
)

_BAND_POOL = (
    "j_joker",
    "j_greedy_joker",
    "j_lusty_joker",
    "j_wrathful_joker",
    "j_gluttenous_joker",
    "j_fibonacci",
    "j_scholar",
)


def test_bands_respect_ante_restrictions() -> None:
    cfg = HandPlayConfig(joker_pool=_BAND_POOL, joker_count_bands=_TEST_BANDS)
    for i in range(120):
        adapter = HandPlayAdapter(cfg)
        state = adapter.reset(BACK, STAKE, f"BANDS_{i}")
        count = len(adapter.raw_state.get("jokers", []))
        if count in (0, 1):
            assert state.ante <= 2, f"{count}-joker state at ante {state.ante}"
        if count == 5:
            assert state.ante >= 3, f"5-joker state at ante {state.ante}"


def test_bands_weight_distribution_is_roughly_honored() -> None:
    cfg = HandPlayConfig(joker_pool=_BAND_POOL, joker_count_bands=_TEST_BANDS)
    counts: dict[int, int] = {}
    n = 400
    for i in range(n):
        adapter = HandPlayAdapter(cfg)
        adapter.reset(BACK, STAKE, f"BANDDIST_{i}")
        c = len(adapter.raw_state.get("jokers", []))
        counts[c] = counts.get(c, 0) + 1
    # Coarse sanity: the 30% band should dominate the two 10% bands.
    assert counts.get(5, 0) > counts.get(0, 0)
    assert counts.get(5, 0) > counts.get(1, 0)
    # Every configured count should actually occur.
    assert set(counts) == {0, 1, 2, 3, 4, 5}


def test_band_count_exceeding_pool_raises() -> None:
    cfg = HandPlayConfig(
        joker_pool=("j_joker", "j_scholar"),
        joker_count_bands=(JokerCountBand(count=5, weight=1.0),),
    )
    adapter = HandPlayAdapter(cfg)
    with pytest.raises(ValueError, match="joker_pool"):
        adapter.reset(BACK, STAKE, SEED)


def test_band_count_exceeding_joker_slots_raises() -> None:
    cfg = HandPlayConfig(
        joker_pool=_BAND_POOL,
        joker_count_bands=(JokerCountBand(count=6, weight=1.0),),
    )
    adapter = HandPlayAdapter(cfg)
    with pytest.raises(ValueError, match="joker_slots"):
        adapter.reset(BACK, STAKE, SEED)


def test_bands_reset_is_deterministic() -> None:
    cfg = HandPlayConfig(joker_pool=_BAND_POOL, joker_count_bands=_TEST_BANDS)
    a = HandPlayAdapter(cfg)
    b = HandPlayAdapter(cfg)
    state_a = a.reset(BACK, STAKE, "BAND_DET")
    state_b = b.reset(BACK, STAKE, "BAND_DET")
    assert state_a == state_b
    keys_a = [j.center_key for j in a.raw_state.get("jokers", [])]
    keys_b = [j.center_key for j in b.raw_state.get("jokers", [])]
    assert keys_a == keys_b


# ---------------------------------------------------------------------------
# Scaling-joker accumulation randomization
# ---------------------------------------------------------------------------


def _reset_with_joker(key: str, ante: int, seed: str) -> object:
    cfg = HandPlayConfig(
        ante_range=(ante, ante),
        joker_pool=(key,),
        joker_count_range=(1, 1),
    )
    adapter = HandPlayAdapter(cfg)
    adapter.reset(BACK, STAKE, seed)
    jokers = adapter.raw_state["jokers"]
    assert len(jokers) == 1
    return jokers[0]


def test_scaling_joker_fresh_at_ante_1() -> None:
    # (ante - 1) == 0 elapsed antes -> zero trigger opportunities -> always base.
    j = _reset_with_joker("j_ride_the_bus", 1, "SCALE_A1")
    assert j.ability.get("mult", 0) == 0


def test_scaling_joker_accumulates_within_cap_at_ante_8() -> None:
    # cap = min(10*7, 30 streak window) * 0.70 = 21
    values = set()
    for i in range(60):
        j = _reset_with_joker("j_ride_the_bus", 8, f"SCALE_A8_{i}")
        mult = j.ability.get("mult", 0)
        assert 0 <= mult <= 21
        values.add(mult)
    assert len(values) > 3, "accumulation should actually vary across seeds"


def test_xmult_scaling_joker_quantized_to_trigger_steps() -> None:
    for i in range(40):
        j = _reset_with_joker("j_hologram", 6, f"SCALE_HOLO_{i}")
        x = j.ability.get("x_mult", 1)
        steps = round((x - 1) / 0.25)
        assert abs((1 + steps * 0.25) - x) < 1e-9, f"x_mult {x} not on 0.25 grid"


def test_decay_joker_stays_alive() -> None:
    # Ice Cream: destroyed when chips would hit 0 -- injected values must
    # be states the engine could have kept on the board.
    for i in range(40):
        j = _reset_with_joker("j_ice_cream", 8, f"SCALE_ICE_{i}")
        chips = j.ability["extra"]["chips"]
        assert 0 < chips <= 100
        assert chips % 5 == 0


def test_campfire_flat_range_even_at_ante_1() -> None:
    values = set()
    for i in range(40):
        j = _reset_with_joker("j_campfire", 1, f"SCALE_CAMP_{i}")
        x = j.ability.get("x_mult", 1)
        assert 1.0 <= x <= 2.5
        values.add(x)
    assert len(values) > 1, "Campfire range is ante-independent, must vary at ante 1"


def test_randomize_joker_state_false_gives_base_jokers() -> None:
    cfg = HandPlayConfig(
        ante_range=(8, 8),
        joker_pool=("j_ride_the_bus",),
        joker_count_range=(1, 1),
        randomize_joker_state=False,
    )
    adapter = HandPlayAdapter(cfg)
    adapter.reset(BACK, STAKE, "SCALE_OFF")
    j = adapter.raw_state["jokers"][0]
    assert j.ability.get("mult", 0) == 0
    assert "skips" not in adapter.raw_state or adapter.raw_state["skips"] == 0


def test_run_stat_priors_seeded_for_formula_jokers() -> None:
    cfg = HandPlayConfig(ante_range=(8, 8))
    seen_skips = set()
    seen_tarots = set()
    for i in range(40):
        adapter = HandPlayAdapter(cfg)
        adapter.reset(BACK, STAKE, f"RUNSTAT_{i}")
        gs = adapter.raw_state
        seen_skips.add(gs["skips"])
        seen_tarots.add(gs["consumable_usage_tarot"])
        assert gs["consumable_usage_total"]["tarot"] == gs["consumable_usage_tarot"]
        assert 0 <= gs["skips"] <= 4
    assert len(seen_skips) > 1
    assert len(seen_tarots) > 1


def test_all_scaling_spec_keys_exist_and_fields_resolve() -> None:
    """Every curated spec must name a real joker and a field path that
    exists on (or is legitimately absent from) the freshly-created joker --
    guards against engine data renames drifting away from this map."""
    from jackdaw.engine.card_factory import create_joker

    for key, spec in _SCALING_SPECS.items():
        joker = create_joker(key)  # raises on unknown key
        if spec.kind == "loyalty":
            continue
        value = joker.ability
        for part in spec.field:
            if part == "caino_xmult":
                # written lazily by the handler; absent on a fresh joker
                break
            assert isinstance(value, dict) and part in value, (
                f"{key}: field path {spec.field} broken at {part!r}"
            )
            value = value[part]


# ---------------------------------------------------------------------------
# Boss round-history injection (The Eye / The Mouth)
# ---------------------------------------------------------------------------


def _force_boss(monkeypatch, boss_key: str) -> None:
    """Pin blind selection to a specific boss key regardless of ante/RNG --
    ``get_new_boss`` is imported lazily inside ``reset()``, so patching the
    module attribute before calling reset() is picked up correctly."""
    monkeypatch.setattr(
        "jackdaw.engine.blind.get_new_boss", lambda *a, **k: boss_key, raising=True
    )


def _reset_boss(monkeypatch, boss_key: str, cfg: HandPlayConfig, seed: str = SEED):
    _force_boss(monkeypatch, boss_key)
    adapter = HandPlayAdapter(cfg)
    adapter.reset(BACK, STAKE, seed)
    return adapter


def test_boss_history_noop_at_hands_played_zero(monkeypatch) -> None:
    cfg = HandPlayConfig(blind_stages=("Boss",), boss_history_hands_played_range=(0, 0))
    adapter = _reset_boss(monkeypatch, "bl_eye", cfg)
    blind = adapter.raw_state["blind"]
    assert blind.name == "The Eye"
    assert blind.hands_used == {}
    assert blind.only_hand is None


def test_eye_history_marks_k_distinct_hand_types(monkeypatch) -> None:
    cfg = HandPlayConfig(blind_stages=("Boss",), boss_history_hands_played_range=(2, 2))
    adapter = _reset_boss(monkeypatch, "bl_eye", cfg)
    blind = adapter.raw_state["blind"]
    assert len(blind.hands_used) == 2
    assert set(blind.hands_used).issubset(_ALL_HAND_TYPES)


def test_eye_history_weight_one_always_blocks_best_hand(monkeypatch) -> None:
    cfg = HandPlayConfig(
        blind_stages=("Boss",),
        boss_history_hands_played_range=(1, 1),
        boss_history_best_hand_weight=1.0,
    )
    for i in range(20):
        adapter = _reset_boss(monkeypatch, "bl_eye", cfg, seed=f"EYE_W1_{i}")
        gs = adapter.raw_state
        best_type = estimate_best_hand_type(gs["hand"], gs["jokers"])
        assert best_type in gs["blind"].hands_used


def test_eye_history_weight_zero_never_blocks_best_hand(monkeypatch) -> None:
    cfg = HandPlayConfig(
        blind_stages=("Boss",),
        boss_history_hands_played_range=(1, 1),
        boss_history_best_hand_weight=0.0,
    )
    for i in range(20):
        adapter = _reset_boss(monkeypatch, "bl_eye", cfg, seed=f"EYE_W0_{i}")
        gs = adapter.raw_state
        best_type = estimate_best_hand_type(gs["hand"], gs["jokers"])
        assert best_type not in gs["blind"].hands_used


def test_mouth_history_locks_one_hand_type(monkeypatch) -> None:
    cfg = HandPlayConfig(blind_stages=("Boss",), boss_history_hands_played_range=(1, 1))
    adapter = _reset_boss(monkeypatch, "bl_mouth", cfg)
    blind = adapter.raw_state["blind"]
    assert blind.only_hand in _ALL_HAND_TYPES


def test_mouth_history_is_uniform_not_correlated_with_best_hand(monkeypatch) -> None:
    """The Mouth's lock represents an unseen, different hand (the round's
    actual first hand) -- it must NOT be weighted toward the current hand's
    best-detectable line the way The Eye's history is."""
    cfg = HandPlayConfig(blind_stages=("Boss",), boss_history_hands_played_range=(1, 1))
    matches = 0
    n = 60
    for i in range(n):
        adapter = _reset_boss(monkeypatch, "bl_mouth", cfg, seed=f"MOUTH_UNIFORM_{i}")
        gs = adapter.raw_state
        best_type = estimate_best_hand_type(gs["hand"], gs["jokers"])
        if gs["blind"].only_hand == best_type:
            matches += 1
    # Uniform over ~12 types -> ~n/12 matches by chance; a weighted
    # implementation would saturate near n. Generous upper bound to avoid
    # flaking on the small sample while still catching a correlation bug.
    assert matches < n * 0.4, f"only_hand looks correlated with best_type ({matches}/{n})"


def test_randomize_boss_history_false_disables_feature(monkeypatch) -> None:
    cfg = HandPlayConfig(
        blind_stages=("Boss",),
        boss_history_hands_played_range=(3, 3),
        randomize_boss_history=False,
    )
    adapter = _reset_boss(monkeypatch, "bl_eye", cfg)
    blind = adapter.raw_state["blind"]
    assert blind.hands_used == {}
    assert blind.only_hand is None


def test_non_history_boss_unaffected_by_range(monkeypatch) -> None:
    cfg = HandPlayConfig(blind_stages=("Boss",), boss_history_hands_played_range=(3, 3))
    adapter = _reset_boss(monkeypatch, "bl_flint", cfg)
    blind = adapter.raw_state["blind"]
    assert blind.name == "The Flint"
    assert blind.hands_used == {}
    assert blind.only_hand is None


def test_the_ox_unaffected_by_boss_history(monkeypatch) -> None:
    """The Ox is deliberately excluded -- its debuff depends on
    HandLevels.most_played (run-cumulative usage counts), the pre-existing
    deferred gap, not Blind-instance state this feature can fake."""
    cfg = HandPlayConfig(blind_stages=("Boss",), boss_history_hands_played_range=(3, 3))
    adapter = _reset_boss(monkeypatch, "bl_ox", cfg)
    blind = adapter.raw_state["blind"]
    assert blind.name == "The Ox"
    assert blind.hands_used == {}
    assert blind.only_hand is None

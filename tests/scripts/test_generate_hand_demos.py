"""Tests for the offline demonstration-generation pipeline
(`scripts/generate_hand_demos.py`).

Covers the correctness-critical, easy-to-get-backwards pieces identified
during design: mapping the solver's `AnteClearChoice` back to an action
label (the `hold` field means different things for "play" vs "discard"
actions), identity-based (not value-equality) index recovery, padding/mask
shapes, shard round-tripping, and the seed-partitioning math that makes the
dataset reproducible independent of worker count.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from generate_hand_demos import (
    DEFAULT_COUNT_BANDS,
    MAX_JOKERS_V2,
    STAGE2_JOKER_POOL,
    Example,
    GenerationError,
    _worker_run,
    acted_cards_from_choice,
    all_joker_keys,
    generate_one_example,
    indices_by_identity,
    load_dollar_marginals,
    partition_indices,
    stage_presets,
    write_shard,
)
from hand_solver import AnteClearChoice

from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.env.action_space import ActionType
from jackdaw.env.hand_play_adapter import HandPlayConfig
from jackdaw.env.hand_play_gym import MAX_CONSUMABLES_V2
from jackdaw.env.observation import D_CONSUMABLE, D_HAND_CARD, D_HAND_GLOBAL

HAND_WIDTH = 8


def _cards(n: int) -> list:
    ranks = list(Rank)
    suits = list(Suit)
    return [create_playing_card(suits[i % 4], ranks[i % len(ranks)]) for i in range(n)]


def _example_extras(hand_width: int = HAND_WIDTH) -> dict:
    """The v2 arrays every Example carries. Synthetic-Example tests keep
    using tiny shapes for the float blocks they assert on; these just need
    to exist and round-trip."""
    return dict(
        joker_ids=np.zeros(MAX_JOKERS_V2, dtype=np.int64),
        copy_active=np.zeros(MAX_JOKERS_V2, dtype=np.float32),
        copy_target_ids=np.zeros(MAX_JOKERS_V2, dtype=np.int64),
        trigger_match=np.zeros((hand_width, MAX_JOKERS_V2, 2), dtype=bool),
        consumables=np.zeros((MAX_CONSUMABLES_V2, D_CONSUMABLE), dtype=np.float32),
        consumable_mask=np.zeros(MAX_CONSUMABLES_V2, dtype=bool),
    )


# ---------------------------------------------------------------------------
# acted_cards_from_choice
# ---------------------------------------------------------------------------


def test_acted_cards_for_play_action_is_hold_not_discard() -> None:
    played = _cards(3)
    choice = AnteClearChoice(
        action="play",
        template_name=None,
        hold=played,
        discard=[],
        p_clear=1.0,
        immediate_value=10.0,
    )
    assert acted_cards_from_choice(choice) is played


def test_acted_cards_for_discard_action_is_discard_not_hold() -> None:
    kept = _cards(3)
    tossed = _cards(2)
    choice = AnteClearChoice(
        action="discard",
        template_name="flush_Hearts",
        hold=kept,
        discard=tossed,
        p_clear=0.8,
        immediate_value=5.0,
    )
    assert acted_cards_from_choice(choice) is tossed


def test_acted_cards_raises_on_unknown_action() -> None:
    choice = AnteClearChoice(
        action="fold",
        template_name=None,
        hold=[],
        discard=[],
        p_clear=0.0,
        immediate_value=0.0,
    )
    with pytest.raises(GenerationError):
        acted_cards_from_choice(choice)


# ---------------------------------------------------------------------------
# indices_by_identity
# ---------------------------------------------------------------------------


def test_indices_by_identity_basic() -> None:
    hand = _cards(5)
    selected = [hand[3], hand[0]]
    assert indices_by_identity(selected, hand) == [3, 0]


def test_indices_by_identity_handles_duplicate_valued_cards() -> None:
    """Two cards with identical field values (Erratic-deck style) must
    still resolve to their own distinct positions, not collide."""
    a = create_playing_card(Suit.HEARTS, Rank.TWO)
    a_dup = create_playing_card(Suit.HEARTS, Rank.TWO)  # same value, different object
    b = create_playing_card(Suit.SPADES, Rank.KING)
    hand = [a, a_dup, b]

    assert indices_by_identity([a_dup], hand) == [1]
    assert indices_by_identity([a], hand) == [0]


def test_indices_by_identity_raises_for_foreign_card() -> None:
    hand = _cards(3)
    foreign = create_playing_card(Suit.CLUBS, Rank.ACE)
    with pytest.raises(GenerationError):
        indices_by_identity([foreign], hand)


# ---------------------------------------------------------------------------
# generate_one_example (exercises the real solver, kept cheap)
# ---------------------------------------------------------------------------


def test_generate_one_example_shapes_and_valid_action() -> None:
    # No discards allowed -- skips the expensive recursive discard-chain
    # search, keeping this test fast while still exercising the full
    # sample -> solve -> label pipeline end to end.
    config = HandPlayConfig(discards_range=(0, 0), hands_range=(1, 1))
    example = generate_one_example("PIPELINE_SMOKE_TEST", config)

    assert isinstance(example, Example)
    assert example.global_context.shape == (D_HAND_GLOBAL,)
    assert example.hand_cards.shape == (len(example.hand_mask), D_HAND_CARD)
    assert example.hand_mask.shape == (len(example.hand_cards),)
    assert example.jokers.shape[0] == MAX_JOKERS_V2
    assert example.joker_mask.shape == (MAX_JOKERS_V2,)
    assert example.joker_ids.shape == (MAX_JOKERS_V2,)
    assert example.copy_active.shape == (MAX_JOKERS_V2,)
    assert example.copy_target_ids.shape == (MAX_JOKERS_V2,)
    assert example.trigger_match.shape == (len(example.hand_cards), MAX_JOKERS_V2, 2)
    assert example.consumables.shape == (MAX_CONSUMABLES_V2, D_CONSUMABLE)
    # Stages 1-4 inject no consumables; the block is real but empty here.
    assert example.consumable_mask.sum() == 0
    assert example.card_indices.shape == (5,)
    assert example.action_type in (int(ActionType.PlayHand), int(ActionType.Discard))
    # No discards possible -> the solver must recommend playing.
    assert example.action_type == int(ActionType.PlayHand)
    # At least one card selected, and only within the real (unpadded) hand.
    selected = example.card_indices[example.card_indices >= 0]
    assert 1 <= len(selected) <= 5
    assert selected.tolist() == sorted(selected.tolist())
    assert selected[-1] < len(example.hand_mask)
    assert np.all(example.card_indices[len(selected) :] == -1)


# ---------------------------------------------------------------------------
# write_shard round-trip
# ---------------------------------------------------------------------------


def test_write_shard_round_trip(tmp_path) -> None:
    examples = [
        Example(
            global_context=np.full(5, float(i), dtype=np.float32),
            hand_cards=np.zeros((HAND_WIDTH, 3), dtype=np.float32),
            hand_mask=np.zeros(HAND_WIDTH, dtype=bool),
            jokers=np.zeros((MAX_JOKERS_V2, 2), dtype=np.float32),
            joker_mask=np.zeros(MAX_JOKERS_V2, dtype=bool),
            action_type=int(ActionType.PlayHand),
            card_indices=np.array([0, -1, -1, -1, -1], dtype=np.int64),
            p_clear=0.5 + i,
            seed=f"SEED_{i}",
            **_example_extras(),
        )
        for i in range(3)
    ]
    path = tmp_path / "shard_00000.npz"
    write_shard(path, examples)

    loaded = np.load(path, allow_pickle=False)
    assert loaded["global_context"].shape == (3, 5)
    assert loaded["p_clear"].tolist() == pytest.approx([0.5, 1.5, 2.5])
    assert list(loaded["seed"]) == ["SEED_0", "SEED_1", "SEED_2"]
    assert loaded["schema_version"][0] == 3
    assert loaded["trigger_match"].shape == (3, HAND_WIDTH, MAX_JOKERS_V2, 2)
    assert loaded["card_indices"].shape == (3, 5)
    assert "card_target_mask" not in loaded.files
    assert loaded["joker_ids"].dtype == np.int64
    assert loaded["consumables"].shape == (3, MAX_CONSUMABLES_V2, D_CONSUMABLE)


def test_write_shard_uses_its_widest_actual_hand(tmp_path) -> None:
    def example_with_hand_width(width: int) -> Example:
        return Example(
            global_context=np.zeros(1, dtype=np.float32),
            hand_cards=np.zeros((width, D_HAND_CARD), dtype=np.float32),
            hand_mask=np.ones(width, dtype=bool),
            jokers=np.zeros((MAX_JOKERS_V2, 1), dtype=np.float32),
            joker_mask=np.zeros(MAX_JOKERS_V2, dtype=bool),
            action_type=int(ActionType.PlayHand),
            card_indices=np.array([8, -1, -1, -1, -1], dtype=np.int64),
            p_clear=1.0,
            seed=f"WIDE_{width}",
            **_example_extras(width),
        )

    path = tmp_path / "wide_shard.npz"
    write_shard(path, [example_with_hand_width(9), example_with_hand_width(11)])
    loaded = np.load(path, allow_pickle=False)
    assert loaded["hand_cards"].shape == (2, 11, D_HAND_CARD)
    assert loaded["hand_mask"].sum(axis=1).tolist() == [9, 11]
    assert loaded["trigger_match"].shape == (2, 11, MAX_JOKERS_V2, 2)


# ---------------------------------------------------------------------------
# partition_indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "num_workers"),
    [(100, 4), (100, 3), (1, 8), (0, 4), (7, 7), (10, 1)],
)
def test_partition_indices_covers_every_index_exactly_once(total, num_workers) -> None:
    ranges = partition_indices(total, num_workers)
    covered: list[int] = []
    for start, end in ranges:
        assert start < end
        covered.extend(range(start, end))
    assert covered == list(range(total))


def test_partition_indices_independent_of_worker_count_for_seed_set() -> None:
    """The union of indices covered must be identical regardless of how
    many workers the same total is split across -- this is what makes the
    seed scheme reproducible across reruns with a different --num-workers.
    """
    total = 53
    for num_workers in (1, 2, 5, 10, 53, 100):
        ranges = partition_indices(total, num_workers)
        covered = set()
        for start, end in ranges:
            covered.update(range(start, end))
        assert covered == set(range(total))


# ---------------------------------------------------------------------------
# _worker_run (in-process, no actual subprocess spawn)
# ---------------------------------------------------------------------------


def test_worker_run_writes_shards_and_logs_failures(tmp_path, monkeypatch) -> None:
    import generate_hand_demos as gen_mod

    def fake_generate(seed: str, config) -> Example:
        if seed.endswith("3"):
            raise RuntimeError("simulated solver failure")
        return Example(
            global_context=np.zeros(1, dtype=np.float32),
            hand_cards=np.zeros((HAND_WIDTH, 1), dtype=np.float32),
            hand_mask=np.zeros(HAND_WIDTH, dtype=bool),
            jokers=np.zeros((MAX_JOKERS_V2, 1), dtype=np.float32),
            joker_mask=np.zeros(MAX_JOKERS_V2, dtype=bool),
            action_type=int(ActionType.PlayHand),
            card_indices=np.array([0, -1, -1, -1, -1], dtype=np.int64),
            p_clear=1.0,
            seed=seed,
            **_example_extras(),
        )

    monkeypatch.setattr(gen_mod, "generate_one_example", fake_generate)

    out_dir = tmp_path / "stage_test"
    _worker_run(
        worker_id=0,
        start_idx=0,
        end_idx=5,
        stage_name="unit_test",
        config=HandPlayConfig(),
        output_dir=out_dir,
        shard_size=2,
    )

    shard_files = sorted(out_dir.glob("worker_000_shard_*.npz"))
    total_examples = 0
    for path in shard_files:
        data = np.load(path)
        total_examples += data["seed"].shape[0]
    # indices 0,1,2,4 succeed (index 3 -> seed ending "3" fails); 4 successes.
    assert total_examples == 4

    failures_path = out_dir / "worker_000_failures.jsonl"
    lines = failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    failure = json.loads(lines[0])
    assert failure["seed"] == "unit_test_00000003"
    assert "simulated solver failure" in failure["error"]


# ---------------------------------------------------------------------------
# Curriculum stage presets
# ---------------------------------------------------------------------------


def test_default_count_bands_weights_sum_to_one() -> None:
    assert sum(b.weight for b in DEFAULT_COUNT_BANDS) == pytest.approx(1.0)


def test_default_count_bands_encode_agreed_restrictions() -> None:
    by_count = {b.count: b for b in DEFAULT_COUNT_BANDS}
    assert set(by_count) == {0, 1, 2, 3, 4, 5}
    # 0/1-joker states confined to antes 1-2; 5-joker boards to ante 3+.
    assert by_count[0].ante_range == (1, 2)
    assert by_count[1].ante_range == (1, 2)
    assert by_count[5].ante_range == (3, 8)
    assert by_count[5].weight == pytest.approx(0.30)


def test_stage2_pool_keys_all_exist_in_engine_data() -> None:
    valid = set(all_joker_keys())
    unknown = [k for k in STAGE2_JOKER_POOL if k not in valid]
    assert not unknown, f"stage 2 pool names unknown joker keys: {unknown}"


def test_stage3_pool_is_all_150_jokers() -> None:
    keys = all_joker_keys()
    assert len(keys) == 150
    presets = stage_presets()
    assert presets["stage3_full"].config.joker_pool == keys


def test_stage4_boss_preset_is_boss_only_full_pool() -> None:
    presets = stage_presets()
    cfg = presets["stage4_boss"].config
    assert cfg.blind_stages == ("Boss",)
    assert cfg.joker_pool == all_joker_keys()
    assert cfg.joker_count_bands == DEFAULT_COUNT_BANDS
    assert presets["stage4_boss"].total_examples == 8_000


def test_stage4_boss_preset_always_samples_a_boss(monkeypatch) -> None:
    """blind_stages=("Boss",) means sampler.choice has only one option --
    every reset() must land on a boss, never Small/Big."""
    from jackdaw.env.hand_play_adapter import HandPlayAdapter

    cfg = stage_presets()["stage4_boss"].config
    for i in range(10):
        adapter = HandPlayAdapter(cfg)
        state = adapter.reset("b_red", 1, f"STAGE4_SMOKE_{i}")
        assert state.blind_on_deck == "Boss"
        assert adapter.raw_state["blind"].boss is True


def test_stage_presets_are_generatable() -> None:
    """Every preset must produce a valid HandPlayConfig that reset() accepts
    (band counts within pool and joker_slots) — catches preset drift."""
    from jackdaw.env.hand_play_adapter import HandPlayAdapter

    for name, preset in stage_presets().items():
        adapter = HandPlayAdapter(preset.config)
        adapter.reset("b_red", 1, f"PRESET_SMOKE_{name}")
        assert preset.total_examples > 0


# ---------------------------------------------------------------------------
# Resume after interruption
# ---------------------------------------------------------------------------


def _fake_example(seed: str) -> Example:
    return Example(
        global_context=np.zeros(1, dtype=np.float32),
        hand_cards=np.zeros((HAND_WIDTH, 1), dtype=np.float32),
        hand_mask=np.zeros(HAND_WIDTH, dtype=bool),
        jokers=np.zeros((MAX_JOKERS_V2, 1), dtype=np.float32),
        joker_mask=np.zeros(MAX_JOKERS_V2, dtype=bool),
        action_type=int(ActionType.PlayHand),
        card_indices=np.array([0, -1, -1, -1, -1], dtype=np.int64),
        p_clear=1.0,
        seed=seed,
        **_example_extras(),
    )


def test_worker_resumes_after_interruption(tmp_path, monkeypatch) -> None:
    """Killing a run and relaunching with identical settings must continue
    after the last stored index — no duplicated seeds, no skipped indices,
    and shard numbering continues rather than overwriting."""
    import generate_hand_demos as gen_mod

    calls: list[str] = []

    def fake_generate(seed: str, config) -> Example:
        calls.append(seed)
        return _fake_example(seed)

    monkeypatch.setattr(gen_mod, "generate_one_example", fake_generate)
    out_dir = tmp_path / "stage_resume"

    # First (interrupted) run: covers indices 0-4 with shard_size 2 -> two
    # full shards (0-1, 2-3) plus index 4 flushed as a final short shard.
    _worker_run(0, 0, 5, "resume_test", HandPlayConfig(), out_dir, shard_size=2)
    first_run_calls = list(calls)
    assert [s[-1] for s in first_run_calls] == ["0", "1", "2", "3", "4"]

    # Relaunch the full range 0-10: must resume at index 5, not redo 0-4.
    calls.clear()
    _worker_run(0, 0, 10, "resume_test", HandPlayConfig(), out_dir, shard_size=2)
    assert [s[-1] for s in calls] == ["5", "6", "7", "8", "9"]

    # All shards together: every index exactly once, no overwrites.
    all_seeds: list[str] = []
    for path in sorted(out_dir.glob("worker_000_shard_*.npz")):
        all_seeds.extend(str(s) for s in np.load(path)["seed"])
    assert sorted(all_seeds) == [f"resume_test_{i:08d}" for i in range(10)]


def test_resume_ignores_seeds_outside_worker_range(tmp_path, monkeypatch) -> None:
    """Seeds from a run with different partitioning must not gate resume."""
    import generate_hand_demos as gen_mod

    monkeypatch.setattr(gen_mod, "generate_one_example", lambda s, c: _fake_example(s))
    out_dir = tmp_path / "stage_repart"

    # Simulate an old shard for worker 0 holding an out-of-range index (50).
    out_dir.mkdir(parents=True)
    write_shard(out_dir / "worker_000_shard_00000.npz", [_fake_example("repart_test_00000050")])

    from generate_hand_demos import _resume_point

    resume_idx, next_shard = _resume_point(out_dir, 0, 0, 10)
    assert resume_idx == 0  # index 50 is outside [0, 10) -> ignored
    assert next_shard == 1  # but numbering still continues past the file


# ---------------------------------------------------------------------------
# Generation-time label executability validation (added after the 5-card
# discard-cap bug -- see CLAUDE.md open items)
# ---------------------------------------------------------------------------


class TestValidateLabelExecutability:
    """Tier 1 (schema-native) + tier 2 (real engine execution) guards."""

    def _adapter(self, discards: tuple[int, int] = (1, 3)):
        from jackdaw.env.hand_play_adapter import HandPlayAdapter

        adapter = HandPlayAdapter(HandPlayConfig(discards_range=discards))
        adapter.reset("b_red", 1, "validate_test_0001")
        return adapter

    def test_valid_play_executes_through_engine(self) -> None:
        from generate_hand_demos import validate_label_executability

        adapter = self._adapter()
        hands_before = adapter.raw_state["current_round"]["hands_left"]
        validate_label_executability(adapter, ActionType.PlayHand, [0, 1, 2])
        # Tier 2 really ran: the engine consumed a hand.
        assert adapter.raw_state["current_round"]["hands_played"] >= 1 or (
            adapter.raw_state["current_round"]["hands_left"] < hands_before
        )

    def test_valid_discard_executes_through_engine(self) -> None:
        from generate_hand_demos import validate_label_executability

        adapter = self._adapter(discards=(2, 3))
        discards_before = adapter.raw_state["current_round"]["discards_left"]
        validate_label_executability(adapter, ActionType.Discard, [0, 4])
        assert adapter.raw_state["current_round"]["discards_left"] == discards_before - 1

    def test_six_card_discard_rejected_at_schema_tier(self) -> None:
        from generate_hand_demos import GenerationError, validate_label_executability

        adapter = self._adapter(discards=(2, 3))
        with pytest.raises(GenerationError, match="1-5 cards"):
            validate_label_executability(adapter, ActionType.Discard, [0, 1, 2, 3, 4, 5])

    def test_discard_with_no_budget_rejected(self) -> None:
        from generate_hand_demos import GenerationError, validate_label_executability

        adapter = self._adapter(discards=(0, 0))
        with pytest.raises(GenerationError, match="no discards"):
            validate_label_executability(adapter, ActionType.Discard, [0])

    def test_out_of_range_index_rejected(self) -> None:
        from generate_hand_demos import GenerationError, validate_label_executability

        adapter = self._adapter()
        with pytest.raises(GenerationError, match="outside actual hand length"):
            validate_label_executability(adapter, ActionType.PlayHand, [0, 9])

    def test_big_hand_index_passes_tier_one_and_executes_through_engine(self) -> None:
        from generate_hand_demos import validate_label_executability

        from jackdaw.env.hand_play_adapter import HandPlayAdapter

        adapter = HandPlayAdapter(
            HandPlayConfig(
                hands_range=(1, 1),
                discards_range=(0, 0),
                hand_size_delta_range=(4, 4),
                hand_size_tail_prob=1.0,
            )
        )
        adapter.reset("b_red", 1, "wide_label_validation")
        assert len(adapter.raw_state["hand"]) > 8

        hands_before = adapter.raw_state["current_round"]["hands_left"]
        validate_label_executability(adapter, ActionType.PlayHand, [8])
        assert adapter.raw_state["current_round"]["hands_left"] < hands_before


# ---------------------------------------------------------------------------
# h1 regen config wiring: dollar-marginal loading + preset hand-size tail
# ---------------------------------------------------------------------------


class TestRegenConfigWiring:
    def test_load_dollar_marginals_reductions_format(self, tmp_path):
        path = tmp_path / "reductions.json"
        path.write_text(
            json.dumps(
                {
                    "dollar_marginals_by_ante": {"1": {"4": 2, "9": 1}, "2": {}},
                    "hand_size_histogram": {"8": 100},
                    "note": "x",
                }
            ),
            encoding="utf-8",
        )
        marginals = load_dollar_marginals(path)
        # string keys -> ints; the empty ante-2 histogram is dropped
        assert marginals == {1: {4: 2, 9: 1}}

    def test_load_dollar_marginals_bare_format(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"3": {"0": 5}}), encoding="utf-8")
        assert load_dollar_marginals(path) == {3: {0: 5}}

    def test_load_dollar_marginals_empty_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"dollar_marginals_by_ante": {}}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_dollar_marginals(path)

    def test_stage_presets_carry_hand_size_tail(self):
        """Every stage preset bakes in the user-locked flat hand-size tail
        (prob 0.1, delta uniform +1..+4) — width-40 / Candidate-B coverage."""
        for name, preset in stage_presets().items():
            assert preset.config.hand_size_tail_prob == 0.1, name
            assert preset.config.hand_size_delta_range == (1, 4), name

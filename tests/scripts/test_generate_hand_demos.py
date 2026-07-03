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
    MAX_HAND_CARDS,
    MAX_JOKERS,
    Example,
    GenerationError,
    acted_cards_from_choice,
    generate_one_example,
    indices_by_identity,
    partition_indices,
    write_shard,
    _worker_run,
)
from hand_solver import AnteClearChoice

from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.env.action_space import ActionType
from jackdaw.env.hand_play_adapter import HandPlayConfig


def _cards(n: int) -> list:
    ranks = list(Rank)
    suits = list(Suit)
    return [create_playing_card(suits[i % 4], ranks[i % len(ranks)]) for i in range(n)]


# ---------------------------------------------------------------------------
# acted_cards_from_choice
# ---------------------------------------------------------------------------


def test_acted_cards_for_play_action_is_hold_not_discard() -> None:
    played = _cards(3)
    choice = AnteClearChoice(
        action="play", template_name=None, hold=played, discard=[], p_clear=1.0,
        immediate_value=10.0,
    )
    assert acted_cards_from_choice(choice) is played


def test_acted_cards_for_discard_action_is_discard_not_hold() -> None:
    kept = _cards(3)
    tossed = _cards(2)
    choice = AnteClearChoice(
        action="discard", template_name="flush_Hearts", hold=kept, discard=tossed,
        p_clear=0.8, immediate_value=5.0,
    )
    assert acted_cards_from_choice(choice) is tossed


def test_acted_cards_raises_on_unknown_action() -> None:
    choice = AnteClearChoice(
        action="fold", template_name=None, hold=[], discard=[], p_clear=0.0,
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
    assert example.hand_cards.shape[0] == MAX_HAND_CARDS
    assert example.hand_mask.shape == (MAX_HAND_CARDS,)
    assert example.jokers.shape[0] == MAX_JOKERS
    assert example.joker_mask.shape == (MAX_JOKERS,)
    assert example.card_target_mask.shape == (MAX_HAND_CARDS,)
    assert example.action_type in (int(ActionType.PlayHand), int(ActionType.Discard))
    # No discards possible -> the solver must recommend playing.
    assert example.action_type == int(ActionType.PlayHand)
    # At least one card selected, and only within the real (unpadded) hand.
    assert 1 <= example.card_target_mask.sum() <= 5
    assert example.card_target_mask[example.hand_mask.sum() :].sum() == 0


# ---------------------------------------------------------------------------
# write_shard round-trip
# ---------------------------------------------------------------------------


def test_write_shard_round_trip(tmp_path) -> None:
    examples = [
        Example(
            global_context=np.full(5, float(i), dtype=np.float32),
            hand_cards=np.zeros((MAX_HAND_CARDS, 3), dtype=np.float32),
            hand_mask=np.zeros(MAX_HAND_CARDS, dtype=bool),
            jokers=np.zeros((MAX_JOKERS, 2), dtype=np.float32),
            joker_mask=np.zeros(MAX_JOKERS, dtype=bool),
            action_type=int(ActionType.PlayHand),
            card_target_mask=np.zeros(MAX_HAND_CARDS, dtype=bool),
            p_clear=0.5 + i,
            seed=f"SEED_{i}",
        )
        for i in range(3)
    ]
    path = tmp_path / "shard_00000.npz"
    write_shard(path, examples)

    loaded = np.load(path, allow_pickle=False)
    assert loaded["global_context"].shape == (3, 5)
    assert loaded["p_clear"].tolist() == pytest.approx([0.5, 1.5, 2.5])
    assert list(loaded["seed"]) == ["SEED_0", "SEED_1", "SEED_2"]
    assert loaded["schema_version"][0] == 1


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
            hand_cards=np.zeros((MAX_HAND_CARDS, 1), dtype=np.float32),
            hand_mask=np.zeros(MAX_HAND_CARDS, dtype=bool),
            jokers=np.zeros((MAX_JOKERS, 1), dtype=np.float32),
            joker_mask=np.zeros(MAX_JOKERS, dtype=bool),
            action_type=int(ActionType.PlayHand),
            card_target_mask=np.zeros(MAX_HAND_CARDS, dtype=bool),
            p_clear=1.0,
            seed=seed,
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

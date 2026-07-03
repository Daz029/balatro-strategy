"""Offline demonstration-generation pipeline for hand-agent BC.

Samples domain-randomized hand-play states via `HandPlayAdapter`, labels
each with `hand_solver.solve_hand_for_ante_clear` (the P(clear)-not-EV
oracle -- see CLAUDE.md "Ante-play (hand/discard) track" for why raw EV is
the wrong objective here), and writes padded `.npz` shards for
behavior-cloning training.

Design decided in a /grilling session (see CLAUDE.md open items):
  - Run once per curriculum stage: pass a `HandPlayConfig` + `stage_name`,
    get one dataset in `{output_dir}/{stage_name}/`. Don't mix stages into
    one pool -- BC training consumes stages as separable units.
  - Workers write their own `.npz` shards independently. No centralized
    collector -- avoids a serialization bottleneck and single point of
    failure for a run that's ~12s/example even after the permutation-search
    speed fix.
  - Seeds are `f"{stage_name}_{global_index:08d}"`, partitioned into fixed
    per-worker index ranges. The seed *set* depends only on
    (stage_name, total_examples), not on how many workers happen to run the
    job -- reruns with a different --num-workers still reproduce the same
    dataset.
  - A solver exception on one sampled state is logged to
    `{stage_dir}/worker_{id}_failures.jsonl` (seed + traceback) and skipped,
    not fatal to the whole run -- deterministic seeding makes a later
    retry-only-failures pass possible once the underlying bug is fixed.
  - Card selection labels are stored as a multi-hot mask over the padded
    hand width (matching `ActionMask.card_mask` in `action_space.py`), not
    as raw variable-length index tuples.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hand_solver import AnteClearChoice, DeckComposition, solve_hand_for_ante_clear  # noqa: E402

from jackdaw.engine.hand_eval import get_hand_eval_flags  # noqa: E402
from jackdaw.env.action_space import ActionType  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayAdapter, HandPlayConfig  # noqa: E402
from jackdaw.env.observation import (  # noqa: E402
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    encode_global_context,
    encode_jokers_batch,
    encode_playing_cards_batch,
)

SCHEMA_VERSION = 1
MAX_HAND_CARDS = 8
MAX_JOKERS = 5
BACK_KEY = "b_red"
STAKE = 1


class GenerationError(Exception):
    """Raised for a sampled state the pipeline can't turn into a valid
    label -- callers must catch this per-example, not let it kill a
    worker's whole index range."""


@dataclass
class GenerationJobConfig:
    stage_name: str
    hand_play_config: HandPlayConfig
    total_examples: int
    num_workers: int
    output_dir: Path
    shard_size: int = 500


@dataclass
class Example:
    global_context: np.ndarray
    hand_cards: np.ndarray
    hand_mask: np.ndarray
    jokers: np.ndarray
    joker_mask: np.ndarray
    action_type: int
    card_target_mask: np.ndarray
    p_clear: float
    seed: str


def _pad_entities(arr: np.ndarray, max_n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    if n > max_n:
        raise GenerationError(f"entity count {n} exceeds max {max_n}")
    padded = np.zeros((max_n, dim), dtype=np.float32)
    mask = np.zeros(max_n, dtype=bool)
    if n > 0:
        padded[:n] = arr
        mask[:n] = True
    return padded, mask


def acted_cards_from_choice(choice: AnteClearChoice) -> list:
    """The cards the label's action actually operates on.

    NOT simply `choice.hold` -- `hold`/`discard` are named from the
    "keep vs. discard" framing of a *discard* decision. For `action=="play"`,
    `hold` is repurposed by `hand_solver.py` to mean "the subset chosen to
    play" (see `best_immediate_play`'s callers), not "cards kept in hand".
    Getting this backwards silently produces a Discard label for what the
    solver actually recommended as a Play, or vice versa.
    """
    if choice.action == "play":
        return choice.hold
    if choice.action == "discard":
        return choice.discard
    raise GenerationError(f"unknown solver action: {choice.action!r}")


def indices_by_identity(cards: list, hand: list) -> list[int]:
    """Map `cards` back to their positions in `hand` by object identity,
    not value equality. `Card` is a plain @dataclass (value-based `__eq__`),
    so an Erratic deck's duplicate-valued cards would otherwise collide
    under `in`/`.index()` -- see the /grilling session and the matching fix
    in `hand_solver.best_immediate_play`.
    """
    id_to_index = {id(c): i for i, c in enumerate(hand)}
    try:
        return [id_to_index[id(c)] for c in cards]
    except KeyError as exc:
        raise GenerationError(
            f"solver returned a card not present (by identity) in the original hand: {exc}"
        ) from exc


def generate_one_example(seed: str, config: HandPlayConfig) -> Example:
    """Sample one domain-randomized hand-play state via `HandPlayAdapter`
    and label it with the exact P(clear)-not-EV solver. Raises
    `GenerationError` (or lets a solver exception propagate) on any
    failure -- callers must catch and log, not swallow silently.
    """
    adapter = HandPlayAdapter(config)
    adapter.reset(BACK_KEY, STAKE, seed)
    gs = adapter.raw_state

    hand: list = gs["hand"]
    jokers: list = gs["jokers"]
    hand_levels = gs["hand_levels"]
    blind = gs["blind"]
    rng = gs["rng"]
    cr = gs["current_round"]

    hands_left = cr.get("hands_left", 0)
    discards_left = cr.get("discards_left", 0)
    blind_chips = getattr(blind, "chips", 0) if blind else 0
    chips_needed = max(0.0, blind_chips - gs.get("chips", 0))

    deck = DeckComposition.from_deck(gs.get("deck", []))
    flags = get_hand_eval_flags(jokers)

    choice = solve_hand_for_ante_clear(
        hand,
        jokers,
        hand_levels,
        blind,
        rng,
        deck,
        chips_needed,
        hands_left,
        discards_left,
        game_state=gs,
        blind_chips=blind_chips,
        four_fingers=flags["four_fingers"],
        shortcut=flags["shortcut"],
    )

    acted_cards = acted_cards_from_choice(choice)
    selected_indices = indices_by_identity(acted_cards, hand)

    card_target_mask = np.zeros(MAX_HAND_CARDS, dtype=bool)
    for idx in selected_indices:
        if idx >= MAX_HAND_CARDS:
            raise GenerationError(f"hand index {idx} exceeds MAX_HAND_CARDS={MAX_HAND_CARDS}")
        card_target_mask[idx] = True

    action_type = ActionType.PlayHand if choice.action == "play" else ActionType.Discard

    global_ctx = encode_global_context(gs)
    hand_arr = encode_playing_cards_batch(hand, gs)
    joker_arr = encode_jokers_batch(jokers, gs)

    hand_padded, hand_mask = _pad_entities(hand_arr, MAX_HAND_CARDS, D_PLAYING_CARD)
    joker_padded, joker_mask = _pad_entities(joker_arr, MAX_JOKERS, D_JOKER)

    return Example(
        global_context=global_ctx,
        hand_cards=hand_padded,
        hand_mask=hand_mask,
        jokers=joker_padded,
        joker_mask=joker_mask,
        action_type=int(action_type),
        card_target_mask=card_target_mask,
        p_clear=float(choice.p_clear),
        seed=seed,
    )


def write_shard(path: Path, examples: list[Example]) -> None:
    """Write one `.npz` shard. `schema_version` lets a future consumer
    detect stale data if `observation.py`'s encoding shape ever changes."""
    np.savez_compressed(
        path,
        schema_version=np.array([SCHEMA_VERSION]),
        global_context=np.stack([e.global_context for e in examples]),
        hand_cards=np.stack([e.hand_cards for e in examples]),
        hand_mask=np.stack([e.hand_mask for e in examples]),
        jokers=np.stack([e.jokers for e in examples]),
        joker_mask=np.stack([e.joker_mask for e in examples]),
        action_type=np.array([e.action_type for e in examples], dtype=np.int64),
        card_target_mask=np.stack([e.card_target_mask for e in examples]),
        p_clear=np.array([e.p_clear for e in examples], dtype=np.float32),
        seed=np.array([e.seed for e in examples]),
    )


def _worker_run(
    worker_id: int,
    start_idx: int,
    end_idx: int,
    stage_name: str,
    config: HandPlayConfig,
    output_dir: Path,
    shard_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / f"worker_{worker_id:03d}_failures.jsonl"

    buffer: list[Example] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not buffer:
            return
        shard_path = output_dir / f"worker_{worker_id:03d}_shard_{shard_idx:05d}.npz"
        write_shard(shard_path, buffer)
        shard_idx += 1
        buffer.clear()

    with open(failures_path, "a", encoding="utf-8") as failures_file:
        for global_idx in range(start_idx, end_idx):
            seed = f"{stage_name}_{global_idx:08d}"
            try:
                example = generate_one_example(seed, config)
            except Exception as exc:  # noqa: BLE001 -- one bad sample must not kill the worker
                failures_file.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    + "\n"
                )
                failures_file.flush()
                continue
            buffer.append(example)
            if len(buffer) >= shard_size:
                flush()
        flush()


def partition_indices(total: int, num_workers: int) -> list[tuple[int, int]]:
    """Split [0, total) into up to `num_workers` contiguous, non-overlapping
    (start, end) ranges covering every index exactly once. The range
    boundaries depend only on (total, num_workers) -- not on wall-clock
    scheduling -- which is what makes the seed set reproducible.
    """
    if total <= 0 or num_workers <= 0:
        return []
    chunk = -(-total // num_workers)  # ceil division
    ranges = []
    for worker_id in range(num_workers):
        start_idx = worker_id * chunk
        end_idx = min(start_idx + chunk, total)
        if start_idx >= end_idx:
            break
        ranges.append((start_idx, end_idx))
    return ranges


def run_generation_job(job: GenerationJobConfig) -> None:
    """Top-level entry point: partitions [0, total_examples) into fixed
    per-worker index ranges and spawns one independent process per worker.
    Each worker writes its own shards and failures file -- no results are
    passed back through this process.
    """
    job.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage_name": job.stage_name,
        "schema_version": SCHEMA_VERSION,
        "total_examples": job.total_examples,
        "num_workers": job.num_workers,
        "shard_size": job.shard_size,
        "back_key": BACK_KEY,
        "stake": STAKE,
        "hand_play_config": asdict(job.hand_play_config),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(job.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    processes = []
    for worker_id, (start_idx, end_idx) in enumerate(
        partition_indices(job.total_examples, job.num_workers)
    ):
        p = multiprocessing.Process(
            target=_worker_run,
            args=(
                worker_id,
                start_idx,
                end_idx,
                job.stage_name,
                job.hand_play_config,
                job.output_dir,
                job.shard_size,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _parse_int_range(spec: str) -> tuple[int, int]:
    lo, hi = spec.split(",")
    return (int(lo), int(hi))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-name", required=True, help="e.g. stage1_no_jokers")
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument(
        "--num-workers", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1)
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/hand_agent_demos"))
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--ante-range", type=_parse_int_range, default="1,8")
    parser.add_argument("--hands-range", type=_parse_int_range, default="1,4")
    parser.add_argument("--discards-range", type=_parse_int_range, default="0,3")
    parser.add_argument("--dollars-range", type=_parse_int_range, default="0,50")
    parser.add_argument("--blind-stages", default="Small,Big")
    parser.add_argument("--joker-pool", default="", help="comma-separated joker keys")
    parser.add_argument("--joker-count-range", type=_parse_int_range, default="0,0")
    args = parser.parse_args()

    joker_pool = tuple(k for k in args.joker_pool.split(",") if k)
    config = HandPlayConfig(
        ante_range=args.ante_range,
        joker_pool=joker_pool,
        joker_count_range=args.joker_count_range,
        hands_range=args.hands_range,
        discards_range=args.discards_range,
        dollars_range=args.dollars_range,
        blind_stages=tuple(args.blind_stages.split(",")),
    )

    job = GenerationJobConfig(
        stage_name=args.stage_name,
        hand_play_config=config,
        total_examples=args.total_examples,
        num_workers=args.num_workers,
        output_dir=args.output_dir / args.stage_name,
        shard_size=args.shard_size,
    )
    run_generation_job(job)


if __name__ == "__main__":
    main()

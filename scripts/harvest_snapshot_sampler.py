"""Sample repaired hand snapshots for h1 training.

The 2026-07-19 staleness grep found that every engine commit since harvest
capture (03e288d, 92c6e27, bd33d4d, c07cfff, 86bb8a6, 9af408a, 5b9ab27,
d891db0) changes COMPUTATION, which restored blobs inherit for free. The
Idol id cache (dfa80b0) remains the only STORED-STATE skew, repaired in
``harvest_restore``. Any future engine fix landing before h1 PPO gets the
same grep.

The sampler keeps the config-anchor mixture in the training layer. A
nonzero fresh-config branch preserves coverage of states the harvest cannot
contain, while the snapshot branch is uniform over all eligible hand
records, independent of source.
"""

from __future__ import annotations

import json
import pickle
import random
from collections import OrderedDict
from pathlib import Path

from harvest_restore import restore_state


class HarvestSnapshotSampler:
    """Draw repaired hand states from a harvested corpus or a fresh config."""

    _CACHE_SIZE = 8

    def __init__(
        self,
        harvest_dir: Path,
        config_anchor_frac: float = 0.5,
        seed: int = 0,
        sources: tuple[str, ...] = ("det", "sampled"),
    ) -> None:
        self._harvest_dir = Path(harvest_dir)
        if not self._harvest_dir.is_dir():
            raise FileNotFoundError(f"harvest directory does not exist: {self._harvest_dir}")
        if not 0.0 < config_anchor_frac < 1.0:
            raise ValueError("config_anchor_frac must be strictly between 0 and 1")

        self._config_anchor_frac = config_anchor_frac
        self._rng = random.Random(seed)
        allowed_sources = set(sources)
        self._records: list[tuple[str, str]] = []
        metadata_path = self._harvest_dir / "metadata.jsonl"
        with metadata_path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("kind") == "hand" and record.get("source") in allowed_sources:
                    self._records.append((str(record["run_seed"]), str(record["record_id"])))

        if not self._records:
            raise ValueError("harvest corpus has no eligible hand records")

        self._shard_cache: OrderedDict[str, dict[str, bytes]] = OrderedDict()

    def __call__(self) -> bytes | None:
        """Return ``None`` for an anchor episode or a repaired snapshot blob."""
        if self._rng.random() < self._config_anchor_frac:
            return None

        run_seed, record_id = self._rng.choice(self._records)
        shard = self._load_shard(run_seed)
        try:
            blob = shard[record_id]
        except KeyError as exc:
            raise KeyError(f"record {record_id!r} missing from shard {run_seed!r}") from exc

        gs = restore_state(blob)
        return pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_shard(self, run_seed: str) -> dict[str, bytes]:
        cached = self._shard_cache.get(run_seed)
        if cached is not None:
            self._shard_cache.move_to_end(run_seed)
            return cached

        shard_path = self._harvest_dir / "blobs" / f"{run_seed}.pkl"
        with shard_path.open("rb") as fh:
            shard = pickle.load(fh)
        if not isinstance(shard, dict):
            raise TypeError(f"blob shard {shard_path} is not a record dictionary")

        self._shard_cache[run_seed] = shard
        self._shard_cache.move_to_end(run_seed)
        if len(self._shard_cache) > self._CACHE_SIZE:
            self._shard_cache.popitem(last=False)
        return shard

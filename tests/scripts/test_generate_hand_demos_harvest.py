"""Tests for C2, the snapshot-fed labeling front-end.

C2's whole value is that it labels REAL mid-run states through the SAME solver,
encoder and validation tiers as stages 1-4. So the properties under test are the
seams around that shared body, not the body itself (covered by
test_generate_hand_demos.py):

* the sha gate never silently proceeds (pitfall #7),
* resume is by record-id membership, not index-parsing (a harvested seed's
  trailing number is a turn index, so parsing would resume in the wrong place),
* a systematically thinned stage fails loudly rather than shipping,
* the blob store reads one shard per run, not one per record.

The end-to-end labeling run against real blobs is the smoke test at the bottom;
it is the phase-C definition of done.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from generate_hand_demos import (
    BlobStore,
    GenerationError,
    HarvestJobConfig,
    _completed_record_ids,
    _harvest_worker_run,
    check_corpus_sha,
    load_selection_manifest,
    run_harvest_job,
    summarize_run,
)

CORPUS = Path("data/harvest_s0")
REAL_CORPUS = pytest.mark.skipif(
    not (CORPUS / "metadata.jsonl").exists(),
    reason="harvest corpus not present (data/ is gitignored)",
)


def _write_manifest(path: Path, records: list[dict]) -> Path:
    path.write_text(
        json.dumps({"manifest_version": 1, "params": {"seed": 0}, "records": records}),
        encoding="utf-8",
    )
    return path


def _record(record_id="HARVEST_00000000_0", run_seed="HARVEST_00000000", sha="abc123"):
    return {
        "record_id": record_id,
        "run_seed": run_seed,
        "source": "det",
        "ante": 1,
        "git_sha": sha,
    }


class TestManifestLoading:
    def test_rejects_unknown_manifest_version(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest_version": 99, "records": [_record()]}))
        with pytest.raises(GenerationError, match="unsupported manifest_version"):
            load_selection_manifest(path)

    def test_rejects_empty_manifest(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest_version": 1, "records": []}))
        with pytest.raises(GenerationError, match="no records"):
            load_selection_manifest(path)

    def test_loads_a_valid_manifest(self, tmp_path):
        path = _write_manifest(tmp_path / "m.json", [_record()])
        assert len(load_selection_manifest(path)["records"]) == 1


class TestShaGate:
    """Pitfall #7: blobs are pickles of live engine objects, so a capture/label
    engine skew can load cleanly while behaviour has drifted. The gate makes
    that decision explicit -- it must never silently proceed."""

    def test_mismatch_is_fatal_by_default(self):
        with pytest.raises(GenerationError, match="different engine sha"):
            check_corpus_sha([_record(sha="deadbeef")], allow_mismatch=False)

    def test_mismatch_error_names_the_override(self):
        """The gate has to tell the operator how to make the call explicitly."""
        with pytest.raises(GenerationError, match="--allow-sha-mismatch"):
            check_corpus_sha([_record(sha="deadbeef")], allow_mismatch=False)

    def test_mismatch_can_be_accepted_explicitly_and_is_reported(self, capsys):
        report = check_corpus_sha([_record(sha="deadbeef")], allow_mismatch=True)
        assert report["n_mismatched"] == 1
        assert report["allowed"] is True
        assert "WARNING" in capsys.readouterr().out

    def test_matching_sha_passes_without_the_flag(self):
        from harvest_s0_rollouts import git_sha

        report = check_corpus_sha([_record(sha=git_sha())], allow_mismatch=False)
        assert report["n_mismatched"] == 0

    def test_report_records_every_distinct_sha(self):
        records = [_record(sha="aaa"), _record(sha="bbb"), _record(sha="aaa")]
        report = check_corpus_sha(records, allow_mismatch=True)
        assert report["record_shas"] == {"aaa": 2, "bbb": 1}


class TestBlobStore:
    def _store(self, tmp_path, runs=2, per_run=3):
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        for r in range(runs):
            run_seed = f"HARVEST_{r:08d}"
            shard = {f"{run_seed}_{i}": pickle.dumps({"i": i}) for i in range(per_run)}
            (blob_dir / f"{run_seed}.pkl").write_bytes(pickle.dumps(shard))
        return BlobStore(blob_dir)

    def test_reads_a_record(self, tmp_path):
        store = self._store(tmp_path)
        assert pickle.loads(store.get("HARVEST_00000000", "HARVEST_00000000_1")) == {"i": 1}

    def test_missing_shard_is_a_generation_error(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(GenerationError, match="blob shard missing"):
            store.get("HARVEST_99999999", "HARVEST_99999999_0")

    def test_missing_record_in_shard_is_a_generation_error(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(GenerationError, match="not in blob shard"):
            store.get("HARVEST_00000000", "HARVEST_00000000_99")

    def test_shard_is_read_once_per_run(self, tmp_path, monkeypatch):
        """Labeling a run's records must not re-read its shard per record."""
        store = self._store(tmp_path)
        reads = []
        real = BlobStore._shard

        def counting(self, run_seed):
            if run_seed not in self._cache:
                reads.append(run_seed)
            return real(self, run_seed)

        monkeypatch.setattr(BlobStore, "_shard", counting)
        for i in range(3):
            store.get("HARVEST_00000000", f"HARVEST_00000000_{i}")
        assert reads == ["HARVEST_00000000"]

    def test_cache_evicts_beyond_its_size(self, tmp_path):
        store = self._store(tmp_path, runs=2)
        store._cache_size = 1
        store.get("HARVEST_00000000", "HARVEST_00000000_0")
        store.get("HARVEST_00000001", "HARVEST_00000001_0")
        assert list(store._cache) == ["HARVEST_00000001"]


class TestResume:
    def test_resume_is_by_record_id_not_parsed_index(self, tmp_path):
        """A harvested seed is a record_id whose trailing number is a TURN
        index, so the stage 1-4 index-parsing resume would land in the wrong
        place. Membership is the only safe key."""
        np.savez_compressed(
            tmp_path / "worker_000_shard_00000.npz",
            seed=np.array(["HARVEST_00000000_7", "HARVEST_00000005_2"]),
        )
        done, next_shard = _completed_record_ids(tmp_path, 0)
        assert done == {"HARVEST_00000000_7", "HARVEST_00000005_2"}
        assert next_shard == 1

    def test_no_shards_means_nothing_done(self, tmp_path):
        assert _completed_record_ids(tmp_path, 0) == (set(), 0)

    def test_other_workers_shards_are_not_counted(self, tmp_path):
        np.savez_compressed(
            tmp_path / "worker_001_shard_00000.npz", seed=np.array(["HARVEST_00000000_7"])
        )
        assert _completed_record_ids(tmp_path, 0) == (set(), 0)


class TestFailureSummary:
    def test_counts_written_and_failed(self, tmp_path):
        np.savez_compressed(tmp_path / "worker_000_shard_00000.npz", seed=np.array(["a", "b"]))
        (tmp_path / "worker_000_failures.jsonl").write_text(
            '{"seed": "c", "error_type": "GenerationError"}\n'
        )
        summary = summarize_run(tmp_path, 3)
        assert summary == {
            "n_requested": 3,
            "n_written": 2,
            "n_failed": 1,
            "failure_rate": round(1 / 3, 4),
            "errors_by_type": {"GenerationError": 1},
        }

    def test_empty_dir_summarizes_cleanly(self, tmp_path):
        assert summarize_run(tmp_path, 0)["failure_rate"] == 0.0

    def test_errors_are_tallied_by_tag_across_workers(self, tmp_path):
        """A big count under ONE tag means a systematic fault, which eats a
        whole class of states rather than a random sample."""
        (tmp_path / "worker_000_failures.jsonl").write_text(
            '{"seed": "a", "error_type": "CaptureSkewError"}\n'
            '{"seed": "b", "error_type": "GenerationError"}\n'
        )
        (tmp_path / "worker_001_failures.jsonl").write_text(
            '{"seed": "c", "error_type": "CaptureSkewError"}\n'
        )
        summary = summarize_run(tmp_path, 10)
        assert summary["n_failed"] == 3
        assert summary["errors_by_type"] == {"CaptureSkewError": 2, "GenerationError": 1}

    def test_tags_are_ordered_most_frequent_first(self, tmp_path):
        (tmp_path / "worker_000_failures.jsonl").write_text(
            '{"error_type": "Rare"}\n' + '{"error_type": "Common"}\n' * 3
        )
        assert list(summarize_run(tmp_path, 4)["errors_by_type"]) == ["Common", "Rare"]

    def test_untagged_and_corrupt_rows_still_count(self, tmp_path):
        """A failure row must never vanish from the tally just because it is
        malformed -- that would under-report exactly when things are worst."""
        (tmp_path / "worker_000_failures.jsonl").write_text(
            '{"seed": "a"}\nnot json at all\n'
        )
        summary = summarize_run(tmp_path, 2)
        assert summary["n_failed"] == 2
        assert summary["errors_by_type"] == {"unknown": 1, "unparseable_failure_row": 1}

    def test_failures_are_tagged_and_logged_by_the_worker(self, tmp_path, capsys):
        """Every skipped record must be counted by tag AND surfaced; a silent
        skip is how a systematically thinned stage ships unnoticed."""
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        (blob_dir / "HARVEST_00000000.pkl").write_bytes(pickle.dumps({}))
        out = tmp_path / "out"
        _harvest_worker_run(
            0,
            [_record(f"HARVEST_00000000_{i}") for i in range(3)],
            blob_dir,
            out,
            shard_size=500,
        )
        rows = [
            json.loads(line)
            for line in (out / "worker_000_failures.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 3
        assert {r["error_type"] for r in rows} == {"GenerationError"}
        assert all(r["traceback"] for r in rows)
        assert "FAILED" in capsys.readouterr().err

    def test_excess_failures_stop_the_run(self, tmp_path):
        """A silently thinned stage is worse than a loud stop: a systematic
        failure would preferentially eat the unusual states worth having."""
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()
        # Shard exists but holds no usable records -> every record fails.
        (blob_dir / "HARVEST_00000000.pkl").write_bytes(pickle.dumps({}))
        manifest = _write_manifest(
            tmp_path / "m.json", [_record(f"HARVEST_00000000_{i}") for i in range(4)]
        )
        with pytest.raises(GenerationError, match="records failed"):
            run_harvest_job(
                HarvestJobConfig(
                    manifest_path=manifest,
                    blob_dir=blob_dir,
                    output_dir=tmp_path / "out",
                    num_workers=1,
                    allow_sha_mismatch=True,
                )
            )


@REAL_CORPUS
class TestEndToEnd:
    """Phase C's definition of done: manifest records -> labeled shard ->
    loaded back by the BC loader."""

    def _manifest_slice(self, tmp_path, n=3):
        real = json.loads(Path("manifests/h1_harvested.json").read_text(encoding="utf-8"))
        return _write_manifest(tmp_path / "m.json", real["records"][:n]), n

    def test_labels_real_records_into_a_loadable_shard(self, tmp_path):
        manifest, n = self._manifest_slice(tmp_path)
        out = tmp_path / "stage5_harvested"
        summary = run_harvest_job(
            HarvestJobConfig(
                manifest_path=manifest,
                blob_dir=CORPUS / "blobs",
                output_dir=out,
                num_workers=1,
                allow_sha_mismatch=True,
            )
        )
        assert summary["n_failed"] == 0
        assert summary["n_written"] == n

        shards = sorted(out.glob("worker_*_shard_*.npz"))
        assert shards
        data = np.load(shards[0])
        assert int(data["schema_version"][0]) == 3
        # Labels are the v3 index-set encoding: ascending, -1 padded.
        for row in data["card_indices"]:
            picks = [i for i in row if i >= 0]
            assert 1 <= len(picks) <= 5
            assert picks == sorted(picks)
        assert data["hand_cards"].shape[2] == 18  # v2/v3 hand-card width

    def test_job_manifest_records_the_skew_decision_and_worker_count(self, tmp_path):
        manifest, n = self._manifest_slice(tmp_path)
        out = tmp_path / "stage5_harvested"
        run_harvest_job(
            HarvestJobConfig(
                manifest_path=manifest,
                blob_dir=CORPUS / "blobs",
                output_dir=out,
                num_workers=1,
                allow_sha_mismatch=True,
            )
        )
        written = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert written["source"] == "harvested"
        assert written["num_workers"] == 1  # pitfall #5: recorded, not just used
        assert written["engine_sha"]["allowed"] is True
        assert written["engine_sha"]["n_mismatched"] == n

    def test_rerun_resumes_instead_of_duplicating(self, tmp_path):
        manifest, n = self._manifest_slice(tmp_path)
        out = tmp_path / "stage5_harvested"
        config = HarvestJobConfig(
            manifest_path=manifest,
            blob_dir=CORPUS / "blobs",
            output_dir=out,
            num_workers=1,
            allow_sha_mismatch=True,
        )
        run_harvest_job(config)
        run_harvest_job(config)

        seeds: list[str] = []
        for shard in sorted(out.glob("worker_*_shard_*.npz")):
            seeds.extend(str(s) for s in np.load(shard)["seed"])
        assert len(seeds) == n
        assert len(set(seeds)) == n

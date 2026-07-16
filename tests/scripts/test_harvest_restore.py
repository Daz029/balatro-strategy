"""Tests for the harvest restore/repair seam (C2's front door).

The Idol repair reconstructs a cached field an older engine never wrote. Two
things have to hold for that to be legitimate rather than a fudge:

1. The reconstruction is EXACT — ``id`` is a pure function of the ``rank`` the
   blob already stores, via the engine's own table. Pinned here against the
   engine's real card-construction path, so the repair cannot drift from it.
2. Unknown shapes hard-fail. A key-set diff structurally cannot catch this bug
   class (the missing field lived inside a cached dict value), so the shape
   guard is what stops the next skew of this class from passing silently.
"""

from __future__ import annotations

import pickle

import pytest
from harvest_restore import (
    CaptureSkewError,
    repair_capture_skew,
    repair_idol_card,
    restore_state,
)

from jackdaw.engine.card import _RANK_ID, CardBase
from jackdaw.engine.data.enums import Rank


def _gs(idol_card) -> dict:
    return {"current_round": {"idol_card": idol_card}}


class TestRankIdIsTheEnginesOwn:
    def test_rank_id_table_matches_the_engines_card_construction(self):
        """The repair's mapping IS the engine's: pin it against the real
        CardBase path, so moving/changing _RANK_ID breaks CI rather than
        silently mislabeling The Idol."""
        for rank in Rank:
            base = CardBase.from_card_key("dummy", "Hearts", rank.value)
            assert _RANK_ID[rank.value] == base.id

    def test_every_rank_is_covered(self):
        assert set(_RANK_ID) == {r.value for r in Rank}
        assert len(_RANK_ID) == 13


class TestIdolRepair:
    def test_backfills_id_from_stored_rank(self):
        gs = _gs({"suit": "Hearts", "rank": "King"})
        assert repair_idol_card(gs) is True
        assert gs["current_round"]["idol_card"] == {
            "suit": "Hearts",
            "rank": "King",
            "id": 13,
        }

    @pytest.mark.parametrize(
        ("rank", "expected_id"),
        [("2", 2), ("10", 10), ("Jack", 11), ("Queen", 12), ("King", 13), ("Ace", 14)],
    )
    def test_reconstructs_every_rank_exactly(self, rank, expected_id):
        gs = _gs({"suit": "Spades", "rank": rank})
        repair_idol_card(gs)
        assert gs["current_round"]["idol_card"]["id"] == expected_id

    def test_matches_what_the_fixed_engine_would_have_cached(self):
        """The fixed engine writes rank/suit/id off ONE drawn card. Repairing
        the pre-fix capture of that same card must reproduce it exactly."""
        base = CardBase.from_card_key("dummy", "Diamonds", "Queen")
        fixed_engine_cache = {"suit": "Diamonds", "rank": base.rank.value, "id": base.id}
        pre_fix_capture = {"suit": "Diamonds", "rank": base.rank.value}

        gs = _gs(pre_fix_capture)
        repair_idol_card(gs)
        assert gs["current_round"]["idol_card"] == fixed_engine_cache

    def test_suit_is_untouched(self):
        gs = _gs({"suit": "Clubs", "rank": "7"})
        repair_idol_card(gs)
        assert gs["current_round"]["idol_card"]["suit"] == "Clubs"

    def test_post_fix_capture_passes_through_untouched(self):
        """Idempotent + version-agnostic: a re-harvest needs no change here."""
        gs = _gs({"suit": "Hearts", "rank": "King", "id": 13})
        assert repair_idol_card(gs) is False
        assert gs["current_round"]["idol_card"]["id"] == 13

    def test_repair_is_idempotent(self):
        gs = _gs({"suit": "Hearts", "rank": "9"})
        assert repair_idol_card(gs) is True
        assert repair_idol_card(gs) is False
        assert gs["current_round"]["idol_card"]["id"] == 9

    def test_report_names_the_applied_repair(self):
        assert repair_capture_skew(_gs({"suit": "Hearts", "rank": "9"})) == {
            "idol_card_id": True
        }
        assert repair_capture_skew(_gs({"suit": "Hearts", "rank": "9", "id": 9})) == {
            "idol_card_id": False
        }


class TestShapeGuard:
    def test_unknown_field_is_fatal(self):
        """The guard that stops the NEXT cached-field drift from being silent."""
        gs = _gs({"suit": "Hearts", "rank": "King", "enhancement": "m_gold"})
        with pytest.raises(CaptureSkewError, match="unknown field"):
            repair_idol_card(gs)

    def test_missing_rank_is_fatal_not_defaulted(self):
        gs = _gs({"suit": "Hearts"})
        with pytest.raises(CaptureSkewError, match="missing required field"):
            repair_idol_card(gs)

    def test_unknown_rank_is_fatal(self):
        gs = _gs({"suit": "Hearts", "rank": "Knight"})
        with pytest.raises(CaptureSkewError, match="not a rank the engine knows"):
            repair_idol_card(gs)

    def test_absent_idol_card_is_fatal(self):
        with pytest.raises(CaptureSkewError, match="no current_round\\['idol_card'\\]"):
            repair_idol_card({"current_round": {}})

    def test_missing_current_round_is_fatal(self):
        with pytest.raises(CaptureSkewError, match="no current_round"):
            repair_idol_card({})

    def test_non_dict_idol_card_is_fatal(self):
        with pytest.raises(CaptureSkewError, match="not a dict"):
            repair_idol_card(_gs(["Hearts", "King"]))


class TestRestoreState:
    def test_restores_and_repairs_in_one_step(self):
        blob = pickle.dumps(_gs({"suit": "Hearts", "rank": "Ace"}))
        gs = restore_state(blob)
        assert gs["current_round"]["idol_card"]["id"] == 14

    def test_non_state_blob_is_fatal(self):
        with pytest.raises(CaptureSkewError, match="did not restore to a game state"):
            restore_state(pickle.dumps(["not", "a", "state"]))

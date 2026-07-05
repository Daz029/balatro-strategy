"""Tests for the canonical hand-play action space.

The 436-index layout is a frozen contract (append-only): BC labels, trained
checkpoints, and the gym env all depend on index *i* meaning the same
(action_type, combo) forever. These tests pin the enumeration itself, the
round-trip encodings, and the vectorized legality mask against a
brute-force reference.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from jackdaw.agents.hand_action_space import (
    COMBOS,
    MAX_HAND_CARDS,
    NUM_COMBOS,
    NUM_HAND_ACTIONS,
    action_to_combo,
    combo_to_action,
    legal_action_mask,
    mask_to_action,
)
from jackdaw.env.action_space import ActionType


class TestEnumeration:
    def test_combo_count(self):
        assert NUM_COMBOS == sum(
            len(list(combinations(range(8), k))) for k in range(1, 6)
        )  # 218
        assert NUM_HAND_ACTIONS == 436

    def test_size_lexicographic_order_is_frozen(self):
        # Spot-pin the contract at the boundaries; a change to enumeration
        # order silently corrupts every existing label and checkpoint.
        assert COMBOS[0] == (0,)
        assert COMBOS[7] == (7,)
        assert COMBOS[8] == (0, 1)
        assert COMBOS[-1] == (3, 4, 5, 6, 7)

    def test_combos_unique(self):
        assert len(set(COMBOS)) == NUM_COMBOS


class TestRoundTrip:
    def test_every_action_round_trips(self):
        for action in range(NUM_HAND_ACTIONS):
            action_type, combo = action_to_combo(action)
            assert combo_to_action(action_type, combo) == action

    def test_play_vs_discard_blocks(self):
        at, combo = action_to_combo(0)
        assert at == ActionType.PlayHand
        at, combo = action_to_combo(NUM_COMBOS)
        assert at == ActionType.Discard
        assert combo == (0,)

    def test_unsorted_indices_accepted(self):
        assert combo_to_action(ActionType.PlayHand, (4, 0, 2)) == combo_to_action(
            ActionType.PlayHand, (0, 2, 4)
        )

    def test_mask_label_round_trips(self):
        mask = np.zeros(MAX_HAND_CARDS, dtype=bool)
        mask[[1, 3, 6]] = True
        action = mask_to_action(int(ActionType.Discard), mask)
        action_type, combo = action_to_combo(action)
        assert action_type == ActionType.Discard
        assert combo == (1, 3, 6)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            action_to_combo(NUM_HAND_ACTIONS)
        with pytest.raises(ValueError):
            action_to_combo(-1)
        with pytest.raises(ValueError):
            combo_to_action(ActionType.PlayHand, ())  # zero cards
        with pytest.raises(ValueError):
            combo_to_action(ActionType.PlayHand, (0, 1, 2, 3, 4, 5))  # six cards
        with pytest.raises(ValueError):
            combo_to_action(ActionType.SelectBlind, (0,))  # wrong family


class TestLegalityMask:
    def _brute_force(self, hand_size: int, hands_left: int, discards_left: int) -> np.ndarray:
        ref = np.zeros(NUM_HAND_ACTIONS, dtype=bool)
        for action in range(NUM_HAND_ACTIONS):
            action_type, combo = action_to_combo(action)
            if any(i >= hand_size for i in combo):
                continue
            if action_type == ActionType.PlayHand and hands_left > 0:
                ref[action] = True
            if action_type == ActionType.Discard and discards_left > 0:
                ref[action] = True
        return ref

    @pytest.mark.parametrize("hand_size", [0, 1, 3, 5, 8])
    @pytest.mark.parametrize("hands_left,discards_left", [(0, 0), (1, 0), (0, 2), (3, 3)])
    def test_matches_brute_force(self, hand_size, hands_left, discards_left):
        got = legal_action_mask(hand_size, hands_left, discards_left)
        expected = self._brute_force(hand_size, hands_left, discards_left)
        assert np.array_equal(got, expected)

    def test_full_hand_all_plays_legal(self):
        mask = legal_action_mask(8, 1, 0)
        assert mask[:NUM_COMBOS].all()
        assert not mask[NUM_COMBOS:].any()

    def test_labels_are_always_legal(self):
        # Any solver label (1-5 cards within the hand, action family with
        # budget remaining) must be a legal action under the mask.
        mask = legal_action_mask(8, 2, 1)
        action = combo_to_action(ActionType.Discard, (0, 7))
        assert mask[action]

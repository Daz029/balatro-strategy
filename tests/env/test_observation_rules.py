"""The obs meta-joker flags read live rules (engine overhaul Phase 2).

Before Phase 2 the encoder read flat ``gs["four_fingers"]`` / ``"shortcut"``
/ ``"smeared"`` / ``"splash"`` keys that nothing ever wrote, so the global
context flag (index 29) was 0 for every state.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.observation import encode_global_context


def _gs_with(*keys: str) -> dict:
    gs = initialize_run("b_red", 1, "OBS_RULES")
    gs["jokers"] = [create_joker(k) for k in keys]
    return gs


def test_meta_joker_flags_reflect_owned_jokers() -> None:
    v = encode_global_context(_gs_with("j_four_fingers", "j_splash"))
    assert v[29] == pytest.approx((1 + 8) / 15)


def test_debuffed_rule_joker_is_inactive() -> None:
    gs = _gs_with("j_smeared")
    gs["jokers"][0].debuff = True
    assert encode_global_context(gs)[29] == 0.0

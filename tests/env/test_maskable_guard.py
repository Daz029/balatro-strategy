"""Tests for the shared MaskableCategorical stale-probs guard."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical

from jackdaw.env.maskable_guard import install_stale_probs_guard


def test_stale_probs_cache_does_not_fail_the_simplex_check():
    # The s1 crash (~320k steps). MaskableCategorical caches probs from the
    # UNMASKED logits, and apply_masking's re-init re-validates that cache
    # instead of the distribution it is building -- so the check fires on a
    # tensor layer 3's clamp never touches, while the masked logits here are
    # finite. Fails pre-guard with the reported Simplex ValueError.
    install_stale_probs_guard()
    dist = MaskableCategorical(logits=torch.randn(2, 8))
    dist.probs = torch.full((2, 8), float("nan"))
    dist.apply_masking(torch.ones(2, 8, dtype=torch.bool))
    assert torch.isfinite(dist.probs).all()
    assert torch.allclose(dist.probs.sum(-1), torch.ones(2), atol=1e-6)


def test_genuinely_nonfinite_masked_probs_are_repaired_not_raised():
    # Real poison must survive as a near-no-op update rather than killing a
    # multi-hour run -- and must stay a legal distribution.
    install_stale_probs_guard()
    dist = MaskableCategorical(logits=torch.randn(2, 8))
    # Poison arriving AFTER construction: layer 3 keeps non-finite logits
    # out of the constructor, so this is the shape real poison must take.
    dist._original_logits = torch.full((2, 8), float("nan"))
    dist.apply_masking(torch.ones(2, 8, dtype=torch.bool))
    assert torch.isfinite(dist.probs).all()
    assert torch.allclose(dist.probs.sum(-1), torch.ones(2), atol=1e-6)


def test_install_is_idempotent():
    install_stale_probs_guard()
    patched_apply_masking = MaskableCategorical.apply_masking
    install_stale_probs_guard()
    assert MaskableCategorical.apply_masking is patched_apply_masking

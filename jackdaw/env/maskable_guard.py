"""Shared safeguards for sb3-contrib's maskable distributions."""

import sys

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical

_STALE_PROBS_PATCHED = False


def install_stale_probs_guard() -> None:
    """Guard against sb3-contrib validating a stale ``probs`` cache.

    It drops the cache before masking and repairs genuinely non-finite logits.
    """
    global _STALE_PROBS_PATCHED
    if _STALE_PROBS_PATCHED:
        return
    _STALE_PROBS_PATCHED = True

    original_apply_masking = MaskableCategorical.apply_masking
    stats = {"catches": 0}

    def _guarded_apply_masking(self, masks):  # type: ignore[no-untyped-def]
        # Drop the stale cache: validation must judge the distribution being
        # built, not the previous one. Categorical repopulates it downstream.
        self.__dict__.pop("probs", None)

        # Repair genuine poison BEFORE delegating -- the re-init validates the
        # logits constraint too, so a post-hoc check never gets to run.
        logits = self._original_logits
        if not torch.isfinite(logits).all():
            stats["catches"] += 1
            n = stats["catches"]
            bad_rows = (~torch.isfinite(logits).all(dim=-1)).nonzero().flatten().tolist()
            if n == 1 or n % 100 == 0:
                print(
                    f"[stale-probs-guard] non-finite logits reached masking; "
                    f"repaired (count={n}, rows={bad_rows[:8]}). Run continues.",
                    file=sys.stderr,
                    flush=True,
                )
            self._original_logits = torch.nan_to_num(
                logits, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(-30.0, 30.0)

        original_apply_masking(self, masks)

    MaskableCategorical.apply_masking = _guarded_apply_masking
    MaskableCategorical._stale_probs_guard_stats = stats  # type: ignore[attr-defined]

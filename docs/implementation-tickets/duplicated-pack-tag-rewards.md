# Deliver duplicated pack-tag rewards sequentially

Status: implemented

## What to build

Deliver pack-creating rewards duplicated by Double Tag sequentially. The first pack must remain active until the player exhausts its choices or skips it; closing that pack must open the duplicated reward as a second pack. Closing the final pack must return the run to blind selection.

The implementation must not overwrite an active pack or record `PACK_OPENING` as that pack's return phase. Keep ordinary, non-duplicated shop and tag-pack behavior unchanged.

## Acceptance criteria

- [x] Double Tag followed by Meteor Tag produces two distinct pack-opening cycles rather than replacing the first pack.
- [x] Exhausting all choices in the first pack opens the duplicated pack; exhausting the duplicated pack returns to blind selection.
- [x] Skipping the first pack opens the duplicated pack; skipping the duplicated pack returns to blind selection.
- [x] Single tag packs and shop-purchased packs still close to their existing return phases.
- [x] Regression tests exercise the real Skip Blind -> duplicated pack reward -> pick/skip engine path and fail on the previous infinite `PACK_OPENING` loop.

## Blocked by

None - can start immediately.

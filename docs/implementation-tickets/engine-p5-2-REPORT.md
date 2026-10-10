# Engine P5-2 report (written at review; codex hit its usage limit before reporting)

## What changed

- `effects.py`: `EffectQueue` carries the pass-local money view
  (`pending_dollars` = this pass's `dollar_buffer`, `instant_dollars` =
  committed-view deltas such as The Ox); `committed()` / `with_buffer()`.
  `EaseDollars(instant=, buffered=)` reserves at emission, commits on apply.
  New `AddChips` effect (the queued `G.GAME.chips` ease). `gs["dollar_buffer"]`
  stays 0 at every stable point (pinned over the seeded rollout).
- `read.py`: `money_committed` / `money_with_buffer` / `StateView.money*`
  read through the pass queue (no longer cached per view).
- `jokers.py`: Bull and Bootstraps read `money_with_buffer`; Vagabond
  committed. `individual_hand_end` deleted; Vampire and Obelisk run in
  `before` (`card.lua:3411-3563`). DNA returns `copy_held_card`, emplaced
  synchronously into the hand so the held-card loop scores it
  (`card.lua:3501-3511`).
- `blind.py`: The Ox inside `debuff_hand`, compares the stored
  `current_round.most_played_poker_hand`, instant drain via the queue
  (`blind.lua:560-567`).
- `scoring.py`: every scoring payout is a buffered `EaseDollars` at its
  emission point; blocked hands run the `after` pass (D12);
  `last_hand_played` / `visible` written once before the blind branch
  (live path only); destruction and chips are F2 effects; `dollars_earned`
  is derived from the effects. Mr. Bones untouched (P5-4).
- `game.py`: play is F1 (press_play queue, Tooth as per-card unbuffered
  eases) -> score -> F2 (played cards to discard, both hand counters, then
  the scoring queue). Post-scoring Ox block deleted. NEW-P5-1-05 (selection
  order) kept and commented, pending the user decision.

## Findings fixed (tests in `tests/engine/test_play_sequencing.py`)

C01, C03 (7 chips), C05 (Bull/Bootstraps 21; Vagabond committed view incl.
Ox; seven buffered producers), D12, D29 (stored target; drain before
scoring; Gold money survives; Matador sees the trigger), DNA residual,
solver isolation for money consumers, `dollar_buffer == 0` over the
rollout. All 21 tests verified FAILING on the parent `0049381` at review.
Oracle xfails flipped: C01 (now final-state only — NEW-P5-1-06 has no
synchronous HAND_PLAYED point; intermediate timing is pinned by
`test_play_sequencing`), C03, C05, D12, D29. 12 strict xfails remain.

## Verification (at review)

- `tests/engine`: 1502 passed, 12 xfailed. Ruff clean on `jackdaw/engine`.
- `tests/env tests/scripts`: 856 passed, 9 skipped; 6 modules fail to
  import (`torch` / `scipy` not installed in this environment), unrelated.
- Perf gate (codex, vs `2bdbcc2`, 20 seeds/stage): stage2 135.79 s ->
  131.44 s (-3.2%), stage3 22.18 s -> 21.51 s (-3.0%).

## Review notes

- DNA's solver path appends a copy to the caller's `held_cards`; safe
  because `scripts/hand_solver.py` builds fresh clone lists per call.
- `last_hand_played` is written only when the scorer is given the live
  play/hand lists (identity check), keeping solver probes isolated.

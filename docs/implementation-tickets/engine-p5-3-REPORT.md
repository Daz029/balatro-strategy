# Engine P5-3 report

## Outcome

P5-3 is implemented without a commit. Round start, post-play, and
post-discard now use one synchronous `draw_from_deck_to_hand` / `draw_card` /
DRAW_TO_HAND port. Voluntary discards and The Hook use one discard port, and
`new_round` / `Blind:set_blind` now preserve the scoped F5 ordering.

For the synchronous F4 translation, `first_hand_drawn` handlers are dispatched
after the number of deck draws has been snapshotted but before any card is
moved. Their effects stay queued while the deck cards land, then apply before
`Blind:drawn_to_hand`. This gives Certificate Lua's placement: normal deck
draws first, Certificate's card next (and potentially over hand capacity), then
the blind callback (`game.lua:3219-3241`, `card.lua:2462-2481`).

## Files changed

- `jackdaw/engine/game.py`: shared draw/card-draw controller; shared voluntary
  and Hook discard; F3/F4/F5 ordering; set-blind card/Joker debuff passes;
  queued Water/Needle/Manacle/Acorn effects; Hook routing and Tooth trigger.
- `jackdaw/engine/blind.py`: set-blind `prepped` initialization, Fish
  `stay_flipped`, gated Crimson Heart, unconditional `prepped` clear, and the
  Manacle disable draw through the shared per-card primitive.
- `jackdaw/engine/effects.py`: typed `ShuffleArea` effect used for Acorn's
  deferred `aajk` shuffles.
- `jackdaw/engine/jokers.py`: `hook` context field and Burnt Joker's real
  `pre_discard and not context.hook` branch.
- `jackdaw/engine/run_init.py`: new-round `wheel_flipped` reset.
- `jackdaw/engine/state.py`: registers the already-existing challenge key
  `modifiers.flipped_cards`, now read by the real draw path.
- `jackdaw/bridge/backend.py`: mid-round hand-size top-ups use the shared
  per-card draw primitive after `_draw_hand` was removed.
- `tests/engine/test_draw_discard_sequencing.py`: new step-level P5-3 suite.
- `tests/engine/test_transition_oracle.py`: flipped the C09, D28, D31, Acorn
  D32, and discard-marker strict xfails; the D32 defeat half remains strict.
- `tests/engine/test_effect_structure.py`, `test_game.py`, and
  `test_state_completeness.py`: effect coverage and Lua-cited expectation
  updates.

## Findings fixed and proving tests

The mandated new test file was run before implementation on parent `8afb782`:
all 11 collected cases failed. After implementation it has 12 passing cases
(the twelfth also pins the Joker redebuff/Certificate ordering).

- **C09:** House and forced-success Wheel initial draws are face-down with
  `wheel_flipped`; Mark flips only its face card; Fish flips only newly drawn
  cards and leaves held cards face-up. Proved by
  `test_c09_house_and_wheel_flip_real_initial_draws`,
  `test_c09_mark_flips_only_faces_on_a_real_initial_draw`, and
  `test_c09_fish_flips_only_cards_drawn_after_play`
  (`blind.lua:605-622`, `common_events.lua:386-423`, `cardarea.lua:32-43`).
- **D28:** an unprepared Crimson Heart discard neither changes Joker debuffs
  nor advances `crimson_heart`; `prepped` clears even when no effect fires.
  Proved by `test_d28_discard_does_not_reroll_crimson_heart_or_advance_stream`
  (`blind.lua:588-603`).
- **D31:** Hook's two `hook`-stream selections go through the shared discard
  pipeline. A Hook-discarded Purple Seal creates a Tarot, Mail-In Rebate pays,
  Burnt does not level, and voluntary counters remain unchanged. Proved by
  `test_d31_hook_uses_discard_pipeline_but_burnt_guard_blocks_level`
  (`blind.lua:466-487`, `state_events.lua:379-446`, `card.lua:2748-2755`).
- **D32, Acorn half:** Jokers flip immediately and three deferred shuffles
  advance `aajk` three times. The exact `P53_ACORN` derivation, confirmed by
  the Lua oracle, is `abdce -> eabdc -> ecbad`; the final stream state is
  `0.5256759670741`. Proved by
  `test_d32_amber_acorn_runs_three_aajk_shuffles`
  (`blind.lua:190-205`, `misc_functions.lua:206-218`).
- **NEW-P5-1-04:** `start_round` clears `wheel_flipped` on every owned playing
  card. Proved by `test_new_p5_1_04_new_round_clears_wheel_flipped_markers`
  (`state_events.lua:307-309`).
- **NEW-P5-1-07:** a zero-capacity empty initial hand loses in the shared draw
  path. Proved by `test_new_p5_1_07_zero_capacity_empty_initial_hand_loses`
  (`state_events.lua:355-360`). The former post-action helper was folded into
  this controller; Lua's additional exhausted hand/deck/play guard was found
  at `game.lua:3056-3065`, so the env terminal boundary remains faithful.
- **Serpent:** both a discard and a play invoke the same shared controller and
  draw exactly up to three after the prior action. Proved by the two cases of
  `test_serpent_draws_exactly_three_through_the_shared_draw`
  (`state_events.lua:362-376`, `game.lua:3208-3244`).
- **NEW-P5-O2-01:** surviving discarded cards retain
  `ability.discarded = True`; the matching oracle xfail is now green
  (`state_events.lua:410-419`, `cardarea.lua:66-79`).
- **Set-blind completeness / NEW-P5-1-08:** every playing card and Joker is
  re-debuffed, and Certificate lands after the snapshotted deck draws. Proved
  by `test_set_blind_redebuffs_jokers_and_certificate_lands_after_draws`
  (`blind.lua:207-215`, `game.lua:3219-3241`, `card.lua:2462-2481`).

Discard money now uses P5-2's `EffectQueue` ledger exclusively:
Mail-In/Trading payouts are ordinary FIFO `EaseDollars`; Faceless is
`order=1`, preserving its nested event after every first-level discard event
and before redraw. No discard observer can distinguish the intermediate
balance: all seal/Joker dispatches finish before F3 applies any queued money,
and the next gameplay callback is DRAW_TO_HAND. The nested position is still
represented rather than coalesced.

Only Burnt Joker reads `context.hook` anywhere in vanilla `card.lua`; the
other discard handlers intentionally receive Lua's ordinary per-card discard
context.

## Changed pre-existing expectations

- `test_game.py::TestCardFlipping::test_the_fish_flips_cards` now expects held
  cards to remain face-up and only post-play draws to be face-down
  (`blind.lua:618-620`, `common_events.lua:386-423`).
- `test_state_completeness.py` now expects `Blind.create` to start a newly set
  non-Fish blind with `prepped=True`; the old-pickle class fallback remains
  false (`blind.lua:78-95,176-177`).
- The exhausted hand/deck game-over expectation is retained with its newly
  located Lua authority (`game.lua:3056-3065`).
- `test_effect_structure.py` gained the required application case for the new
  typed `ShuffleArea`; no gameplay number changed.
- Five transition-oracle xfail groups were removed because they now match Lua;
  the remaining six xfails stay strict.

## Verification

- Parent-red check: `tests/engine/test_draw_discard_sequencing.py` before the
  implementation: **11 failed**.
- `pytest tests/engine tests/env -q`: **2064 passed, 1 skipped, 6 xfailed**.
- `pytest tests/env -q`: **543 passed, 1 skipped**. The legal-action mask suite
  passes unchanged; no env mask or action-space code changed.
- `pytest tests/scripts` excluding the six unavailable optional-dependency
  modules: **313 passed, 8 skipped**.
- The requested combined `tests/env tests/scripts` command reaches the same
  environment limitation as P5-2: six collection errors because `torch`
  and/or `scipy` are not installed
  (`test_analyze_boss_clear_estimators.py`, `test_eval_shop_policy.py`,
  `test_extract_shop_joker_embeddings.py`, `test_extract_v_curve.py`,
  `test_train_bc_v3.py`, `test_train_hand_ppo_b.py`).
- Ruff on every touched Python file: clean. `git diff --check`: clean.
- The engine handler/state structural tests pass, including the prohibition on
  direct handler `gs[...]` reads.

## New divergences found

None. The exhausted-area check at `game.lua:3056-3065` was a source location
missing from the P5-1 sweep, not a divergence; it justifies preserving the
existing terminal behavior inside the new shared draw controller. The known
D32 boss-defeat half and round-end cleanup remain assigned to P5-4 and were not
changed here.

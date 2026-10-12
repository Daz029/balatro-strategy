# Engine CO-1 report: local scoring close-out

## Result

Implemented all in-scope CO-1 findings against Balatro 1.0.1o. The new
integration suite had 11 failures and 4 passes on the unmodified parent; it now
has 16 passes. No legality work was touched and no commit was made.

## Findings

- **D01 — failed on parent: yes.** `Card.get_nominal` is now the only nominal
  implementation. High Card and both CardArea sorts use it; the duplicated
  CardArea helper was deleted. This preserves Lua's Stone suit scaling and the
  `unique_val` tiebreak (`card.lua:950-955`). Proving tests:
  `TestD01Nominal::test_score_hand_and_area_sort_use_unique_val_tiebreak` and
  `test_stone_nominal_scales_suit_terms`. The Stone half of `Card.get_nominal`
  itself was already correct; the failing part was the callers. The solver,
  play-ordering, and observation code have no separate card-nominal ordering
  implementation, and all 176 solver tests retained their outputs.

- **D05 — failed on parent: yes.** Brainstorm now targets exactly slot one and
  no-ops when it occupies that slot, through `_find_leftmost`; both scoring and
  `resolve_copy_targets` share that path (`card.lua:2321-2327,4225-4237`).
  Proving test: `TestD05Brainstorm::test_leftmost_brainstorm_is_inactive_everywhere`.

- **D06 — failed on parent: yes.** Removed the non-Lua Blueprint guards from
  Hiker (`card.lua:3067-3074`), DNA's first-hand copy branch
  (`card.lua:3501-3522`), and Burnt Joker (`card.lua:2748-2755`). DNA's separate
  UI-pulse guard at `card.lua:2483` remains represented separately, and Sixth
  Sense keeps the outer guard from `card.lua:2603`. Perkeo was already fixed by
  Phase 4 commit `b5a2cea` and its regression passed on the parent. Proving
  tests: the four methods in `TestD06BlueprintGuards`. A full audit of every
  other `not ctx.blueprint` in `jokers.py` found no further mismatch.

- **D08 — failed on parent: yes.** Raised Fist now replaces its candidate on
  equal rank and therefore selects the last tied lowest held card
  (`card.lua:3320-3326`). Proving test:
  `TestD08RaisedFistTie::test_last_tied_low_card_gets_red_seal_retrigger`.

- **D09 — failed on parent: yes.** Blackboard now accepts an empty held hand,
  matching the source's `0 == 0` count comparison (`card.lua:3951-3963`).
  Proving test: `TestD09BlackboardEmptyHand::test_playing_the_whole_hand_triggers_blackboard`.

- **D13 — failed on parent: yes.** Playing a secret hand was already revealing
  it from Phase 5 commit `8afb782`; that half passed on the parent. Removed the
  incorrect reveal from `HandLevels.level_up`, so Black Hole and other level-up
  paths do not reveal unplayed secret hands (`common_events.lua:464-468`), while
  scoring still reveals at `state_events.lua:574-578`. To Do List, Telescope,
  and observation continue to consume `visible`; the close-out tests cover all
  three. Proving tests: the three methods in `TestD13SecretHandVisibility`.

- **D21 — failed on parent: yes.** Gift Card now emits one typed effect covering
  owned Jokers and consumables; the effect increments `extra_value` and calls
  `set_cost(gs)` for every card (`card.lua:2993-3005`). Proving test:
  `TestD21GiftCardConsumables::test_end_round_increases_consumable_sell_value`.

- **D34 — failed on parent: yes.** Successful Ectoplasm uses now spend the
  current run-wide `ecto_minus` and increment it through a typed effect,
  producing costs 1, 2, 3, ... (`card.lua:1494-1498`). Proving test:
  `TestD34EctoplasmPenalty::test_successive_uses_cost_one_then_two_hand_size`.

- **D38 — failed on parent: yes.** Grim assigns Ace directly and advances
  `grim_create` only for each suit; enhancement still uses the independent
  `spe_card` stream (`card.lua:1319-1327`). The fixed-seed test derives and
  asserts Spades then Diamonds for `D38_GRIM`. Proving test:
  `TestD38GrimRng::test_fixed_seed_exact_created_suits`.

- **D40 — failed on parent: no; already fixed.** Phase 2 commit `6c39f32`
  ported `set_consumeable_usage`; Phase 4 commit `977d263` completed the live
  use path. Tarot increments Fortune Teller's tally, Black Hole counts as
  Spectral, and Black Hole preserves the prior `last_tarot_planet`, so Fool
  copies that prior Tarot/Planet. Proving test:
  `TestD40ConsumableUsage::test_tarot_black_hole_and_fool_bookkeeping`.

## Changed pre-existing expectations

- Brainstorm-leftmost handler and trigger-resolution expectations now assert
  inactive/no effect (`card.lua:2321-2327`).
- Blueprint copies of DNA and Hiker now assert the copied mutation
  (`card.lua:3501-3522`, `3067-3074`).
- Empty-hand Blackboard now asserts X3 (`card.lua:3951-3963`).
- Secret-hand level-up and serializer expectations now remain hidden
  (`common_events.lua:464-468`).
- Gift Card unit coverage now applies its queued typed effect; its gameplay
  expectation remains +$1 sell value (`card.lua:2993-3005`). Structural effect
  and consumable-schema expectations were extended for the two new effects.

## New divergence found (not fixed)

- **NEW-CO1-01 — To Do List may reroll to the same target.** Its engine handler
  filters to visible hands but does not exclude the current target; Lua excludes
  it at `card.lua:2975-2980` (specifically line 2978). This is outside the listed
  D13 visibility correction, so it is reported without expanding scope.
  **FIXED in review** (2026-10-12): the handler now excludes the current
  target. Proving test `test_to_do_list_reroll_excludes_current_target` loops
  20 seeds (a single seed passed on the buggy code by 1/2 luck); seen failing at
  seed 5 with the exclusion removed.

No further Blueprint-guard mismatch was found. No new nominal-ordering
implementation was found in the solver, play-ordering, or observation paths.

## Verification

- Parent close-out baseline: **11 failed, 4 passed**.
- Current close-out file: **16 passed**.
- `BALATRO_SOURCE=... uv run pytest tests/engine tests/env -q`:
  **2107 passed, 1 skipped**; all **29/29** transition-oracle tests ran and
  passed.
- Script tests not requiring unavailable optional ML/analysis packages:
  **313 passed, 8 skipped**.
- Bridge serializer: **37 passed**.
- Ruff on every changed Python file: **clean**. `git diff --check`: **clean**.
- Solver timing, same 176 files: parent **13.197s**, current **12.902s**
  (about **2.2% faster**); no test-output changes.

The exact mandated combined command could not collect because this checkout
lacks optional `torch` and `scipy` packages (six pre-existing script modules).
Retrying with `uv run --all-extras` was blocked by the sandbox's unavailable
network while fetching `scikit-learn`. Engine/env/oracle and all unaffected
script tests are green as reported above.

# Engine P5-O2 report

The transition oracle now runs explicitly on `lupa.lua51`, drains the real
`evaluate_round`/`add_round_eval_row`, records `{type, amount}` evaluation rows,
normalizes only documented cosmetic pile-facing state, and compares final
states independently of timing probes. The focused file runs in 0.22 seconds.

## Harness state and stubs

- The cached runtime resets `G.STATE`, `G.STATE_COMPLETE`, `G.E_MANAGER`, and
  the round-evaluation UI for every scenario. These are gameplay/event-control
  globals; resetting them prevents one scenario's draw/UI state from affecting
  the next scenario.
- `G.round_eval` is a geometry-free `null_ui`. A wrapper records every row and
  delegates to the source `add_round_eval_row`; the source events still write
  `current_round.dollars`, and source cash-out still writes `G.GAME.dollars`.
- The `CardArea` stand-in owns gameplay-visible area membership, ordering,
  highlights, limits, flips, draw/removal, shuffle RNG, and sort behavior. Its
  sort modes now match `cardarea.lua:577-590`; the old unconditional suit sort
  caused the Crimson Heart hand-order artifact.
- `update_hand_text` writes the real `current_round.current_hand` mirror because
  scoring reads it. `card_eval_status_text` remains inert except for the source
  function's gameplay tail, `playing_card_joker_effects`.
- `math.randomseed`/`math.random` remain backed by Jackdaw's LuaJIT-compatible
  RNG implementation. The LÖVE, animation, sound, localization, UI, and
  profile-persistence stubs do not mutate run gameplay state.

Lua 5.4 compatibility aliases for `loadstring` and `unpack` were removed. The
README now documents the remaining Lua 5.1-versus-LuaJIT `pairs` caveat.

## `pairs` audit

- `state_events.lua:129-138` has the gameplay-visible Boss most-played-hand tie
  traversal (NEW-P5-1-03). The D18 test now compares only payout. A direct
  `/opt/homebrew/bin/luajit` cross-check selected `Flush`, while `lupa.lua51`
  selected `Flush House`; both produced `$12` and identical evaluation rows.
- `game.lua:814-842` builds center pools with `pairs` but sorts every gameplay
  pool by `order`. `pseudorandom_element` also sorts collected keys
  (`misc_functions.lua:253-267`). D31 compares created-card count, not identity.
- Other scoped `pairs` loops reset flags, count/sum values, copy snapshots, or
  traverse UI children and are order-independent. Event work stays in the base
  queue in these seeded scenarios, so cross-queue `pairs` order is not observed.

No retained assertion depends on raw `pairs` order.

## Comparison exclusions

`_COSMETIC` is the single exclusion table:

- `areas.deck[*].facing` — `cardarea.lua:410-424` flips deck cards for layout.
- `areas.discard[*].facing` — `cardarea.lua:425-428` flips discard cards for
  layout.

Hand/play facing remains compared for C09. `ability.discarded` also remains:
`cardarea.lua:66-79` reads it when `discarded_only` removal is requested. Its
missing engine write is covered by NEW-P5-O2-01.

## Strict xfail evidence

- C01: final counters agree; at Lua `synchronous`, both hand counters are `0`,
  while Jackdaw's public transition has already committed `hands_played=1`.
- C03: Vampire scenario chips are Lua `7`, engine `42`.
- C05: Lua synchronously exposes dollars/buffer `4/3`; Bull scores Lua `21`,
  engine `15` chips.
- C09 House: both drawn hand cards are back-facing with `wheel_flipped`; engine
  leaves both front-facing without the marker.
- C09 Mark: the drawn face card is back-facing with `wheel_flipped`; engine
  leaves it front-facing without the marker.
- C10: held Gold + Red seal + Mime ends at Lua `$13`, engine `$7`.
- C11: below-threshold Mr. Bones ends `game_over` in Lua, `round_eval` in engine.
- D12: blocked-hand Ice Cream chips decay to Lua `95`, engine remains `100`.
- D18: Rocket round payout is Lua `$12`, engine `$10`.
- D22: losing rental balance is Lua `$7`, engine `$10`.
- D28: Lua does not create `pseudorandom.crimson_heart`; engine consumes
  `0.4633335546794` on discard.
- D29: Lua synchronously has committed/buffered money `0/3` and finishes at
  `$3`; engine finishes at `$0`.
- D30: Lua clears `played_this_ante`; engine leaves it `true`.
- D31: two Purple seals create two consumables in Lua; engine creates none.
- D32 shuffle: source Acorn flips all three Jokers; Lua's third `aajk` state is
  `0.1122206418401`, while engine's single shuffle state is `0.8308205886149`.
- D32 defeat cleanup: from a controlled post-flip/post-debuff Acorn state, Lua
  restores the Joker to front/non-debuffed and clears the playing-card debuff;
  engine leaves the Joker back-facing and debuffed.
- NEW-P5-O2-01: normal Lua discard sets `ability.discarded=true`; engine omits
  it. This was separated from D28/D31 because the field has a gameplay read.

The Crimson Heart hand-order difference was not a new engine divergence: it
was the oracle's incorrect suit sort and disappears with the faithful sort.

## Validation

- `pytest tests/engine -q`: **1474 passed, 17 xfailed** in 3.23 seconds.
- `pytest tests/engine/test_transition_oracle.py -q --runxfail`: **12 passed,
  17 expected real failures** in 0.23 seconds.
- `ruff check` and `ruff format --check` pass on both touched Python files.
  Repository-wide Ruff still reports five unrelated pre-existing findings in
  `prototypes/shop-slideshow-prototype/server.py` and the discard-ranking
  scripts.

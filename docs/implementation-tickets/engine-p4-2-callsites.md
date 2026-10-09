# Engine Phase 4, ticket P4-2: move every joker call site onto the Effect API

Branch `engine-phase1-state`, base commit `b5a2cea` (P4-1 WIP). Read first:
`docs/engine_fixing_plan_2026-10-06.md` (S1, Phase 4), section 6 of
`docs/engine_class_design_2026-10-06.md`, and `jackdaw/engine/effects.py`.
Lua source: `~/Code/Code/balatro-strategy/balatro_source/Balatro` (read-only).

## What P4-1 already did (do not redo)

- `jackdaw/engine/effects.py`: `Effect` (``reserve`` = Lua emission-time half,
  ``apply`` = event-time half), 18 subclasses, `EffectQueue` (pass-local room
  reservations; `room(area)`), `apply_effects(gs, effects)` (the ONLY applier;
  stable-sorted by `order`).
- `jackdaw/engine/jokers.py`:
  - `JokerContext` is kw-only, `game` is REQUIRED, new `queue`, `game_over`,
    `booster` fields; InitVar snapshot path and the `smeared` / `pareidolia`
    test overrides are gone; `ctx.rules` reads `ctx.game.rules`;
    `ctx.room(area)`.
  - `JokerResult.extra` is gone; `effects: list[Effect]`. `remove=True` now
    means ONLY Lua's "destroy the card this pass is looking at"
    (`destroying_card` -> the scored card; `discard` -> the discarded card).
    Every self-destruct (Seltzer, Ice Cream, Popcorn, Ramen, Gros Michel,
    Cavendish, Mr. Bones, Turtle Bean) is a `DestroyCard(card=self)` effect.
  - `calculate_joker` queues the handler's effects on `context.queue` and
    RAISES `UnappliedEffectError` if there are effects and no queue.
    `_dispatch` is the non-queueing inner call (Blueprint/Brainstorm).
  - `context_for(gs, *, queue, **fields)`: the single builder for live-state
    contexts. `fire_jokers(gs, queue, **fields)`: left-to-right loop over
    `gs["jokers"]`, skipping debuffed, returns the results.
  - `on_end_of_round` is DELETED. `round_dollar_bonus(jokers, view)` sums
    `calc_dollar_bonus`.
  - `GameSnapshot` stays only as a hand-built frozen stand-in for tests; it
    gained `hands_played_total` and `ante`. `StateView` gained
    `hands_played_total` and `ante`.

The suite is red because nothing calls the new API yet. This ticket makes it
green with every effect applied through `apply_effects` / `EffectQueue.apply`.

## Tasks

### 1. `scoring.py::score_hand`

- One `EffectQueue(gs)` per call (`gs` = `game_state or {}`); every
  `JokerContext` it builds passes `queue=queue` (put it in `_shared` and in
  the `debuffed_hand` context too).
- `ScoreResult`: add `effects: list[Effect]` = `queue.effects` (NOT applied
  here: the solver scores clones against the live state and never commits).
  Delete the `joker_creates` field and its collection in phase 9. Make
  `jokers_removed` a derived property: the cards of `DestroyCard` effects in
  `effects` (keep `tests/engine/test_scoring.py:429,447` passing).
- After pass and Mr. Bones: stop treating `remove` as self-removal; the
  handlers now queue `DestroyCard`. Mr. Bones: build its context with
  `game_over=True` (the attribute-poke `bones_ctx.game_over = True` must go).
  Keep `saved`.
- Destroying-card pass: unchanged (`remove` = destroy the scored card), but
  its effects (Sixth Sense's Spectral) now reach the queue automatically.
- **C01 (pulled forward from Phase 5, needed so DNA/Sixth Sense/Loyalty can
  fire):** call `hand_levels.record_play(hand_type)` ONCE, right after hand
  detection and BEFORE the blind debuff check (Lua `state_events.lua:574-578`
  updates history before the debuff branch). Delete the second call in
  `game.py::_handle_play_hand` (step 9). Blocked hands now record too.
- **D11 trigger assignments:** `Blind.debuff_hand` (`blind.py`) sets The Arm's
  `triggered = True` unconditionally; it must be True only when the hand's
  level > 1 (Lua `blind.lua:537-545`). `debuff_hand` has no `HandLevels`, so
  set The Arm's `triggered` in `score_hand`'s phase 3b where the level-down
  already happens (and leave `debuff_hand` leaving it False). In phase 7,
  a debuffed scoring card sets `blind.triggered = True` (Lua
  `state_events.lua:655-656`) before it is skipped. Matador's main-pass
  branch is already in the handler.

### 2. `game.py::_handle_play_hand`

- **C01:** move `cr["hands_played"] += 1` and `gs["hands_played"] += 1` to
  AFTER `score_hand` returns (Lua `state_events.lua:523-524`, the event after
  `evaluate_play`). `hands_left -= 1` stays before scoring (Lua
  `ease_hands_played(-1)` at `:476`).
- Replace the `jokers_removed` loop and `_resolve_create_descriptors(...joker_creates)`
  with `apply_effects(gs, result.effects)`, at the point the old code applied
  them (after the dollars, after the playing-card destruction).

### 3. `game.py` setting_blind (`_handle_select_blind`, `_fire_setting_blind`, `_apply_setting_blind_mutations`)

- Delete `_apply_setting_blind_mutations` and the `disable_requested` path.
- **D14 (pulled forward): Lua order.** Lua `new_round` (`state_events.lua:290-337`)
  resets the round counters, then `Blind:set_blind` (boss set-time effects),
  THEN loops `calculate_joker({setting_blind = true})`. Move the setting_blind
  pass to after `start_round(gs)` and after `_apply_boss_blind_effects`, and
  before the boss card-debuff loop and the `nr` shuffle:
  `queue = EffectQueue(gs); fire_jokers(gs, queue, setting_blind=True); queue.apply()`.
  This fixes Burglar (D14: `start_round` overwrote its +hands / zero
  discards) and orders Chicot's `DisableBlind` after The Water/Needle/Manacle
  set-time effects, which `Blind.disable` then reverses.
- Make sure `context_for` passes the NEW blind (`gs["blind"]` is set in
  step 1, so it does).

### 4. `game.py::_round_won` (end of round)

- Replace `on_end_of_round` with:
  `joker_dollars = round_dollar_bonus(jokers, StateView(gs, jokers=jokers))`
  computed at the same point as before (BEFORE the end_of_round pass, as
  today), then `queue = EffectQueue(gs); fire_jokers(gs, queue, end_of_round=True); queue.apply()`.
  Delete the hand-written mutation loop (hand_size_delta / pool_flag) and the
  `jokers_removed` loop.
- Pass `joker_dollars` to `calculate_round_earnings(joker_dollars=...)` exactly
  as now.
- Seals at end of round (same function): route the Gold-card held dollars
  through `EaseDollars` and the Blue seal Planet through
  `CreateCard(set="Planet", forced_key=planet_key, append="blusl")` on one
  `EffectQueue` (Lua `card.lua:2264`: room = `#consumeables + buffer < limit`,
  so several Blue seals share the reservation), applied at the same point.

### 5. `game.py::_handle_discard`

- One `EffectQueue(gs)` for the whole discard; build every joker context with
  `context_for(gs, queue=queue, ...)`.
- `remove` from ANY joker in the `discard` context destroys the discarded
  card (Lua `discard_cards_from_highlighted`: `if eval.remove then removed =
  true end`). Delete the Ramen-vs-Trading `extra["destroy"]` branch; Ramen
  now queues its own `DestroyCard`.
- Purple seal: `CreateCard(set="Tarot", append="8ba")` on the same queue
  (Lua `card.lua:2254`: room + buffer gate the creation; no RNG drawn
  without room). Delete the inline `_resolve_create_descriptors` call.
- Apply the queue where the old code applied side-effects (step 5).
- `_build_discard_snapshot` is no longer needed; delete it.

### 6. Other existing joker call sites

Each becomes `context_for` + a queue that is applied, or `fire_jokers`:

- `game.py::_handle_skip_blind` (skip_blind context).
- `game.py::_use_consumable_card` (using_consumeable notification).
- `shop.py` selling (`:730`, `selling_self` / `selling_card`) and reroll
  (`:876`). Selling_self is dispatched today but its result is dropped
  (D23): with the queue, Luchador / Diet Cola / Invisible Joker now take
  effect. Keep Lua's order: `selling_self` on the sold card, then
  `selling_card` on the others, then removal. Apply the queue BEFORE the
  sold card is removed (Invisible's copy must see the row as Lua does) — or
  after, if you can show Lua's order differs; cite the line.
- `lifecycle.fire_joker_context`: return nothing; build a queue, dispatch,
  apply. Its callers (`_fire_shop_joker_context`, `_apply_shop_mutations`
  for Perkeo at `ending_shop`, `add_playing_cards`, `destroy_playing_cards`)
  stop passing mutation dicts around. Delete `_apply_shop_mutations`; Perkeo's
  `CopyCard(pick_area=...)` handles itself. Watch for re-entrancy:
  `CreatePlayingCard`/`CopyCard`/`DestroyPlayingCards` call
  `lifecycle.add_playing_cards`/`destroy_playing_cards`, which fire a joker
  context with its own queue. That nesting is intended (Lua's
  `playing_card_joker_effects`); it must not reuse the outer queue.
- `jackdaw/env/trigger_match.py:520` builds `JokerContext(jokers=jokers)`:
  give it a `game` (`read.StateView(gs, jokers=jokers)`), no queue.
- `economy.py::calculate_round_earnings` fallback (when `joker_dollars is
  None`): use `round_dollar_bonus(jokers, StateView(game_state or {}, jokers=jokers))`.
  It must NOT fire end_of_round handlers (the old fallback did, rolling Gros
  Michel's extinction a second time). Drop the `GameSnapshot` import.
- `_resolve_create_descriptors` in `game.py` stays for the consumable path
  only (ticket P4-3 replaces it); its joker callers are gone.

### 7. Tests

- New `tests/engine/_joker_ctx.py`: `make_ctx(**kwargs) -> JokerContext`.
  Keyword args that are `GameSnapshot` fields build `game=GameSnapshot(...)`;
  `smeared=` / `pareidolia=` build `GameSnapshot.rules`; everything else goes
  to `JokerContext`. Default `queue=EffectQueue(gs)` where `gs` holds
  `consumables` / `consumable_slots` / `jokers` / `joker_slots` taken from
  optional `consumables_owned=` / `consumable_slots=` / `joker_count`-style
  kwargs, so room-gated handlers work. It is a TEST fixture: production code
  must not import it.
- Migrate `tests/engine/test_jokers.py` and `tests/engine/test_jokers_integration.py`
  to `make_ctx` and replace every `.extra` assertion with an assertion on
  `result.effects` (type + fields). Where an old test pinned behavior the
  ticket deliberately changed (Madness excluding `jokers[0]`, Riff-raff count
  computed in the handler, Turtle Bean's `hand_size_delta`, Mr. Bones
  `remove`, Matador only on blocked hands, Loyalty round-local count,
  Hallucination key), update it to the Lua behavior and cite the Lua line in
  the test.
- `tests/engine/test_read.py`: the GameSnapshot/StateView attribute parity
  test must still pass (both gained `hands_played_total`, `ante`).
- Any other test that breaks: fix the test only if the new behavior is the
  Lua behavior (cite the line); otherwise fix the code.

### 8. New regression tests (each must FAIL on `b5a2cea^` = `576ff28`, check it)

Write them through `step()` / `score_hand` (integration), not the handler:
`tests/engine/test_effect_pipeline.py`.
- DNA: first hand, one card -> a copy lands in hand; deck+hand count +1.
- Sixth Sense: first hand, single 6 -> the 6 is destroyed and a Spectral is
  created (with room); with a full consumable row the 6 still dies and
  nothing is created.
- 8 Ball with a full consumable row draws NO '8ball' RNG (compare the RNG
  stream state with and without the joker, or assert the stream counter).
- Riff-raff + Cartomancer in one setting_blind: room shared, row never
  overfills; Riff-raff with 1 free slot creates 1.
- Ceremonial Dagger removes its right neighbor and gains 2x its sell cost;
  an Eternal neighbor survives; Riff-raff to the right of the victim (in the
  same pass) sees the freed slot.
- Madness never destroys itself or an Eternal, at any position.
- Burglar: Red Deck round starts at hands+3 and 0 discards (D14).
- Loyalty Card fires on the 6th hand of the RUN across rounds (D07).
- Matador pays on a non-blocked hand under The Arm with a level-2 hand; not
  with a level-1 hand (D11).
- Gros Michel extinction sets the pool flag through `step()` (existing test
  may cover it; keep it green).
- Turtle Bean decays hand size by 1 per round through `step()`.

## Rules

- Never apply an effect anywhere but `apply_effects` / `EffectQueue.apply`.
- Do not change engine behavior beyond what is listed; if you find another
  bug, write it in your final report instead of fixing it.
- Keep the lifecycle lint (`tests/engine/test_engine_lint.py`) and the state
  registry test green. `effects.py` writes `card.getting_sliced` (not a
  banned attribute) and `ability["invis_rounds"]` via `new.ability[...]`; if
  the lint flags `effects.py`, move that write into `lifecycle.py` rather
  than allowlisting it.
- Run: `uv run pytest tests/engine tests/env -q -p no:cacheprovider` (the 2
  `test_hand_eval_oracle.py::TestLiveOracle` errors are pre-existing on this
  machine), and `uv run ruff check jackdaw tests`.
- Do NOT commit. Report: files changed, each new test and whether you saw it
  fail on `576ff28`, any behavior you were unsure about, any bug found but
  not fixed.

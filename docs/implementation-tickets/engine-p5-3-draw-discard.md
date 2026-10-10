# Engine Phase 5, ticket P5-3: one draw, one discard (with The Hook), new_round

Branch `engine-phase1-state`, after P5-2. Read first:
`docs/engine_s4a_event_timing.md` section 1 (flush points F3, F4, F5) and
section 4 (`discard_cards_from_highlighted`, `draw_from_deck_to_hand` /
`draw_card`, `new_round`, `Blind:set_blind`, `Blind:press_play`,
`Blind:drawn_to_hand` / `stay_flipped`); plan Phase 5; audit C09, D28, D31,
D32 (the Amber Acorn half); P5-2's report
(`docs/implementation-tickets/engine-p5-2-REPORT.md`) for the money-ledger
API you must use. Lua: `~/Code/Code/balatro-strategy/balatro_source/Balatro`.
Lua wins over this ticket; cite `# L: file:line` per ported step.

## 1. One shared draw

Replace `_draw_hand`, the post-play redraw block in `_handle_play_hand`,
the post-discard redraw block in `_handle_discard`, and both Serpent loops
with ONE port of `G.FUNCS.draw_from_deck_to_hand` + `draw_card` + the
DRAW_TO_HAND state-controller step (`game.lua:3208-3244`), used by round
start, post-play and post-discard:

- zero-capacity empty-hand loss (NEW-P5-1-07, `state_events.lua:355-360`);
  keep `_end_round_if_hand_empty`'s post-action guard only if Lua has an
  equivalent, otherwise fold it into this path.
- fill-space, Serpent's "exactly up to 3 after any prior play/discard";
- per card: pop from deck end, `Blind.stay_flipped` (C09 — House first
  action, Wheel with its RNG key, Mark faces, Fish when prepped; implement
  Fish in `stay_flipped`, it is a no-op today), set `facing`, set
  `ability.wheel_flipped` where Lua does, emplace, sort;
- re-debuff as Lua does on emplace / `set_blind`;
- `first_hand_drawn` dispatched at Lua's point (after the draws are
  queued, before they land — NEW-P5-1-08: decide what that means for a
  synchronous port, state it, and make Certificate's card arrive where
  Lua's arrives);
- then `Blind.drawn_to_hand`: Cerulean Bell, Crimson Heart ONLY when
  `prepped` (D28), and `prepped = None/False` afterwards even when the
  blind is disabled (`blind.lua:601-603`). Delete the callers' copies of
  this logic.

`Blind.set_blind` sets `prepped = True` (`blind.lua:78-118`); Fish clears it
there and sets it in `press_play`. Make our blind construction match.

## 2. One discard, The Hook through it (D31)

Port `discard_cards_from_highlighted(gs, cards, *, hook=False)` per the
sweep's step list: forced-selection clear, sort, `pre_discard` with `hook`
propagated to the context (Burnt Joker's `not context.hook` guard — check
every discard-context handler for a hook guard in Lua and mirror it), seal
then per-joker `discard` per card, destroy vs move, `remove_playing_cards`
notification, `ability.discarded = True` on discarded cards (NEW-P5-O2-01; `cardarea.lua:66-79` reads it), stats, and the VOLUNTARY-only tail (discard cost, discards
decrement, `discards_used`, DRAW_TO_HAND). Money through the P5-2 ledger;
Faceless's payout is a nested event (NEW-P5-1-01) — keep its FIFO
position relative to Mail-In / Trading if any observer can see the
difference; otherwise say why it cannot.

`_press_play`'s Hook branch calls this function with `hook=True` on the 2
randomly selected cards (keep the `hook` RNG key and selection order
exactly; cite `blind.lua:466-487`), so Purple seals, Mail-In Rebate,
Trading Card etc. fire as in Lua. The Tooth: per-card queued money, final
value unchanged.

## 3. new_round / set_blind

Port `new_round` (`state_events.lua:290-352`) and `Blind:set_blind`'s boss
section in order, with F5:

- `start_round` resets every `wheel_flipped` marker (NEW-P5-1-04).
- Water / Needle subtraction as queued events at their Lua point; Manacle
  hand size; Amber Acorn: flip immediately and THREE shuffles on the
  `aajk` stream (D32 first half; `blind.lua:190-205`), deterministic seed
  per Lua.
- Re-debuff every playing card AND every joker at set_blind
  (`blind.lua:207-215`) — the joker pass is missing today.
- setting_blind joker pass after set_blind (D14, already ordered in
  Phase 4 — keep), then the `nr..ante` shuffle, then the shared draw.

## 4. Tests (red first)

New `tests/engine/test_draw_discard_sequencing.py`, through `step()`; each
must FAIL on the parent (check and say so): C09 House / Mark / Wheel /
Fish facing on real draws; D28 (a discard does not reroll Crimson Heart,
and does not advance its stream); D31 (Hook-discarded Purple-seal card
creates a Tarot; Mail-In pays for a Hook discard; Burnt Joker does NOT
level on a Hook discard); D32 Acorn three shuffles (exact joker order for a
fixed seed — derive the expected order from the Lua oracle if merged,
else from the Lua code by hand and show the derivation); NEW-P5-1-04;
NEW-P5-1-07; Serpent draw count after a discard and after a play.
Flip the matching transition-oracle xfails (`tests/engine/test_transition_oracle.py`,
if present) and keep them strict.

## Rules

As P5-2 (suite green incl. `tests/env tests/scripts`, ruff clean, every
changed pre-existing expectation cited to a Lua line, no handler `gs[...]`
reads). The env's legal-action masks must be unchanged by this ticket —
run `tests/env` and report. Do not commit. Report to
`docs/implementation-tickets/engine-p5-3-REPORT.md` and repeat it as the
final message: files, findings fixed + proving tests, changed expectations,
new divergences found (do not fix out-of-scope ones).

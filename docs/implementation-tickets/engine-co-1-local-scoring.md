# Engine close-out, ticket CO-1: local findings in jokers, scoring, consumables, hand levels

Branch `engine-closeout` (off `engine-phase1-state` after Phase 5). Read first:
`docs/engine_audit.md` (the finding text for each ID below),
`docs/engine_fixing_plan_2026-10-06.md` Part 2 (the fix column) and the Phase 2-5
status blocks (what the overhaul already changed, so you build on it), and
`CLAUDE.md`'s critical-errors header. Lua 1.0.1o source:
`~/Code/Code/balatro-strategy/balatro_source/Balatro` (set
`BALATRO_SOURCE` to it for the oracle tests). **Lua wins over this ticket and over
the audit text**; cite `# L: file:line` at every changed line of logic.

These are LOCAL bugs: wrong in one handler/helper body, not structural. The
overhaul's structures (getters in `read.py`, setters + `lifecycle.py`, `Effect`s
via `effects.py`, the Phase 5 flows) are the only allowed way to read/mutate state
— no new raw `gs[...]` reads in handlers, no direct card-field assignment (the
lints in `tests/engine/test_state_registry.py` / `test_engine_lint.py` /
`test_effect_structure.py` must stay green).

## Findings

For EACH finding: (1) write an integration test that goes through `step()` or
`score_hand` (never a hand-built `JokerContext` asserting a return value — that
test class cannot fail on the bug class this project keeps hitting); (2) run it on
the unmodified code and record whether it FAILS. If it already passes, the
finding was fixed incidentally by the overhaul — keep the test, record
"already fixed" with the commit you believe fixed it, change no code. (3) Fix,
see it pass.

- **D01** — one `get_nominal` port of Lua `Card:get_nominal` (`card.lua:950`,
  incl. the Stone suit-nominal scaling and the `unique_val` tiebreak — the engine
  has `Card.unique_val` as an order-preserving property over `sort_id`, see the
  Phase 1 status block). Delete `hand_eval._card_nominal` and
  `card_area._card_nominal` and route every caller (High Card `get_highest`,
  sorting) through it. Watch the solver: `scripts/hand_solver.py`,
  `scripts/play_ordering`/`jackdaw/engine/play_ordering.py`,
  `jackdaw/env/observation.py` may use nominal ordering — keep them consistent
  and report any change to their outputs.
- **D05** — Brainstorm targets `G.jokers.cards[1]`; when that is Brainstorm
  itself it does nothing (`card.lua:2321`, and `Card:update` around
  `card.lua:4225`). Check whether `resolve_copy_targets` (used by the solver /
  trigger_match) shares the bug and fix both through one path.
- **D06** — remove the `not ctx.blueprint` guards that Lua does not have on
  Hiker, DNA (the `hands_played == 0` copy branch, `card.lua:3501`; DNA's other
  branch at `card.lua:2483` DOES have the guard — keep that one), Burnt Joker
  (`card.lua:2749`), Perkeo (`card.lua:2413`). Sixth Sense keeps its guard.
  Before removing ANY guard, read the Lua branch; audit every other
  `not ctx.blueprint` in `jokers.py` against Lua while you are there and report
  any further mismatch (fix only ones that are clearly the same bug).
- **D08** — Raised Fist picks the LAST tied lowest held card (`card.lua:3320`,
  `>=` scan). Test with a Red seal on only one of two tied cards.
- **D09** — Blackboard passes on an empty held hand (black count == total, 0 ==
  0). Test: play the whole hand.
- **D13** — secret hands (Five of a Kind, Flush House, Flush Five) become
  visible when PLAYED during evaluation (`state_events.lua:578`); `level_up_hand`
  does NOT reveal (`common_events.lua:464`). Check every consumer of `visible`
  (To Do List re-roll, Telescope, observation) still behaves.
- **D21** — Gift Card adds sell value to jokers AND consumables
  (`card.lua:2993`), through the cost setter.
- **D34** — Ectoplasm's hand-size cost grows 1, 2, 3, … run-wide
  (`G.GAME.ecto_minus`, which Phase 1 added as storage — wire its writer/reader).
- **D38** — Grim: assign Ace directly and draw ONLY the suit from
  `grim_create` (`card.lua:1319-1327`). Test the exact suits for a fixed seed,
  derived from the Lua code by hand (show the derivation in a comment).
- **D40 (remaining half)** — Phase 2 ported `set_consumeable_usage`. Verify by
  test that: Tarot use increments Fortune Teller's tally; Black Hole counts as
  Spectral, not Planet; and Black Hole does NOT become `last_tarot_planet` (so
  The Fool after Black Hole creates the previous Tarot/Planet). Fix if not.

## Rules

- Tests go in a new `tests/engine/test_closeout_local.py`, one test class per
  finding ID, named `TestD08...` etc., each docstring quoting the Lua line.
- Full suite green: `uv run pytest tests/engine tests/env tests/scripts -q`
  (with `BALATRO_SOURCE` set, so the 29 transition-oracle tests RUN, not skip).
  Ruff clean on changed files (`uv run ruff check <files>`); report pre-existing
  ruff errors separately, do not fix unrelated ones.
- Every changed pre-existing test expectation must be justified by a Lua line in
  the report.
- Solver performance: if you touch `get_nominal` or anything on the
  `score_hand` path, time `uv run pytest tests/scripts -q -k solver` before and
  after and report.
- Do not commit. Do not touch legality (`actions.py` enumeration, Spectral
  picks, boss reroll, sell phases) — deferred by the user.
- Report to `docs/implementation-tickets/engine-co-1-REPORT.md` and repeat it as
  the final message: per finding — failed-on-parent yes/no, fix summary, proving
  test; changed expectations; any NEW divergences found (describe, do not fix
  out-of-scope ones).

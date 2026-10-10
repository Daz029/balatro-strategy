# Engine Phase 5, ticket P5-O2: transition-oracle fixes (review of P5-O)

Branch `engine-p5-oracle` @ `73e8631` (P5-O WIP, rebased on P5-1 `db3dbb7`).
Files: `scripts/lua_transition_oracle/`, `tests/oracle/transition.py`,
`tests/engine/test_transition_oracle.py`. Read `docs/engine_s4a_event_timing.md`
(P5-1's sweep) for the Lua event model and observation points. New/oracle
files only — still no `jackdaw/engine` changes.

Review found the harness is promising but not yet trustworthy. Run
`uv run pytest tests/engine/test_transition_oracle.py -q --runxfail` to see
every xfail's REAL failure reason; an xfail is only valid if its reason is
an engine-vs-Lua gameplay diff for the finding it names.

## 1. Lua runtime: 5.1 semantics, not 5.4

`run_lua_oracle` uses `lupa.LuaRuntime` (defaults to `lupa.lua54`). Balatro
runs on LuaJIT (5.1 semantics: all numbers are doubles, no integer subtype,
`loadstring`/`unpack`/`setfenv`, different `#`/`pairs` behavior). `lupa.lua51`
is installed (LuaJIT is not available in-process). Switch to `lupa.lua51`
explicitly (import the submodule) and remove any 5.4 shims that become
unnecessary. Document in the README that `pairs` iteration order may still
differ from LuaJIT's, and audit the scoped code paths for any `pairs` whose
order is gameplay-visible (e.g. most-played-hand selection, pool building):
if a scenario depends on one, either avoid it or cross-check that scenario
with the `/opt/homebrew/bin/luajit` subprocess. Report what you found.

## 2. `evaluate_round` crashes the harness

`common_events.lua:929 add_round_eval_row` indexes `G.round_eval` (UI) and
raises, so C10, D18, D22 (and anything reaching `evaluate_round`) are red
for a harness reason. Stub the round-eval UI so the real
`evaluate_round` / `add_round_eval_row` run to completion with their gameplay
writes intact (`current_round.dollars`, `G.GAME.dollars` at cash-out, the
per-row amounts). Capture the round-eval rows (type, amount) in the
snapshot. Then re-check those three xfails fail for their finding.

## 3. Normalise non-gameplay differences in `compare`

Current diffs mix real findings with presentation:
- `areas.discard[*].facing` — Lua flips cards face down when they move to
  the discard pile; that is cosmetic for the discard area. Ignore `facing`
  in `discard` and `deck` (keep it in `hand`/`play`, where it is C09).
- Verify `ability.discarded` (Lua sets it on discard): grep the source for
  any gameplay READ of `ability.discarded`. If none, drop it from the
  comparison; if there is one, keep it and say where.
- Anything else you find that is visual-only: same rule, each exclusion
  listed in one place (`_COSMETIC` table) with the Lua line justifying it.
Never normalise a field a gameplay function reads.

## 4. Engine-side observation points

`test_c01_hand_counter_observation_points` currently fails because the
engine trace has 1 observation vs Lua's 18 — that is a schema mismatch,
not C01. Restructure: compare the engine FINAL state to the Lua FINAL
state for every test, and for the timing-sensitive findings (C01, C05,
D29) ALSO assert the Lua-side intermediate values directly (e.g. at the
`synchronous` point `hands_played` has not advanced and `dollar_buffer==3`)
so the oracle self-documents what Phase 5 must reproduce. Provide a hook
(`run_engine(..., probe=callable)` or similar) that Phase 5 can later wire
to internal engine points; do not require engine changes now.

## 5. Every xfail fails for its own finding

After 1-4, run `--runxfail` and for each of the 15 xfails confirm the diff
is the finding's gameplay value (chips/dollars/counters/facing in hand/
card identity), not a harness artifact or an unrelated divergence. If a
test shows an UNRELATED real divergence too, narrow its `fields=` so it
tests one finding, and add a separate xfail named `NEW-...` for the other
divergence. Known ones from the sweep (`docs/engine_s4a_event_timing.md`
section 5): e.g. the Crimson Heart test shows hand ORDER differences —
determine whether that is the hand sort (NEW-P5-1-05 territory) or the
draw, and split it accordingly.

Also: the D32 test — confirm what it asserts. Amber Acorn's three shuffles
are in `Blind:set_blind`; "defeat cleanup is incomplete" is the other half
of D32 — add a scenario for it (a boss that flips jokers/cards, defeated,
facing and debuffs restored).

## Rules

- `uv run pytest tests/engine -q` green (strict xfails allowed), `uv run
  ruff check` clean on touched files, the oracle test file < ~30 s.
- Write your final report to `docs/implementation-tickets/engine-p5-o2-REPORT.md`
  in the repo (NOT to a path outside the workspace) and keep it concise:
  every stub touching gameplay state and why, the `pairs` audit, each
  xfail with its one-line real diff, every exclusion in `_COSMETIC`, and
  any new divergences.
- Do not commit.

# Engine Phase 4, ticket P4-4: structural tests (the Phase 4 exit gate)

Branch `engine-phase1-state`, after P4-2 and P4-3. Phase 4's exit criterion is
"producer/consumer test goes green". These tests make the S1 bug class (a
handler returns a change that its call site never applies) impossible to
reintroduce silently. Each must be shown to FAIL against a deliberately
broken variant (describe the mutation you tried in the report).

New file `tests/engine/test_effect_structure.py`:

1. **Every Effect applies.** For each class in `effects.EFFECT_TYPES`, build
   a minimal live state (`initialize_run` with a fixed seed), construct the
   effect, apply it with `apply_effects`, and assert the documented state
   change. Also: `Effect` cannot be instantiated (abstract `apply`); a
   non-Effect in `apply_effects` / `EffectQueue.add` raises `TypeError`;
   `EFFECT_TYPES` equals the set of concrete `Effect` subclasses defined in
   `effects.py` (so a new subclass cannot be left out).
2. **No effect without a queue.** A handler returning effects through
   `calculate_joker` with `queue=None` raises `UnappliedEffectError`.
3. **Producer/consumer through `step()`.** For every registered joker whose
   handler can return a non-empty `effects` list (find them statically: AST
   scan of `jokers.py` for `effects=` in a `JokerResult(...)` call inside a
   registered handler), name the context flag(s) under which it emits
   (same scan: the enclosing `if ctx.<flag>` condition), and assert the flag
   is dispatched by production code (an AST scan of `jackdaw/engine` for
   `<flag>=True` passed to `context_for` / `fire_jokers` / `JokerContext`, or
   set inside `score_hand`). Keep an explicit allowlist ONLY for contexts the
   engine deliberately does not dispatch yet, each with the phase that adds
   it; an empty allowlist is the goal. This is the test that would have
   caught C02 (`first_hand_drawn`, `open_booster`), D23 (`selling_self`) and
   D15.
4. **Context construction lint.** In `jackdaw/` (not tests), `JokerContext(`
   is constructed only in `jokers.py` (`context_for`) and `scoring.py`
   (`score_hand`, the counterfactual builder); `dataclasses.replace` on a
   context only in `jokers.py` (copy jokers). Everything else uses
   `context_for` / `fire_jokers`. `GameSnapshot(` is never constructed in
   `jackdaw/` (test fixture only). `tests/engine/_joker_ctx.py` is never
   imported from `jackdaw/` or `scripts/`.
5. **No untyped channel.** `JokerResult` has no `extra` field and no field
   of type `dict`; no module in `jackdaw/engine` reads `.extra` on a handler
   result (AST: attribute `extra` on a name bound from `calculate_joker` /
   `use_consumable` / `Tag.apply`). `ability["extra"]` reads are fine.
6. **Solver isolation.** Score a hypothetical play with 8 Ball (forced
   roll), Vagabond, DNA, Sixth Sense and Riff-raff-style creators against a
   LIVE `gs` via `score_hand` (as `scripts/hand_solver.py` does with clones),
   and assert `gs` is unchanged afterwards apart from what scoring itself is
   documented to change (`docs/engine_class_design_2026-10-06.md` section 8
   "What one scored hand changes": on cloned objects only). Concretely:
   consumables, jokers, deck, hand, dollars, `used_jokers`, `joker_buffer`,
   `consumeable_buffer` all identical; `result.effects` non-empty.

Also update `docs/engine_class_design_2026-10-06.md` section 6 only where the
built code differs from the draft (e.g. `EffectQueue` holding reservations
instead of `SlotReservations` in `gs`; `CreatePlayingCard`, `SetDollars`,
`DestroyPlayingCards`, `ChangeHandSize` added; `SaveRun` not an effect — Mr.
Bones' `saved` stays a pipeline field until Phase 5's C11). Keep it short.

Rules as P4-2. Do not commit; report as P4-2.

# Engine Phase 6, ticket P6-1: one legality predicate, full vanilla action set (ENGINE ONLY)

Branch `engine-legality` (off `engine-closeout`). Read first:
`docs/engine_fixing_plan_2026-10-06.md` S5 + the Phase 6 section (incl. the
2026-10-12 update), `docs/engine_audit.md` D23, D27, D50, D52, D56, D57,
`docs/engine_audit_verification.md`, and `CLAUDE.md` TODO items 1-4. Lua 1.0.1o:
`~/Code/Code/balatro-strategy/balatro_source/Balatro` (set `BALATRO_SOURCE` to it).
**Lua wins over this ticket**; cite `# L: file:line` at every legality rule.

User decision 2026-10-12: **the engine is fully vanilla; the environments hold
the constraints.** Env masks are reconstructed LATER (CLAUDE.md TODO 4) — this
ticket must not change any env mask or env behavior.

## 0. Pin the env first (before any engine change)

Write `tests/env/test_mask_pin.py`: drive seeded rollouts (reuse
`tests/engine/_rollout.py`'s shop-heavy driver, plus `ShopGymEnv` and
`HandPlayGymEnv` episodes with fixed seeds and a fixed/greedy policy) and record,
at every decision point, `get_action_mask` (as arrays), `shop_gym`'s
`action_masks()`, and `hand_play_gym.hand_action_mask`. Store a compact digest
fixture (e.g. a sha256 per step plus the step count) generated ON THE PARENT,
committed under `tests/fixtures/`. After your change the digests must be
byte-identical. Also pin the action sequence each env adapter auto-resolves
(`ShopRunAdapter`, `HandPlayAdapter`, `DirectAdapter`).

## 1. `jackdaw/engine/legality.py`

`is_legal(gs, action) -> bool` and `illegal_reason(gs, action) -> str | None`,
the ONLY source of legality. `step()` calls it first and raises
`IllegalActionError(reason)`; `get_legal_actions` enumerates candidate actions
and filters with the same predicate (keep its marker convention for
PlayHand/Discard — empty indices = "playing is legal"; `is_legal` on a concrete
PlayHand/Discard checks the concrete indices). Remove the scattered validation in
the `_handle_*` functions it replaces (keep only what is not legality). A
structural test asserts every action `get_legal_actions` returns passes `is_legal`,
over the rollout states.

## 2. Findings (each: red integration test through `step()` first, on the parent)

- **D56** — reject: duplicate play/discard indices; out-of-range or negative
  indices anywhere (targets included — reject, never silently filter); buying or
  Buffoon-picking a joker / consumable into a full area (Negative edition and
  Lua's room rules apply: read `check_for_buy_space`, `button_callbacks.lua:2393`);
  `SellCard.area` other than `"jokers"` / `"consumables"`; selling an Eternal.
- **D50** — affordability per Lua (`button_callbacks.lua` `can_buy`,
  `can_reroll`, `can_open`/voucher; `G.GAME.bankrupt_at`): `dollars - cost >=
  bankrupt_at`, zero-cost always affordable even in debt. Credit Card sets
  `bankrupt_at = -20` through its passive (already ported — verify).
- **D23 (phases)** — selling is legal in every phase Lua allows
  (`Card:can_sell_card`, `G.FUNCS.can_sell_card`: SHOP, SELECTING_HAND,
  BLIND_SELECT, PACK_OPENING, ROUND_EVAL? — read Lua, do not guess), with the
  full sell path (selling_self, Verdant Leaf) already in `sell_card`.
- **D27** — Cerulean Bell: a PlayHand/Discard whose indices omit the
  `forced_selection` card is illegal (`blind.lua:575-586`, `cardarea.lua`
  highlight rules). Hook's internal discard is engine-chosen and exempt if Lua
  exempts it — check.
- **D52** — new action `RerollBoss()` appended to the Action union (do not
  renumber anything). Legal in BLIND_SELECT when Retcon (unlimited) or Director's
  Cut (once per ante) is owned, affordable at Lua's price ($10 each — Retcon is
  NOT free; fix the voucher's recorded price), and per Lua's other conditions
  (`button_callbacks.lua` `reroll_boss` / `reroll_boss_button`). Port the reroll
  itself line-faithfully (RNG keys, `bosses_used` bookkeeping, the boss-blind
  state reset). Boss Tag's automatic reroll is separate — share the reroll
  primitive if Lua does.
- **D57** — pack opening:
  - Spectral-pack picks are LEGAL in the engine (delete the balatrobot comment
    and restriction in `actions.py`; the env keeps its own mask untouched).
  - `PickPackCard` checks `can_use_consumeable` (Lua `Card:can_use_consumeable`)
    for usable-on-pick cards; an unusable pick is illegal, never a silent no-op.
  - While a pack is open, the vanilla extra actions are legal: use an owned
    consumable, sell owned cards, reorder hand/jokers — read
    `button_callbacks.lua:2155-2247` and `can_use_consumeable`/`can_sell_card`
    for exactly which.
  - Buy-and-use: a new action `BuyAndUse(shop_index, target_indices=())`
    appended to the union, legal for a consumable in the shop when Lua's
    `can_buy_and_use` holds (read it), executing buy then use through the
    existing paths.

## 3. Callers of the engine list

`jackdaw/env/shop_run_adapter.py`, `hand_play_adapter.py`, `game_interface.py`,
`jackdaw/engine/runner.py`, `scripts/find_seeds.py` call `get_legal_actions`.
Any of them that picks from or counts that list (auto-resolve "only one legal
action", random agents) must keep EXACTLY its current behavior: filter the new
actions out at the call site with a named env-side restriction and a comment
pointing at CLAUDE.md TODO 4. The step-0 pin proves it.

## Rules

- Tests: `tests/engine/test_legality.py` (one class per finding ID) +
  `tests/env/test_mask_pin.py`; every new test seen failing on the parent
  (pin tests excepted — they are generated there).
- Full suite green with `BALATRO_SOURCE` set:
  `uv run pytest tests/engine tests/env tests/bridge tests/scripts -q` (6 script
  modules need torch/scipy and fail to collect in this checkout — pre-existing).
- Ruff clean on touched files; lints in `tests/engine/test_state_registry.py`,
  `test_engine_lint.py`, `test_effect_structure.py` stay green.
- Update `docs/engine_audit_verification.md` rows D23/D27/D50/D52/D56/D57 and the
  plan's Phase 6 status block.
- Do not commit. Report to `docs/implementation-tickets/engine-p6-REPORT.md` and
  repeat it as the final message: per finding (failed on parent, fix, test), new
  actions added, every call site filtered in §3, Lua lines that contradicted this
  ticket, new divergences found.

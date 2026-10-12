# Engine Phase 6 report: legality (engine only)

Branch `engine-legality`, commit `3067cf7`. Ticket:
`engine-p6-legality.md`. Codex was usage-capped (twice), so this ticket was
implemented by hand, following the ticket.

## Result

The engine exposes the full vanilla action set through one predicate,
`jackdaw/engine/legality.py::illegal_reason`. `step()` raises
`IllegalActionError(reason)`; `get_legal_actions` enumerates candidates and
keeps exactly those the predicate accepts (PlayHand / Discard stay markers;
targeted consumable uses and pack picks are listed per legal target set,
which is at most 92 subsets for an 8-card hand since vanilla targets at most
3 cards). All handler-side validation and `_require_phase` are gone.

## Findings (tests in `tests/engine/test_legality.py`)

| ID | Red on parent | Fix | Lua |
|---|---|---|---|
| D56 | 6/7 (sell from a non-owned area was already rejected; message only) | duplicate / negative / out-of-range indices rejected, targets included; room per `check_for_buy_space`; owned rows only | `button_callbacks.lua:2392-2401`, `2112-2120` |
| D50 | 3/4 (1 control: the credit floor still blocks) | `dollars - bankrupt_at`, free always affordable; redeem without the free exemption | `button_callbacks.lua:55-125`, `2076-2090` |
| D23 | 4/4 | sell in every stable phase | `card.lua:1640-1652` |
| D27 | 3/4 (1 control: a play with the forced card is legal) | forced card required in every play/discard; `with_forced_card` | `blind.lua:575-586` |
| D52 | 5/5 | `RerollBoss`: $10, Retcon unlimited, Director's Cut once per ante; shared `_reroll_boss` with the Boss Tag, which now sets `boss_rerolled` | `button_callbacks.lua:2784-2848`, `tag.lua:284-289` |
| D57 | 5/5 | Spectral picks legal; picks need `can_use_consumeable`; owned consumables usable/sellable in packs; `BuyAndUse` | `button_callbacks.lua:2155-2247`, `77-79`, `2407`; `card.lua:1523-1579` |
| D39 residual | n/a (new test) | `can_use_consumable` counts debuffed editionless jokers | `card.lua:4209-4223` |

Plus a structural test: over 8 seeded shop-heavy rollouts, every listed
action passes `is_legal`.

New actions: `RerollBoss`, `BuyAndUse` (appended to the `Action` union and
the package exports; nothing renumbered).

Deleted: `_require_phase`, the invented `boss_blind_rerolls` /
`boss_blind_reroll_cost` keys (also dropped on restore by `migrate_state`),
the Director's Cut / Retcon `apply_voucher` branches (Lua has no
`apply_to_run` for them).

## Lua lines that contradicted the ticket or audit

- `Card:can_sell_card` allows BLIND_SELECT (the exclusion is commented out).
- `can_redeem` has no free-card exemption, unlike `can_buy` / `can_open`.
- The Boss Tag goes through `G.FUNCS.reroll_boss`, so it also consumes
  Director's Cut's once-per-ante reroll. The audit did not mention this.

## Env and call sites (user: env masks are rebuilt later, TODO 4)

- `tests/env/test_mask_pin.py`: the ShopGymEnv (default and s1) and
  HandPlayGymEnv canonical masks, hand masks and trajectories are
  byte-identical to a fixture generated on the pre-Phase-6 commit (5,450
  shop + 213 hand steps). Verified sensitive by unmasking Spectral picks.
- `jackdaw/env/interim_constraints.with_cerulean_bell`, applied at
  `HandPlayAdapter.step`, at `ShopRunAdapter._advance`'s hand-partner step,
  and in `scripts/probe_boss_clear_spread.py`.
- The generic `BalatroEnvironment` / `BalatroGymnasiumEnv`
  (`scripts/train_ppo.py`) sent untargeted or unusable pack picks that the old
  engine swallowed as silent no-ops. Its pick mask (`_pack_pick_mask`) now
  offers only engine-legal picks, and `RandomAgent`, the gymnasium action
  table and `factored_to_engine_action` now supply and forward targets. The
  generic mask is no longer in the pin for that reason.
- `get_legal_actions` consumers: `runner.random_agent` /
  `greedy_play_agent` complete selections with `with_forced_card`;
  `scripts/find_seeds.py` is unaffected (its agent picks blind, cash-out and
  pack actions explicitly). `test_action_space::test_forward_coverage`
  asserts that `BuyAndUse` and `RerollBoss` do NOT convert to the generic env
  space.

## Follow-ups (not fixed)

- The hand solver does not know Cerulean Bell's forced card;
  `generate_hand_demos.py`'s engine execution check will reject such labels
  (logged to `failures.jsonl`). Teaching the solver the forced card is a
  label-semantics change for the next regen.
- Env-mask rebuild (TODO 4): Spectral picks, Credit Card affordability,
  Cerulean Bell, selling outside the shop, boss reroll and buy-and-use are
  all reachable in the engine but not exposed to the training envs.

## Verification

- `tests/engine tests/env tests/bridge`: 2,301 passed (29/29 transition
  oracle).
- `tests/scripts`: 313 passed, 8 skipped. 6 modules need torch or scipy and
  fail to collect in this checkout; this is pre-existing.
- Ruff: clean on `jackdaw/` and `tests/`. 4 errors remain in two untouched
  validation scripts; they are pre-existing.

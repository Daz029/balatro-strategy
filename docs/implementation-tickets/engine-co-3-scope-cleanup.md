# Engine close-out, ticket CO-3: scope guards, NEW-P5-1-05, delete superseded code

Branch `engine-closeout`, after CO-1 and CO-2 (read both REPORTs). Same rules
as CO-1 ("Rules" section of `engine-co-1-local-scoring.md`).

## 1. Scope guards (user decisions 2026-10-12)

Things the engine does not implement must RAISE, never silently run a different
game. Use one exception type, `UnsupportedGameModeError(NotImplementedError)`,
in `jackdaw/engine/state.py` or a small new module, with a message naming the
mode.

- **Challenges (C15, D43 bans, D44 voucher-timing half)**: `initialize_run`
  with any `challenge` raises. Keep the challenge DATA module only if something
  outside the engine still imports it for non-run purposes; otherwise delete
  `apply_challenge` and every challenge-only code path in the engine (ban-list
  plumbing, `challenge_jokers`, challenge-only modifiers) — grep `jackdaw/`,
  `scripts/`, `tests/` for every consumer first and list them in the report.
  If a consumer would break, stop and report instead of deleting.
  Challenge-only `modifiers` keys that vanilla decks/stakes also use must stay.
- **Endless (D49)**: after the win-ante boss is beaten (`gs["won"] = True`,
  `game.py` round-end), any transition that would start a blind at an ante
  greater than `win_ante` raises. Must NOT fire for the horizon curriculum
  (`ShopRunAdapter` sets `win_ante` 2/4/8 and ends the episode on `won`), nor for
  `HandPlayAdapter`, which samples antes 1-8 directly — check both and add a
  test for each. Also assert in `get_blind_amount` for ante > 16 (belt and
  braces; nothing legitimate reaches it).
- **Profiles (D44)**: all play assumes an ALL-UNLOCKED, all-discovered profile.
  `apply_profile_to_game_state` (and any other profile entry) raises on a
  profile that is not fully unlocked and fully discovered; `fresh_profile()` and
  locked-item helpers are deleted unless a non-engine consumer needs them
  (report). Pool filtering must not read profile state at all afterwards.

## 2. NEW-P5-1-05 — DECIDED

Lua scores played cards in physical hand-position order
(`state_events.lua:459-483`). Decision: **the bot's pick order IS the
left-to-right placement** — vanilla lets a player rearrange the hand for free
before playing, so the index sequence in `PlayHand.card_indices` is read as the
cards' left-to-right order in the play area. The solver's internal ordering does
not matter (its labels are sets; envs decide the submitted order, e.g.
`hand_play_gym` via `best_play_order`). No engine behavior change is expected.
Do: replace the "pending decision / intentional divergence" comments at
`game.py` (~522) and `scoring.py` (~566) with the decision and its rationale;
update the docstring of `PlayHand` / `actions.py` to state the contract; add a
test through `step()` proving an order-sensitive joker (Photograph: first
scoring face card) follows the index order (`[q, k]` vs `[k, q]` with differently
enhanced faces give different scores); mark NEW-P5-1-05 RESOLVED in
`docs/engine_s4a_event_timing.md` and the plan's Phase 5 status block.

## 3. Delete superseded code

- **D58**: delete `jackdaw/engine/card_area.py`'s exported helpers if the run
  loop does not use them (it uses raw lists) — move anything genuinely used into
  `lifecycle.py` / `read.py`; delete the rest and their tests.
- `shop.buy_card` (unused duplicate of `_handle_buy_card`).
- Any compatibility shim the overhaul left behind: search for comments saying
  "legacy", "backward compat", "deprecated", "bridge", "kept for", "TODO
  Phase", "until Phase", and for functions with no production caller (use
  `rg` over `jackdaw/ scripts/` — a function only called from tests is dead
  unless it is a documented public API). For each candidate: show the evidence
  it is dead, delete it with its tests. KEEP: `migrate_state` and the pool /
  sticker migrations (old pickled blobs still need them), `tests/engine/_joker_ctx.py`
  (`GameSnapshot` test fixture) unless no test imports it.
- List everything deleted (file, symbol, why dead) in the report.

## Rules

As CO-1. Full suite green with `BALATRO_SOURCE` set, `tests/env` masks
unchanged, ruff clean on touched files. Do not commit. Do not touch legality,
Spectral picks or boss reroll. Report to
`docs/implementation-tickets/engine-co-3-REPORT.md` and repeat it as the final
message.

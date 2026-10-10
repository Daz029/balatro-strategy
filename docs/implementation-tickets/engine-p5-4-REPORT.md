# Engine P5-4 report

Codex ran about half of P5-4 (the WIP on branch `engine-p5-4-wip`) before
its usage limit; the rest was finished and reviewed by hand.

## What changed

- `game.py`
  - `end_round(gs)` replaces `_round_won` and the play handler's
    loss branch. Every round end goes through it: blind beaten, no hands
    left, and the exhausted hand/deck loss.
    - `game_over` comes from committed accumulated chips.
    - Per joker, interleaved: `end_of_round(game_over)`, `saved`, rent,
      perishable (`state_events.lua:92-110`).
    - An ordinary LOSS still runs that maintenance before GAME_OVER.
    - Won/saved branch: unused discards; Boss `most_played_poker_hand`
      with Lua's stale-`_order` tie code (NEW-P5-1-03).
    - Held-card EOR effects with Red seal / Mime repetitions; the Blue-seal
      Planet uses the last played hand (C10).
    - Hand and discard return to the deck; Boss: ante +1 and next voucher.
    - Pillar markers cleared on Boss defeat (D30); Juggle revert.
    - Target cards reset here, not at cash-out (NEW-P5-1-09).
    - Flush F6, then `evaluate_round`.
  - `evaluate_round(gs)` (F7): the blind row only if committed chips beat
    the target ($0 row on a Mr. Bones save, NEW-P5-1-10); Back eval; hands,
    discards, joker `calc_dollar_bonus` AFTER the EOR pass (D18), eval tags,
    interest on committed dollars. Then `Blind.defeat`. Lua QUEUES defeat,
    so every row is computed against the live blind.
  - `_handle_cash_out` (F8): shuffle; reset `jokers_purchased` and
    hands/discards before SHOP (NEW-P5-1-02); commit `current_round.dollars`
    through the ledger; snapshot `previous_round.dollars`; chips to 0. After a
    Boss: Small tag, Big tag, then `get_new_boss` (`reset_blinds`), the same
    RNG order as the old `assign_ante_blinds`.
  - `won` is forced False on GAME_OVER. Lua sets `G.GAME.won` before the
    game-over branch even when the win-ante boss is lost; ours means "run
    won".
- `blind.py`: `Blind.defeat` (face-down jokers back to face-up, Manacle
  restore, then the reset `set_blind` that re-runs card + joker debuffs) for
  D32's second half.
- `jokers.py`
  - `_dispatch` enforces Lua's `end_of_round` precedence
    (`card.lua:2874-2887`): with `individual`, nothing fires; with
    `repetition`, only Mime.
  - Mr. Bones saves only at >= 25% of the blind target, measured on
    committed chips (C11).
  - Gros Michel / Cavendish no longer return `saved` (Lua: a "Safe!"
    message only). Under `end_round` that flag would rescue a lost run.
- `scoring.py`: the Mr. Bones block and `ScoreResult.saved` are deleted.
- `economy.py`: `calculate_round_earnings` is a pure reward-row function.
  Rent is committed by `end_round` and passed in only as metadata; eval-tag
  rows are part of `total`.
- `round_lifecycle.py`: `process_perishable` per joker.
- `tests/oracle/transition.py`: stickers in a scenario's `ability` now map to
  the engine `Card` attributes, so D22 compares a real rental.

## Review fixes on top of the codex WIP (each mutation-checked)

1. `Blind.defeat` ran BEFORE the reward rows, so a Crimson-Heart-debuffed
   Golden Joker would pay. It now runs after them.
   Test: `test_defeat_runs_after_reward_rows_so_boss_debuffed_golden_pays_nothing`.
2. A lost win-ante boss kept `won=True`.
   Test: `test_losing_the_win_ante_boss_is_not_a_win`.
3. The held-card EOR loop re-fired every plain end-of-round handler once per
   held card: Turtle Bean decayed 4x in one round.
   Test: `test_held_card_eor_passes_do_not_refire_plain_end_of_round_handlers`
   (plus the existing Turtle Bean tests).
4. Investment and Anaglyph read `blind.boss` (Lua's `last_blind.boss`), not
   `get_type()`.

## Changed pre-existing expectations (Lua-cited in place)

- Rent is off the balance at ROUND_EVAL and outside the cash-out total
  (`test_cashout_ordering`, `test_economy`, `test_consumables`;
  `state_events.lua:108`).
- The Investment Tag is an eval row inside `earnings.total`, never in the
  interest base (`test_cashout_ordering`, `test_tag_wiring`;
  `state_events.lua:1183-1191`).
- Gros Michel / Cavendish survival is not `saved` (`card.lua:3043`).
- `score_hand` never consumes Mr. Bones (`test_scoring`; C11).
- The structural producer/consumer scan treats `game_over` as a value context
  (`state_events.lua:101`).
- `test_shop_run_adapter::test_empty_hand_after_draw_is_terminal_before_policy_call`
  now gives its forged SELECTING_HAND state a real blind target. The
  adapter resets into SHOP with the empty 0-chip blind, which 0 chips "beat"
  under Lua's rule.

## Verification

- `tests/engine/test_end_round_sequencing.py`: 19 tests.
  - 15 fail on the parent `a2722a5`.
  - Of the other 4: 2 cover behavior that was already correct (Blue-seal
    Pluto, Manacle restore), and 2 guard against bugs the WIP introduced,
    shown by mutation.
- `tests/engine`: 1546 passed, 0 xfailed. All 29 transition-oracle tests pass;
  all 17 original strict xfails are gone.
- `tests/env` and `tests/scripts`: all pass apart from 6 modules that fail to
  import because `torch` / `scipy` are not installed here.
- Ruff: clean.
- Smoke: 40 random-shop `ShopGymEnv` runs ran without an error. 30
  forced-clear runs reach ante 8 and win, meeting 8 bosses each. Every
  cash-out commits exactly `current_round.dollars`; chips are 0 after
  cash-out; `dollar_buffer` stays 0 after every play.
- Solver hot path: P5-4 only deletes work from `score_hand` (the Mr. Bones
  loop) and adds one boolean check in `_dispatch`; not re-timed.

## Phase 5 data impact (for plan Part 6)

- Economy: rent timing, the Investment row, post-pass dollar bonuses, $0
  blind rows on saves, and losing-round maintenance change money totals.
  Interest is unchanged in rule but sees post-rent balances, as before.
- In SHOP, `current_round.hands_left` / `discards_left` now hold the
  next-round values (NEW-P5-1-02). Shop obs that encode the global context
  see a different value than s0/s1 trained on.
- Labels: every Phase 5 change sits on the play/score path (money views,
  Vampire/Obelisk, The Ox, DNA, blocked-hand after pass), so hand-solver
  labels change. Full regen and re-harvest, as planned.

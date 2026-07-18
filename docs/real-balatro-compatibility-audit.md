# Real Balatro compatibility audit

Audit target: local `post-regen-wave0` at `f43e43b` (the newest checked-out
branch after fetching `origin`). This is an audit only; no engine code was
changed.

## Current confidence

- `tests/engine`: **1,054 passed, 14 skipped**.
- The passing suite covers hand detection, scoring, all 150 Joker registrations
  and integration paths, 52 consumable handlers, bosses, shop generation,
  tags, seeded fixtures, and recent cash-out ordering fixes.
- The 14 skipped tests require a local Lua/LuaJIT executable, so the RNG and
  hand-evaluation Lua cross-checks were not run in this environment.
- The live comparison harness exposes 254 scenarios across Jokers, modifiers,
  planets, spectrals, tarots, tags, and grouped boss-blind cases. See
  [`docs/validation.md`](validation.md).

Passing Python tests establish internal consistency and agreement with the
checked-in fixtures; they do not prove agreement with the user's installed
Balatro build.

## Confirmed deviations

### 1. Three unseeded `math.random` choices are deterministic in Jackdaw

Vanilla uses raw `math.random(1, 2)` for:

- Charm Tag's Mega Arcana pack variant;
- Meteor Tag's Mega Celestial pack variant;
- the first Buffoon pack variant.

Jackdaw always chooses variant 1. This is explicitly documented in
[`rng.py`](../jackdaw/engine/rng.py:59), [`tags.py`](../jackdaw/engine/tags.py:242),
and [`shop.py`](../jackdaw/engine/shop.py:230). This is a deliberate
reproducibility tradeoff, not an accidental test failure.

### 2. Stone-card rank identity is simplified

Vanilla returns a random negative ID for a Stone Card. Jackdaw returns the
constant `-1` from [`Card.get_id`](../jackdaw/engine/card.py:559). Normal hand
classification excludes negative IDs, so ordinary scoring should agree. It can
still matter in unusual code paths that observe the exact negative ID or rely
on repeated random calls.

## Deliberate non-vanilla behavior outside the core game engine

The solver and RL environments may auto-order Jokers or start from synthetic
states. Those are policy/training conveniences, not claims about the player's
vanilla UI. The distinction is recorded in
[`docs/engine_changes.md`](engine_changes.md:95).

## Live observations for Update 1.0.1o

The user reports the following behavior from the installed Update 1.0.1o build:

- Photograph follows hand order, not selection/click order.
- Stone Cards do not trigger Idol, Fibonacci, Hack, or Raised Fist.
- When the entire deck is Stone, Idol appears to use Ace of Spades as its
  fallback target.
- The held/scored Gold Card, Gold Seal, Rental, Golden Joker, and interest
  ordering checks all agree with the engine.

These observations close the ordering, Stone interaction, and economy questions
below. The all-Stone Idol result should be treated as a vanilla fallback rule,
not as evidence that Stone Cards have an Ace rank.

The tag difference remains technically real but low priority for gameplay
equivalence: both implementations create the same tag-granted pack category;
Jackdaw fixes an internal variant choice to `_1`, while vanilla can choose `_1`
or `_2`. It matters primarily for bit-exact seeded replay and pack contents,
not for the tag's visible effect or strategic category.

## Questions requiring live tests

Please run these against the same Balatro version you want the simulator to
model, ideally through the existing BalatroBot validator:

1. Run `jackdaw validate` and report the failed scenario names and diffs. The
   validator compares the simulator and live game state after identical
   actions; BalatroBot provides the live-game control/API surface
   ([repository](https://github.com/coder/balatrobot)).
2. Check whether the game build is the same balance/data revision represented
   by this checkout. Official patch notes have changed Joker, tag, blind,
   sticker, and voucher behavior across revisions
   ([Steam patch notes](https://store.steampowered.com/news/posts/?enddate=1714581521&feed=steam_community_announcements)).
3. For the known RNG deviation, compare repeated runs of the same seed using
   Charm Tag, Meteor Tag, and the first Buffoon pack if bit-exact pack contents
   become important. Record whether the live variant can be 2 while Jackdaw
   always returns 1.

Based on the Update 1.0.1o observations, the honest claim is: **the core engine
matches the tested gameplay behaviors, including Photograph ordering, Stone
Card exclusions/fallback behavior, and cash-out economy ordering. It is still
not bit-equivalent in the low-priority tag-variant and Stone-ID implementation
details, which matter mainly for exact seeded replay.**

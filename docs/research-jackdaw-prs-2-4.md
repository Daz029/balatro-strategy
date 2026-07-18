# Jackdaw upstream PRs #2–#4

Research date: 2026-07-18. Sources are the open GitHub PRs and their head
commits from `TylerFlar/jackdaw-balatro`. The local `enginge-fixes` branch name
is misspelled in the repository; the active checkout was left unchanged while
that branch was inspected directly.

## PR #2 — live-validated engine fixes

[PR #2](https://github.com/TylerFlar/jackdaw-balatro/pull/2) is authored by
Khetnen and targets `main`. Its head commit is
[`aaf24f9`](https://github.com/TylerFlar/jackdaw-balatro/commit/aaf24f93b4f22d3ee70a9099a211a7a6a93bef7e).
The PR reports live BalatroBot validation improving from 266/275 to 270/275,
with the remaining five failures attributed to three Steamodded RNG artifacts
and one unresolved `used_jokers`/pool divergence affecting two scenarios. It
also reports 1480 offline tests passing, including LuaJIT cross-validation.

The patch contains ten related vanilla-compatibility fixes:

1. Planet pools only soft-lock Planet X, Ceres, and Eris, and only after their
   associated hand has never been played. The existing code filtered regular
   planets too broadly and used hand visibility rather than play counts.
2. `create_card` records every created center key in `used_jokers`, matching
   vanilla `Card:set_ability`; this affects shop displays, packs, and
   consumable/joker-created cards, not only purchased Jokers. The PR updates
   the affected shop fixtures.
3. Purple Seal creates a Tarot selected through the Tarot pool with append
   `8ba`, instead of always creating The Fool.
4. Riff-Raff, Cartomancer, and 8 Ball use the normal pool resolver with their
   respective append keys instead of hardcoded cards.
5. Top-up Tag creates a forced-Common Joker with append `top`.
6. Idol, Mail, Ancient, and Castle targets roll at run start and round end,
   not at every round start; target selection considers all playing-card zones.
7. Castle reads its suit from `current_round.castle_card` through the scoring
   snapshot rather than from an ability field that is never populated.
8. Cryptid copies are put into the hand with fresh sort IDs when used during
   hand selection, rather than always being appended to the draw pile.
9. The Fool's can-use check receives game state, allowing it to see
   `last_tarot_planet`.
10. Descriptor-created cards use `soulable=False`; Soul/Black Hole rolls are
    reserved for pack-created cards rather than consuming `soul_*` RNG streams
    during consumable/joker effects.

### Relationship to `enginge-fixes`

This is the important PR, but it is not a clean cherry-pick onto
`enginge-fixes`. The branch already changes several of the same files for
economy ordering, Idol wiring, Blueprint compatibility, Riff-Raff capacity,
and related lifecycle behavior. A direct patch check failed across all eight
PR #2 code/fixture files. The sensible integration shape is a manual, focused
merge, preserving the branch's existing fixes and adding the missing upstream
semantics. In particular, recheck the interaction between PR #2's target-roll
timing and the branch's cash-out lifecycle, then add regression tests for pool
filtering, creation-time `used_jokers`, target timing/zone coverage, Cryptid,
and descriptor Soul rolls before accepting the fixture updates.

## PR #3 — annotate Steamodded-only validation failures

[PR #3](https://github.com/TylerFlar/jackdaw-balatro/pull/3) is authored by
Khetnen; its head commit is
[`278ef1f`](https://github.com/TylerFlar/jackdaw-balatro/commit/278ef1f7d66cd2d7e5f85a7a5a65d31ecdd7da41).
It changes only the descriptions of the Familiar, Grim, and Incantation live
scenarios. The rationale is that Steamodded takes ownership of those spectrals
and consumes suit/rank RNG in a different order, while BalatroBot requires
Steamodded. The simulator is intended to follow vanilla semantics, so these
three failures cannot be treated as a vanilla-oracle failure.

There is no engine or test-runner behavior change: `EXPECTED-FAIL` is plain
text in the scenario description. It helps a human reading validation output,
but does not automatically exclude or classify failures in CI.

## PR #4 — fix shop-oracle script import path

[PR #4](https://github.com/TylerFlar/jackdaw-balatro/pull/4) is authored by
Khetnen; its head commit is
[`dec6ab0`](https://github.com/TylerFlar/jackdaw-balatro/commit/dec6ab0d116cbe3cef66479e38544ef066ee0aec).
It changes one line in `scripts/generate_fixtures/run_shop_oracle.py`:
`Path(__file__).resolve().parent.parent` becomes `.parents[2]`, correctly
reaching the repository root from the nested `scripts/generate_fixtures/`
directory so standalone execution can import `jackdaw`.

This is low-risk and independent of the engine semantics. It is worth taking
if that generator is part of the workflow, but it is not an engine fix.

## Initial recommendation

Consider PR #2 for selective integration after focused tests and a manual
merge. Take PR #4 independently. Take PR #3 only if clearer validation output
is useful; it should eventually become structured expected-failure metadata if
the validation tooling needs to act on it automatically.

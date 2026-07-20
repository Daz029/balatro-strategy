# Engine PR-2 fixes — downstream impact assessment

Branch: `engine-pr2-fixes` (off `main` @ `67c3e8d`)
Commits: `b74ff35` (PR port), `297fc47` (Showman), `b17e91a` (tests)
Date: 2026-07-20

Ports [TylerFlar/jackdaw-balatro#2](https://github.com/TylerFlar/jackdaw-balatro/pull/2)
plus two adaptations our copy required. Scope of this document: **which banked
artifacts these fixes dirty, and why**. Nothing downstream was regenerated.

## Verification status

| Check | Result |
|---|---|
| Engine suite | 1073 passed, 14 skipped (baseline 1054 + 19 new) |
| Non-engine suite | 1300 passed, 5 deselected (pre-change baseline, re-run pending) |
| Ruff on touched files | clean; repo-wide count unchanged from `main` |
| New tests vs parent commit | every one verified FAILING in an isolated worktree |
| Oracle fixtures | **regenerated**, independently reproduce all 5 collaborator values |

The fixture corroboration is the strongest external evidence the port is
correct: our engine, changed independently, produces exactly the values the
collaborator hand-wrote (`c_ceres`→`c_saturn`, `c_eris`→`c_uranus`,
`c_uranus`→`c_neptune`, `c_eris`→`c_uranus`, `c_ceres`→`c_pluto`).

## The two rules that decide blast radius

Both are this project's own, from the C2 capture-skew work:

1. **Snapshots.** An engine fix that changes *computation* is inherited by a
   restored harvest blob for free — the blob is re-scored by current code. A fix
   that changes *stored state* is **not**: the blob's stale cache wins, because
   fidelity to the capture is what preserves the old bug.
2. **Labels.** A label is *never* re-scored by current code. Every
   label-semantics change since a corpus was generated dirties that corpus.

## Per-fix classification

| Fix | Changes RNG/pool? | Snapshot class |
|---|---|---|
| `create_card` registers `used_jokers` | **yes, run-wide** | **stored state** |
| `has_showman` derived from joker list | yes (when Showman owned) | computation |
| `resolve_create_descriptor` `soulable=False` | yes (drops a soul roll) | computation |
| Top-up Tag `soulable=False`, `append="top"` | yes | computation |
| Purple Seal rolls the real Tarot pool | **yes (new RNG draw)** | computation |
| Riff-raff / Cartomancer roll real pools | **yes (new RNG draws)** | computation |
| `reset_round_targets` round-start → round-**end** | yes (stream timing) | **stored state** |
| `reset_round_targets` all zones (deck+hand+discard) | yes (different card drawn) | **stored state** |
| Planet softlock → only Planet X / Ceres / Eris | yes (pool contents) | computation |
| `_sync_played_hand_types` `visible` → `played > 0` | yes (pool contents) | computation |
| `can_use_consumable(game_state=…)` (The Fool) | no | computation |
| Castle suit via `GameSnapshot` | no | computation |
| Cryptid copies → hand, fresh `sort_id` | no | computation |

### The one irreparable item

`used_jokers` is **stored state, and the lost information is not recoverable**.
Blobs in `data/harvest_s0/blobs` were captured at sha `57f1088`, when only
buy/pick sites registered. Under the fixed engine those same run-states would
carry many more registered keys — every joker, tarot, planet and spectral ever
*displayed*. Restoring an old blob therefore under-registers, so any pool roll
made from it is more permissive than a fresh run at the same point.

This is **not** repairable the way the Idol `id` bug was. That one was a pure
function of stored state (`id` derivable from `rank`), so it could be
recomputed on restore. Here, the set of keys merely displayed at past shop
visits is genuinely gone — the blob never recorded shop contents from prior
antes.

Practical severity is **bounded, not zero**: hand-play labeling rarely rolls
consumable/joker pools. It does so at Purple Seal discards, mid-round
consumable creation, and any joker-triggered creation — a minority of states,
but a real one. It is *not* bounded for anything that resumes shop play from a
harvested blob.

## Artifact verdicts

| Artifact | Verdict |
|---|---|
| `data/harvest_s0/` blobs | **Restore fine; continuations diverge.** Metadata, manifest and record IDs stay valid. Carries the irreparable `used_jokers` skew above. |
| `manifests/h1_harvested.json` | **Valid.** Record IDs unaffected; only the labels produced from them change. |
| `data/hand_agent_demos_v4/` (stages 1–5) | **Stale.** Generated pre-fix, and labels are never re-scored. Affected wherever a solve touched a pool roll, a targeting card, Castle, The Idol, or The Fool. |
| stage2 brute-exact rows | **Stale on the same grounds**, and no longer re-derivable as brute-exact more cheaply than the rest. |
| `runs/shop_ppo/` (`s0_a4_v4`) | **Not corrupt, but its values are "given the old engine."** It was trained where duplicates could repeat indefinitely, Ceres/Eris were mis-gated, and Showman was inert. Its critic — which the h1 `V_curve` money term consumes — inherits that. |
| `data/reductions.json` dollar marginals | **Usable.** Money marginals are a coarse group-by; these fixes do not plausibly move them enough to matter. |
| Validation JSONs (B7 sweep, K3 arms, fingerprint A3) | **Stale in principle.** All were measured on the pre-fix engine. The B7 depth-gate verdict is qualitative and probably survives, but it is no longer *measured* on shipping code. |

## Open decisions (not taken here)

1. **Does h1 regenerate?** The v4 pool is stale by rule 2. Cheapest honest
   measurement before committing to a full regen: relabel a small stratified
   sample (~200 states) on the fixed engine and diff against the banked labels.
   A near-zero delta argues for keeping the pool with the skew documented; a
   large delta forces the regen.
2. **Does `s0` retrain?** Its pool behaviour is now wrong in a way that
   directly affects shop value estimates (duplicate availability is a shop-side
   concern above all). Arguably the strongest case for regeneration of anything
   listed here — but it is also the most expensive.
3. **Harvest recapture.** The only way to clear the `used_jokers` skew is a
   fresh capture pass. Worth deferring until (2) is decided, since a recapture
   against a stale `s0` buys little.

## Known-open, not fixed here

- The aliasing between `gs["jokers"]` and the `jokers` parameter of
  `_apply_setting_blind_mutations` is load-bearing and undocumented:
  `_resolve_create_descriptors` appends to `gs["jokers"]`, so if a caller ever
  passes a list that is not that object, Riff-raff/Cartomancer creations land
  in the wrong list silently. Production is safe (`game.py:179` passes
  `gs.get("jokers", [])`, which is the same object whenever the key exists).
- `hand_potential_features` still takes `four_fingers`/`shortcut` but not
  `smeared` — an informativeness gap on the same axis, pre-existing and not
  label-corrupting.

# Pre-Regeneration Handoff

**Audience:** any agent implementing the pre-regen work. This document assumes you
have NOT read the grilling transcript. It tells you the sequence, the exact work,
and — most importantly — the specific ways this work goes wrong. Read the whole
Pitfalls section before writing any code. The decision record (the *why* behind
every choice here) is CLAUDE.md → "Pre-regeneration build plan — GRILLED AND LOCKED
(2026-07-12)". If this document and CLAUDE.md ever disagree, CLAUDE.md wins; fix the
drift.

## Context in three paragraphs

The hand agent (h1) is about to be retrained from scratch: new BC demonstration
data, new observation schema, new action head (Candidate B, an autoregressive
pointer). All demonstration shards get regenerated ONCE, on the 9600X, at
~12s/example × ~42k examples. Because that regen is expensive and happens exactly
once, **every change that affects label semantics or shard schema must land before
it**. That set of changes is this handoff.

The new data has two sources: the existing domain-randomized stages 1-4 (regenerated
with the new schema and fixes), and a NEW "harvested" stage (~8k examples) — real
mid-run states captured from rollouts of the final shop agent (s0 = `s0_a4_v4`,
deterministic) paired with the current hand agent (h0.5 =
`runs/hand_ppo/hand_ppo_2000000_steps.zip`), then labeled by the exact solver.

Work splits into three phases. **A** (harvest + evidence) is schema-independent and
runnable immediately on this machine. **B** (schema bump + solver changes) is where
all label-semantics changes live. **C** (selection + labeling front-end) is gated on
B. The regen itself is out of scope.

## Sequence and gating

```
A1 harvest script ──> A2 run passes + reductions ──> (corpus banked)
                                └─> A3 fingerprint eval ──> gates B6

B1 add_to_deck fix ─> B2 feature bump ─> B3 best_joker_order ─> B4 labels/width/tier-1
                                                                    └─> B5 prescreen+validation
A3 result ──────────────────────────────────────────────────────────> B6 (conditional)
B5's ranking precedent ─────────────────────────────────────────────> B7 discard-ranking fidelity

[all of B done] ──> C1 selection manifest ──> C2 snapshot-fed labeling front-end
                                                   └──> regen (9600X, not this scope)
```

Hard rules:
- A1/A2 can run in parallel with all of B (blobs are engine state, not encoded obs).
- Nothing in C runs until ALL of B is merged (B6 only if A3 triggered it; B7 always).
- A3 must be resolved (triggered or cleared) before B6's go/no-go, and B6 —
  if it happens — must finish before C2.

## Status (A1/A2/B1 + B2 slices 1–3 on `main`; slice 4 on `worktree-pre-regen-b2-slice4-schema-v2`)

- **A1 — DONE** (`scripts/harvest_s0_rollouts.py` + tests): capture pipeline,
  dual pass, per-run blob shards + blob-free metadata, reductions + readout.
- **A2 — DONE** (run on the 9600X, corpus transferred to `data/harvest_s0/`):
  26,978 det hand records (+18k shop), ante coverage to 7, 141 distinct jokers.
  **KEY READOUT: `max_hand_size = 8`, zero hands > 8** — the circular gate,
  confirmed: s0/h0.5 never reach +hand-size builds. So (a) it vindicates
  committing Candidate B without the evidence gate, and (b) the harvested BC
  stage is BLIND to wide hands — wide-hand coverage for BC comes from stages
  1-4 (the B1 `add_to_deck` mechanism + the flat tail set from GAME KNOWLEDGE,
  not the harvest histogram, which is a circular zero here). Money marginals are
  valid (money isn't circular).
- **B1 — DONE** (`hand_play_adapter.py` + tests): `add_to_deck` passives applied
  once per injected joker; flat hand-size tail knob (off by default, modest range
  set from game knowledge per the A2 finding above).
- **B2 slices 1–3 — DONE** (2026-07-13): slice 1 `hand_potential_features` /
  `encode_hand_potential` in observation.py (D_HAND_CARD=18, D_HAND_GLOBAL=256;
  shared shop constants untouched; engine-mirror test pins the window model
  against `get_straight`/`get_flush`); slice 2 `jackdaw/env/trigger_match.py`
  (match matrix + 4-class taxonomy over all 150 vocab jokers, import-time
  coverage hard-fail — it caught `j_cloud_9` immediately); slice 3
  `resolve_copy_targets` + copy-column inheritance via the engine's own
  resolution helpers. TWO engine bugs found + fixed on the branch, both the
  Throwback class (handler unit-tested, integration broken): The Idol could
  never fire (`reset_round_targets` stored idol_card without the "id" the
  handler compares against), and Blueprint/Brainstorm ignored
  `blueprint_compat` (all 29 incompatible jokers were copyable — e.g. a
  Blueprint beside an Egg doubled its end-of-round growth).
- **B2 slice 4 — DONE** (2026-07-13, branch
  `worktree-pre-regen-b2-slice4-schema-v2`): `SCHEMA_VERSION = 2` in the demo
  writer (18-wide hand cards, 256 GC, trigger_match, joker/copy id arrays,
  real consumable block); `build_observation_v2` / `observation_space_v2` as a
  VERSIONED SEAM in `hand_play_gym.py` (v1 stays byte-identical and the
  `HandPlayGymEnv` default — `obs_version=2` opt-in — so every h0.5 consumer
  and A3 keep working); `train_bc.py` originally loaded v2 only (v1 =
  pre-regen data, rejected loudly) with a width-generic axis-1 up-pad
  covering the 4-D trigger_match. Decisions made while building:
  - **v2 was NOT FROZEN until B4 landed.** B4 promotes the completed shard
    schema to **v3**: index-set labels replace the v2 card-target mask and
    hand blocks retain their actual shard width. Do not generate v2 datasets
    from the gap; the loader rejects v1/v2 loudly.
  - **Consumable block = 8 PER-INSTANCE rows in engine slot order**
    (`MAX_CONSUMABLES_V2`), tail-truncating: 2 (the v1 dormant width) is too
    narrow for harvested states (Crystal Ball = 3 slots; Perkeo negatives
    exceed any slot count), and width is nearly free. Stacked (type+count)
    rows REJECTED: row index must stay engine slot index for the h2
    UseConsumable/SellConsumable addressing (the shop obs invariant), and a
    stacked shard would force exactly the h2 re-regen the rider prevents —
    the stacked view is a lossy projection the model can compute internally.
  - **Copy-target fields store the frozen-vocab key id, not the 24-dim
    descriptor** (a pure function of the id; storing vectors would be a
    drift surface — the same id-not-vector pattern as `joker_ids`).
  - The current `HandPlayBCModel` trains against the v2 space by consuming
    the widened float blocks and ignoring the new keys; the embedding-gather
    encoder that consumes them is post-regen scope (recorded in CLAUDE.md).
  - **GOAL — v2 joker cap raised 5 -> 15, dual-counter expand-not-truncate**
    (2026-07-15, branch `worktree-joker-cap-15`): the v2 joker block inherited
    the v1 width-5 cap, which TRUNCATES negative-edition jokers (Negative shop
    buys, the Negative tag) — exactly the wide/negative builds a shop agent should
    value. Worse, the demo writer's joker `_pad_entities` was a BLANKET raise
    (not the negative-aware overfill check), so phase-2 harvest labeling would
    QUIETLY DROP every harvested >5-joker state to `failures.jsonl` (the
    Tier-1-executability drop class, but for jokers). Fix: a SEPARATE
    `MAX_JOKERS_V2 = 15` constant (v1 `MAX_JOKERS = 5` stays FROZEN — it is
    h0.5's exact obs and must not move, mirroring the
    `MAX_CONSUMABLES`/`MAX_CONSUMABLES_V2` split), applied to
    `build_observation_v2` / `observation_space_v2` AND the demo writer as a
    single imported source of truth (the BC loader does NOT up-pad the joker
    axis, so writer width must equal obs width; `trigger_match`'s joker axis
    rides this too). Dual counter: the game-view capacity check
    (`_check_joker_overfill`: `nonneg_count > joker_slots` -> loud engine-bug
    raise) is unchanged; the model-view array now holds up to 15 REAL jokers
    (negatives included), truncating lowest-slot-first only past 15 as a pure
    safety valve (same discipline as the width-40 hand tail). 15 = generous
    past any realistic negative stack. Shop obs (`MAX_JOKER_ROWS = 8`, coupled
    to SellJoker x8) is a SEPARATE follow-up, deliberately NOT bundled.
- **A3 — DONE, verdict CLEARED (2026-07-14): B6 is NOT built.**
  `scripts/fingerprint_discard_bias.py`, report `data/fingerprint_a3.json`
  (main checkout). h0.5 deterministic, 800 EVAL_ episodes/stage; teacher
  stats from shard labels (buckets recovered from GC[13]/GC[14]).
  - Signal 1 (locate): FIRES — recovery in discards>=2 buckets far below
    discards==0: deficits +30.2pts (stage2, CI [8.0, 54.6]), +79.0pts
    (stage3, CI [42.3, 119.5]), +104.9pts (stage4, CI [50.2, 165.4]);
    stage1 +17.0pts (CI includes 0).
  - Signal 2 (attribute): EXONERATES — h0.5 discards BELOW the teacher's
    rate in exactly those buckets (gap −0.113 / −0.080 / −0.067 in stages
    2/3/4, CIs entirely below zero). Under-discarding cannot be caused by
    the play-only label bias (pitfall #15) → conjunction fails → the
    solver's banked-discard estimator stays untouched; the deficit is a
    TRAINING problem (h1's PPO/BC), not a label problem.
  - Magnitude caveat for future readers: d0 recovery exceeds 1.0 in
    stages 3/4 (1.24/1.37) — the MC future-hand labels are documented-
    pessimistic, so recovery ratios inflate in buckets whose labels lean
    on MC; part of the deficit is metric asymmetry. The greedy control
    (stage1: deficit 62.3pts with a policy that barely banks discards)
    supports that read. None of this changes the verdict — attribution
    already failed on sign.
  - Archetype decomposition (calibrates the B2 hand-potential bump):
    pair-family recovery beats flush/straight in EVERY stage — stage2
    0.562 vs 0.482/0.476, stage3 0.675 vs 0.467/0.410, stage4 0.542 vs
    0.368/0.374. The flush/straight gap is real and is exactly what the
    v2 features target; expect post-bump recovery in those buckets to
    close toward the pair level.
- **B5 — DONE (2026-07-14):** `prescreen_play_candidates` +
  `best_immediate_play` prescreen path in `scripts/hand_solver.py`
  (n > PRESCREEN_HAND_LIMIT=8), validation harness
  `scripts/validate_prescreen.py`, 12 tests in
  `tests/scripts/test_hand_solver_prescreen.py`. Three decisions made
  while building (all user-grilled in-session):
  - **Family = the candidate's REALIZED scoring line** (hand type +
    scoring-card identity from the cheap eval), NOT the source template:
    kicker padding lets every weak template piggyback the dominant line
    (a lone Queen padded with four Kings scores as the same quads), so
    template-keyed diversity fills every slot with relabeled copies of
    one line — the exact pitfall-#13 crowding it exists to prevent.
  - **Ranking is JOKER- and HELD-AWARE** (user call): one fixed-order
    `score_hand` per candidate with true held cards, clones throughout.
    Jokerless ranking filters joker-favored lines before exact eval.
    The discard-side twin (`rank_templates_cheaply`, still jokerless/
    held-empty) is now B7 — user-locked as a pre-regen requirement.
  - **Validated k = 3, set to 4** (user margin call): sampled-distribution
    regret 0.0 at every tested k (3/5/8) vs noise floor 0.022; regret AND
    best-in-cut-rate (0.646) identical across k, so misses live in the
    candidate GENERATOR, not the cut depth — raising k buys nothing, and
    the lever for shrinking the boundary-stress exposure (mean 0.12
    p_clear with a synthetic blind at exactly the best play's total;
    0.02-0.03 at 1.1-1.5x) is candidate generation, not k. Report:
    `data/prescreen_validation.json` (main checkout).
  - **Pair pin** (user call): the best already-realized rank line (pair
    or better) is promoted to index 1, so every k>=2 cut evaluates it —
    weak cheap rank, but consistent (no draw, no luck) and rank-joker
    upside. Ordering stays prefix-stable in top_k (the harness slices
    k-cuts from one call).
  - Solver cost with prescreen: 0.7–10.4s full labels at hand sizes
    12–17 (within the ~12s/example budget; unprescreened n=14 is ~3.5k
    exact evals per recursion node).
- **B5 addendum — GENERATOR WIDENED, k=5 (2026-07-15):** all 17/48 misses
  were cross-rank-group generation holes (two pair / full house never
  proposed); rank-combination pass added, revalidated best-in-cut
  0.646 → 0.958, regret 0.0; 2 kicker-choice misses ACCEPTED (user call).
  Full record at the B5 spec below.
- **B3 — DONE (2026-07-15,** same slice-4 branch): `best_joker_order` +
  classification + `objective` hook in `play_ordering.py`; env mutation in
  `action_to_engine_action`; solver entry sort + per-candidate copy argmax
  in `evaluate_value`; brute-force-validated on constructed boards; 25 new
  tests. Full decision record at the B3 spec below.
- **B7 — code + tests DONE; validation REDESIGNED to faithful-MC
  (2026-07-15).** A diagnostic showed the solver's own value model CANNOT
  judge B7 (optimistic `_fill_hand_to_size` refill saturates the flipped-away
  branch at p_clear=1.0), so both the final-p_clear and shortlist-coverage
  harnesses were dead ends. New gate: disagreement-filtered, faithful Monte
  Carlo over real redraws, paired best-in-box, adaptive goal-line sweep, MC
  noise floor; `solve_hand_turn` gained a `joker_aware` comparison arm;
  constructed full-solver existence proofs (Greedy flip, faithfully better)
  are now the primary argument. 7/7 tests green; the ~200-state 9600X run is
  the remaining gate. Full record below.
- **B4 — DONE** (2026-07-15): shard schema v3 stores `(5,)` ascending
  `card_indices` with `-1` padding, replacing the width-bound mask; demo
  hand blocks are actual-width and the loader up-pads them to the width-40
  observation. Loader validation hard-fails malformed/unsorted/duplicate,
  out-of-hand-mask, empty, or budget-illegal labels. Wide labels remain in
  the dataset; the legacy flat trainer rejects them explicitly until the
  Candidate-B pointer trainer consumes the index sequence. Tier 1 is
  schema-native and the pinned position-8 synthetic hand executes through
  Tier 2. **C1–C2 — not started. B6 — SKIPPED, citing A3** (the conjunction failed on attribution; do not build it
  without a NEW fingerprint showing teacher-mirroring over-discarding).

## Task specifications

### A1 — `scripts/harvest_s0_rollouts.py` (new)

Drive full runs and capture every hand-turn decision state.

- Construct `ShopGymEnv(win_ante=8)` with `hand_policy` = h0.5 via
  `HandCheckpointPolicy`, shop policy = PPO checkpoint `s0_a4_v4`, deterministic
  (argmax) by default; `--sample-shop` switches the shop policy to sampling
  (h0.5 stays deterministic in BOTH modes — it is the partner being targeted).
- Capture mechanism: wrap the hand policy in `HarvestingHandPolicy(inner, sink)`.
  The interception point already exists — `shop_run_adapter.py` `_advance()` calls
  `self._hand_policy(self._gs)` on every SELECTING_HAND step. The wrapper pickles
  `self._gs` (same mechanism as `ShopRunAdapter.snapshot_state`: full state,
  RNG included) BEFORE delegating to the inner policy, then returns the inner
  policy's action unchanged.
- A record = engine blob + metadata row: `record_id` (= `{run_seed}_{turn_idx}`),
  `run_seed`, `turn_idx`, `ante`, `blind_type`/`boss_key`, `hand_size`, `dollars`,
  `hands_left`, `discards_left`, `source` (`det`/`sampled`), `git_sha`,
  `schema_note` (free text, e.g. engine test count at capture time).
- **Capture EVERYTHING** — no subsampling, no per-run caps at capture time.
  Blobs go in per-worker/per-run shard files (append-friendly); metadata goes in ONE
  flat table (JSONL is fine) so later queries never unpickle a blob.
- Also capture SHOP-state snapshots (a second sink at shop decision points, same
  runs) tagged `kind=shop` — these feed V_curve later; bank them now, one pass.
- Seeds: `HARVEST_{i:08d}` for the deterministic pass, `HARVEST_S_{i:08d}` for the
  sampled pass. NEVER any seed with the `EVAL_` prefix.
- Tests: blob restores byte-identically and continues (reuse the RNG round-trip test
  pattern from `tests/env/test_shop_run_adapter.py`); metadata row matches its blob
  (unpickle a sample and cross-check ante/dollars/hand_size); wrapper is transparent
  (inner policy's actions unchanged, run outcome identical with/without wrapper).

### A2 — run passes + reductions/readout

- ~1200 deterministic runs + ~500 sampled runs. Seconds per run; runs locally.
- Emit free reductions as group-bys over the metadata table, **deterministic corpus
  only**: per-ante `$` marginals at hand-turn entry (feeds stage 1-4 money regen)
  and the hand-size histogram (informs B's max decode length).
- Emit the coverage readout (both corpora, split by source): ante marginal
  (state counts at ante ≥2, ≥3), distinct jokers owned, distinct owned-joker
  key-sets (build families), hand-size tail. This readout has NO pass/fail
  threshold — it tunes the manifest's det:sampled ratio in C1 (default 75:25).

### A3 — discard-bias fingerprint eval (+ archetype decomposition)

Decides whether B6 (the banked-discard solver fix) happens. Two signals; the
trigger is their **conjunction**:

1. **Locate:** h0.5 deterministic clear rate ÷ shard label-mean `p_clear`
   (the free solver-ceiling trick), bucketed by starting
   `(hands_left, discards_left)`. Symptom = recovery in discards≥2 buckets
   materially below discards=0 buckets (>5 recovery points, outside a bootstrap CI).
2. **Attribute:** h0.5's discard *frequency* per bucket vs the solver labels'
   discard frequency (from shards, free). The label bias has a known sign — it can
   only push toward discarding TOO EAGERLY. Bias survived = h0.5 discards at or
   above the teacher's rate in the deficit buckets. Directionless error or
   UNDER-discarding = NOT the label bias; do not build B6.

Greedy (`GreedyHandPolicy`) as a control on stage 1 only. The flush/straight/pair
archetype decomposition of eval recovery runs in this same pass (calibrates how much
recovery to expect from the B2 features; it gates nothing).

### B1 — `add_to_deck` at injection (`jackdaw/env/hand_play_adapter.py`)

`reset()` builds jokers with bare `create_joker(key)` and never applies acquisition
passives — injected Juggler/Troubadour/Turtle Bean/Stuntman states carry a hand size
the real engine would never produce. Fix:

- After `_apply_scaling_state` has run for all jokers, call each injected joker's
  `add_to_deck(gs)` exactly once. Full passive — do NOT cherry-pick `h_size`.
- Add a flat hand-size tail knob to `HandPlayConfig` (e.g.
  `hand_size_delta_range` + a tail probability, mirroring the 20% flat money-tail
  pattern) for coverage beyond what sampled builds produce.
- Tests: Juggler injection deals 9 cards; decayed Turtle Bean deals base + decayed
  h_size (NOT +5); Stuntman deals 6; passives applied exactly once (no
  double-application on any path); Drunkard bumps discards on top of the sampled
  value; deterministic per seed.

### B2 — observation feature bump (`jackdaw/env/observation.py`, demo writer)

**Schema is HAND-AGENT-ONLY — shop obs stays frozen (DEFERRED ISSUE for the merge).**
s0 is frozen at `D_PLAYING_CARD=15` / `D_GLOBAL=235`, and the shop obs uses both
(`encode_global_context` verbatim + `D_PLAYING_CARD` for pack-targeting rows), so
`s1 --init-from s0` breaks if the shared constants move. B2's new features
therefore land on HAND-SPECIFIC paths only: a hand card width of 18 and a
hand-only `encode_hand_potential` appended to the hand global — `encode_global_context`
and `D_PLAYING_CARD` are NOT touched. This is also semantically correct (the shop
has an empty hand and picks pack cards, not poker hands — flush/straight potential
is dead input for it). CONSEQUENCE / OPEN ISSUE: the hand (18-wide cards, +21 GC)
and shop (15-wide, 235 GC) obs schemas now DIVERGE; reconciling them into one card/
global encoder is deferred to the in-blind MERGE (h2, out of scope here) — the
already-documented place a shop-side schema bump happens. Flag it there when the
merge is built.

**Build slices** (staged into separate tested commits, each stacking on the last):
1. **Flush/straight potential features** — per-card +3 + GC +21 (below), on the
   hand paths only per the note above. Pure O(n) obs additions, known-hand
   fixtures. Self-contained, lowest risk — first.
2. **Trigger-match matrix** — `trigger_match[card, joker, {scored,held}]` +
   `joker_center_key_id`, 4-class taxonomy with a build-time coverage check that
   classifies EVERY vocab joker (unclassified = hard error). Class-2 reads live
   state (pitfall #8); class-3 all-zero by design (pitfall #9).
3. **Copy resolution** (Blueprint/Brainstorm) — resolved-target ids + active-copy
   bit via the ENGINE's own resolution path (pitfall #11), match rows inherited.
4. **`schema_version` bump + loader up-pad** — hard-fail on unknown schema.
   **SEQUENCING FLAG (2026-07-13, found while building slices 1-3):**
   `build_observation` is consumed by `HandCheckpointPolicy` (the h0.5
   partner in the shop env) and `eval_hand_policy` — switching it to the
   v2 schema IN PLACE breaks h0.5's checkpoint obs width, and A3 (an h0.5
   eval) hasn't run yet. Land v2 as a versioned seam: the v1 path stays
   available and remains what the h0.5 checkpoint paths build, and the
   default flips only at h1 BC/PPO (whose nets are fresh anyway). Do NOT
   let slice 4 make A3 unrunnable; alternatively run A3 first.
   **RIDER (2026-07-13, locked):** shards STORE the real consumable block
   (encode owned consumables via `encode_consumable`; harvested states
   carry real ones, stages 1-4 write it empty) instead of the BC loader
   synthesizing zeros. Labels stay consumable-blind — the solver ignores
   consumables; say so in the writer docstring. This removes the one
   plausible pressure to re-regenerate shards at the h2 in-blind merge.
   In-blind consumable SELECTION stays at h2: P(clear-this-blind) is a
   dishonest objective for consumables (cross-blind value — the
   shop-visit-episode myopia argument), and it's PPO-side anyway, so
   nothing about it belongs to the label-semantics scope.

- Per-card +3: suit-count-of-my-suit /5; rank-count-of-my-rank /4; best
  straight-window occupancy among 5-rank windows containing my rank (occupancy =
  distinct ranks present ÷ ranks needed). Wheel window (A-2-3-4-5) included. Under
  Four Fingers, window length 4 (flushes also count at 4 — reflect in the suit-count
  normalization ONLY if the engine flag path confirms it; verify, don't assume).
  Under Shortcut, windows tolerate single-rank gaps. Flags via
  `get_hand_eval_flags(jokers)`.
- GC +21: per-suit counts /hand_size (4); per-rank counts /4 (13); max suit count
  /5 (1); best window occupancy (1); Four Fingers bit, Shortcut bit (2).
- Trigger-match matrix `trigger_match[card, joker_slot, {scored, held}]` (bool,
  written to shards) + `joker_center_key_id` int array (frozen vocab ids — from the
  same sorted-centers.json mapping the shop obs uses; the VOCABULARY FREEZE test
  applies).
- Trigger predicate: `(card, joker, gs) -> (scored: bool, held: bool)`. Implement a
  4-class taxonomy with a build-time coverage check that CLASSIFIES EVERY joker key
  in the vocabulary (unclassified = hard error at import/test time, not silent
  zeros):
  - Class 1 static per-card (suit/rank/face/parity conditions).
  - Class 2 state-dependent per-card (Ancient, Idol, Castle, Mail...) — reads gs.
  - Class 3 set-level (Flower Pot, Seeing Double, Blackboard, hand-type-conditional
    jokers) — all-zero rows BY DESIGN.
  - Class 4 non-card (economy etc.) — all-zero rows.
- Blueprint/Brainstorm: resolve the copy chain via the ENGINE's own resolution code
  path; joker rows gain resolved-target `center_key_id`, target descriptor, and an
  active-copy bit (inactive when pointing at nothing/incompatible → zeroed target
  fields); match rows inherit from the resolved target.
- `schema_version` bump; `train_bc.py` loader updated (up-pads old widths, hard-fails
  on unknown schema).
- Encode budget: everything above must stay O(n) single-pass; the <500µs budget from
  the obs-limitation decision record still applies.
- Tests: known-hand fixtures for each feature (4-flush draw, open-ended straight
  draw, wheel draw, Four Fingers/Shortcut variants); taxonomy coverage (every vocab
  key classified); Ancient predicate tracks the rotating suit across states;
  Photograph marks ALL faces; Flower Pot rows all-zero; Blueprint inherits
  Photograph's face matches; inactive Blueprint zeroed.

### B3 — `best_joker_order` (`jackdaw/engine/play_ordering.py`)

- One function used by BOTH the env execution path and the solver. Signature ~
  `best_joker_order(gs, played_subset) -> ordering`.
- Algorithm: stable sort putting additive-mult jokers before x-mult jokers
  (closed-form optimal for the independent-joker chain); if Blueprint/Brainstorm
  owned, argmax over ≤4 copy-target placements, each scored by one cheap evaluation
  of the candidate play.
- Env (`HandPlayGymEnv` and the shop env's auto-resolved hand phase): compute once
  per COMMITTED play, apply as a persistent mutation of `gs["jokers"]`, then execute
  through `best_play_order` as before.
- Solver: apply the sort once per hand-turn (it is subset-independent); on the exact
  current-hand path, re-run the copy-target argmax per candidate subset; on the MC
  future-hand path, use the hand-turn-entry target (fixed).
- Solver hypotheticals must CLONE the joker list before mutating (extend the
  `fast_clone_*` family). This is the stage-4 shared-blind bug class — assume any
  in-place mutation of shared state during hypothetical evaluation is a bug until
  proven otherwise.
- Tests: brute-force 120-permutation ground truth on constructed boards — pure
  additive, pure x-mult, mixed, with/without Blueprint, with per-card-phase jokers
  (Greedy) present. The closed form is only PROVEN for the independent phase;
  the per-card-phase interleaving is verified empirically here. Also: solver/env
  consistency test (the committed action's label value equals env execution value
  under the same subset).
- **BUILT 2026-07-15** (same slice-4 branch). Decisions grilled with the user
  in-session, on top of the locked spec:
  - Classification: `MAIN_PHASE_XMULT_JOKER_KEYS` (28 keys, jokers whose main
    handler can return Xmult_mod) is hand-written for import cheapness but
    PINNED to the handler SOURCE by a regenerating scan in
    `tests/engine/test_play_ordering.py` — a handler drift breaks CI, not the
    sort. Polychrome edition on any joker classifies it x-mult (9d applies at
    its position). Misclassification only costs order-optimality, never label
    honesty (the engine scores whatever order is submitted).
  - Copy-joker placement: argmax over ALL insertion slots (board size does not
    cap it — 6+ jokers just means more slots); with MULTIPLE copy jokers, FULL
    cross-product of ordered placements up to 10 total jokers (user call;
    2 copies among 8 = 90 candidates), sequential-greedy beyond. Candidate
    exclusion is id()-based (two Blueprints via negatives — the Erratic-deck
    equality bug class).
  - `objective` hook (user call, decided over my initial doc-note-only
    recommendation): `best_joker_order(..., objective=)` re-targets the
    placement argmax; default None = raw score. Channel that justified it:
    copy-joker placement DECIDES what gets duplicated, and a score-argmax
    never copies economy effects. The copyable money flows THROUGH scoring
    (Business Card per-card $, lucky-money via copied retriggers) so it lands
    in ScoreResult.dollars_earned; end-of-round payers (Golden/Rocket/Egg) are
    blueprint-INCOMPATIBLE and never copyable. The double-agent env passes a
    money-aware objective at the h1 seam via `action_to_engine_action`'s
    `ordering_objective` param; solver labels stay score-only — loose
    label/env convergence accepted (PPO-against-real-game corrects). Pinned:
    a dollars-first objective flips Blueprint's neighbor to Business Card.
  - Env: mutation lives in `action_to_engine_action` (the single shared decode
    path — HandPlayGymEnv AND checkpoint partners), plays only, gated by
    `joker_order_matters` (copy joker owned, or x-mult coexisting with any
    other joker). Solver: context-free sort once at `solve_hand_for_ante_clear`
    entry; `evaluate_value` re-runs the full best_joker_order per candidate on
    the exact path only (MC tier keeps the entry order — its approximation
    tier). Solver/env consistency + mutation-safety pinned in
    `tests/scripts/test_hand_solver_joker_order.py`.
  - Known-deferred: Baseball Card contributes x-mult at OTHER uncommon jokers'
    positions (9c) — the binary sort ignores the rider (second-order, labels
    stay engine-honest).

### B4 — index-set labels, width 40, tier-1 rework (`scripts/generate_hand_demos.py`)

- Labels: `(5,)` int array, ascending, -1 pad; delete `card_target_mask` and the
  `idx >= MAX_HAND_CARDS` GenerationError.
- `MAX_HAND_CARDS_OBS` 12 → 40 in `hand_play_gym.py`; demo writer writes ACTUAL hand
  width; loader up-pads (existing pattern from the 12-width fix).
- Rewrite `validate_label_executability` tier 1 schema-native: indices unique,
  sorted, within actual hand length; 1-5 cards; PlayHand requires hands_left ≥ 1,
  Discard requires discards_left ≥ 1. Tier 2 (engine execution) unchanged.
- Tests: a synthetic big-hand state (hand > 8) with a label touching position ≥ 8
  passes tier 1 and executes through tier 2 (this exact case silently vanished
  under the old tier 1 — pin it).

### B5 — solver prescreen + validation harness (`scripts/hand_solver.py`, new script)

- Prescreen: for hands with n > 8, `best_immediate_play` evaluates only top-k
  template-derived subsets from `rank_templates_cheaply`, selected FAMILY-DIVERSE:
  best candidate per template family first (each flush suit, each straight window,
  rank-line groups), then fill remaining k by rank.
- Validation harness (one-time script): ~50 hands flat over sizes 9-12 (dealt via
  B1's hand-size knob), run FULL brute force once per hand, score prescreen cuts at
  k = 3, 5, 8 (say) from the same evaluations. Metric = regret:
  `p_clear(brute best) − p_clear(prescreen choice)`, both valued by brute force.
  Noise floor: re-solve a sample of n=8 states with different `mc_seed`s, compute
  the same regret between seeds. Accept the smallest k with mean regret ≤ ~1.33×
  the floor. Record the chosen k in CLAUDE.md.
- If no k passes: raise k / widen families; if still failing, the prescreen design
  returns to review — do NOT ship a failing k because the schedule wants it.
- **GENERATOR WIDENED + REVALIDATED 2026-07-15** (rode in on the B7 harness
  session): reading the first validation run's per-hand data showed ALL 17/48
  misses were GENERATION holes (`best_in_candidates=False` in every one, at
  every k) of a single class — cross-rank-group combinations (Two Pair from
  two separate pairs, Full House from trips + pair). Root cause: kicker
  padding ranks INDIVIDUAL cards by nominal priority, so a second rank group
  never wins a padding contest against a loose Ace — the docstring's claim
  that FH/two-pair "emerge from padding" was false in practice. Fix: a
  rank-combination pass in `prescreen_play_candidates` — every ordered pair
  of 2+-card rank groups proposes 2+2 (bare AND kicker-padded; bare matters
  when the 5th card is better held — Baron/Blackboard class) and 3+2 when a
  group has trips. Revalidation (48 hands): best-in-cut 0.646 -> 0.958,
  regret 0.0 on 44/48 MC-active states, stress regret f=1.0 0.12 -> 0.02 /
  f=1.1 -> 0.0, wall-time cost ~free, minimal k still 3.
  `PRESCREEN_TOP_K` raised 4 -> 5 (user call, preemptive margin).
  **ACCEPTED RESIDUAL (user call 2026-07-15)**: 2/48 kicker-CHOICE misses
  (right line, wrong kicker — keep-priority pads nominal-best where a joker
  wants a suit/enhancement; ratios 0.92/0.71, measured regret 0.0). Named
  lever if it ever matters: kicker VARIANTS per combination, not k.
  Regression tests modeled on the failing hands:
  `test_hand_solver_prescreen.py::TestRankCombinationCandidates`.

### B6 (CONDITIONAL on A3) — banked-discard credit in `estimate_future_hand_distribution`

Do not design this until A3 triggers. If it triggers: the fix must go through B5's
regret harness (prescreen-style validation vs brute force on discard-rich states)
before any label is generated with it. Budget expectation: it will slow labeling;
that cost lands on all ~42k examples — keep the implementation's cost profile
explicit in the PR.

### B7 — joker/held-aware discard-branch ranking (`scripts/hand_solver.py`)

USER-LOCKED 2026-07-14 (rode in on the B5 session): `rank_templates_cheaply`
— the top-k selection of which discard templates the recursion explores —
still ranks with jokerless, held-empty `score_hand_base`. Same failure shape
the B5 play prescreen just fixed: a joker- or held-favored line (Greedy's
suit, Baron's held Kings) can rank below its true value and never reach the
exact hit/miss recursion. This is a LABEL-SEMANTICS change for every hand
size (the discard path runs at n<=8 too), so it must land before C2/regen.

- Mechanics (BUILT 2026-07-14, same session as B5): `_ranking_score` — the
  shared ranking-tier scorer (one fixed-order `score_hand`, fast-clones
  everything, returns a clone→original identity map) — now backs BOTH the
  B5 play prescreen and `rank_templates_cheaply`. Per-branch held =
  `kept` + any (hold+completion) overflow beyond the played 5; unknown
  replacement draws contribute nothing (same representative-completion
  tier). Only the RANKING scorer changed — reachability math and the
  exact recursion valuation are untouched, so labels shift only where a
  different template set gets explored. `jokers=None` / `joker_aware=False`
  keep the old jokerless scorer (legacy callers + the harness's
  comparison arm). Tests:
  `tests/scripts/test_hand_solver_discard_ranking.py` (Greedy flips the
  ranking toward its suit's flush; Baron raises a branch whose discard
  cap keeps a King in hand; escape-hatch equivalence; Eye mutation
  safety).
- The completion-card hypothetical (`_best_completion_cards`) stays — only
  the SCORER upgrades, not the reachability math.
- **Validation design REPLACED TWICE.** (1) The original old-vs-new
  final-`p_clear` compare tested the shared outer solver (the max
  saturates once either ranker keeps a certain-clear branch). (2) The
  2026-07-15 shortlist-coverage replacement was itself discarded after a
  30-state run and a diagnostic exposed a deeper flaw — see below.
- **THE FINDING (2026-07-15): the solver cannot judge B7 with its own
  value model.** A discard branch is valued by the two-point representative
  hit/miss recursion, and the "hit" hand is refilled by `_fill_hand_to_size`
  with the HIGHEST-remaining-rank optimistic filler. A `still_needed==0`
  branch (e.g. discard around trip Kings) therefore refills to a monster and
  clears EVERY goal line — its p_clear pins at 1.0. When B7 flips the emitted
  discard from that trips-refill line to a joker-favored flush line, BOTH read
  p_clear=1.0: the solver records a different, genuinely better ACTION but is
  structurally blind to the improvement. Any metric built on the solver's own
  recursion reads ~0 (and can read a FALSE regression, since the optimistic
  refill pins the old box at 1.0). The 30-state shortlist-coverage run
  ("new loses rank-2/3 coverage, rank-1 100%/100%, regret 0/0") was exactly
  this artifact — and its rank-2/3 / reach-bucket metrics measured the
  NON-argmax tail the solver's `max` discards, which is not label-relevant.
- **New gate = FAITHFUL Monte Carlo, disagreement-filtered, paired.**
  `scripts/validate_discard_ranking.py`, rewritten 2026-07-15.
  - *Box-set agreement ⟹ identical label.* Verified `solve_hand_turn`
    iterates all `top_k` with a strict-`>` max and no order-dependent early
    exit, so B7 can only move a label where the old (jokerless) and new
    (joker/held-aware) top-k SETS differ. Disagreement is a cheap two-pass
    filter (the box is chosen by `cheap_value`, goal-line-independent);
    agreement states are recorded exactly-zero and never valued.
  - *Faithful ground truth.* Value each candidate discard by REAL random
    redraws from the deck — discard, draw, `best_immediate_play`, fraction
    clearing — NOT the solver's optimistic recursion. Real draws de-saturate
    the trips-refill line (random draws rarely make a monster) while the
    flush completes at its true probability and the joker mult carries it.
  - *Paired best-in-box.* Report `best_in_new_box − best_in_old_box`; the
    global-best term cancels, so only `old_box ∪ new_box` is valued.
    Absolute regret vs ALL template branches is a B5 (generator-coverage)
    question, deliberately out of scope.
  - *Adaptive goal-line sweep.* `hands_left` AND `discards_left` forced to 1
    (one discard node, valued exactly as `P(best play ≥ chips_needed)`; no
    downstream solver model). Per-draw best-play totals are goal-line-
    independent, so sample ONCE per action and threshold for free. Goal
    lines = quantiles of the pooled achievable totals ABOVE the best
    play-now `P` (below `P` the label is not a discard). This adapts the
    band to each state's real reachable range — the "adaptive goal line"
    the flat sampled blind never provides.
  - *MC-reseed noise floor* (yes, it is needed — faithful values are MC;
    an earlier note claiming the `hands_left=1` isolation removed all MC
    was WRONG: it removes the FUTURE-HAND boundary, but the redraw MC is the
    whole point). A separate n≤8 pass reseeds the best action; a state
    counts as a B7 win only if `max_help > accept_factor × floor`; a
    `min_paired < −floor` is a real regression (the safety gate).
- **`solve_hand_turn` gains `joker_aware: bool = True`** threaded through the
  recursion (default = production). It is the comparison arm: the constructed
  existence proofs toggle it to get the two arms' EMITTED discards, then judge
  which is better by faithful MC.
- **Existence proofs are the PRIMARY argument** (not the aggregate):
  `tests/scripts/test_hand_solver_discard_ranking.py::TestFullSolverExistenceProof`
  — a constructed Greedy board where the full solver's emitted discard flips
  jokerless→joker-aware toward `flush_Diamonds`; the solver's own p_clear does
  NOT prefer it (`new.p_clear ≤ old.p_clear` — the saturation documented);
  faithful MC (200 draws) shows the flush clears >5pts more often above
  play-now; a d=2 variant pins the `joker_aware` threading through one level
  of recursion. B7 is a consistency/correctness fix (mirror the B5 play
  prescreen) — the aggregate answers how-often / does-it-regress, and is NOT
  expected to show large lift (argmax is robust; the effect is rare-but-real).
- **Status:** code + all 7 discard-ranking tests green; harness plumbing
  smoke (3 states, n_samples=20) completes and emits valid JSON. The real
  gate is a background/9600X run (`--n-states ~200 --n-samples ~80`) —
  headline numbers = disagreement rate, `frac_helped` above floor, and
  `n_regressions_below_floor` (must be 0). Do not read the tiny smoke as the
  gate.
- **RESOLVED 2026-07-16 — KEEP B7 + depth-gate the shortlist (branch
  `b7-topk-depth-gate` off main).** The 200-state gate was net-positive but
  NOT clean (25 helped / 9 regressed at `top_k=4`,
  `data/discard_ranking_validation.json`). A top_k sweep
  (`scripts/validate_discard_ranking_sweep.py`, k=4/6/8/12/64,
  `data/discard_ranking_sweep.json`) showed every regression lives at the
  rank-k cut: widening to 6 heals ~all hard cases (9->6 regressions), k=64 =>
  0 disagreements (no-op sanity check passes). Fix: production discard
  `top_k` is now depth-gated — **6 at `discards_left <= 2`, 4 at
  `discards_left >= 3`** (`_discard_shortlist_k`; `solve_hand_turn` /
  `solve_hand_for_ante_clear` take `top_k=None` = gate, explicit int = fixed
  box for the harness/tests; `generate_hand_demos` uses the default). Gated
  because solve cost scales `(k/4)^discards_left` (flat-6 ~2x the regen wall,
  flat-8 ~4x, with multi-minute deep-hand stragglers); the regressors all sit
  at shallow depth where the wide box is cheap, so keeping 4 at d_left>=3 caps
  the tail (~1.7x blended). Label-semantics change => still a pre-regen lock.
  Full reasoning + worked example (seed 247, Shoot the Moon) in CLAUDE.md
  "B7 discard-shortlist DEPTH-GATED WIDENING". Verified: ruff-clean, 168
  solver/generation tests green.

### C1 — selection script → manifest (new, small)

- Input: the metadata table. Output: a MANIFEST file — ordered list of `record_id`s
  (~8k) — plus the parameters used (seed, strata, ratios) embedded in a header.
- Selection rules: ante-stratified; ≤8 records per run with ≤2-3 per ante bucket
  within a run; det:sampled ratio default 75:25 (tune from A2's readout); fixed RNG
  seed so the same inputs always emit the identical manifest.
- The manifest is a versioned artifact — check it in (it's a few hundred KB).

### C2 — snapshot-fed labeling front-end (`scripts/generate_hand_demos.py`)

- New mode: consume a manifest + blob store instead of `HandPlayAdapter` sampling.
  Restore blob → `gs` → the EXISTING solve/encode body (`generate_one_example`'s
  logic from `gs = adapter.raw_state` down runs unchanged on a restored state) →
  `write_shard` into a new stage dir (e.g. `stage5_harvested`).
- `mc_seed` = the `record_id`. Worker partitioning: fixed ranges over the manifest
  order, `--num-workers` EXPLICIT and recorded (see Pitfalls #5).
- On load, compare each record's `git_sha` stamp to the current checkout; mismatch =
  hard error with an override flag (`--allow-sha-mismatch`) that logs loudly.
- Failures (solver exceptions, tier-2 rejects) go to `worker_N_failures.jsonl` and
  skip, per the existing design — but ALSO emit a summary count at end; if >2-3% of
  the manifest fails, stop and investigate rather than shipping a silently-thinned
  stage.

## Pitfalls — read before coding

1. **Tier-1 validation silently deletes the point of the harvest.** The old tier 1
   maps labels into the Discrete(436) mask; any label touching hand position ≥8
   raises and the example is *skipped, not failed*. If B4 ships without the rework,
   phase 2 completes "successfully" and every big-hand example is quietly missing.
   The failure summary in C2 is the backstop; the pinned test in B4 is the fix.
2. **`add_to_deck` ordering and idempotence.** Apply passives AFTER
   `_apply_scaling_state` (decayed Turtle Bean), exactly ONCE per joker (a second
   call doubles hand size), and BEFORE the deal (injection already precedes
   `SelectBlind`, keep it that way). Do not cherry-pick fields out of the passive.
3. **Solver/env divergence on joker order is a label-poisoning bug.** One shared
   function. If the solver values a play under an order the env won't reproduce (or
   vice versa), labels systematically overpay — the 5-card-discard-cap class of bug,
   which wasn't caught until BC consumed the data. The consistency test in B3 is
   non-negotiable.
4. **Hypothetical evaluation must not mutate shared state.** The joker list joins
   the blind/hand-levels/RNG in the `fast_clone` discipline. The stage-4 bug
   (shared `Blind` mutated by hypothetical `score_hand` calls) was invisible for
   months because earlier stages never exercised the mutating branch. Same shape
   here: order mutations during candidate ranking must happen on clones.
5. **Resume partitioning is fixed by (total, num_workers).** 2026-07-05 incident:
   resuming a transferred partial run with a different default worker count
   silently re-partitioned — 925 duplicated + 321 missing examples. ALWAYS pass
   `--num-workers` explicitly; C2 partitions over the manifest, which makes the
   record list stable, but the worker count must still be pinned and recorded.
6. **Seed discipline.** `HARVEST_`/`HARVEST_S_` prefixes are reserved; `EVAL_` seeds
   must NEVER be harvested (leaks the held-out eval suite into training data).
   `mc_seed` for a harvested example is its `record_id`, not the run seed alone
   (two records from one run must not share MC draws).
7. **Blobs are pickles of live engine objects.** An engine class change between
   capture and labeling can break unpickling — or worse, load fine while behavior
   has drifted. That's what the `git_sha` stamp is for: check it loudly in C2.
   If an engine bug fix lands mid-stream (history says it will), decide explicitly:
   re-harvest, or accept and document the skew. Never silently proceed.
8. **Class-2 trigger predicates must read live state.** Ancient Joker's suit
   rotates every round; The Idol's card changes; Castle's suit changes. A
   config-derived static table gives confidently wrong match bits that no test on
   fresh states will catch — test against mid-run states with rotated values.
9. **Class-3 all-zero rows are a decision, not a gap.** Flower Pot / Seeing Double /
   Blackboard / hand-type jokers have NO honest per-card bit ("which card triggers
   Flower Pot?" has no answer). Their signal is the GC set-structure features. Do
   not "helpfully" mark all cards, the scarcest suit, or anything else — fabricated
   per-card signal was explicitly rejected.
10. **Match bits mean "candidate", not "will fire".** Photograph marks every face
    card; only the first scored face gets the ×2. The bit is class membership;
    whether it fires depends on the chosen set/order and is the policy's job.
11. **Copy resolution must reuse the engine's path.** Reimplementing
    Blueprint/Brainstorm chain rules in feature code WILL drift from the engine
    (compatibility rules, chain termination). Resolve via the engine, inherit match
    rows from the resolved target, zero the fields when the copy is inactive.
12. **Regret, not disagreement.** In B5 and B6 validation, two near-tied plays
    "disagree" harmlessly. The metric is the p_clear cost of the choice under
    brute-force valuation. Calibrate acceptance against the n=8 MC-reseed noise
    floor, not against zero.
13. **Family-diverse top-k.** Naive top-k off `rank_templates_cheaply` can return k
    variants of one flush line and starve discard-toward-straight candidates whose
    cheap rank is systematically lower than their exact value. Best-per-family
    first, then fill.
14. **Deterministic-only reductions.** The `$` marginals and hand-size histogram
    describe the DEPLOYED policy's distribution; including sampled-pass records
    smears a deliberately-worse policy into the money prior. Filter on the source
    tag.
15. **The fingerprint trigger is a conjunction.** A recovery deficit in
    discard-rich buckets alone does NOT justify B6 — it's equally consistent with
    plain learning weakness. Only deficit + directional teacher-mirroring (h0.5
    discarding at/above the teacher's known-inflated rate) indicates the solver.
    Under-discarding can NOT be caused by this bias; if you see it, the solver is
    exonerated.
16. **Width 40 is obs-only.** No parameter scales with it, shards write actual
    width, the loader up-pads. Do not regenerate anything for width; the FEATURE
    changes are what force regen. Truncation beyond 40 stays lowest-first (no
    finite width is provably safe — Serpent compounding).
17. **B's decode grammar (post-regen, recorded here so it isn't lost):** picks in
    strictly ascending index order enforced by the per-step mask; explicit stop
    token as a fixed extra logit slot (not positioned at hand size); stop illegal
    at zero picks, forced at five. The mask constraint must be identical at BC,
    PPO, and eval — pin with a parity test when B is built.
    Two riders (agreed 2026-07-13, slice-4 session):
    - **Flat-head control on big-hand labels:** the BC-only validation's flat
      436-head control structurally cannot represent labels touching position
      ≥8 — it DROPS those examples, and the dropped fraction must be reported
      alongside the CE comparison so the numbers are read with that caveat.
    - **B's env interface = the label encoding itself:** step() takes the
      (type + up-to-5 ascending picks, -1 pad) vector — Discrete(436) cannot
      carry variable-length picks. Per-step masks are built POLICY-side (the
      prefix-dependent monotone/stop constraints only exist mid-decode; the
      base facts — hand_mask, hands/discards-left — are already in the v2
      obs). The v1 Discrete(436) path survives alongside, same seam pattern
      as `obs_version` (h0.5 stays the shop partner through s1).
18. **Coverage criteria that can never pass are not criteria.** Do not gate the
    harvest on things like "every joker seen N times" — an argmax policy will never
    buy all 150 jokers, and vocabulary breadth is stages 1-4's job. The readout
    tunes the manifest mix; it doesn't pass or fail.
19. **Where things run.** Harvest, reductions, fingerprint eval, all tests: this
    machine. Brute-force prescreen validation (n=11-12) and the regen itself: 9600X
    if local runtime is painful. `runs/` is gitignored — checkpoints move by manual
    transfer; verify `s0_a4_v4` and the h0.5 zip exist locally before A2.

## Definition of done (per phase)

- **A:** corpus banked (both passes), reductions + readout emitted, fingerprint
  verdict written down (triggered / cleared, with the bucket tables), all A1 tests
  green.
- **B:** all feature/label/solver tests green including the brute-force ordering
  test and the big-hand tier-1 pin; prescreen k chosen and recorded; B6 resolved
  (built+validated, or explicitly skipped citing A3); B7 remains to be built —
  phase B is NOT done until its joker/held-aware discard ranking is in and its
  old-vs-new regret check has passed.
- **C:** manifest checked in; a smoke labeling run (a few dozen manifest records
  end-to-end into a shard, loaded back by `train_bc.py`'s loader) passes before
  the full 9600X job is queued.

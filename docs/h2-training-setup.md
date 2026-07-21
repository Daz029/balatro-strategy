# h2 training setup — harvest→h2 tooling + recipe decisions

Decision record (2026-07-21) for the harvest→h2 bootstrap step, triggered by
the `s1_a4_pr2` collapse (see `docs/s1-training-hiccups.md`, final verdict:
"stop shop rungs and go to harvest → h2"). That doc names the *trigger*; this
doc is the *procedure* + the decisions locked in the pre-build grilling session.

h2 = the next bootstrap-loop iteration of the **hand** agent: fine-tune h1
against the hand-state distribution the s1 shop policy actually induces
(CLAUDE.md bootstrap loop, "h1: fine-tune hand-agent against that induced
distribution" — the same step, one iteration later). Schema is unchanged from
h1 (still v3), so **no schema bump, no B-phase work, no re-label** this
iteration — a much lighter pass than h1 was.

## LOCKED — recipe A: pure PPO fine-tune from h1

h2 is a **pure PPO fine-tune** against the s1-induced start-state distribution,
NOT a re-BC. Rationale: the failure the survival curve shows is deep-ante
*conversion* (a PPO/partner problem), not an imitation problem — there is no
evidence the BC prior is what's wrong. Re-BC would only buy a new prior at the
cost of dragging the whole C1/C2 solver-labeling path back in, for no expected
gain while the schema is unchanged.

**What the recipe consumes:** `train_hand_ppo_b.py --harvest-dir` reads ONLY the
harvest's `metadata.jsonl` + `blobs/*.pkl` (via `HarvestSnapshotSampler` →
`restore_state`) as PPO episode start states. It does NOT touch solver labels,
the C1 manifest, or `reductions.json`. So the harvest only has to emit **raw
blobs + metadata** — which the current script already does. The tooling change
below exists solely to make the rollout *drive* correctly (load the s1 model,
deploy the right partner) so the captured blobs reflect the s1+h1 distribution.

## LOCKED — the harvest tooling change (the Codex ticket)

**STATUS 2026-07-21 — BUILT + VERIFIED** (implemented via `codex exec`, architect
review + re-verified in the real env). Diff was in-scope (the two permitted files
only), all locked points met, `ShopModelPolicy`/reductions/capture untouched.
`ruff` clean; `pytest tests/scripts/test_harvest_s0_rollouts.py` = 14/14. Real-model
acceptance smoke (`s1_a3_pr2/best_model` 694-width + h1 money-ordering partner,
`--s1-schema --partner-money-ordering --seed-prefix HARVEST_H2 --n-det 2
--n-sampled 1 --win-ante 8`): model loaded at 694, partner deployed, 149 records
captured (87 hand / 62 shop), shard names confirmed the seed derivation
(`HARVEST_H2_*` det, `HARVEST_H2_S_*` sampled), and a captured hand blob restored
at `SELECTING_HAND` through `harvest_restore.restore_state` — the exact path h2's
PPO sampler uses. No correction ticket needed. The four PARKED items below still
gate the actual run.

All in `scripts/harvest_s0_rollouts.py`. **Behavior with no new flags stays
byte-identical** (the s0 harvest path is unchanged).

- `s1_schema: bool = False` param on `harvest_runs` → `ShopGymEnv(config=
  ShopRunConfig(win_ante=…, s1_schema=s1_schema))`.
- `--s1-schema` CLI flag → threaded to both `harvest_runs` call sites (det +
  sampled).
- `--partner-money-ordering` CLI flag → `HandCheckpointPolicy(str(args.
  hand_policy), money_aware_ordering=True)`; **errors if `--hand-policy` is
  absent** (mirror `eval_shop_policy.py:216`).
- `--seed-prefix` CLI flag, default `"HARVEST"` (backward-compatible);
  `det = prefix`, `sampled = f"{prefix}_S"`.
- De-lie the module docstring + the `--shop-policy` / `--hand-policy` help
  strings (they currently claim "s0" / "h0.5"). **No rename** — renaming the
  file breaks `from harvest_s0_rollouts import …` in the test + churns git for
  no behavior gain.

### Determined facts (not decisions — verified in the grill)

- **`ShopModelPolicy` needs NO change.** `train_shop_ppo.py:787` bakes
  `features_extractor_kwargs={"s1_schema": True}` into `policy_kwargs` at model
  construction, and SB3 persists `policy_kwargs` in the zip, so
  `MaskablePPO.load` reconstructs the 694-width extractor automatically. The
  only requirement is that the ENV it steps against emits 694-width obs+masks —
  i.e. the `s1_schema` thread into `ShopRunConfig`.
- **Partner mode is forced, not chosen.** `s1_a3_pr2` was trained with
  `--partner-money-ordering`, so to reproduce the distribution h1 actually
  induced when paired with that shop, the harvest partner must be h1 *with*
  money-ordering. Determined by the checkpoint choice.

### Seed-prefix design (distinct seeds — locked)

The s1 harvest gets a **distinct** seed namespace (chosen over reuse for
provenance clarity + fresh blind/deck sequences rather than re-policying s0's
exact runs). Rules that keep it safe:

- The invariant that matters is **disjoint from `EVAL_`** (harvesting eval seeds
  leaks the held-out suite into training), not disjoint from `HARVEST_`. Any
  fresh string namespace satisfies it by construction (different string →
  different hashed run). Only rule: never pick a prefix colliding with a
  reserved space (`EVAL`, `SHOPRUN`).
- **Suffix footgun:** because `sampled = f"{prefix}_S"`, do NOT pass
  `--seed-prefix HARVEST_S` (its det pass would collide with the default sampled
  space). For h2 use `--seed-prefix HARVEST_H2` → det `HARVEST_H2`, sampled
  `HARVEST_H2_S`, clean of both `HARVEST_*` and `EVAL_`.

### Testing split

- **Committed, hermetic (Codex must write + pass) — no checkpoints:**
  1. `harvest_runs(s1_schema=True)` with `NextRoundPolicy` + `GreedyHandPolicy`
     → blobs + `metadata.jsonl` written, a blob restores at `SELECTING_HAND`.
     (Both stubs are schema-agnostic, so this proves the 694-width env drives +
     captures with no synthesized fixture.)
  2. `--partner-money-ordering` without `--hand-policy` → argparse error.
  3. `--seed-prefix` default reproduces `HARVEST` / `HARVEST_S`; a custom prefix
     flows to both passes.
- **Real-model acceptance smoke (reviewer, NOT committed — `runs/` is
  gitignored so it can't be a pytest):** real `s1_a3_pr2/best_model` + h1 run,
  `--s1-schema --partner-money-ordering --seed-prefix HARVEST_H2 --n-det 2
  --n-sampled 1 --win-ante 8`. Only this exercises `ShopModelPolicy.load(694)` +
  `HandCheckpointPolicy(money_aware_ordering=True)` + the partner actually
  deploying — the seam the hermetic test structurally can't reach.

**Write scope:** `scripts/harvest_s0_rollouts.py` +
`tests/scripts/test_harvest_s0_rollouts.py` only.

## PARKED — must resolve before the h2 run (NOT this ticket)

These do not gate the tooling; they gate the training run. Resolve after the
tooling lands and the harvest corpus is banked.

1. **PPO start-from + KL-leash target.** `train_hand_ppo_b.py --bc-checkpoint`
   both initializes the policy AND is the KL leash target. Open question:
   - start PPO from h1's **BC** `.pt` (`runs/bc_v3_full/pointer/bc_v3_pointer.pt`)
     again — i.e. re-run h1's recipe with the sampler swapped to the s1 harvest
     (produces an h2 that is a *sibling* of h1), OR
   - continue from h1's **trained** `best_model.zip` (produces a *descendant*),
     and in either case decide whether the leash targets the BC prior or h1.
   The bootstrap loop says "fine-tune the hand-agent" (→ descendant), but the
   wave-2 command started PPO from the BC checkpoint. Not locked anywhere.

2. **Which s1 checkpoint to harvest from.** Leaning `s1_a3_pr2/best_model` — the
   strongest honest fixed-engine shop (ante-3 ≈ 0.46–0.47, ante-4 zero-shot
   0.22). `s1_a4_pr2` collapsed *below its own 25k warm start*, so it induces a
   strictly worse distribution — do NOT harvest from it. Caveat:
   `s1_a3_pr2/best_model` is a lottery-selected 950k peak (`s1-training-hiccups.md`
   "max-of-N is fractal"); harvesting the induced distribution from a lottery
   peak vs. a representative neighbor (e.g. 900k) is a real methodological
   choice. Confirm before the run.

3. **V_curve provenance for the terminal-$ term.** h1 used `data/v_curve.json`
   derived from **s0's** critic. For h2 the honest move is to re-derive from
   **s1's** critic (`scripts/extract_v_curve.py`) — Future-worry #6 in
   `s1-training-hiccups.md` (Φ/V_curve provenance drifts each iteration).
   Reusing s0's is defensible (documented second-order drift) but is a choice,
   not a default.

4. **Run params:** `--n-det` / `--n-sampled` (s0 used 1200/500) and `--win-ante`
   (s0 used 8 so runs die/win naturally; keep 8 so s1_a3_pr2's early death is
   captured realistically). Set at run time.

## The h2 run shape (once tooling lands — on the 9600X, where the s1 zips live)

```
# 1. Harvest the s1-induced hand-state distribution (heavy rollout)
uv run python scripts/harvest_s0_rollouts.py \
  --output-dir data/harvest_s1 \
  --shop-policy runs/shop_ppo/s1_a3_pr2/best_model/best_model.zip \
  --hand-policy runs/hand_ppo_b/h1/best_model/best_model.zip \
  --s1-schema --partner-money-ordering --seed-prefix HARVEST_H2 --win-ante 8

# 2. Fine-tune h1 -> h2 against that distribution (recipe + leash per parked #1)
uv run python scripts/train_hand_ppo_b.py \
  --bc-checkpoint <per parked #1> \
  --v-curve <s0's data/v_curve.json OR re-derived per parked #3> \
  --harvest-dir data/harvest_s1 \
  --config-anchor-frac 0.5 --total-timesteps 2000000 \
  --log-dir runs/hand_ppo_b/h2 --seed 0
```

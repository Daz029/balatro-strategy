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

> **⚠ RECIPE BLOCKER (found 2026-07-21) — RESOLVED 2026-07-21.** The recipe is
> **descendant** — init AND KL-leash from h1's *trained* `.zip` (locked #1) — but
> the original `train_hand_ppo_b.py` `--bc-checkpoint` flag loaded *only* the
> pointer BC `.pt` format for both, so it supported **only the sibling recipe**.
> The descendant path is now **BUILT + VERIFIED** (see the "PARKED — descendant
> tooling" section, which records the build): `--init-from <trained .zip>` loads
> a trained `PointerPPOPolicy` as both the policy init and a frozen KL-leash
> reference. Both *harvest ready* and *run ready* are now **yes**; the h2 run is
> unblocked.

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

## LOCKED — the run decisions (grilled 2026-07-21)

The four items below were PARKED pending the tooling; all four are now resolved
in a grilling session. The reasoning is the load-bearing part — each decision
has a failure mode that only shows if the *reason* is forgotten. Two framing
corrections that fell out of the grill are recorded at the end because they
would silently mislead if misremembered.

**Mechanism framing (the premise the four decisions serve).** h2 sharpens h1 on
the **concentrated, realistic build manifold** s1 actually induces — domain
randomization (h1's stages 1-4) gives *broad* coverage over uniform-random joker
sets; a real optimizing shop concentrates on a *narrower* synergistic manifold,
and h1 saw those states only under a broader/worse build distribution. Two
routes to value: (a) distributional-shift correction on that manifold (primary);
(b) conversion is multiplicative, so lifting the dense early/mid conditionals
raises the *reach rate* into deep antes, which mechanically hands PPO more
deep-ante episodes per rollout. **Deep-ante coverage does NOT come from the
harvest** — s1 barely reaches deep antes, so they are a thin tail in the corpus;
deep coverage comes from `--config-anchor-frac 0.5` (half the PPO episodes start
from domain-randomized stages 1-4). **Pre-registered expectation, s1-doc style:
h2 buys mid-ante conversion on s1's build distribution; it is NOT a deep-ante
breakthrough mechanism.** Reading it as a fix for the 0.286 ante-4 conditional
will misread the result.

1. **PPO start-from + KL-leash target → DESCENDANT.** Init *and* leash both =
   `runs/hand_ppo_b/h1/best_model/best_model.zip` (init and leash are one choice;
   you leash to where you start). Rejected the sibling route (BC `.pt` init + BC
   leash). **TOOLING GAP (found 2026-07-21 during this grill — see the new PARKED
   item below):** the current `--bc-checkpoint` flag *only* loads the pointer BC
   `.pt` format (`load_bc_model`/`load_bc_weights` hard-require
   `metadata["head"]=="pointer"` + `HandPointerBCModel.load_state_dict`), so it
   supports only the *sibling* recipe today. Descendant needs a code change
   (load a trained `.zip` as init + a frozen trained `PointerPPOPolicy` as the
   leash reference). The *decision* is descendant; the *build* is pending.
   Reasons:
   h1's trained refinements (terminal-$ money valuation, real-training pointer
   decoding, general hand-play PPO gains) are **skill conditional on a build**,
   not fits to s0's *choice* of builds, so they transfer — re-deriving them from
   BC each iteration is pure waste. Additionally, BC carries *documented* biases
   (the A3 discard bias, the kicker-fill → size-5 prior) that PPO exists to
   correct and h1 has *already* partially corrected against real reward; leashing
   to BC would re-anchor to the biased prior. **Why the warm start is safe here
   when it sank the s1 a4 rung:** h2 does **not** extend the horizon — same antes,
   same difficulty, only s1's build manifold instead of s0's/randomized. It's a
   distribution *shift*, not a horizon *extension*, so the plasticity risk that
   collapsed a4 (warm-started sharp policy into a genuinely harder distribution)
   is much milder. The leash to h1 **decays to zero** (project standard): h1 is
   the init and the early-stability anchor, s1's distribution is the objective —
   a non-decaying h1 leash would just reproduce h1.

2. **Harvest checkpoint → `s1_a3_pr2/best_model`.** The strongest honest
   fixed-engine shop (ante-3 ≈ 0.46–0.47, ante-4 zero-shot 0.22); NOT the
   collapsed `s1_a4_pr2` (a *decision-level* defect — a collapsed shop induces a
   genuinely degraded manifold). The lottery-peak caveat is *resolved, not
   waved*: for harvesting you don't score the win-rate number (that's the
   seed-dependent part, and it doesn't matter here), but the induced **state
   distribution** is decision-*dependent* — the pivotal-flip theory says
   neighbors take different argmax decisions and induce different manifolds, so
   the choice is not a no-op. It settles cleanly on **ship-what-you-harvest**:
   h2's job is to fit the distribution s1 *deploys*, so harvest whatever ships as
   s1. It "doesn't matter much" only because fresh blind/deck seeds + the sampled
   pass + the 0.5 config anchor smear the fine pivotal-flip distinctions — any
   *non-collapsed* representative shop gives a fine corpus.

3. **Harvest mode + params → deterministic-primary, 1200:500, win-ante 8**
   (corpus BANKED on the 9600X). Deterministic is the backbone because s1
   deploys argmax and h2 must fit the *deployed* distribution. The sampled
   fraction stays **nonzero** for a structural reason: h2 episodes are
   **single-blind** (`HandPlayAdapter.done` fires at `ROUND_EVAL`/`GAME_OVER`),
   so PPO explores *actions*, not *start states* — start-state coverage does
   **not** self-heal, and a pure-argmax corpus would be a razor-thin trajectory
   sliver with no on-manifold neighborhood. `--config-anchor-frac 0.5` backstops
   coverage regardless, which is why det-*heavy* (not det-*only*) is safe.
   win-ante 8 so runs die/win naturally and s1_a3_pr2's early death is captured
   realistically.

4. **V_curve provenance → RE-DERIVE from s1's critic** over the banked s1
   shop-state snapshots (`scripts/extract_v_curve.py`; the harvest already
   captured shop snapshots for exactly this). Reason: the terminal-$ term pays
   the *hand* agent for banking money on a clear, and money's future value must
   be computed under the policy that will actually **spend it downstream** —
   s1, the deployed teammate, not s0. Using s0's curve is a teammate-mismatch
   (same class as the Φ-provenance drift, FW#6). **Hold the magnitude loosely:**
   Issue 3's low-action-leverage finding implies s1's critic attributes outcomes
   to the build-state, so the $-only sweep (build held fixed) may come out
   **flatter** than s0's, and the ante-average already erases the
   build-conditional money value (economy builds — To the Moon, Bull) where a
   held dollar is worth most. Re-derive because it's cheap and *correct*, not
   because it will move h2's numbers much; if money play looks wrong the fix is
   the **contextual critic** (CLAUDE.md's named upgrade), not another provenance
   swap.

5. **Checkpoint selection → pre-registered bar, honest baseline + 2σ** (n=200 →
   ~0.029), s1-doc style, so `best_model`'s max-of-N lottery is not
   re-institutionalized. Re-verify any headline checkpoint on a second disjoint
   seed set before believing it.

**Interest-ordering for the $ term — VERIFIED, no run-time check needed.** The
CLAUDE.md rider (in-blind earnings land *before* interest and can cross a
bracket; end-of-round payouts land *after* and must not inflate it) is pinned:
`tests/engine/test_cashout_ordering.py` (`test_held_gold_card_pays_and_crosses_interest_bracket`,
`test_golden_joker_pays_once_and_never_bumps_interest`,
`test_investment_payout_lands_after_earnings_total`) plus the env mirror
`tests/env/test_cashout_mirror.py`.

## PARKED — descendant tooling — BUILT + VERIFIED 2026-07-21

**STATUS 2026-07-21 — BUILT + VERIFIED** (implemented via `codex exec`, architect
review + independently re-run in the real env). The descendant path is live in
`train_hand_ppo_b.py`: a **required mutually-exclusive** flag pair —
`--bc-checkpoint <.pt>` (sibling, unchanged, byte-identical) XOR `--init-from
<trained .zip>` (descendant). `--init-from` loads the trained hand-agent zip via
`load_trained_pointer_policy` (`KLToBCPointerPPO.load(zip).policy`, frozen), uses
it as the KL-leash reference (`set_leash_policy` → `KLToBCPointerPPO` now branches
its reference distributions between a `bc_model` and a frozen `leash_policy`, with
an exactly-one XOR guard at every layer), and copies its weights into the fresh
`PointerPPOPolicy` via `load_state_dict(strict=True)` — full warm start including
the calibrated `value_net` critic. `ruff` clean; `pytest
tests/scripts/test_train_hand_ppo_b.py` = **9/9** (independently re-run in the real
env), incl. a round-trip test proving descendant init reproduces the source
policy's deterministic outputs and the frozen-policy leash reads ≈0 for the
source's own actions and >0 after a head perturbation. Write scope held to
`scripts/train_hand_ppo_b.py` + its test file; `pointer_ppo_policy.py` untouched
(it already exposed `teacher_forced_step_distributions`, the exact reference
surface the leash needs). Every future bootstrap iteration `hN → h(N+1)` reuses
this path, so it is paid once.

**Historical record — the change that was needed** (the decision `--bc-checkpoint`
→ `load_bc_model`, pointer variant `jackdaw/agents/pointer_ppo_policy.py:293`,
expected a BC `.pt` for BOTH init and the frozen KL-leash reference; a trained
`.zip` is a different format):

1. **Init from a trained `.zip`.** An init path that does `PPO.load(zip)` (the
   hand-b model is ordinary SB3 `PPO`, **not** MaskablePPO — earlier drafts here
   said `MaskablePPO.load`, a slip; corrected in build) and copies
   `loaded.policy.state_dict()` into the new `PointerPPOPolicy` (the
   `load_bc_weights_into_policy` precedent, but source = a trained policy).
   Architectures match (both `PointerPPOPolicy` at the v3 schema), so it's a
   direct `strict=True` state-dict load.
2. **Leash reference = the frozen trained policy.** For descendant the reference
   is a frozen copy of h1's *trained* `PointerPPOPolicy` producing the same
   per-step logits; the leash's reference-distribution computation now accepts
   either a `HandPointerBCModel` (sibling) or a frozen `PointerPPOPolicy`
   (descendant).

Options considered before the build (recorded — the recommended one shipped):
- **Build it** (recommended, SHIPPED — it's what the lock says, and every future
  bootstrap iteration needs the same descendant path, so it's paid once).
- **Sibling fallback** (zero code): init + leash from `bc_v3_pointer.pt`. Loses
  h1's trained refinements (money term, corrected BC biases) — the exact thing #1
  argued is worth keeping — so only if the descendant build had been deferred.
- **Decoupled compromise REJECTED**: init from the trained zip but keep the BC
  leash. The leash would pull h2 back toward BC early, actively eroding the
  warm-start gains before reward re-establishes them — the "leash fights the warm
  start" failure. Do not ship this as a shortcut.

**Two framing corrections (recorded because they would silently mislead):**
- **Marginal-$ sign.** A *better* shop makes the marginal dollar worth **more**,
  not less — the critic measures win-probability, not propensity-to-spend, and
  s0 "blindly buying jokers" is the case where a dollar buys *junk*. Do NOT
  reason the V_curve magnitude from "s1 is pickier."
- **Best-response definition.** The best response is defined by **s1's
  distribution** (argmax over policies of return under it), NOT by where h1 ended
  up. h1 is the init and eval baseline, not the target — which is *why* the leash
  must decay. Treating "where h1 ended up" as the target argues for a permanent
  h1 leash, which reproduces h1 and gains nothing.

## The h2 run shape (once tooling lands — on the 9600X, where the s1 zips live)

```
# 1. Harvest the s1-induced hand-state distribution (heavy rollout) — BANKED
uv run python scripts/harvest_s0_rollouts.py \
  --output-dir data/harvest_s1 \
  --shop-policy runs/shop_ppo/s1_a3_pr2/best_model/best_model.zip \
  --hand-policy runs/hand_ppo_b/h1/best_model/best_model.zip \
  --s1-schema --partner-money-ordering --seed-prefix HARVEST_H2 \
  --n-det 1200 --n-sampled 500 --win-ante 8

# 1b. Re-derive V_curve from s1's critic over the banked s1 shop snapshots (locked #4)
#     (flags verified against extract_v_curve.py: --checkpoint / --harvest-dir / --out)
uv run python scripts/extract_v_curve.py \
  --checkpoint runs/shop_ppo/s1_a3_pr2/best_model/best_model.zip \
  --harvest-dir data/harvest_s1 --out data/v_curve_s1.json

# 2. Fine-tune h1 -> h2 (DESCENDANT, locked #1) -- descendant tooling BUILT.
#    --init-from loads h1's trained .zip as BOTH the policy init and the frozen
#    KL-leash reference. (Sibling fallback, if ever needed: swap --init-from for
#    --bc-checkpoint runs/bc_v3_full/pointer/bc_v3_pointer.pt.)
uv run python scripts/train_hand_ppo_b.py \
  --init-from runs/hand_ppo_b/h1/best_model/best_model.zip \
  --v-curve data/v_curve_s1.json \
  --harvest-dir data/harvest_s1 \
  --config-anchor-frac 0.5 --total-timesteps 2000000 \
  --log-dir runs/hand_ppo_b/h2 --seed 0
```

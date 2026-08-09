# Shop joker embedding analysis — measured findings, open question, and fix options

**Date:** 2026-07-28. **Subject:** the shop policy's `nn.Embedding` identity table
(`shop_policy.py:82`, `EMBED_DIM=16` over the frozen `centers.json` vocabulary).
**Trigger:** a UMAP of the learned joker embeddings
(`scripts/extract_shop_joker_embeddings.py`, `data/joker_embeddings_s2_a3.npz`,
`.png`) showed a featureless blob — no interpretable clusters, semantically
random nearest neighbours (blueprint↔mad, rocket↔sixth_sense).

**Verdict in one line:** the plot is correct, the embedding is fine, and the
DIAGNOSTIC was measuring the wrong object — the table is an untrained
random-projection identity code that is nonetheless load-bearing.

A condensed version of the measurements lives in `docs/s1-training-hiccups.md`
under "The joker embedding table is a RANDOM IDENTITY CODE". This document is
the full record plus the forward plan.

**UPDATE 2026-08-09 — §5's open question is ANSWERED (memorization dominates),
the §7 fixes are unblocked, and §5.1 records a descriptor-extraction BUG found
en route: 76 of 150 jokers are falsely flagged as scaling jokers. Read §5
before acting on §§1-4, which pre-date it.**

---

## 1. Measured — the table never trains

### 1.1 Fresh-init control

Trained `s2_a3` embeddings vs a freshly constructed untrained `nn.Embedding`,
both sliced to 150 joker rows:

| metric | fresh init | trained s2_a3 |
|---|---|---|
| std | 1.0106 | 0.9883 |
| row-norm median | 3.961 | 3.960 |
| pairwise cosine std | 0.2490 | 0.2496 |
| effective rank (participation ratio) | 14.55 / 16 | 14.46 / 16 |
| top-3 PC variance | 0.278 | 0.282 |

Statistically indistinguishable on every axis. UMAP with `metric="cosine"` over
near-isotropic 16-d noise produces exactly the observed uniform blob.

### 1.2 Displacement is a random walk — confirmed to 1%

Rows DO receive gradient: ~0.32 median displacement over
`s1_a3_pr2 → s2_a4`. But it is diffusion, not learning.

Adam normalizes step size to ~`lr` per update regardless of gradient magnitude,
which turns "distance travelled" into a readable function of update count. From
the checkpoint's own optimizer state — **Adam, `lr=3e-4`, betas `(0.9, 0.999)`,
`eps=1e-5`, `weight_decay=0`, 85,504 steps** — and the measured minibatch hit
rate of §1.3 (0.812 → ~69,400 effective updates):

| hypothesis | formula | predicted | observed |
|---|---|---|---|
| coherent drift | `lr · N · √d` | ~83 | 0.32 |
| random walk | `lr · √N · √d` | **0.316** | **0.32** |

Match within 1%; coherent learning is off by 260×.

### 1.3 Appearance rate — the objection this survives

A per-shop-slot model ("~0.7 jokers per slot, so rows update rarely")
undercounts by two orders of magnitude, and would have rescued the coherent
hypothesis, which needs only a 0.3% hit rate to fit the observed displacement.

**The gradient unit is the MINIBATCH, not the shop visit.** Owned jokers sit in
`joker_ids` in *every observation for the rest of the run*, and a minibatch
pools `batch_size=256` timesteps across 8 concurrent envs at different antes and
builds. Measured over 5,523 eval observations:

- median P(joker present in one observation) = **0.011**
- **P(present in a 256-obs minibatch) = 0.812** (empirical, resampled honoring
  the real 8×256 buffer structure; naive-independent estimate 0.942)
- 93% of jokers seen at least once

Coupon-collector does the work: even 1% per-observation presence lands in
`1 − 0.99²⁵⁶` = 92% of minibatches.

### 1.4 Movement is uncorrelated with appearance frequency

Supporting evidence, using rarity as a frequency proxy (commons appear far more
often than rares): mean displacement 0.318 / 0.328 / 0.316 for rarity 1 / 2 / 3,
Pearson `r(rarity, movement) = −0.076`. Real learning would show commons pulling
far ahead. They don't.

---

## 2. Measured — and yet the table is LOAD-BEARING

Ablation on `s2_a4/best_model`, 200 episodes, identical eval seeds,
`--win-ante 4 --s1-schema`, partner h2@2.0M `--partner-money-ordering`:

| arm | win rate | mean final ante |
|---|---|---|
| **base** | **32.5%** | 3.42 |
| zeroed (embedding block = 0) | 6.5% | 2.27 |
| **permuted** (rows shuffled among ids) | **6.5%** | **2.27** |
| `nextround` floor | 0.0% | 1.64 |

**The permuted arm is the entire result.** It preserves the input distribution
EXACTLY and destroys only the identity→vector mapping — and it costs precisely
what zeroing costs (0.065 vs 0.065; ante 2.265 vs 2.270). Had the network merely
adapted to "some nonzero input", the in-distribution permutation would have hurt
far less than the wildly-OOD zeroing. It didn't.

**What the policy depends on is the MAPPING** — not the values, not the
distribution, not the geometry.

Raw data: `data/embedding_ablation.json`.

---

## 3. Interpretation — a random-projection identity code

Both findings are true and consistent. A random 16-d vector is already a
perfectly good unique identifier — random projections separate 150 items in 16
dimensions trivially. **The rows never needed to move because they were born
useful**; `joker_encoder` learned to decode those fixed random codes, so all the
actual learning went into the DECODER weights, which are not per-joker and
cannot be plotted this way. This is a random-features architecture, arrived at
by accident, and it works.

Value decomposition against the floor: joker identity is worth ~26 of the 32.5
points (~80% of the policy's value over `nextround`). The residual 6.5% is what
the frozen descriptors and numeric features buy — buying by cost, rarity and
effect family without knowing WHICH joker.

> **Do not over-read that 6.5%.** It is a network trained to expect identity
> codes, measured with them removed — an out-of-distribution measurement. It
> says nothing about how a descriptor-only policy would perform if trained from
> scratch, and nothing about generalization.

---

## 4. What this changes

### 4.1 The CLAUDE.md diagnostic is unanswerable as written

The shop-obs design offers: *"Embedding table is inspectable mid-training
(t-SNE = is synergy learning happening at all)."* The geometry carries no
information BY DESIGN, so no amount of training makes it cluster and no choice
of UMAP hyperparameters will help. **Amend that line rather than re-running the
plot.** Proposed replacement: a pointer to this document plus "probe the critic,
not the table."

### 4.2 A wrong fix, recorded so it is not re-proposed

The pre-ablation reading was: *"the embedding is noise drowning the descriptors
(78% of identity-signal variance), so shrink the init to `std≈0.02` and
retrain."* **This is backwards.** Small init collapses the 150 codes toward each
other, destroys separability, and would reproduce the measured 5× collapse. The
large `N(0,1)` init is load-bearing. Do not "fix" it.

### 4.3 Which half of the design broke

CLAUDE.md justified the embedding as: *"a scalar ordinal ID has false geometry
... and makes synergy a 150×150 pointwise lookup; embeddings give learnable
geometry, descriptors give day-one generalization; each covers the other's
failure mode."*

**The learnable-geometry half is not delivering.** A random code gives identity
with zero transfer — nothing learned about joker A moves to a similar joker B
through that channel. Two distinct gaps result:

- **Gap 1 — no learnable cross-joker geometry.** All cross-joker transfer rests
  on the 24-d FROZEN descriptors. The capability the embedding was supposed to
  add — discovering a similarity axis the hand-authored descriptors missed —
  is simply absent, and since descriptors are frozen, training cannot recover
  it.
- **Gap 2 — no explicit pairwise term.** Verified: `joker_encoder` is a per-row
  MLP and `masked_pool` returns masked mean ⊕ masked max (DeepSets). No
  attention, no outer product, no pair features. The trunk never sees "A and B
  are both present"; it sees a summary of the joker set.

Gap 2 cuts both ways and is not purely bad news. Because synergy must be
represented as a *function of the aggregate*, it is inherently compositional: if
the encoder maps raw-mult jokers into one region of the 64-d row space and xmult
jokers into another, a **novel** pair pools into the same place a seen pair did
and the learned value transfers for free. The honest counterpoint is that
**max**-pooling preserves per-joker spikes, so specific jokers can leave
distinctive marks in specific dimensions — a limited, bandwidth-constrained form
of pair memorization is possible. Which mode dominates is empirical.

---

## 5. THE OPEN QUESTION — ANSWERED 2026-08-09: MEMORIZATION DOMINATES

> Does the shop policy generalize to joker pairings it has rarely or never seen
> (e.g. a fresh raw-mult + xmult combination), or has it memorized the pairs its
> own rollouts happened to cover?

**Memorized.** Option A (§6.2) ran on `s2_a4/best_model`;
`scripts/probe_joker_synergy.py`, artifacts `data/joker_synergy_probe.json` +
three seed replicates. The gate is therefore OPEN for the §7 fixes — this is no
longer a "do nothing" outcome.

**Headline — twin/non-twin gap ratio 0.87** (seeds 0/1/2/3: 0.882, 0.934,
0.805, 0.867; 95% bootstrap CIs exclude 1.0 in three of four). Making two
jokers descriptor-IDENTICAL shrinks the difference in their learned synergy by
only ~13%. If the descriptor channel drove generalization the ratio would
approach 0; ~0.87 means the non-descriptor identity channels (random embedding
+ the ordinal `center_key_id` in row feature 0) account for ~87% of it.

**The positive control is what makes that readable.** If the critic's second
differences were noise, twins and non-twins would BOTH be noise and the ratio
would sit at ~1.0 for reasons unrelated to memorization — a vacuous pass. So
every run splits base states into disjoint halves and correlates each cell's
mean synergy across them: **r = 0.79-0.92** over ~950 cells. Synergy is a
reproducible function of the pair; the null is real. Mean |synergy| 0.023-0.05
in P(win) units (max 0.30), so the quantity has real magnitude too.

**Structure inside the result** (seed 0, 27 twin groups): 21 of 27 fall below
1.0, and which ones is the tell. The tightest are jokers that genuinely ARE
mechanically alike — `mime/dusk` 0.35, `wily/devious` 0.39, `crazy/zany` 0.51
(the last two being the same hand-type-conditional family with different
trigger hands). The loosest are pure descriptor COLLISIONS —
`blue_joker/photograph` 2.05, `onyx_agate/arrowhead` 1.41, `juggler/drunkard`
1.38. So the descriptor channel transfers when it happens to be right, and
transfers nothing when it is merely degenerate.

### 5.1 Why generalization failed — the descriptor channel is thin AND wrong

Two measurements taken while interpreting the above; together they explain the
result and re-rank §7.

**Thin.** Of 24 descriptor dims, cols 3-7 are other-item type flags
(structurally 0 on every joker) and col 2 `is_joker` is constant 1. The
effect-family columns are extremely sparse across the 150-joker pool: flat
+mult 9, +chips 3, xmult 10, suit-conditional 4, hand-type-conditional 17,
economy 5, probabilistic 4, hand-size 6. Only `scaling` (87),
`scaling_rate` (87) and `blueprint_compat` (121) are dense. **103 distinct
descriptor rows for 150 jokers — 77 jokers (51%) sit in one of 30 exact
duplicate groups**, e.g. `(four_fingers, shortcut, midas_mask, smeared)`.

**Wrong.** `j_onyx_agate` (Clubs) and `j_arrowhead` (Spades) are exact twins
despite a dedicated suit column, which exposed the defect: both configs are a
BARE NUMERIC `extra` (`{'extra': 7}` / `{'extra': 50}`), so the suit branch
(which requires `extra` to be a dict) never fires, and the bare-numeric branch
at `joker_descriptors.py:114-122` flags them as SCALING jokers with an
arbitrary rate. **76 of 150 jokers use the bare-numeric idiom and all 76 are
flagged scaling — 76 of the 87 total scaling flags.** The idiom carries
wildly different meanings: Credit Card's $20 debt allowance, Banner's 30 chips
per remaining discard, Mime's retrigger count, Photograph's x2 mult (whose
xmult ALSO never registers, since col 10 reads `cfg["Xmult"]` which Photograph
lacks). It cannot be fixed by excluding the idiom either — Obelisk
(`{'Xmult': 1, 'extra': 0.2}`) genuinely scales through it while Runner and
Green Joker scale through dict keys. The column does not discriminate.

Consequence: with `scaling` mostly false-positive, `blueprint_compat`
near-constant and `is_joker` constant, a typical joker's descriptor reduces to
**(rarity, cost)**. That is the mechanism behind the 51% duplicate rate, and it
closes the loop on §3 — the embedding had to serve as a pure identity code
because nothing else was carrying identity.

The module docstring calls this layer "coarse by design" and says "precision
here is NOT critical, coverage of broad effect families is." Coarse is fine;
76/76 false positives on the densest column is not coarseness, it is a column
that actively misinforms.

---

## 6. PROBE OPTIONS

### 6.1 What to measure — the second difference, not ΔV

The wrong version: `V(state + joker c) − V(state)`. That is a MAIN EFFECT — how
good joker c is alone. Synergy is not a main effect; this would measure joker
quality and reveal nothing about pairings.

The right object is the discrete second difference / interaction term:

```
synergy(a, b) = V(O ∪ {a,b}) − V(O ∪ {a}) − V(O ∪ {b}) + V(O)
```

Both main effects cancel; what survives is the non-additive part. Four critic
forwards per pair, singletons cache, so P pairs cost `1 + N + P` forwards. The
critic is a small MLP and this batches — thousands of pairs in seconds.

**State construction, two designs, both needed:**

- **Observational** — real shop states from rollouts, owned sets as the policy
  actually built them. In-distribution so V is trustworthy, but the pairs of
  interest are by definition rare or absent. Use as the SANITY ANCHOR.
- **Constructed** — inject chosen jokers into a base state. The only way to
  present never-co-occurred pairs.

  **CORRECTED 2026-08-09 (this sentence originally said "same
  counterfactual-obs-edit method the `V_curve` money sweep already uses").
  `extract_v_curve.py` states the OPPOSITE as its one hard rule:
  counterfactuals edit ENGINE STATE and re-encode, never obs vectors.** A joker
  fans out into far more derived obs features than dollars does — the joker's
  own 15 row features, `get_hand_eval_flags` in the GC (`observation.py:1107`;
  inject Four Fingers by obs-edit and the state claims Four Fingers while every
  hand-structure feature disagrees), the hand-analysis block
  (`observation.py:837-843`), and the per-shop-row buyability flag
  `len(jokers) < joker_slots` (`observation.py:554`). The implemented path is
  restore blob → mutate `gs["jokers"]` → `build_shop_observation` → critic.

**Implementation traps (revised against the built probe):**

1. **Injected jokers need their `add_to_deck` passives** (`card.py:746`).
   Skipping it is the pre-regen B1 bug verbatim — `HandPlayAdapter.reset`
   injected with a bare `create_joker` and never applied acquisition passives,
   so every demo sampling Juggler/Turtle Bean carried a hand size the engine
   would never deal. (This SUPERSEDES the original trap 1, which was about
   keeping a hand-written obs edit self-consistent — re-encoding makes mask
   bits and derived counts correct for free.)
2. **Cardinality artifact — solved by construction, not by subtraction.**
   `masked_pool`'s mean denominator changes when the joker COUNT changes, so an
   addition-form second difference carries a structural offset. The original
   remedy was to measure `synergy(a, a′)` as a floor and subtract it. The
   probe instead uses the REPLACEMENT form against a fixed filler `f`:
   `V(O∪{a,b}) − V(O∪{a,f}) − V(O∪{f,b}) + V(O∪{f,f})`. Every arm injects
   exactly two jokers, so cardinality is pinned at 2 and the artifact never
   arises. Still an exact second difference — `f` is just the baseline level
   instead of absence, and enters all four terms symmetrically.
3. **Slot positions must match across arms.** `masked_pool` is
   permutation-invariant but `encode_joker` feature 9 is `position / 20`
   (`observation.py:443`), so rows are not. Inject at the same indices in every
   arm, and filter base states to those with two free joker slots so
   `_check_joker_overfill` (`shop_obs.py:190`) never truncates an arm.
4. **Dollars are not decremented** for the injected joker. This is FINE — the
   inconsistency is identical across all four terms and cancels in the second
   difference to first order. Stated explicitly so nobody "fixes" it and
   reintroduces a money confound.

### 6.2 Option A — descriptor twins (RECOMMENDED FIRST)

Cheapest, needs no functional-form assumption, and is a paired test so it
controls for everything the regression needs covariates for.

Find joker pairs `(a, a′)` with near-identical 24-d descriptors but very
different co-occurrence with some third joker `b`. Compare `synergy(a,b)`
against `synergy(a′,b)`.

- **Close** ⟹ knowledge transferred through the descriptor channel; the random
  code is not blocking generalization. Gap 1 is benign.
- **Far apart** ⟹ identity-memorization dominates; each joker's synergy was
  learned separately, and sparse pair coverage bites.

This is the structure of CLAUDE.md's own "Hack-vs-descriptor-twin" objection
from the trigger-matrix design, repurposed as a measurement.

**Two amendments from the build (2026-08-09):**

- **No similarity threshold is needed** — "near-identical" turned out to
  understate it. 77 of 150 jokers share a BIT-IDENTICAL descriptor row with
  another joker (30 exact groups), so the descriptor channel is provably unable
  to distinguish a twin pair and any measured difference is attributable to the
  non-descriptor identity channels. One knob avoided.
- **"Close" and "far apart" need a SCALE, so the probe adds a reference arm.**
  A raw gap of 0.042 is uninterpretable alone. Every run therefore computes the
  same gap statistic over random NON-twin pairs, and reports the ratio. ~1.0 =
  twins no more alike than strangers = memorization; ≪1.0 = descriptor-driven
  transfer. Without this arm the verdict would rest on eyeballing an absolute
  number against nothing.
- **A positive control is MANDATORY, not optional** — see §5. A noise-only
  critic yields ratio ~1.0 trivially, so the split-half reliability check is
  what separates "memorization confirmed" from "measurement failed". Report it
  alongside the ratio every time.

### 6.3 Option B — nested regression (if twins are ambiguous)

Two variables that must NOT be used naively:

- **Descriptor SIMILARITY is the wrong functional form.** Synergy is usually
  COMPLEMENTARITY, not similarity — raw-mult + xmult is synergistic precisely
  because they differ, while two xmult jokers are similar and merely stack. A
  cosine-similarity regressor would find nothing and produce a falsely negative
  verdict. Use **pair features**: `[desc(a), desc(b), desc(a) ⊙ desc(b)]`,
  symmetrized over pair order. The elementwise product is what expresses "a is
  xmult AND b is +mult".
- **Raw co-occurrence is confounded by marginals.** Two common jokers co-occur
  often mechanically, with no pair-specific learning involved. Use the PMI
  residual `log[ P(a,b) / (P(a)·P(b)) ]` and include `log freq(a)`,
  `log freq(b)` as covariates regardless. Without this the regression is
  garbage.

Fit and compare on **held-out pairs stratified by co-occurrence**, especially
the zero/near-zero stratum:

- **M1:** `synergy ~ descriptor pair features + marginal controls`
- **M2:** `M1 + PMI(a,b)`

M1 predicting unseen pairs well ⟹ composition works, coverage sparsity
survivable. M1 collapsing there while M2 fits ⟹ memorization confirmed.

### 6.4 What the probe cannot tell you

- **V may not be in P(win) units.** If Φ shaping or a nonzero blend beta was
  live for `s2_a4`, V is in blended-reward units. Only RELATIVE comparisons are
  meaningful; check that run's flags before interpreting magnitudes.
- **Constructed states are off-distribution**, and the critic was only trained
  on states the policy reaches. A poor V on a never-built pair is ambiguous —
  failed generalization, or merely an unvisited region. Partly this ambiguity IS
  the finding: a critic that cannot value combinations the policy never builds
  is the coverage problem in its own right.
- **The six gated jokers are unprobeable** (§8) — random code plus an untrained
  decoder path, so their values are meaningless.

---

## 7. FIX OPTIONS — GATE PASSED 2026-08-09, re-ranked

The gate condition was: *"If the probe shows composition works, do nothing;
the options below apply only if memorization dominates."* **Memorization
dominates (§5), so the options are live.**

§5.1 also re-ranks them, and the gap between rungs 1 and 2 is much larger than
the original ordering implies:

- **Rung 1 is no longer merely "cheapest" — it has the best evidence, and part
  of it is a BUG FIX rather than enrichment.** The transfer channel is ~5
  informative dims wide for a typical joker, collapses half the pool to exact
  duplicates, and its densest column (`scaling`) is 76/87 false positives.
- **Rung 2 is strictly DOWNSTREAM of rung 1, not an independent alternative.**
  A factored embedding `A·desc + residual` gives twins the same `A·desc` by
  construction — so while half the pool shares a descriptor row, the entire
  distinction falls back into the free residual, i.e. back into a random code.
  Doing 2 before 1 buys transfer only among the 73 descriptor-unique jokers.
- **Rung 3's contrastive-on-co-occurrence variant is partly self-defeating**:
  it would shape geometry from the co-occurrence statistics of the policy's own
  rollouts, which is the very coverage sparsity §5 just confirmed. The
  predict-descriptors-from-embedding variant is the safer one — and is likewise
  bounded by rung 1's content.
- **Rung 0, free, not previously listed**: §11's first caveat (the arms ablated
  all ~300 center keys, so the joker-vs-consumable/voucher split is not
  isolated) is one line of the harness. Settle it before scoping any fix.

**Note what is already available and NOT missing:** `joker_encoder` is shared
across all jokers and already receives the descriptors concatenated, so a
trainable shared function of descriptors ALREADY exists. Adding an
`MLP(descriptors)` channel would be redundant. What is missing is narrower: a
way for two jokers that behave alike to *become* alike in a channel the
descriptors do not already encode.

Ordered by cost, cheapest first:

1. **Enrich the descriptors** (no architecture change, no retrain risk).
   Descriptors are the transfer channel that demonstrably works; if they are the
   bottleneck, widen them. Cheapest real fix, and it uses the working pathway.
   Cost: a shop obs schema bump + retrain, but no novel machinery.
2. **Factored embedding** — `embedding = A · descriptor + free residual`, with
   the projection learned and shared. Similar jokers START close (restoring
   learnable geometry) while the residual preserves the separability that §2
   proved load-bearing. The most direct repair of Gap 1 and it does not risk the
   collapse that a small init would.
3. **Auxiliary loss on the table** — contrastive on co-occurrence, or predict
   descriptors from the embedding. Shapes geometry without changing the forward
   architecture. Adds a loss-weight hyperparameter and a tuning burden.
4. **Attention or a bilinear term over joker rows** — the only option that
   addresses Gap 2 directly by giving the architecture an explicit pairwise
   term. CLAUDE.md already defers attention to h2 as evidence-gated, and the
   pointer-head work at the in-blind merge brings the machinery. Most invasive;
   full shop retrain.
5. **REJECTED — shrink the embedding init.** See §4.2. Measured to be harmful.

Constraints binding any of these: the `centers.json` VOCABULARY FREEZE (ids come
from sorted keys; reordering corrupts every shop checkpoint), the append-only
action-space contract, and the fact that any obs-schema or trunk change is a
shop retrain that must be sequenced against the s1/s2 rung plan rather than
dropped mid-run.

Existing mitigation already live, worth accounting for before adding more:
`CountBonus` (`train_shop_ppo.py:240`) applies `1/sqrt(N)` novelty over sorted
owned-joker key-sets, decayed to zero — i.e. there is already pressure toward
covering unseen combinations. What is unmeasured is whether it is enough.

---

## 8. Byproduct — six jokers never receive a single gradient

Bit-identical rows across every consecutive checkpoint pair
(`s1_a3_pr2 → s1_a4_pr2 → s2_a3 → s2_a4`). All six are conditionally pooled by
`pools.py::_filter_joker`, so they never appeared in ANY observation:

| joker | gate |
|---|---|
| `cavendish` | needs flag `gros_michel_extinct` |
| `glass` | needs `m_glass` in deck |
| `lucky_cat` | needs `m_lucky` in deck |
| `steel_joker` | needs `m_steel` in deck |
| `stone` | needs `m_stone` in deck |
| `ticket` | needs `m_gold` in deck |

Exactly zero rather than merely small because Adam's moment buffers stay at
zero, so the update is exactly `0/(0+ε)`.

Two riders: (a) the agent has **never held a deck containing a
glass/steel/stone/gold/lucky card at shop-generation time** across all of s1+s2
— it is essentially never using enhancement tarots, a behavioural finding
independent of the embedding question; (b) those six have random codes AND an
untrained decoder pathway, so if a deck ever does acquire those enhancements the
policy reads them as noise rather than as unknown-but-typed items.

**Zero movement is a POOL-AVAILABILITY signal, not a broken-effect signal.** A
joker whose passive is inert would still move — the row is looked up whenever
the joker is merely PRESENT, so it would just learn "worthless". The embedding
table structurally CANNOT detect the dead-joker bug class (cf. the five
integration-seam joker bugs). Checked explicitly: the hand-size family and
Showman (`j_ring_master` — there is no `showman` key) all move at or above the
median 0.304 — `turtle_bean` 0.540 (3rd largest of all 150), `stuntman` 0.391,
`ring_master` 0.367, `troubadour` 0.329, `juggler` 0.313.

---

## 9. Process lessons

- **An ablation needs a distribution-preserving arm.** Zeroing alone proves
  nothing: any large OOD input perturbation degrades a policy whether or not the
  removed content was meaningful. The PERMUTATION arm — same marginals, broken
  mapping — is what converted "the policy got worse" into "the MAPPING is what
  matters". Build it into every future ablation here.
- **"Never moved from init" and "essential" are compatible.** The intuition that
  they conflict is what made this look strange. Static ≠ unused.
- **Adam's normalized step size is a measurement instrument**, not just an
  optimizer. Because step ≈ `lr` per update, observed displacement compared
  against `lr·N·√d` vs `lr·√N·√d` discriminates learning from diffusion. It
  needs the update count (checkpoint optimizer state) and the row hit rate —
  MEASURE the hit rate, do not model it from game rules.
- **A confounded statistic that was nearly load-bearing:** Adam's
  `|exp_avg|/√exp_avg_sq` (median 0.0012) was initially read as direct evidence
  of gradient cancellation. It is confounded — `exp_avg` decays over ~10 steps,
  so rows whose joker was absent recently read near-zero from SPARSITY, not
  noise. The displacement arithmetic is the load-bearing evidence; that ratio is
  not.

---

## 10. Reproduction

- Extraction + UMAP: `scripts/extract_shop_joker_embeddings.py`
  (`--checkpoint`, `--output`, `--3d`, `--interactive-output`).
- **Option A probe (§5): `scripts/probe_joker_synergy.py`.** Defaults target
  `runs/shop_ppo/s2_a4/best_model/best_model.zip` + that run's `reservoir.pkl`
  (2048 shop snapshots; 40 sampled with >=2 free joker slots, pack-pending
  strata excluded). `--n-states`, `--n-partners`, `--max-groups` (smoke),
  `--filler`, `--seed`, `--n-boot`. ~26 s and ~41k critic forwards for the
  full run. `s1_schema` is RECOVERED from the checkpoint's observation space
  rather than passed, since a mismatch there is a silent garbage-in
  measurement. Artifacts: `data/joker_synergy_probe.json` (+ `_seed{1,2,3}`).
- Φ-shaping check for any run whose critic is being read for magnitudes:
  `shop/phi_beta` / `shop/blend_beta` / `shop/count_beta` are logged to
  TensorBoard by `train_shop_ppo.py:156`. For `s2_a4`: phi_beta0=0.1 decaying
  to 6e-5, blend_beta identically 0 — so V is effectively in P(win) units at
  the final checkpoint. §6.4's "you may not be able to check" is resolved for
  any run that still has its event file.
- Floor: `scripts/eval_shop_policy.py --policy nextround --win-ante 4
  --n-episodes 200 --s1-schema --hand-policy
  runs/hand_ppo_b/h2/checkpoints/hand_ppo_b_2000000_steps.zip
  --partner-money-ordering`.
- Artifacts: `data/embedding_ablation.json`,
  `data/joker_embeddings_s2_a3.npz` / `.png` (`data/` is gitignored).

**TODO — PARTLY CLOSED 2026-08-09.** The probe is now checked in
(`scripts/probe_joker_synergy.py`) and owns a documented state-construction
path (restore → mutate `gs` → re-encode). Still ad-hoc and NOT checked in: the
three-arm ablation / appearance-rate harness and the fresh-init control from
the original session. When either is next needed (rung 0's joker-ids-only
variant is the likely trigger), promote it to `scripts/` and make it consume
the probe's construction path rather than growing a second one — a divergence
between them is the solver/env-divergence bug class applied to analysis
tooling. Note the probe deliberately does NOT reimplement the ablation's
permutation arm for this reason.

## 11. Caveats

- The ablation zeroed/permuted the WHOLE table (all ~300 center keys), so
  consumables/vouchers/boosters were ablated alongside jokers. "Identity mapping
  is load-bearing" holds; the split ACROSS entity types is NOT isolated. A
  joker-ids-only variant would separate them.
- All measurements are on `s2_a3` / `s2_a4`. Nothing here has been checked
  against the h-track's hand policy, which has its own embedding usage.
- §1.4's rarity-as-frequency-proxy is indirect; §1.3's direct hit-rate
  measurement supersedes it and is what the arithmetic uses.

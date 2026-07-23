# a4 reward alteration — boss-clear-scaled shop reward

Status: DESIGN (findings validated 2026-07-23; plan not yet built).
Scope: the shop agent (s2 line, a4 horizon). Hand agent untouched.

## Motivation

The shop agent is stalling. Working hypothesis: **too many roughly-viable
strategies**, so the ante scaffolding (`c_ante` per-blind bonus) is nearly
uniform across builds and gives the agent no gradient toward *good* builds.
Idea: scale the reward by **how likely each build is to clear**, so the signal
discriminates builds by boss-clearing power — concentrated at the **boss blind**,
the obstacle-conditioned checkpoint where build quality actually gets tested
(economy / scaling / banking all pay off *into* the boss, so P(clear boss) is a
far less myopic build-quality proxy than P(clear a small blind)).

Before building it we measured whether the signal is real. It is (in the antes
that matter). Findings first, then the plan.

### Why *pure* p_clear — the load-bearing rationale

The alteration only earns its place if it injects a signal **orthogonal to what
the reward already carries**. The reward is already money-aware: terminal
P(win), `c_ante`, and the critic's `V ≈ P(win)` all encode money. So scaling the
reward by any **money-inclusive** metric — `V`, or `min(V,1)` — merely
re-expresses `V`, which the agent already has in its advantage estimates. It
adds **no new discrimination** (two builds the critic rates equal stay equal),
so it cannot move the builds problem. It is redundant by construction.

What the reward does **not** carry is **draw-marginalized clear-ability**. The
realized reward is a noisy Bernoulli ("did this build clear this boss *this*
draw"). The redeal-averaged pure p_clear is the counterfactual, luck-discounted
quantity ("how reliably does this *build* clear"), and — being money-free — it
**separates the builds `V` rates identical**: high-clear/poor vs low-clear/rich.
That separation is the entire new gradient. Hence the signal must be (a) the
redeal **average** (the expectation is the new part; the single realized draw is
already the reward) and (b) the reward-split **pure** head (money would just
re-add `V`). This is why §1 below is *necessary*, not merely cleaner.

## Findings — terminal-boss clear-probability probe

Probe: `scripts/probe_boss_clear_spread.py`. Roll out full runs (s2_a4 shop +
h2 partner), snapshot the deepest boss each run reached **pre-`SelectBlind`**,
and **redeal** its opening hand N=40 times (reseed only the `nr{ante}` shuffle
stream — build held byte-identical, boss fixed). Per redeal: the partner plays
the boss out (`cleared ∈ {0,1}`) and its critic value is recorded. Corpus:
**1,880 builds × 40 redeals**. Raw records in `data/boss_clear_probe.jsonl`;
figure in `data/boss_clear_probe_analysis.png`.

### Verdict: GO — build discrimination is real, in antes 1–4

- **Broad, middle-loaded spread**: per-build clear rate mean 0.39, std 0.34.
  Only **20% dead at 0.0** and **5% certain at 1.0** — **75% sit strictly
  between**. Not partner-saturated.
- **Within a FIXED (ante, boss) cell** — the decisive test, obstacle held
  constant — builds still span a wide band: within-cell std ≈ **0.15–0.32**
  (e.g. ante-3 The Wall 0.32, ante-1 Pillar 0.26, ante-4 Wall 0.22). Each
  build's rate is over N=40, so its sampling SE is ≈ 0.08 at p=0.5; removing
  that in quadrature leaves a **true build-driven std ≈ 0.18**. The spread is
  the *build*, not draw noise. → against the same boss, build composition moves
  clear probability by a lot. **The shaping signal exists.**

### The partner ceiling — a depth effect, not fixable by shaping

Clear rate collapses with ante: mean **0.65 (a1) → 0.32 (a4) → 0.13 (a5) →
~0.05 (a6)**; dead-build fraction climbs **0% → 21% → 49% → 58%**. Deep antes
(6–8) are both thin (126 of 1,880 records) and saturated near 0 — the partner
just dies regardless of build. So:

- Shaping helps in **antes 1–4**, where 75% of terminal bosses land and s2
  actually operates.
- It **cannot** manufacture signal past ante 5 — that's the h-partner
  bottleneck (consistent with the s0 "a4 plateaued at the hand-partner
  bottleneck" note). A deep wall is a partner problem, addressed by the next
  hand-agent bootstrap round, not by shop reward.

### The h2 critic is NOT a clean P(clear) — it's money-contaminated

`critic_mean` ranges **−0.07 to 1.76**. Values > 1 mean the h2 value head
carries the terminal **`v_curve` $ term** (`reward = 1 + v_curve` on a win), so
it estimates **P(win)+money**, not pure P(clear). Rank agreement with the
sampled clear rate (unit-free): Spearman **0.65 overall**, but **decays with
depth** — 0.64/0.60/0.55 (a1–3) → 0.44 (a4) → 0.34 (a5) → ~0 (a7).

Consequence: **the raw h2 value head is not a usable reward multiplier**, and
it **cannot be de-contaminated by subtraction**. The reward is additively
separable — `G = 1{clear}·(1 + v_curve) = 1{clear} + v_curve·1{clear}` — but the
value head learned `E[G|s] = P(clear)·(1 + E[v_curve | clear, s])`: the money
sits *inside* the clear-conditioned expectation and **multiplies** P(clear)
(which is why critic values reach ~1.76, i.e. ≈ P(clear)·1.76, not
P(clear)+0.76). There is no constant to subtract off the single trained head.
The clean recovery is at the reward, not the output (Plan §2).

### Coverage — good enough, no synthetic fallback needed

**143 / 150 jokers** appear at a terminal boss; 26 distinct bosses; ante
distribution concentrated 1–4 (1,408 of 1,880). The on-policy sample is diverse
enough that the synthetic random-subset fallback isn't required, and per-joker
attribution downstream will be well-populated.

## Plan — sample a pure P(clear) value head, then scale the boss reward

The signal is real but the only cheap estimator we have (the h2 critic) is
contaminated by the `v_curve` money term. So: obtain a **pure P(clear) head** by
splitting the reward (§1), then at training time **redeal the head** and scale
the boss reward by the averaged p_clear (§2–3). The probe's play-out sampling
becomes the head's **validation**, not its in-loop source.

### 1. A pure P(clear) head via reward decomposition ("p_clear before v_curve")

The partner reward splits additively — `G = 1{clear} + v_curve·1{clear}` — so
the clean way to get p_clear is to split at the **reward**, not to post-process
the contaminated single head:

- Train an **auxiliary critic head on the `1{clear}` component alone**. Its value
  is exactly `P(clear|s)`, pure, no `v_curve`. The existing head keeps the full
  `P(win)+money` return.
- Co-trained with the partner (or bolted on as an aux head + short fit) — **no
  separate sampled-label regression pass needed**; the reward already carries
  the clean target.
- The existing single head **cannot** be de-contaminated post-hoc (Findings:
  money is multiplicative-inside-the-expectation, not additive).
- h2 trained **with** v_curve (confirmed; consistent with the >1 critic values),
  so the decomposition **is** required for this partner. A future `v_curve=None`
  partner variant would already have a pure-p_clear head and skip this step.

The redeal **play-out** sampling (`probe_boss_clear_spread.py`) is now the
**validation** of this head — a reliability curve of head-p_clear vs
redeal-sampled clear, stratified by ante — not its training source. Expect it
trustworthy in antes 1–4 and weak past a5 (matching the critic-decay finding).

### 2. In-training reward: redeal the head, don't play it out

The in-loop p_clear is the head **queried over redeals**, never a play-out:

- **On a boss clear** (post-clear), snapshot the boss entry (pre-`SelectBlind`).
- **Redeal the opening hand 40×** — reseed only `nr{ante}` (the probe's exact
  mechanism) — and **query the pure-p_clear head** on each redealt opening.
- **Average the 40 head outputs** → the build's p_clear, marginalized over the
  opening-hand draw (a single query would carry the dealt-hand variance).
- Cost = 40 forward passes, **not** 40 boss play-outs — cheap enough inline.

### 3. Scale the boss reward

- **Post-clear**, scale the boss-blind reward by that averaged p_clear →
  rewards builds that clear *reliably*, discounts lucky clears, sharpens the
  gradient across the many roughly-viable strategies.
- Restrict scaling to the regime where the head is calibrated (antes 1–4);
  don't let a near-zero, near-random deep-ante estimate distort deep reward.

### 4. Keep the objective honest

Scaling *realized* reward by P(clear) is a **real objective change** (not
potential-based — it doesn't telescope under γ=1). Two admissible routes,
consistent with the project's shaping discipline:

- **Decay** the scaling coefficient to zero over training (project standard), so
  the final optimized objective is exactly P(win) regardless of how the P_clear
  scaling is shaped; OR
- express it as **potential-based shaping** `F = γΦ(s') − Φ(s)` with a
  state-only Φ derived from P_clear (policy-invariant by construction, Ng 1999).

Either preserves "clearing/winning dominates"; pick one at build time (see open
questions).

## Open questions / decisions to lock before building

- **Decay vs PBRS** for the reward alteration (§4). PBRS is the safer default
  (provably policy-invariant); decayed multiplicative scaling is simpler but a
  transient objective change.
- **Head sourcing**: aux p_clear head **co-trained** with the partner (retrain
  cost, cleanest) vs an aux head **bolted onto the frozen trunk** and fit briefly
  on the `1{clear}` reward (cheaper, no partner retrain). Confirm h2's v_curve
  status first — a `v_curve=None` partner needs neither.
- **Redeal count at train time**: 40 matches the probe; fewer (cheaper inline)
  may suffice since the head is smooth — tune against inline throughput.
- **Per-bootstrap refresh**: p_clear is partner-specific (measured on h2). The
  head rides with the partner, so it refreshes each bootstrap iteration by
  construction — but the calibration/validation pass must re-run per partner.
- **Depth handling**: hard-gate the scaling to calibrated antes, or soft-weight
  by the head's per-ante calibration confidence.

## Rejected alternatives

- **Clip the contaminated head at 1 (`min(V,1)`) instead of a pure head.**
  Rejected on two grounds:
  1. *Redundant* — `V` is money-inclusive (≈ P(win)), which the reward already
     carries, so scaling by `min(V,1)` adds no discrimination beyond `V` and
     cannot address the builds problem (see "Why pure p_clear").
  2. *Empirically wrong* — on the probe corpus, 32.7% of builds have `V ≥ 1`
     (would clip), and among them the mean actual clear rate is only 0.64 with
     33% clearing < 0.5 and some clearing 0.00. **10.7% of all builds get max
     reward (`V ≥ 1`) while clearing < 50%** — money-inflated builds that hit the
     cap at mediocre p_clear. Clipping also fails the "discount lucky clears"
     purpose for exactly the rich-lucky builds. At a boss, money can't be spent,
     so future-money value (`v_curve`) doesn't substitute for clearing *now*.
     (Computed over `data/boss_clear_probe.jsonl`: `critic_mean` vs
     `sampled_clear`.)
- **Subtract the `v_curve` term from the single head.** Impossible: the money is
  `P(clear)·(1 + E[v_curve|clear])` — multiplicative and inside the
  clear-conditioned expectation, not an additive constant (see Findings).

## Pointers

- Probe / ground-truth sampler: `scripts/probe_boss_clear_spread.py`
  (+ `tests/scripts/test_probe_boss_clear_spread.py`).
- Raw corpus: `data/boss_clear_probe.jsonl` (1,880 × 40, complete per-build
  joker/state records — aggregation is downstream).
- Figure: `data/boss_clear_probe_analysis.png`.
- Shop reward baseline this modifies: CLAUDE.md → "Shop-agent design" reward
  (`r = 1{won} + beta·c_ante·1{cleared}`) and the s1 `Φ = s0-critic` upgrade.

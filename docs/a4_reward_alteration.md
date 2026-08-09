# a4 reward alteration — clear-probability-substituted terminal reward

Status: **DESIGN LOCKED 2026-07-25** (findings validated 2026-07-23; estimator
selection measured 2026-07-25; not yet built).
**Wiring grilled against the code and amended 2026-08-09** — §Plan, §Wiring,
§Cost, §Tests and §Open questions rewritten; Findings 1–3 unchanged. The
2026-07-25 wiring sketch did not survive contact with the source: five of its
specifics were wrong or unbuildable, and one would have silently corrupted the
objective. Each is marked **AMENDED 2026-08-09** below.
Scope: the shop agent (s2 line, a4 horizon). Hand agent untouched.

> **SUPERSEDED SECTIONS.** The 2026-07-23 plan called for (1) training an
> auxiliary pure-`P(clear)` head on the partner and (2) redealing that head 40×
> per boss. Both were measured on 2026-07-25 and both are **dropped** — the head
> is a far worse estimator than sampling, and its redeals buy nothing. The
> original *findings* stand unchanged; the *plan* is rewritten. See
> "Estimator selection" below for what killed them and what remains open.

## Motivation

The shop agent is stalling. Working hypothesis: **too many roughly-viable
strategies**, so the ante scaffolding (`c_ante` per-blind bonus) is nearly
uniform across builds and gives the agent no gradient toward *good* builds.
Idea: make the reward reflect **how likely each build is to clear**, so the
signal discriminates builds by boss-clearing power — concentrated at the **boss
blind**, the obstacle-conditioned checkpoint where build quality actually gets
tested (economy / scaling / banking all pay off *into* the boss, so P(clear
boss) is a far less myopic build-quality proxy than P(clear a small blind)).

Sharper restatement of the hypothesis (2026-07-25): the *builds* are not
roughly-viable — the probe below measures true build-driven clear-probability
spread of ~0.18 against a **fixed** boss, which is large. What is uniform is the
**reward**: clearing at ante 3 pays `3/108` whether you cleared with 3× margin
or on a lucky river. The alteration exists to stop throwing that margin away.

### Relationship to the confirmed a4 diagnosis

`docs/s1-training-hiccups.md` (2026-07-22) confirmed the ante-4 wall is a
**build/economy ceiling** whose root cause is *coverage*: "PPO never sees a
strong build at ante 4 densely enough to learn build→clear." This document is a
**credit-assignment** lever, not a coverage lever. The two are complementary —
this one is much cheaper, and does not substitute for the start-state injection
idea recorded there.

### Why deal-marginalized, money-free p_clear — the load-bearing rationale

The alteration only earns its place if it injects something **the reward does
not already carry**. The reward is already money-aware: terminal P(win),
`c_ante`, and the critic's `V ≈ P(win)` all encode money.

The correct framing is **Rao-Blackwellization**, not "orthogonal information"
(the 2026-07-23 framing, now retired). What we are doing is replacing the
realized clear indicator — a single Bernoulli draw — with its **conditional
expectation given the build**. That conditional expectation *is* p_clear, by
definition. The mechanism is therefore **variance reduction on an unchanged
objective**, not new information in expectation.

This makes two properties mandatory rather than merely preferable:

- **The average, not one draw.** The expectation is the entire new part; the
  single realized draw is already the reward.
- **Money-free.** A money-inclusive `V` is simply **not the conditional
  expectation of the term being replaced**, so substituting it is *wrong*, not
  merely redundant. This is what kills `min(V,1)` on principle, independent of
  (and stronger than) the empirical objection recorded under Rejected
  alternatives.

There *is* an honest information argument available, and it is the reason a
sampled estimate beats the shop critic's own guess: the estimate is produced by
**actually playing the boss with the deployed partner**, which is privileged
information the shop critic can only infer from sparse run outcomes.

## Findings 1 — terminal-boss clear-probability probe (2026-07-23)

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
  the *build*, not draw noise. **The shaping signal exists.**

### The partner ceiling — a depth effect, not fixable by shaping

Clear rate collapses with ante: mean **0.65 (a1) → 0.32 (a4) → 0.13 (a5) →
~0.05 (a6)**; dead-build fraction climbs **0% → 21% → 49% → 58%**. Deep antes
(6–8) are both thin (126 of 1,880 records) and saturated near 0 — the partner
just dies regardless of build. Shaping helps in **antes 1–4**, where 75% of
terminal bosses land; it **cannot** manufacture signal past ante 5. A deep wall
is a partner problem, addressed by the next hand-agent bootstrap round.

### Coverage — good enough, no synthetic fallback needed

**143 / 150 jokers** appear at a terminal boss; 26 distinct bosses; ante
distribution concentrated 1–4 (1,408 of 1,880).

## Findings 2 — estimator selection (2026-07-25)

The signal is real. The open question was **how to estimate it in-loop**: query
a learned value head (cheap, one forward pass) or **sample it by playing the
boss out** (expensive, exact by construction). The 2026-07-23 plan chose the
head. Measurement reversed that.

### The contamination, and unmultiplying it

h2's reward is `1 + v_curve(ante, dollars_after_cashout)` on a clear, 0 on a
loss (`hand_play_gym.py:681`), so PPO fits its value head to

```
V(s) = P(clear|s) · (1 + E[v_curve | clear, s])
```

The money **multiplies** P(clear) — which is why observed critic values reach
1.76 (≈ P(clear)·1.76, not P(clear)+0.76), and why it cannot be *subtracted*
off. But it **can be divided** off, because the artifact h2 trained with is on
disk. Un-multiplying:

```
m̂ = v_curve_s1.value(ante, dollars)          # data/v_curve_s1.json
q̂ = v / (1 + m̂)
```

**It does not rescue the head.** Pooled across antes, correlation with held-out
sampled clear rate:

| correction | corr |
|---|---|
| head, raw (Pearson) | 0.620 |
| head, Spearman (removes *any* monotone distortion, model-free) | 0.637 |
| **head ÷ (1 + v_curve)** — the exact multiplicative model | **0.661** |
| best CV gradient-boosted transform of (head, dollars, ante) | 0.621 |
| `dollars + ante` alone, **no head at all** | 0.522 |

Un-multiplying buys **+0.041** against a **0.289** gap to sampling. Four
variants of `m̂` (entry dollars / ante+1 / crude +\$5 / +\$10 cashout offsets)
all land 0.648–0.661, so the approximation choice is immaterial; mean `m̂` ≈
0.73–0.83, consistent with the observed 1.76 ceiling.

Where the error actually lives — residual structure, antes 1–4:
`corr(residual, dollars) = −0.124` but `corr(residual, ante) = −0.347`. The
error is **ante-shaped, not dollar-shaped**. Money was never the explanation.

**And within (ante, boss) cells the correction buys exactly nothing**: raw
0.548 → corrected 0.546 (ante-4 cells: 0.358 → 0.348). In hindsight obvious —
most of what the correction was removing *was* the ante effect, and cells
already hold ante fixed.

### Why the head fails: it reads the build, not the deal

- Only **5.6%** of the head's total variance is within-build. It barely moves
  across redeals.
- Within a build, `corr(head value, cleared)` is **0.105** pooled (0.134 at a1
  decaying to 0.072 at a4). What movement it has is nearly uninformative about
  which openings clear.

Consequence: **redealing the head buys nothing.** Split-half, pooled, corr with
held-out clear rate: head at N=1 is 0.604, at N=20 it is 0.620. The 2026-07-23
plan's "redeal the opening hand 40× and average the head outputs" is a no-op —
the deal-marginalization it was meant to buy is already baked in, because the
head essentially ignores the deal.

> **RETRACTED — the "83× variance efficiency" figure.** A law-of-total-variance
> decomposition initially suggested the head was ~84× more sample-efficient than
> play-out. That derivation assumed the head's across-deal variance *equals*
> `Var_s(q)` — i.e. that the head is a correct conditional mean. The two
> measurements above show it is not. Its low across-deal variance is
> **insensitivity, not precision**: low variance around the wrong value. Any
> future estimator comparison must be model-free (below), not decomposition-based.

### Methodology — split-half, model-free

Per build, permute the 40 redeals (fixed seed), then:

```
redeals  0-19  ──►  ESTIMATORS   play-out at N = mean(cleared[:N])
                                 head at N     = mean(v[:N])
redeals 20-39  ──►  TARGET       = mean(cleared[20:])
```

Correlate each estimator with the target, across builds. Disjointness is
load-bearing: scored against its own samples, play-out trivially returns 1.0.

**Ceiling.** The target is itself a 20-sample estimate, so nothing can correlate
1.0 with it. Measured by splitting the held-out half 10-vs-10 and applying
Spearman-Brown to N=20 (`2r/(1+r)`). A noiseless estimator tops out near its
square root.

**Stratification.** Pooling across antes inflates *both* estimators, because
ante is itself a strong predictor of clear rate (`dollars + ante` alone scores
0.522). Since the alteration fires at a **fixed horizon boss** — at `win_ante=4`
that is always ante 4 — the pooled number never described the use case.
Within-cell figures center both estimator and target within each (ante, boss)
cell before pooling.

### Results

Within each ante (corr with held-out clear rate):

| ante | builds | play4 | play8 | play20 | **HEAD** | ceiling(20) |
|---|---|---|---|---|---|---|
| 1 | 362 | 0.635 | 0.716 | 0.831 | **0.619** | 0.834 |
| 2 | 283 | 0.879 | 0.917 | 0.942 | **0.619** | 0.943 |
| 3 | 342 | 0.844 | 0.901 | 0.943 | **0.548** | 0.950 |
| **4** | **421** | **0.821** | **0.893** | **0.938** | **0.397** | **0.943** |
| 5 | 346 | 0.783 | 0.835 | 0.912 | **0.266** | 0.931 |
| 6 | 99 | 0.424 | 0.604 | 0.797 | **0.187** | 0.771 |

Within (ante, boss) cells — 53 cells, 1,360 builds:

| | play1 | play2 | play4 | play8 | play20 | **HEAD** |
|---|---|---|---|---|---|---|
| pooled, centered | 0.540 | 0.659 | 0.764 | **0.842** | 0.908 | **0.548** |

**The head averaged over 20 queries is worth about ONE play-out** (0.548 vs
0.540; the head at N=1 is 0.522). At ante-4 cells: head 0.358 vs play-out N=4
0.765, N=8 0.859.

In variance-explained terms at ante 4, the head recovers ~17% of true
build-clear variance (~14% within cells); play-out at N=8 recovers ~85%.
Qualitatively: at ante 4 the head scores **0.084 on The Wall** — the big-chip
boss where build quality should matter *most* — while play-out N=8 gets 0.850.

A regression combining both (`play20 + head20`) scores 0.949 against play-out's
0.948 alone. **The head adds nothing on top of sampling**, so no hybrid exists.

Note this also invalidates the 2026-07-23 mitigation of gating the head to
"antes 1–4 where it is calibrated": within-ante it is already 0.548 at a3 and
0.397 at a4 — weak *inside* the supposedly-calibrated regime.

### What remains genuinely open: a purpose-built head

The evidence above bounds **transforms of h2's existing critic**. It does
**not** bound a head trained from scratch on `1{clear}` with Monte-Carlo
targets, which could plausibly be much better — a PPO critic is fit to reduce
advantage variance against bootstrapped GAE targets, not to be a calibrated
probability estimator, and looseness is expected. The measured result refutes
"the 0.62 is mostly v_curve contamination"; it does not refute "a dedicated head
could work."

It is nevertheless **not built**, for three reasons:

1. **No headroom to win.** Play-out at N=8 already sits at ~95% of the
   measurement ceiling at antes 2–4. A head could only ever be a *speed*
   optimization.
2. **Validating it requires play-out anyway.** Any calibration gate compares the
   head against sampled clear rates.
3. **Sampling produces its training data for free.** Running play-out in
   training emits exactly the `(build → sampled clear rate)` corpus,
   on-distribution, that fitting and calibrating a head requires.

So the head is **deferred, not rejected** — and the ordering is strictly better
than building it first: swap it in later, with its gate already satisfied by
data the pivot was producing anyway. Escape hatch if wall clock hurts: freeze
h2's trunk, attach `MLP(256→64→1)` alongside `value_net`
(`pointer_ppo_policy.py:81`), fit by BCE on the already-emitted
`info["balatro/cleared"]` (`hand_play_gym.py:687`), then re-run the split-half
comparison above against the in-training corpus.

## Findings 3 — sampling noise, and choosing N

Disjoint 20-sample halves of the same build: mean |diff| **0.074**, RMSE 0.112,
corr **0.948**; within 0.05 for 52% of builds, within 0.10 for 75%, within 0.20
for 92%. Implied **SE ≈ 0.079** per 20-sample estimate.

(Comparing a half against the *40-sample* average instead gives mean |diff|
0.037 and corr 0.987 — but the halves are 50% *of* that average, so
`|A − full| = |A − B| / 2` identically. Use the disjoint number.)

Mean SE by N, against a build-driven signal std of ~0.18:

| N | 4 | 8 | 12 | 20 | 40 |
|---|---|---|---|---|---|
| mean SE | 0.145 | 0.102 | 0.084 | 0.065 | 0.046 |

**Estimator variance is exactly `σ²/N`, so N play-outs give an N-fold variance
reduction on the terminal reward** versus the single realized outcome. That is
the entire benefit, stated cleanly. **N=8 is the default** (~95% of the
measurement ceiling at antes 2–4); **N=4 is a legitimate cheap mode** (~87–93%);
N=20 buys the last ~4% for 2.5× the cost and is not worth it.

## Plan — substitute a sampled p̂ for the terminal indicator (W2)

### The form

At the **horizon boss only**, replace the realized win indicator with the
sampled clear probability:

```
terminal reward = (1 − λ) · 1{won} + λ · p̂          λ0 = 1.0
```

where `p̂` is the mean of N play-outs of that boss, redealt from its pre-`SelectBlind`
snapshot. **The realized boss play is itself a valid sample** — the run plays it
for real regardless — so use it: `p̂ = (realized + N redeals) / (N + 1)`, a free
extra sample.

#### "Horizon boss" is NOT "terminal boss" — AMENDED 2026-08-09

The horizon boss is `ante == win_ante`. The **terminal** boss is the deepest one
the run reached, i.e. for a losing run, *where it died* — typically ante 2 or 3
at the a4 rung. These are different bosses and only the first is correct.

This matters because the probe captures the **terminal** one:
`BossCapturingShopRunAdapter` fires at every boss and overwrites, so
`last_boss_capture` is whichever came last. The 2026-07-25 wiring said to
promote that adapter into production — lifting it verbatim yields terminal-boss
semantics, which pays p̂ at the ante-2 boss for a run that died at ante 2. That
is exactly the earlier-boss case that **breaks unbiasedness** (see Open
questions), and it fails silently: p̂ looks reasonable, training runs, the
objective is quietly wrong. The production observer MUST filter on
`ante == win_ante`, and that filter gets its own test.

#### `blind_bonus` is overridden too — AMENDED 2026-08-09

The 2026-07-25 text said "everything else is untouched." That leaves the reward
saying two contradictory things about one event: a build that reaches the
horizon boss at p̂ = 0.6 and loses the draw is paid 0.6 on the win term (correct
— we are integrating out the coin flip) but **0** on `blind_bonus`, because the
blind genuinely was not cleared. The variance the exercise exists to remove is
still sitting in the `c_ante` term.

So p̂ substitutes into both, under **one** λ:

```
win term   = (1 − λ)·1{won}     + λ·p̂
bonus term = blend_beta · c_ante(win_ante) · [ (1 − λ)·1{cleared} + λ·p̂ ]
```

This introduces no new bias: it is the same Rao-Blackwellization applied to the
same indicator, so the term's expectation is unchanged and it still decays under
`blend_beta` as before. λ = 0 reverts *both* terms exactly.

Wiring consequence: the wrapper currently receives only the product
`rc["blind_bonus"] = cleared × blind_clear_bonus(ante_before)`, and `ante_before`
is env-internal — so on a **lost** horizon boss the product is 0 and `c_ante` is
unrecoverable. The env must also emit `rc["blind_bonus_coeff"] =
blind_clear_bonus(ante_before)` every step. Same p̂, drawn once per episode,
feeds both terms.

`Φ` shaping, count bonuses, and the churn/skip-tag terms keep their current
behaviour.

#### Free bias check on the pooled estimate — AMENDED 2026-08-09

`p̂` mixes two draws produced by *different mechanisms*. The realized deal comes
from a naturally-evolved stream state: `nr{ante}` is advanced once per blind
(`game.py:254`), so the ante-4 boss deal is the **third** advance of that LCG.
`prepare_redeal` resets the stream to `pseudohash("nr4" + seed)` — a fresh
state, structurally the **first** advance. Pooling is only legitimate if those
are identically distributed. They almost certainly are, and nobody has checked.

The check is free and requires no seed study: every horizon-boss episode emits
one realized outcome and N redeal outcomes, so log `mean(realized)` and
`mean(redeal-only p̂)` and let them accumulate. Agreement confirms the mechanism;
divergence is a **bias in p̂**, which is not cosmetic — `E[p̂|s] = P(win|s)` is
the entire justification for substitution over multiplication.

> **TRAP — `data/boss_clear_probe.jsonl` CANNOT settle this.** It looks like the
> obvious corpus (1,880 builds with both quantities) and it is structurally
> useless here: the probe snapshots the deepest boss reached, which by
> construction is the boss the run **failed** (except outright wins). Realized is
> ≈ 0 there *by selection*, against `sampled_clear` averaging 0.39. Anyone
> validating from that file will "find" a huge fake discrepancy. In training
> there is no such selection — every episode reaching the horizon boss is
> counted, win or lose.

### Why substitution and not multiplication (W2, not W1)

The rejected form is `r = 1{won} · p̂` (pay p̂ only on a clear), giving
`E = p²`. Worked at the ante-4 boss:

| build | today | W1 (`E=p²`) | **W2 (`E=p`)** |
|---|---|---|---|
| strong, p = 0.6, clears | 1.0 | 0.6 | 0.6 |
| strong, p = 0.6, loses draw | 0.0 | **0.0** | **0.6** |
| expected | 0.60 | 0.36 | **0.60** |
| mediocre, p = 0.15, expected | 0.15 | **0.0225** | 0.15 |

W1 shrinks a strong build's reward 40% and a weak build's 85% — it **sparsifies
exactly where the agent is dying** (measured a4 density ~0.22, ante-4 mean clear
0.32), which is the reward-starvation failure mode that collapsed `s1_a4_pr2`.

W2's property is stronger than "unbiased on average". For **every shop decision
state** — all of which precede boss entry, since the partner plays the boss out
inside one `step()` — the expected return is *identical* to the true objective:

```
E[p̂(s_boss) | s] = E[ P(clear|s_boss) | s ] = P(win | s)
```

Same value function, same Q, same optimal policy, same expected policy gradient,
with strictly lower variance. It is a **control variate, not a shaping term**.

Intuition: a build that arrives at the horizon boss at p̂ = 0.6 and loses the
coin flip **made every right shop decision**. Paying it zero is precisely the
draw noise the exercise exists to integrate out.

### Objective honesty

W2 needs **no decay on bias grounds** — it does not change the objective.
Sampling error is the only bias source, and it is unbiased in expectation (a
mean of iid draws), so `λ` stays a safety valve rather than a correctness
requirement: `λ0 = 1.0` with decay **off** by default, and decaying it to zero
reverts exactly to the honest indicator if anything looks wrong.

This supersedes the 2026-07-23 §4 ("decay vs PBRS"), which was framed for the
biased multiplicative form. PBRS is also no longer needed: it would inject *no*
build prior at all, whereas W2 injects no *bias* while still densifying the
signal.

Because the alteration is horizon-only, the "restrict scaling to calibrated
antes" gate is **moot** — at `win_ante=4` the horizon boss is always ante 4.

**λ is a documented exception to the decay discipline (AMENDED 2026-08-09).**
`blend_beta` and `phi_beta` both decay to zero, so at convergence the reward is
p̂ alone — whose expectation is exactly `P(win)`. λ is therefore the first term in
this project deliberately *not* decayed, and that is legitimate precisely because
it is not a shaping term. Recorded explicitly so it does not later read as an
oversight against the standing "everything decays to zero" rule.

### Monitoring consequence

**AMENDED 2026-08-09 — the original claim here was wrong.** It said `ep_rew_mean`
stops equalling win rate. It does not: episodes that never reach the horizon boss
pay 0, and `E[p̂ | reach] = P(clear | reach)`, so
`E[reward] = P(reach)·P(clear|reach) = P(win)`. The reward series remains an
**unbiased, lower-variance** estimate of win rate — strictly more readable than
before, not less. What muddies it is the other shaping terms, exactly as it was
pre-alteration.

The real monitoring hazard is unchanged and lives elsewhere: "checkpoint
selection is a lottery" is a CONFIRMED issue (`docs/s1-training-hiccups.md`
Issue 2), so honest evaluation still means `eval_shop_policy.py` rollouts on the
reserved `EVAL_` suite. See Open questions for what actually constitutes evidence
that the alteration helped.

## Wiring — three seams — REWRITTEN 2026-08-09

The env-emits-honest-components / wrapper-blends discipline
(`shop_gym.py:389-395`) decides the layout: the sampling loop lives in the
training wrapper, never in the env.

**Timing constraint that forces the shape.** The pre-`SelectBlind` boss state
exists for one instant *inside* `ShopRunAdapter._advance`, mid-`step()`. Before
the wrapper calls `env.step()` the agent is still in the shop; after it returns,
the boss has already been played and the episode is over. So the wrapper can
never reach that state itself — something inside the env must pickle it at that
instant. A lazy thunk does **not** work: the live `gs` mutates the moment
`SelectBlind` runs, so the decision to snapshot has to be taken at fire time.

1. **`jackdaw/env/shop_run_adapter.py`** — add
   `boss_entry_observer: Callable[[int, str, Callable[[], bytes]], None] | None`
   alongside the existing `hand_decision_observer`, fired in `_advance` at
   `on_deck == "Boss"` immediately before `engine_step(SelectBlind())`, receiving
   `(ante, boss_key, self.snapshot_state)`. The observer decides whether to call
   the thunk. Do **not** promote `BossCapturingShopRunAdapter` verbatim — its
   last-boss-wins semantics are wrong in production (see "Horizon boss is NOT
   terminal boss").
2. **`jackdaw/env/shop_gym.py`** — pass the observer straight through, and add
   `rc["blind_bonus_coeff"]`. The env buffers nothing and knows nothing about p̂.
   The boss is played out inside a single shop `step()`, so capture and outcome
   land in the same step.
3. **`scripts/train_shop_ppo.py`** — a `BossEntryCapture` object is built in the
   `make_train_env` factory and handed to **both** the env (as
   `boss_entry_observer`) and the wrapper. It owns the `ante == win_ante` filter
   and holds the blob for the few microseconds until the wrapper's reward code
   runs (`capture.take()`). `ShopRewardWrapper` then replays the boss N times.
   New CLI: `--p-clear-playouts N` (0 = off), `--p-clear-lambda0`,
   `--p-clear-decay`. Blob never crosses a process boundary — the wrapper wraps
   each env inside the factory, and the vec env is `DummyVecEnv`.

**Why a shared observer object rather than `info` or an env attribute.** All
three park the blob for the same instant and none differ in correctness; the
tiebreakers are that (a) the `win_ante` filter stays wrapper-side, so the
widening question is a one-line change rather than an env change, and (b)
`eval_shop_policy.py` builds `ShopGymEnv` too — passing no observer costs
exactly zero, whereas an env-side capture would pickle a full engine state per
eval episode that nothing reads, needing a disable flag. This makes
`boss_entry_observer` a second instance of the existing `hand_decision_observer`
pattern (opt-in, `None` by default, documented as costly) rather than a new one.
Routing bytes through `info` additionally pushes them into `DummyVecEnv`'s info
collection and every callback, and breaks `reward_components` being flat floats.

### Replay driver — engine-direct, NOT `_play_opening` — AMENDED 2026-08-09

The 2026-07-25 text said to lift `prepare_redeal` + `_play_opening` from the
probe. `prepare_redeal` is reusable; **`_play_opening` is not.** It drives
`HandPlayGymEnv` and calls `partner.act(obs, mask)`, which bypasses
`HandCheckpointPolicy.__call__` — the path the *live* boss play actually takes
(`shop_run_adapter.py:174`). Concretely it drops `money_aware_ordering`, the flag
`load_hand_policy` sets for the s1/s2 partner, so p̂ would estimate a **different
policy than the run plays**. That is fatal, not cosmetic: `E[p̂|s] = P(win|s)`
requires p̂ to be the conditional expectation of *the indicator the run actually
realizes*. It also demands the `ProbePartner` protocol
(`obs_version`/`action_version`), which the training partner does not implement,
and builds observations for a critic value no longer used.

Mirror `_advance` instead — restore into a `HandPlayAdapter`, loop
`hand_policy(gs)` until `done`, read `cleared` as `phase == ROUND_EVAL`. That is
byte-identical in policy path to the live run and is exactly how
`hand_play_gym.py:671,687` defines `balatro/cleared` (`hand_play_adapter.py:553-560`).

Corollary: whatever the partner does, the replay must do. The question "is
money-aware ordering optimal at a terminal blind" never arises — though for the
record it is clear-lossless by construction, since
`make_clear_gated_money_objective` (`ordering_objective.py:37-40`) is
lexicographic with clearing dominant and only prefers dollars among orderings
that already win the round.

**Partner handle.** `make_train_env` passes `hand_policy=None` for the greedy
baseline, in which case each `ShopGymEnv` builds its own `GreedyHandPolicy`
internally and the wrapper has no handle. Expose `ShopGymEnv.hand_policy` and
replay with *the env's own* partner — making "same partner as the run"
structurally true rather than by convention, and keeping the greedy ablation
baseline working.

### Landmine — `card._sort_id_counter` — REWRITTEN 2026-08-09

The 2026-07-25 call ("do not restore the counter; let it advance") was reasoned
from the wrong premise and is superseded.

**What the counter is.** Every `Card` takes a creation-order stamp from a
module-global counter (`card.py:176` → `_next_sort_id`), mirroring Balatro's
`G.sort_id`. `reset_sort_id_counter()` zeroes it and is **never called in
production** — only in tests — so in a training process it climbs forever across
every episode and every env.

**What actually has to stay bounded.** Three of the four readers care only about
relative order (Death's rightmost rule `consumables.py:443`, the deterministic
pre-sorts `rng.py:371-395`, `unique_val` in `card.py:587`). The fourth does not:
`_card_nominal` / `_card_nominal_suit` (`card_area.py:153,170`) fold
`0.000001 * sort_id` arithmetically into the **hand sort key**, alongside a
card's rank. A constant offset common to every card in a state cancels, so
absolute magnitude is irrelevant — the bounded quantity is
`1e-6 × (max_sort_id − min_sort_id)` **within one game state**, which must stay
far below the ~1-per-rank gap between `_card_nominal` values.

**Why a bare reset is worse than doing nothing.** Reset to 0 before a replay and
the restored cards still carry their original ids (~10⁶) while newly created ones
get 1, 2, 3 — a 10⁶ spread inside one state, ≈ 1.0 on the tiebreaker, reordering
cards across ranks. You would have manufactured the corruption *inside* the
replay, and on exit the live envs inherit a rewound counter.

**The correct form is scoped, and seeds from the blob:**

```python
saved = card_module._sort_id_counter
card_module._sort_id_counter = _max_sort_id(restored_gs)   # prepare_redeal's existing fallback
try:     ...run the replay...
finally: card_module._sort_id_counter = saved
```

`DummyVecEnv` is single-process and steps envs sequentially, so no other env can
run inside that window — the original collision warning assumed an *unscoped*
assignment, which is what `prepare_redeal` does today (it assigns in **both**
branches, with no "leave it alone" path; it needs an explicit mode).

Scoped, this also buys what "let it advance" cannot: replays contribute zero net
burn to the global, so they cannot inflate the within-episode id spread of the
other seven live envs — and p̂ becomes a pure function of `(blob, redeal seeds)`
rather than of process history, which is what makes the determinism test below
meaningful rather than flaky.

### Redeal seeding — AMENDED 2026-08-09

The probe derives redeal seeds from `run_seed`. That does not transfer: restored
(reservoir) episodes all carry `_episode_seed == "<restored>"`
(`shop_gym.py:322`), so every restored episode would reuse
`<restored>_REDEAL_0000`. Use a **dedicated** per-wrapper RNG drawing a nonce per
episode — kept separate from `ShopRewardWrapper._rng` so the flag-off path leaves
the harvest sampling stream byte-identical to today's.

### Build guard — keep it, but fix what it measures — AMENDED 2026-08-09

Run `_build_guard` in-loop on every redeal. Its cost is a few hundred
microseconds of dataclass serialization against a replay costing several torch
forward passes plus engine scoring — single-digit percent of the thing it guards.

But it must be repaired first. In `build_probe_record`, `predeal` is unpickled
**once, outside the loop**, and `_opening_build(predeal, opening)` computes
`deck_signature` from `predeal` — not from the replayed state. So the expensive
component is *identical across redeals by construction* and has **zero detection
power**; what actually varies is jokers / hand_levels / vouchers / dollars /
hands_left / discards_left, all read off `opening`, and those are the cheap
parts. Computing the deck signature from the **opening** state instead makes it
earn its cost: `_deck_signature` sorts by `sort_id` across deck + hand + discard
+ played, so a reshuffle leaves it invariant while a redeal that adds, drops or
mutates a card changes it. That also catches a `prepare_redeal` that corrupts
every redeal identically — which the current form cannot, since all redeals would
agree. (In the probe the deck signature's job is *recording* build identity in
the JSONL, which is legitimate but different.)

### Failure handling — AMENDED 2026-08-09

The probe raises on every anomaly, correct for an offline script. In training
that exception lands inside `step()` and kills a multi-hour run, so **fall back
to λ = 0 for that episode** — which is not ad hoc, but a special case of the
safety valve λ already defines.

A bare fallback is not enough, because it converts "my sampler is wrong" into
"training looks healthy while the reward quietly reverted for X% of episodes" —
the exact class of bug this project keeps finding late. Apply the C2 labelling
precedent: tag every failure by exception type, count it, surface the rate on
TensorBoard, and hard-stop above ~1% after a warmup (tighter than C2's 3%
precisely because the expected rate is zero).

Two classes, handled identically at the catch site but tagged separately so a
nonzero count in the second reads differently from the first:

- **Deal-dependent engine/env edges.** Genuinely transient — a redeal is a
  different deal against an identical build, so it can walk into an edge the live
  deal missed. `7cb9984` ("Fix empty-hand shop training crash": play or discard
  exhausted both hand and draw pile, leaving `SELECTING_HAND` with no legal
  action) is a real production instance of this class, now fixed to resolve as a
  clean `GAME_OVER`. Here the counter is really an *engine* bug detector.
- **Invariant violations that cannot be deal-dependent** — boss key changed
  across a redeal, phase not `SELECTING_HAND` after `SelectBlind`, build drift.
  These mean the mechanism is wrong, so every p̂ is suspect.

Restoration itself is not a risk: `restore_state` is exercised constantly by the
reservoir already.

### Cost — AMENDED 2026-08-09

Only episodes that actually reach the horizon boss pay. At the measured ~0.22
ante-4 density that is `0.22 × 8 ≈ 1.8` extra boss blinds per episode against the
~7–9 an episode already plays: roughly +20% wall clock, ~+45% at 0.5 density.
**Accepted (user call): ~+40% is fine, training is fast enough.**

Two reasons that is not a ceiling, so measure rather than model:

- **The model prices a replay as one average blind.** A boss is the most
  expensive blind in an episode (more hands and discards consumed than a Small),
  so 8 replays cost more than 8 average blinds of partner forwards. Shop-policy
  forward passes dilute this in the other direction.
- **Density rises with competence, and that is the goal.** 0.22 is today's reach
  rate. The reservoir compounds it: `fresh_frac=0.5` means half of episodes start
  from a harvested snapshot sampled uniformly over ante strata, so a good
  fraction begin one or two blinds from the horizon boss — short episodes that
  are nearly certain to trigger a replay, the worst ratio in the mix.

A timed `N=0` vs `N=8` comparison over a few thousand steps settles it in
minutes. Log the realized horizon-boss density alongside, since that is the term
that drifts. N is a CLI knob if it disappoints.

### Tests — REWRITTEN 2026-08-09

1. Observer fires exactly once per boss, pre-`SelectBlind`, with a blob that
   round-trips; the probe reproduces its published numbers through the production
   hook.
2. **Horizon filter**: an episode that dies at the ante-2 boss with `win_ante=4`
   produces **no** capture and falls through to the honest indicator. This is the
   test that stops the terminal-vs-horizon confusion from shipping.
3. Replay determinism (same blob + seeds → same p̂, independent of process
   history) and the repaired build-guard invariant.
4. sort-id: the counter is byte-identical before and after a replay, and the
   within-state spread of a replayed opening stays bounded. Two interleaved envs.
5. Replay policy path: the replayed boss uses the same partner configuration as
   the live play (pin that `money_aware_ordering` reaches the replay).
6. Failure handling: an injected replay exception falls back to λ = 0 for that
   episode, is counted under its own tag, and does not propagate.
7. `--p-clear-playouts 0` reproduces today's reward **byte-identically** (the
   same flag-off discipline as `s1_schema`); `λ=1` pays p̂ on reach regardless of
   outcome, on **both** the win term and `blind_bonus`; truncation pays 0.

## Open questions

- **N**: 8 by default, 4 if throughput demands it (Findings 3). Revisit only
  against measured wall clock.
- **Wider than the horizon boss? CLOSED 2026-08-09 — no, and for a stronger
  reason than the one recorded.** The original entry filed this under myopia
  (economy / scaling / banking pay off *later* than an ante-2 boss). That
  undersells it: widening is a **correctness** failure, not a preference. At the
  horizon boss, clearing the boss *is* winning, so
  `p̂ = P(clear boss | s_boss) = P(win | s_boss)`, and by the tower property
  `E[p̂ | s] = P(win | s)` at every shop state — which is the whole control-variate
  guarantee. At an earlier boss `p̂ = P(clear that blind)` while `P(win | s)` also
  requires surviving the antes above it, so the identity breaks and p̂ becomes a
  **biased shaping term**, dragging W2 back into exactly the regime this design
  escaped. Earlier states are already served by the existing reward (`c_ante`, Φ,
  terminal win). Not to be revisited on "the signal is too thin" grounds.
- **What counts as evidence it worked? (added 2026-08-09.)** Not a higher best
  checkpoint. Two traps: picking the best checkpoint by eval is a max over a
  noisy estimate, so a bigger eval suite shrinks per-checkpoint noise but not the
  selection bias, and it inflates *more* with more checkpoints — only safe if both
  arms get identical treatment. And W2's own theory says the objective is
  unchanged in expectation, so the predicted effect is **variance, not mean**:
  faster and more *reliable* convergence, not a higher asymptote.
  - `--p-clear-lambda0 0` reverts through the *same code path*, so the A/B is one
    flag rather than branch-vs-main — no confounds.
  - Run several seeds per arm, evaluate on the reserved `EVAL_` suite at matched
    step counts, and read the **across-seed spread**. Shrinking spread is the
    mechanism's fingerprint; one seed landing higher is what the confirmed
    lottery produces by chance.
  - Cheaper precursor: Findings 3 gives estimator variance as `σ²/N`, so log the
    variance of the terminal reward term across episodes with p̂ on and off. If
    the variance reduction is not landing, nothing downstream can work — and that
    shows up in one short run rather than after a full rung.
- **Warm start (added 2026-08-09).** `--init-from` an existing a4 checkpoint stays
  valid, and for a stronger reason than "ante 4 is where the wall is": the
  identity above means the critic's targets are unchanged *in expectation*, so a
  critic fit against the old terminal reward is fit against the same `V`. Its
  targets merely become lower-variance — it was regressing a Bernoulli and now
  regresses a continuous mean, so it should *sharpen* near the horizon boss. If
  the value head does look miscalibrated, reset **only the value head** (trunk and
  policy retained), not a retrain; policy temperature is a different lever and
  does not address calibration.
- **Per-bootstrap refresh**: p̂ is partner-specific by construction (it is
  sampled with the deployed partner), so it refreshes each bootstrap iteration
  for free. No artifact to re-extract.
- **Purpose-built head**: deferred, with the gate and training corpus produced
  by this pivot (Findings 2).

## Rejected alternatives

- **W1 — multiply the terminal indicator by p̂** (`E = p²`). Rejected: shrinks
  every reward and disproportionately at low p, i.e. sparsifies exactly the
  ante-4 regime whose reward starvation collapsed `s1_a4_pr2`. Also pays zero to
  a build that made every right decision and lost a coin flip, which is the
  draw noise the alteration exists to remove.
- **An auxiliary pure-`P(clear)` head as the in-loop estimator.** Deferred, not
  rejected outright — see Findings 2 for the measurements and the escape hatch.
- **Redealing the head 40× per boss.** Dropped: head N=1 scores 0.604 and N=20
  scores 0.620. It reads the build, not the deal.
- **Clip the contaminated head at 1 (`min(V,1)`).** Rejected: it is not the
  conditional expectation of the term being replaced (see "Why deal-marginalized,
  money-free p_clear"), and empirically 32.7% of builds have `V ≥ 1` while their
  mean actual clear rate is only 0.64 — **10.7% of all builds would take max
  reward while clearing < 50%**. At a boss, money cannot be spent, so future-money
  value does not substitute for clearing *now*.
- **Subtract the `v_curve` term from the single head.** Impossible: the money is
  multiplicative and inside the clear-conditioned expectation. *Dividing* it out
  is possible and was measured — it buys +0.041 pooled and nothing at all
  within cells (Findings 2).
- **Add a cash term to the shop reward** (raised 2026-07-25). Rejected. The hand
  agent needs `v_curve` because its episode is **one blind**, so resources it
  spends are free — `v_curve` imports the future value of money into a myopic
  episode. The shop agent's episode is the **whole run**, so money's value is
  endogenous (money → jokers → later clears → win); an explicit cash term would
  double-count what the critic already learns and reward hoarding as a terminal
  good. Specifically at the horizon boss, clearing *ends the episode*, so
  leftover dollars buy nothing — the same reasoning already baked into
  `hand_play_gym.py` ("a loss pays nothing — run over, money worthless").
  Caveat: this is horizon-specific; in the true 8-ante game cash at ante 4 does
  have future value, and the a8 rung restores that by moving the horizon.

## Pointers

- Probe / ground-truth sampler: `scripts/probe_boss_clear_spread.py`
  (+ `tests/scripts/test_probe_boss_clear_spread.py`). **AMENDED 2026-08-09:**
  `prepare_redeal` is the reusable piece — but it needs an explicit sort-id mode
  (today it *assigns* the global in both branches, with no "leave it alone"
  path), and `_build_guard`'s deck signature must be computed from the opening
  state. `_play_opening` is **not** reusable (see "Replay driver"). Extract the
  shared machinery to `jackdaw/env/boss_replay.py` and have the probe import it,
  rather than the training script importing a sibling script from `scripts/`;
  the probe keeps `"capture"`/`"max"` sort-id modes for byte-exact reproduction,
  training uses the scoped save/restore.
- Raw corpus: `data/boss_clear_probe.jsonl` (1,880 × 40, complete per-build
  joker/state records — aggregation is downstream).
- Figure: `data/boss_clear_probe_analysis.png`.
- Money artifact used for the un-multiplying analysis: `data/v_curve_s1.json`
  (the artifact h2 trained with — `docs/h2-training-setup.md:305,313`), loaded
  via `jackdaw/agents/v_curve.py::load_v_curve`.
- Estimator-selection analysis (2026-07-25):
  `scripts/analyze_boss_clear_estimators.py`
  (+ `tests/scripts/test_analyze_boss_clear_estimators.py`). Reproduces every
  Findings 2/3 figure from the corpus; run it with no arguments for the defaults
  used here. Its `retracted-decomposition` section deliberately reproduces the
  withdrawn 84× efficiency claim so the error stays checkable. Needs the
  `analysis` extra (`uv sync --extra analysis`) for scipy / scikit-learn.
- Shop reward baseline this modifies: CLAUDE.md → "Shop-agent design" reward
  (`r = 1{won} + beta·c_ante·1{cleared}`) and the s1 `Φ = s0-critic` upgrade.
- Confirmed a4 diagnosis this complements: `docs/s1-training-hiccups.md`
  (2026-07-22 density measurement).

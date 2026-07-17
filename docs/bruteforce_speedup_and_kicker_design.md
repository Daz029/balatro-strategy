# Brute-force speedup & kicker design

**Status:** GRILLED AND LOCKED 2026-07-16 — the decision record lives in
CLAUDE.md ("Kicker variants + prescreen-at-n=8"). The §6 questions below are
all RESOLVED there; this document remains the measurement record (§1-§5, §7-§8
stay authoritative for the numbers).

**K1 BUILT 2026-07-17** (branch `kicker-variants-k1`) — the variant emitter,
family key, and variants-ride-their-line. K2-K4 remain. Two spec gaps were
caught in review and fixed during the build; both are recorded in the CLAUDE.md
decision record and matter to anyone reading §6 below:

* **Editions were missing from hypothesis 2.** §6's sketch says "joker-favoured:
  kickers matching suit/parity/rank jokers", and the locked key was "chips +
  enhancement + scored-channel candidacy bits". Editions are neither — they are
  absent from `trigger_match` **by design** (it is a card × JOKER matrix), yet
  they fire on the scored channel, so under Splash a Polychrome kicker is ×1.5.
  Without an edition term a Polychrome 2 ranks below a plain King: the exact
  right-line/wrong-kicker miss this document is about. Now a presence bit.
* **The held-value test is config-derived, not a name list.** §6 q4 asks whether
  the variant set is principled or "a hand-written list that will rot". The
  answer for jokers is the B2 taxonomy (via a new public
  `trigger_match.trigger_predicate`). For *enhancements* the answer is the
  engine's own held-channel config — `get_chip_h_x_mult` (Steel),
  `get_chip_h_mult`, `ability["h_dollars"]` (Gold, which has no accessor) —
  not `{"Steel Card", "Gold Card"}`.

**Answering §6 q2 ("does capture actually reach ~1.0? Measure before
believing"):** not yet — that is K3's gate, and this build does not claim it.
What IS measured is that the fix works on purpose-built repros of the documented
signature: Splash+Lusty regret 141 (24.1% of the play's value) → 0, Raised Fist
regret 120 (18.2%) → 0, plain board unchanged, both confirmed FAILING pre-K1.
Fan-out measured at 14-23 candidates vs brute's 218, matching §6's ~15 budget.
A methodology warning for K2/K3, learned the hard way here: a *dominant complete
flush* fixture has no kicker choice to get wrong, so it passes pre-K1 and proves
nothing. The state must have a line SHORTER than 5 with the kicker actually
contested.

**Test state (2026-07-17, commit `b2ec3ff`).** Green: the 49 targeted tests (22
kicker-variant, 27 prescreen) and the full `tests/scripts` + `tests/env` sweep,
821 passed / 0 failures / ~22min, covering the generation and `trigger_match`
suites. Ruff reports 3 errors on K1-touched files, but all 3 PRE-DATE the commit
(the parent already has 2; the diff touches none of those lines) — K1 introduces
no new lint. One is a genuinely dead `COPY_JOKER_KEYS` import in `hand_solver.py`
left from an older draft that gated on raw joker keys; the shipped gates read
resolved identities via `resolve_copy_targets`, so it is safe to delete whenever
someone is in there — it is not K1 debt.

**K2 must fix the harnesses.** `top_k` now counts LINES, so the returned list is
longer than k and prefix stability holds only at line granularity — it is no
longer INDEXABLE by k. Both harnesses currently score k-cuts by slicing `[:k]`
from one max-k call (`validate_prescreen.py:204`,
`validate_prescreen_n8.py:187`); they must switch to one call per k.

**Audience:** whoever builds the fix. Read "Ruled out" before proposing anything —
four plausible hypotheses died to data in the session that produced this, and
re-litigating them wastes time.

---

## 1. The goal

The h1 regen (stages 1-4 + the harvested C2 stage, ~42k labeled examples) was
budgeted at **~12s/example ≈ 140 CPU-h**. It actually costs **~2,400 CPU-h**
— roughly **17x over**, i.e. ~8 days on 12 workers instead of "a couple of
hours". The 12s figure dates from the h0 era and predates all of phase B; it is
stale, not wrong-at-the-time.

**Goal: get the regen back to the order of ~100-200 CPU-h (hours, not days),
without corrupting label semantics.**

### Measured per-stage cost (current code)

| stage | examples | mean | median | max | projected (mean) |
|---|---|---|---|---|---|
| stage1_no_jokers | 2,000 | 15.9s | 12.6s | 50.6s | 8.8 CPU-h |
| stage2_curated | 4,000 | **673.6s** | 108.7s | **3,678.7s** | **748.5 CPU-h** |
| stage3_full | 20,000 | 73.3s | 46.8s | 176.0s | 407.3 CPU-h |
| stage4_boss | 8,000 | **294.9s** | 121.7s | **1,310.0s** | **655.4 CPU-h** |
| stage5_harvested (C2) | 7,891 | 188.4s | 40.2s | 812.9s | 567 CPU-h |

**Health warning: n=6 per stage, savagely right-tailed.** The mean is the
correct estimator for a total but is outlier-dominated at n=6 — stage2's 748
CPU-h rests almost entirely on ONE 61-minute example. The median-based floor for
everything is ~750 CPU-h (~2.6 days). Treat these as "the budget is wrong by an
order of magnitude", not as precise figures. **A ~100-example/stage timing run on
the 9600X is cheap and would settle the real distribution** — worth doing before
any capacity decision rests on these numbers.

Note stage2 is 9x stage3 per example despite being the "small" stage. That is
not an accident; see §4.

### Cost model (profiled, one representative stage2 label, 205s)

```
488 nodes  x  218 subsets  x  ~1.65ms score_hand  ~=  178s   (74% of runtime)
```

* `218 = C(8,1)+C(8,2)+C(8,3)+C(8,4)+C(8,5)` — `best_immediate_play`
  brute-forces EVERY 1-to-5-card subset at EVERY recursion node.
* `score_hand` cum 177.9s of 205s. Leaves: `evaluate_hand` 57.3s,
  `fast_clone_ability` 25.9s (1.07M calls), `get_id` 19.9s (14.1M calls),
  `calculate_joker` 17.9s (4.6M calls), `dict.get` 14.6s (30.4M calls).
* `best_immediate_play` 488 calls / `evaluate_value` 106,384 / `score_hand`
  107,775.

### Cost is exponential in `discards_left`

Measured on harvested states (the `(k/4)^discards_left` model from B7's
depth-gate work, confirmed empirically):

| `discards_left` | measured per label |
|---|---|
| 0 | 1.9s |
| 1 | 2.3s |
| 2 | 29.5s |
| 3 | 102s |
| 4 | **653s** |

Consequence for the harvested stage: **35.2% of the C1 manifest sits at
`discards_left = 4`** (turn 0 of every blind has full discards) and accounts for
**503 of its 567 CPU-h — 89% of the bill from 35% of the records**. Stages 1-4
cap at `discards_range = (0, 3)` and structurally never produce `d=4`; the
harvest is the only source of those states, which is simultaneously why they are
valuable (real, unreachable synthetically) and why they are expensive.

---

## 2. Ruled out (with evidence — do not re-litigate)

| hypothesis | verdict | evidence |
|---|---|---|
| Blueprint/Brainstorm copy-argmax blowup | **NO** | `COPY_JOKER_KEYS = {j_blueprint, j_brainstorm}`; neither is in stage2's 21-joker pool. The path is unreachable there. (They ARE in stage3/4's full-150 pool — untested there.) |
| Permutation search (Photograph/Hanging Chad, 20-120x) | **NO** | `score_hand` calls ÷ `evaluate_value` calls = **1.01**. If the covering set were firing it would be ~20. Not firing on the profiled seed. |
| B3 `best_joker_order` per candidate | **NO** | Without a copy joker it short-circuits to the closed-form `sorted_joker_order` (no argmax). `sorted_joker_order` does not appear in the top-22 hot functions; the only `play_ordering` entries are `fast_clone_*`. |
| Solver drift between 9600X and this checkout | **NO** | 250 shard states regenerated from seed and re-encoded: **0 state-reproduction mismatches**. Both validation arms agree (0.840 vs 0.845). |
| Raising prescreen `k` | **NO** | capture-by-value and regret are **identical at k=3/5/8/12**. Misses are generator-side. |
| Prescreen misses = small (size 1-4) plays under held-card jokers | **NO** | **27/27 misses have `true_size = 5`.** The generator gets the size and the line right. |

Also measured: **`get_x_same` O(n²)→O(n) gave ~1.15x** (108.7s→87.0s,
201.8s→188.7s), committed. It was 18% of runtime and the source of the 14.1M
`get_id` / 30.4M `dict.get` calls. Real, free, permanent — and **nowhere near
enough**. Micro-optimization is not the answer: the next target
(`fast_clone_ability`, 13%) is worth maybe another 1.2x and carries
shared-mutable-state risk (the stage4 `Blind` bug class). **Do not plan around
micro-opts.**

---

## 3. The 40x lever, and why it is currently blocked

`PRESCREEN_HAND_LIMIT = 8`, gate is `n > PRESCREEN_HAND_LIMIT`. So B5's
prescreen — already built — **never fires where the labels live**: the harvest's
`max_hand_size` is exactly 8, and stages 1-4 deal 8 outside B1's 10% hand-size
tail. Enabling it at n=8 would cut 218 candidates to ~k: **~40x on 74% of
runtime**, which alone brings the regen back inside budget.

It is blocked because it is currently **lossy**, measured by
`scripts/validate_prescreen_n8.py` (250 shard states + 265 brute states, stage2):

| k | capture (by value) | mean regret | max regret | max rel regret | speedup |
|---|---|---|---|---|---|
| 3 | **0.845** | 68.0 | 1988 | **90%** | 72.7x |
| 5 | **0.845** | 68.0 | 1988 | 90% | 43.6x |
| 8 | **0.845** | 68.0 | 1988 | 90% | 27.2x |
| 12 | **0.845** | 68.0 | 1988 | 90% | 18.2x |

* The prescreen misses the optimal play **~15.5%** of the time at n=8.
* Misses are **not near-ties**: mean regret 68 points, worst case **90% of the
  play's score** (box's best scores a tenth of the truth).
* **k-invariant.** Only *strict* (set-identity) capture moves, 0.449 → 0.641 at
  k=12. There is no k that rescues this.

**Verdict: do NOT lower `PRESCREEN_HAND_LIMIT` as-is.** Enabling it would make
~15% of every label wrong.

### Methodology notes (reusable, and hard-won)

* **Measure CAPTURE, not just regret.** B7 established the solver's recursive
  `p_clear` **saturates** (`_fill_hand_to_size`'s optimistic refill pins branches
  at 1.0), so regret measured through the recursion reads ~0 and hides real
  differences. This harness measures at the **single node**, on
  `best_immediate_play`'s own objective (`result.total`), which does not
  saturate.
* **Capture by VALUE, not set identity.** The argmax is not unique — a played
  hand's non-scoring kickers are interchangeable, so many subsets tie at the same
  total. Strict membership punishes an equally-optimal twin and reads ~0.45 where
  value-capture reads ~0.845 (pitfall #12: regret, not disagreement).
* **A node's brute force is ~0.36s regardless of `discards_left`.** What makes a
  slow example slow is the RECURSION (488 nodes), not any one node — and capture
  is a per-node question. So expensive states cost the same to check as cheap
  ones, and survivorship bias over "which examples finished" dissolves.
* **Free oracle:** for a `PlayHand` label, `solve_hand_turn` takes its `hold`
  straight from `best_immediate_play`'s 218-way argmax — so a stored label **is**
  the brute-force answer. Costs k+1 evaluations instead of 218.
* **Known gap in the current harness:** it measures the **root node only**. The
  prescreen would fire at EVERY node, including deep inside discard branches on
  post-discard/redraw hands (a different, more conditioned hand distribution),
  where per-node errors could compound. The planned-but-unbuilt arm: wrap
  `best_immediate_play` and run real full solves — every call becomes a measured
  node at its true depth (~488 nodes per solve, so ~10 solves ≈ 5,000 nodes,
  ~25 min). `evaluate_value` fast-clones rng/blind/cards/jokers, so measuring
  inside a live solve is verified safe (it cannot perturb the run).

---

## 4. The kicker finding (the core of this document)

**All 27/27 misses have `true_size = 5`.** The generator proposes the right size
and the right scoring LINE. **It picks the wrong KICKERS.**

Jokers over-represented in misses (of 27, over 200 states, 13.5% miss rate):

| joker | in misses | why it makes kicker choice matter |
|---|---|---|
| **j_raised_fist** | 16 | +mult from the **lowest HELD** card — the cards you DON'T play set the mult |
| **j_splash** | 12 | **all played cards score** — kickers stop being inert |
| j_lusty_joker | 8 | +mult per **Heart scored** — kicker SUIT matters (esp. under Splash) |
| j_hack | 7 | retrigger 2-5 |
| j_even_steven | 7 | +mult per **even rank scored** — kicker RANK matters |
| j_four_fingers | 6 | 4-card flushes/straights — changes which cards the line even needs |
| j_greedy_joker | 5 | +mult per Diamond scored |
| j_wee | 5 | scaling |

Worked examples (k=5, box vs truth):

```
seed 879  : jokers [lusty, raised_fist, wee]                 true 864  box 585   regret 279
seed 1624 : jokers [droll, hack, jolly, raised_fist, ride]   true 1176 box 840   regret 336
seed 2793 : jokers [four_fingers, greedy, joker, lusty, ride] true 2145 box 1674 regret 471
seed 3516 : jokers [even_steven, splash, wee]                true 882  box 690   regret 192
```

**Root cause:** `prescreen_play_candidates` pads a scoring line up to 5 cards by
**keep-priority nominal-best** — it assumes kickers are inert filler and picks
the "best" cards by nominal rank. That assumption is false on exactly these
boards: under Splash the kickers SCORE (so their suit/rank feeds
Lusty/Greedy/Even Steven), and under Raised Fist the kickers you RETAIN set the
mult (so you want to play away low cards, not keep them).

### B5 predicted this precisely

From B5's decision record (CLAUDE.md / handoff):

> **ACCEPTED RESIDUAL (user call 2026-07-15)**: 2/48 kicker-CHOICE misses (right
> line, wrong kicker — keep-priority pads nominal-best where a joker wants a
> suit/enhancement; ratios 0.92/0.71, measured regret 0.0). Named lever if it
> ever matters: **kicker VARIANTS per combination, not k.**

The residual was accepted at 2/48 with regret 0.0. On stage2's deliberately
joker-dense pool it is **13.5% with regret up to 90%**. The residual did not stay
small — **it had only been measured where it doesn't bite.** Two reasons B5's
validation missed it:

1. its regret went through the **saturating** recursive `p_clear` (a fact only
   discovered later, in B7);
2. its distribution was far less joker-dense than stage2's.

The k-invariance measured here independently confirms B5's named lever: the fix
is **kicker variants, not k**.

### Why stage2 is the worst stage (and it is not an accident)

stage2's pool was hand-picked for *"jokers that visibly CHANGE the optimal
play/discard decision, not because they're strong"*. **"Changes the optimal
decision" and "defeats a cheap heuristic" are nearly the same predicate.** The
21-joker pool concentrates Raised Fist / Splash / suit-and-parity jokers at ~3.2
avg jokers per board, so nearly every stage2 example hits them — versus a much
lower density in the 150-joker stage3 pool. That is why stage2 is 9x stage3 per
example and worst on capture. Any fix must be validated on stage2's density, not
stage3's.

---

## 5. This is a LIVE label-quality bug, not only a speed blocker

**The n>8 prescreen is already in production** for B1's 10% hand-size tail
(`hand_size_tail_prob = 0.1`, delta +1..+4). On joker-dense boards those labels
are taking this kicker hit **right now**, and B5's `regret 0.0` sign-off cannot
be trusted for them because of the saturation + density problems above.

**This wants fixing regardless of the cost question.** Re-measure the n>8 path
with the non-saturating harness (`validate_prescreen_n8.py` generalizes — it is
n-parameterised in spirit; the n==8 filter is a CLI-level choice).

---

## 6. Design space for the grill (open — no decisions taken)

The fix direction: per candidate line, emit a small set of **kicker variants**
instead of a single nominal-best padding. Roughly:

* **nominal-best** (current behaviour — keep as one variant),
* **joker-favoured**: kickers matching suit/parity/rank jokers that score
  (Lusty→Hearts, Greedy→Diamonds, Even Steven→evens), which only matters when
  the kickers actually score (Splash) or are retriggered (Hack),
* **keep-highest-held**: play away the LOW cards, for Raised Fist (and any
  held-card joker — Baron class).

~3 variants x ~5 lines ≈ 15 candidates vs 218 ≈ **14x**, if capture reaches ~1.0.

### Questions the grill must answer

1. **Raised Fist is not separable.** It reads the **minimum** held rank, so
   kicker choice is a joint property of the retained set, not a per-card score.
   A greedy per-card kicker rule is not obviously optimal. Does a
   "min-held-rank-aware" variant suffice, or does this need a real argmax over
   retained sets (which is the brute force we are trying to avoid)?
2. **Does capture actually reach ~1.0?** k-invariance says the current misses are
   generator-side, but it does NOT prove that these three variants close them.
   Measure before believing. The harness makes this a minutes-long question.
3. **How many variants can we afford?** The whole point is a candidate budget.
   15 candidates ≈ 14x; 30 ≈ 7x. Where is the knee, and does capture plateau
   before the budget does?
4. **Which jokers generalise?** The miss table is stage2's pool. stage3/4 use all
   150 — Baron, Blackboard, Flower Pot, Seeing Double, Bull etc. impose other
   kicker preferences (and the copy jokers, untested, live there). Is the variant
   set principled (derived from the joker's own trigger/effect data, like the
   B2 trigger-match taxonomy) or a hand-written list that will rot?
5. **Interaction with B7.** `rank_templates_cheaply`'s discard-branch ranking
   shares `_ranking_score`. Does the same kicker blindness distort discard
   ranking too, and does fixing kickers change B7's validated depth-gate result?
6. **Label semantics.** Enabling the prescreen at n=8 changes labels for EVERY
   stage → a pre-regen lock, and it must land before C2 runs. The precedent
   exists (B5's "exact among prescreened candidates", with PPO-against-the-real-
   game as the documented bias corrector), but B5 confined it to the n>8 tail;
   this would make the WHOLE dataset prescreened. Is that acceptable, and at what
   capture threshold?
7. **Root-only vs full-solve validation.** Root capture is necessary, not
   sufficient (§3). Does the confirming instrumented-solve arm gate the change?

### Fallbacks if the kicker fix does not land

* **Cut example counts.** stage2's 4,000 are the most expensive states in the
  project, and stage3's 150-joker pool already contains **all 21** of stage2's
  jokers — so stage2 is archetype DENSITY, not unique coverage. Cutting it is the
  single biggest cheap saving (~748 CPU-h). Costs density, not breadth.
* **Cut the harvested `d=4` share** in C1. 89% of C2's cost for 35% of records.
  Re-running C1 with a cost-aware stratum is a **query, not a re-harvest** —
  which is exactly why C1 exists as a separate artifact. Costs realism at
  blind-start states, which stages 1-4 structurally cannot supply.
* **Accept a multi-day 9600X run** as specced.
* More micro-opt (~1.2x, shared-state risk). Not a plan.

---

## 7. Operational note (unrelated to the design, but bites now)

`--shard-size` defaults to **500**, but stage2 at 12 workers is **334 examples
per worker** — the cap is never reached, so a worker only flushes when it
finishes its ENTIRE range. Observed live: an 8-hour stage2 run showed "2
completed shards", which did not mean ~1,000 done — it meant 2 workers had
finished and the other 10 were holding up to 334 unflushed examples each in
memory, invisible to progress and **lost on any kill/crash** (resume is by
what's on disk).

**Always pass `--shard-size 25` (or anything << per-worker range) on a relaunch.**
Free, no semantics change, makes progress visible and caps crash-loss.

---

## 8. Artifacts

* `scripts/validate_prescreen_n8.py` — the capture/regret harness (both arms).
  Report: `data/prescreen_n8.json`.
* `tests/engine/test_get_x_same_equivalence.py` — oracle pin for the O(n)
  rewrite (exhaustive over all rank ids for short hands).
* Commit `5b9ab27` — the `get_x_same` speedup + the harness + the n=8 verdict.
* Profile reproduction: `cProfile` on `generate_one_example("stage2_curated_00000003", stage2_config)`.

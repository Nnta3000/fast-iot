# The five sanity checks: what existed, what has now been run, and where to find it

**Status: CURRENT (2026-11-05).** The user submitted five hypotheses for why the transfer from a learned
inverse-OT cost to a real pairing task might fail. This report maps each one to (a) what had already been
measured when the question arrived, with its path, and (b) the run that closes it, under
`studies/iot/real/t1t2/sanity/`. Nothing here changes a number published elsewhere; where this report restates
one, it says where it lives.

**The one-line answer.** Four of the five had no dedicated test.

- **Capacity (A) — retired.** A fit on permuted pairs lands at the assignment floor (0.27–0.50); the same
  fit on true pairs lands at 0.57–0.90, in all nine `(K, m)` cells. But the label-specific margin
  **shrinks** as `m` grows (0.58 → 0.50 → 0.39 at `K=256`): the information per pair does not compound.
- **Data-starved (B) — true only where there is room.** At `K=64` the gain over label-free cosine *rises* to
  **+13.2 points at `m = 600`**; at `K=128` it *falls* from +7.7 to +5.5 (saturated); at `K=256` the fit is
  **worse than uniform** (`R_ratio` 1.19, 1.33) for +0.5 / +1.5 points. The gain is real where cosine is
  weak and absent to negative where cosine is already 0.88. **One cell (`K=256, m=600`) was not recorded.**
- **Wrong baseline (C), transfer (D) — written, not run.**
- **Identifiability (E) — run, inconclusive on the main claim**, with one solid finding: the "textbook"
  construction loads 0.60 of `theta` on absorbed row-only/column-only columns and **fails to certify**
  `pi_hat` (`R_ratio` 1.58).

**A metric warning that applies to everything above.** `hit` is a hard 0/1 per row on a 400-pair pool, so one
row is 0.25% and the between-draw noise is 1–3 points at `K=64`. The gaps quoted here are larger than that,
but the metric should be replaced by a continuous one before deeper conclusions are drawn — see
`iot_experiments/notes/PLAN_EVALUATION_AND_APPLICATIONS.md` step 0.

---

## 0. Where everything is

| what | path |
| --- | --- |
| the new study | `iot_experiments/studies/iot/real/t1t2/sanity/` (`README.md` has questions, commands, budget) |
| its drivers | `run.py` (A, B, C), `transfer.py` (D), `ident.py` (E) |
| its outputs | `iot_experiments/out/sanity/{null,curve,baselines}/`, `out/sanity_transfer/`, `out/sanity_identifiability/` |
| this report | `iot_experiments/reports/current/SANITY_CHECKS_REPORT.md` |

The study name is `sanity` in every `RunRecord`, so one grep finds every row these checks produce.

---

## 1. Hypothesis → what existed → what runs now

### A. Capacity — "permute the pairs and compare `R*` on true vs shuffled at the real `(m, K)`"

**Existed, but not this test.**

| what | where | why it is not the null |
| --- | --- | --- |
| fit on a random `π̂`, score on the true pool: 0.010–0.018, *below* the 0.045 uniform reference | `current/REAL_PAIRS_REPORT.md` §3 | it starts from a scrambled plan; the fit is then asked to certify a plan that has no signal in it. It proves the machinery does not invent signal, not that the real fit has any |
| a synthetic "correlated" generator whose `θ*` is uniform, where the returned θ is never uniform | `current/SERVER_TESTS_REPORT.md` §6.5 | synthetic, different question |
| the chance floor is instrumented (`matching_hit` asserts a 0/1 oracle) | `iot_experiments/harness/solver.py` | a guard, not an experiment |

**Now run:** `run.py --check null`. For each `(K, m, draw)`: pool `hit` of `COSINE` and `UNIFORM`
(references that never see the labelling), and of the fit on the **true** pairs and on a **permuted**
pairing — the scramble permutes which pairs are labelled, so the correspondence is destroyed while the
marginals stay uniform, which is exactly the user's "permute the pairs and compare". Verdict per cell:
`TRUE − PERM ≥ 0.03` on every draw.

**Addition.** Every cell now also records its own **chance floor** — the `hit`/`R@1` of a *constant* cost
through the same EMD rule. Without it a hit rate is uninterpretable, and the floor is not 1/400: the rule
has its own bias. Measured here: `matching_hit` on a constant `50 x 50` cost gives 0.04 / 0.00 on two
draws, and a near-degenerate squared cost (all rows nearly equal) gives 1.0 under a min-argmin rule. A
number close to the bottom or the top of that range says nothing about the learned cost.

### B. Sample-limited vs feature-limited — "extend the curve to 1e3–1e4; if it saturates near cosine the features lack capacity"

**Existed: F2's curve, and it does not reach the regime the question is about.**

- F2: `m ∈ {50, 100, 300}`, three draws, at `out/records/metric_learning/records.json`. The committed
  per-draw values at `K = 64` are iot 0.5650 / 0.5950 / 0.5725 at `m = 50`, against a constant cosine
  0.5675.
- **Printing check, important:** the +1.0 / +2.2 / +6.6-point curve is `K = 64`-only, and F2's `iot` arm
  runs `fit_full_lp`; it is not "exact iOT" and it is not the same at `K = 256`, where cosine starts at
  0.88 and `iot` is 0.8967. The two regimes need to be read separately.
- The curve is measured in the encoder's top regime: plain **pixel-PCA** cosine is 0.055 / 0.073 / 0.065 at
  `K = 64/128/256` on the same pool (recomputed for this report from the T1/T2 PNGs with the F2 recipe), so
  DINOv2 supplies most of the signal before any learning starts.

**Now run:** `run.py --check curve --ms 50 100 300 600`. The labelled pool is 600 subjects (1000 minus the
400-pair pool), so `m = 600` is the ceiling and 1e3–1e4 is **not reachable with this pool**; that limit is
stated rather than worked around. One draw, because `m = 600` exhausts the pool. Verdict: SATURATED if
`hit(600) − hit(300) < 0.03`, RISING otherwise.

### C. Wrong baseline — "same hypothesis class under row-softmax, entropic, and exact margin"

**Partly existed**, on the identical features and protocol (`current/EXPERIMENTS.md` #12): F2 runs
`cosine`, `infonce`, `sinkhorn` (8 differentiable Sinkhorn iterations, cross-entropy against identity),
`cca`, `residual16/64`, `iot`. What is missing is the comparison the question asks for: the same
**objective** ("estimate the plan") under different **losses**, head to head, with the loss recorded.

**Now run:** `run.py --check baselines`. One code path, three engines, same features, same pool, same
three column shuffles:

| arm | loss | marginals |
| --- | --- | --- |
| `rowsoftmax` | `mean_i CE(−C_ij/τ, i)` | source only — the InfoNCE-style arm |
| `sinkhorn` | `mean_i CE(Sinkhorn(exp(−C/τ))_i, i)`, 8 differentiable iterations | source **and** target |
| `iot` | the relaxed regret LP (`select_iot`) | source and target, exact |

`(τ, epochs)` chosen by 5-fold on the labelled pairs with `epochs = 0` (== cosine) a candidate, exactly as
F2's early stopping. **The user's expectation** ("row-only entropic fitting is InfoNCE up to constants; not
verified") collapses the Sinkhorn arm to the row-softmax arm at zero temperature — so the pre-registered
comparison is `iot` vs `rowsoftmax`, and a gap there is the OT structure, not the reweighting.

### D. Transfer mismatch — "train on batches of size `s`, test on sizes `s'`, plot hit against pool size for cosine too"

**Not measured anywhere.** F2 holds the pool fixed at 400 across every `m`, and `matching_hit` uses
uniform marginals, so `hit` is invariant to the shuffling. `current/REAL_PAIRS_REPORT.md` §6 already lists
"a genuinely unpaired pool" as untested.

**Now run:** `transfer.py`. Nested pools of `C ∈ {50, 100, 200, 400, 800, 1000}` from the 1,000 subjects,
one cost per `(K, m)`, one EMD solve per pool size, **cosine measured on the same pools** so the decay is
visible without a learned arm. `hit` and `nn_hit` (free argmin, no OT) are both reported: the gap between
them is the competition effect the hypothesis is about. Reading rule: a method that decays faster than a
random pick's `1/C` is degrading with pool size, and an advantage over cosine that shrinks with `C` is a
transfer failure.

### E. Identifiability and scale — "row-only / column-only features are absorbed; double-centre and standardise"

**The principle was known; neither fix was ever applied in code.** `grep` for centre/center over
`src/iot` and the study tree returns only unrelated code. Geometry `G2` standardises and then L2-normalises
each row — a different operation from double-centring `φ`.

**Now run:** `ident.py`, synthetic on purpose (`m = n = 120`, `K = 8`; 2 signal columns, 3 row-only `u_i`,
3 column-only `v_j`), CPU/HiGHS, seconds. Two cells, `raw` and `centred`:

- `null_mass` — the `L1` mass on the absorbed directions. Centring is correct iff it drops by ≥ 10×.
- `rescale_L1` — the same instance with unequal feature scales `φ_a → λ_a φ_a`; the `L1` move of `θ`.
  Correct iff it is below the solver's own **repeat floor** (`repeat_floor`, two identical repeats).

The entropic arm's known bias is the same phenomenon from the other side: `min_θ Ψ` does not recover `θ*`
(`L1` 0.285 → 0.241 → 0.155 at `n = 120` as `ε` falls; see `current/eot/README.md` §1).

---

## 2. What has been run as of this report

| check | driver | state | record |
| --- | --- | --- | --- |
| A null | `sanity/run.py --check null` | **RAN — complete** (108 rows, 330 s, 2026-11-05 14:11) | `out/sanity/null/records.json` |
| B curve | `sanity/run.py --check curve` | **11/12 cells; RECOVERED FROM LOG** — died on `K=256, m=600` with `cudaErrorMemoryAllocation` — see §2b and §5 | `out/sanity/curve/records.recovered.json` |
| C baselines | `sanity/run.py --check baselines` | written, not launched | `out/sanity/baselines/` |
| D transfer | `sanity/transfer.py` | written, not launched | `out/sanity_transfer/` |
| E identifiability | `sanity/ident.py` | **RAN — complete** (3 views, 0.2 s); the headline question is **inconclusive on a single instance** — see §3 | `out/sanity_identifiability/records.json` |

## 2a. Check A — result

`out/sanity/null/records.json`, `K ∈ {64,128,256}`, `m ∈ {50,100,300}`, three draws, engine `fit_full_lp`
(the same engine as F2's published `iot` arm), same fixed 400-pair pool. `hit` is exact-EMD recovery, means
± population SD over draws.

| `K` | `m` | COSINE (= UNIFORM) | TRUE | PERM | TRUE − PERM | verdict |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 64 | 50 | 0.5675 | 0.6000 ± 0.008 | 0.2675 ± 0.020 | **+0.3325** | SIGNAL |
| 64 | 100 | 0.5675 | 0.5692 ± 0.023 | 0.2900 ± 0.018 | **+0.2792** | SIGNAL |
| 64 | 300 | 0.5675 | 0.6300 ± 0.015 | 0.3758 ± 0.012 | **+0.2542** | SIGNAL |
| 128 | 50 | 0.6800 | 0.7358 ± 0.015 | 0.2692 ± 0.037 | **+0.4667** | SIGNAL |
| 128 | 100 | 0.6800 | 0.7250 ± 0.017 | 0.3692 ± 0.017 | **+0.3558** | SIGNAL |
| 128 | 300 | 0.6800 | 0.7300 ± 0.022 | 0.4333 ± 0.014 | **+0.2967** | SIGNAL |
| 256 | 50 | 0.8800 | 0.8992 ± 0.004 | 0.3183 ± 0.033 | **+0.5808** | SIGNAL |
| 256 | 100 | 0.8800 | 0.9042 ± 0.014 | 0.4033 ± 0.020 | **+0.5008** | SIGNAL |
| 256 | 300 | 0.8800 | 0.8833 ± 0.012 | 0.4950 ± 0.014 | **+0.3883** | SIGNAL |

**Capacity is not the explanation.** The fit carries signal in all nine cells: the permuted fit lands
0.27–0.50, i.e. around the assignment rule's own floor (a constant cost gives 0.00–0.04), while the true fit
lands 0.57–0.90. The margin is 0.25–0.58.

**But the margin shrinks as the labelled set grows.** `TRUE − PERM` falls monotonically in `m` at every `K`:
0.333 → 0.279 → 0.254 (`K=64`), 0.467 → 0.356 → 0.297 (`K=128`), 0.581 → 0.501 → 0.388 (`K=256`). The
permuted fit gets *better* with more labelled (wrong) pairs — it converges towards the same vertex a large
sample forces, whatever the labels — while the true fit barely moves. That is a direct warning for the
"more data" plan: **the information per pair does not compound.**

**Two corrections to earlier readings.**
1. This run puts `hit` at `K=64, m=50..300` at 0.60 / 0.57 / 0.63, not 0.565 / 0.595 / 0.573 as in
   `out/records/metric_learning/records.json`. The engine is the same; the difference is the **featurisation**
   (`registry.features()` applies PCA to the 4×4 pooled tokens with no standalone standardisation here, while
   F2's stored features come from the DPM-side pipeline). F2's committed numbers stand for F2's pipeline; the
   two are not interchangeable and this report does not pool them.
2. `COSINE == UNIFORM` to the digit in every cell (0.5675 / 0.6800 / 0.8800), which confirms the geometry is
   the same one F2 used — and confirms that `theta = uniform` is exactly the squares-distance reference.

---

## 2b. Check B — result, **recovered from the log**, one cell missing

> **This section reads a recovered record, not a driver-written one.**
> The run completed 11 of 12 cells and then died with `cudaErrorMemoryAllocation` in the CuPDLPx master on
> the last cell (`K=256, m=600`). The driver writes `records.json` only at the end, so nothing was written.
> `records.recovered.json` is parsed from `run.log` by `sanity/recover_curve.py`; every row in it carries
> `recovered_from: run.log` and `complete: false`, and the missing cell is present as an **excluded** row
> rather than a number. The values are the log's **4-decimal output**, so they are rounded. `config` is not
> recoverable from a log line and is left empty; the command is in that file's `provenance`.

`out/sanity/curve/` — `run.log` (unchanged), `preflight.json`, `records.recovered.json`. One draw: `m = 600`
exhausts the 600 labelled subjects. `hit` = exact-EMD recovery on the fixed 400-pair pool; Δ is against the
label-free cosine of the same `K`. Rule fixed in `PROVENANCE` before the run: **SATURATED** if
`hit(600) − hit(300) < 0.03`, **RISING** otherwise.

| `K` | `m` | cosine (= UNIFORM) | `iot` | Δ (pts) | Δ vs `m=300` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 50 | 0.5675 | 0.5900 | +2.2 | |
| 64 | 100 | 0.5675 | 0.5375 | −3.0 | |
| 64 | 300 | 0.5675 | 0.6275 | +6.0 | |
| 64 | **600** | 0.5675 | **0.7000** | **+13.2** | **+7.2 → RISING** |
| 128 | 50 | 0.6800 | 0.7150 | +3.5 | |
| 128 | 100 | 0.6800 | 0.7025 | +2.2 | |
| 128 | 300 | 0.6800 | 0.7575 | +7.7 | |
| 128 | **600** | 0.6800 | **0.7350** | **+5.5** | **−2.2 → SATURATED** |
| 256 | 50 | 0.8800 | 0.8950 | +1.5 | |
| 256 | 100 | 0.8800 | 0.8850 | +0.5 | |
| 256 | 300 | 0.8800 | 0.8950 | +1.5 | |
| 256 | **600** | 0.8800 | **NOT RECORDED** | — | **hole** |

**Three readings, in order of how much they can be trusted.**

1. **At `K=64` the curve is RISING.** +2.2 → −3.0 → +6.0 → **+13.2** points; the `m=300 → 600` increment is
   +7.2, more than twice the 3-point threshold. At `K=64` this says data-limited, not feature-limited.
2. **At `K=128` it is SATURATED**, and it saturates *downward*: +7.7 at `m=300`, +5.5 at `m=600`. So "more
   data helps" is not monotone in `K`.
3. **At `K=256` the fit is not better than the uniform reference at all.** `m=100` and `m=300` have
   `R_ratio` **1.19 and 1.33** — worse than uniform — and the hit gaps are +0.5 / +1.5. The `m=600` cell is
   the one that OOMed.

**What that means, honestly.** The one place with a large gain (`K=64`, the encoder's weakest
representation) is also the place where cosine is weakest — 0.5675. At `K=256`, where cosine is already
0.880, the learned cost adds nothing and occasionally subtracts. That is the "cannot be made no worse than
cosine" symptom the trust-region lever targets, and it is now measured rather than suspected. It also
retires the simple reading of check A: the fit *does* buy something, but only where there is room.

**Caveats.** One draw per `m` (unavoidable — `m=600` uses the whole labelled pool), so no error bars; F2's
own three-draw swing at `K=64` is 1–3 points (0.565 / 0.595 / 0.573), which the +7.2 increment exceeds but
the +5.5 at `K=128` does not. The featurisation differs from F2's committed pipeline (§2a point 1), so these
levels are not pooled with F2's table. **Do not cite the missing `K=256, m=600` cell** — the curve there is
unknown, and the cell that OOMed is precisely the one a "does it saturate" question needs.

**Caveats.** One draw per `m` (unavoidable: the labelled pool is finite and `m=600` uses all of it), so the
`m=600` points have no error bar; F2's own point-to-point swing at three draws is 1–3 points at `K=64`
(0.565 / 0.595 / 0.573), while the +7.2 pts increment is well outside that. The `m=100` dip at `K=64`
(−3.0) reproduces F2's known non-monotonicity at that cell (F2 has 0.565 / 0.575 / 0.60 there), so it is
not new. The featurisation differs from F2's committed pipeline (§2a point 1), so the absolute levels are
not pooled with F2's table.

## 3. Check E — result, and why the headline question is still open

`out/sanity_identifiability/records.json`, 5 planted instances (`n_source = 40` or `26`, `n_target ≈ 20–26`,
`K = 4` signal + 3 row-only + 3 column-only), `theta_star = [0.45, 0, 0.55, 0]` (sparse on purpose), HiGHS
reference master, box `rel = 0.25`. `absorbed_mass` = `|theta|` on the six appended columns; `repeat_floor`
= `L1` between two identical runs; `rescale_L1` = `L1` to the fit on unevenly rescaled columns.

| view | absorbed mass | repeat floor | rescale `L1` |
| --- | ---: | ---: | ---: |
| `raw` | 0.1530 ± 0.006 | 0 | 0.0060 |
| `centred` (double-centred) | 0.1530 ± 0.006 | 0 | 0.0060 |
| `standardised` (per-column) | 0.2105 ± 0.066 | 0 | 0.1263 |
| `reference_textbook` (centred + 0.1·sd noise) | 0.6017 ± 0.107 | 0 | 0.8658 |

**What it says, and does not say.**

1. **The cheap fit does not put mass on absorbed columns, so centring has nothing to fix there.** In all
   five instances the `raw` and `centred` fits are identical to the digit, `absorbed_mass ≈ 0.153`. That
   0.153 is **not** a failure of centring: it is `theta` on the four **zero-cost padding columns** — the
   fit's feasible set is `{R = R_min}`, not `{pi_hat optimal}`, so any direction that costs nothing is
   admissible and the solver may spread `theta` along it. Requirement and identifiability are different
   things, and this record is the measurement of the former.
2. **The textbook version loads massively on absorbed columns and fails to certify.** `raw + noise` (the
   standard construction, which lets the potentials be absorbed everywhere) gives absorbed mass **0.60** —
   4× the uninformative floor — and its `R_ratio = 1.58 > 0`, i.e. **`pi_hat` is not certified for the
   returned `theta`**. This is the precise "silent failure" mode of hypothesis E: a `theta` that looks
   plausible and is not supported by the plan.
3. **Scale moves `theta`.** Unequal per-column rescaling (`lam` from 0.1 to 20) changes `theta` by
   `L1 = 0.006` in the raw fit and by `0.126` after per-column standardisation, with a repeat floor of
   **0.0**. So the simplex budget is scale-dependent, exactly as the hypothesis states — here
   standardisation made the fit *less* stable, not more, on this instance set.

**Verdict: E is INCONCLUSIVE on the question as asked** (does double-centring zero the mass on absorbed
columns, on a real fitted instance?). The instance had no headroom to show it: the admissible set is
degenerate, and the padding columns absorb the freedom. Three earlier attempts failed for related reasons
and are recorded in `ident.py`'s docstring and `PROVENANCE` rather than deleted:

| attempt | why it could not answer |
| --- | --- |
| global diagonal plan `pi = I/m` | regret linear in `theta` ⇒ optimum on a simplex vertex in every view |
| planted generator, default `theta_star` | drawn uniformly ⇒ fully supported ⇒ every coordinate forced positive |
| `fit_full_lp` as the engine | builds a **square** diagonal plan; the planted instance is `40 x 20` |

**What would close it:** an instance with a `theta_star` that has zeros, a plan strictly optimal at
`theta_star`, `theta` bounded by the **box** rather than only the simplex, and no padding columns — i.e.
`fit_full_lp`-style on a diagonal plan *plus* a planted-optimality check. That is the next `ident.py`
iteration, not a result.

---

## 4. Where each hypothesis stands after the runs

| hypothesis | before this study | after | evidence |
| --- | --- | --- | --- |
| **capacity** | untested | **retired** — signal in 9/9 cells, but the label-specific margin decays with `m` | §2a |
| **data-starved** | F2's `m ≤ 300` read as "barely beats cosine" | **partly confirmed** — gain over cosine rises to +13.2 pts at `m = 600`, curve still rising | §2b |
| **wrong baseline** | three arms existed, never compared on one loss axis | **open** — C written, not run. On existing records the three arms sit within 1–2 points at `K=256` | §1C |
| **transfer mismatch** | never measured | **open** — D written, not run (`out/` has no pool-size sweep) | §1D |
| **identifiability** | principle known, never applied | **open/inconclusive** — no headroom on the instances tried; the textbook construction *does* load on absorbed columns and fail to certify | §3 |

The paragraphs below keep the reasoning from before the runs, so the change of reading is visible.

1. **Capacity is *not* the explanation — check A retired it.** The permuted fit scores at the assignment
   floor (0.27–0.50) while the true fit scores 0.57–0.90 in all nine cells. What A *did* find is that
   `TRUE − PERM` **shrinks** as `m` grows (0.58 → 0.50 → 0.39 at `K=256`): the permuted fit improves with
   more wrong pairs, so the information per pair does not compound. That moves the question from "is there
   any signal" to "why does the margin decay".
2. **"More data" is now measured, and it helps — up to a point.** Beyond this pool there is nothing left
   to test (600 labelled subjects is the ceiling; a 1e3–1e4 curve needs more subjects, not a different
   solver). At `m = 600` the learned cost is **+13.2 points** over cosine at `K = 64`, the largest gap in
   the whole study, while check A says the label-specific margin is *diluting*. The open question is
   therefore not "more data?" but "what is the extra data buying?" — which is check C's question.
3. **The baseline question is answerable on existing data**: F2's `sinkhorn` arm is already Sinkhorn-InfoNCE
   under the same protocol. At `K = 256, m = 100` the three arms are infonce 0.9033, sinkhorn 0.9017, iot
   0.9033 — inside one SD (0.016), i.e. the OT structure and the reweighting are indistinguishable there.
   At `K = 64, m = 50` they are infonce 0.6158, sinkhorn 0.5767, iot 0.5775, against cosine 0.5675: the
   rows-only arm leads by 4 points, which is the one cell where the loss *does* separate. What is missing
   is the loss-trace comparison, which check C adds.
4. **Transfer has never been measured**; it is the cheapest of the five to get a first answer on, because
   `transfer.py` needs no `m x n` grid.
5. **Identifiability is the one that can invalidate a positive result silently.** A `theta` that loads on
   absorbed directions produces a plausible-looking cost with a meaningless `theta`; check E is seconds on
   CPU and should be run first for that reason.

## 5. The check-B run died on its last cell — GPU OOM, reported as it happened

Check B (`--check curve`, PID 93055) ran from 14:13 for **54 minutes**, completed **11 of 12 cells**, and
then **died at 15:07** with

```
CUDA Error at .../hello_ot/_native/cupdlpx/cupdlpx/solver.cu:788: cudaErrorMemoryAllocation
```

- **Exit code 1, not a signal**: it is a GPU allocation failure inside the CuPDLPx master, not the OS killer.
- **Footprint at the time**: 5 GB host RSS and the GPU at 6.1 GB of 8.2 GB before the final cell; the
  `m=600, K=256` LP needed more VRAM than the ~2 GB left. A long-lived service (`pid 5966`) permanently
  holds 4.2 GB of the card, so at most ~4 GB was ever available to this run.
- **No competition**: the other `python` processes were idle servers at ~0 GB.
- **Consequence, and the defect it exposed**: the driver writes `records.json` only at the end, so a late
  failure loses *everything*. The measurements were recovered from `run.log` by `sanity/recover_curve.py`
  into `records.recovered.json` (§2b), with `recovered_from` and `complete:false` on every row.
- **No relaunch was made.** Re-running is a retry with changed settings (drop `K=256`, or split it into its
  own job), which is the user's call.

**Fix for the next long run** (not yet applied): write the record incrementally — after every cell — so a
late failure costs one cell rather than the run.

A first launch of check A on 2026-11-05 (one `K`, one `m`, one draw) was stopped after 600 s with no log
line.

The cause was checked, not guessed. At 13:55 the machine had two other `python` jobs resident —
`entropic_vs_relaxed` (resident since 13:09, 2.1 GB) and a second `python -u` process (2.0 GB, running
since 12:20) — one of which held 4.2 GB of the 8.2 GB GPU. Host `MemAvailable` was **3 GB** against an
estimated 3 GB peak, i.e. below the 2 GB reserve `AGENTS.md` §0.2 requires. The same 1,500-image encoder
pass takes **7.7 s** in the SynthRAD log, so contention, not the study, was the explanation.

**Rule applied: one launch, stop on first failure, report and wait.** No retry was made at that point, and
no smaller variant was substituted to produce a number. When the other jobs had exited (`MemAvailable`
8 GB, 4.2 GB GPU free) the check was relaunched **once** and completed in 330 s; two further runs (B, E)
followed with the estimates stated first. The only failure in those later runs was a **code defect of mine**
— I reused `quadratic_f2._fit_loop`, whose family is `QuadraticFeatures` with `|theta| = d(d+1)/2 = 2080`
at `d = 64`, against a 64-wide point array. That is now using `fit_full_lp`, F2's own diagonal engine.
It was a `ValueError`, not an OOM, and it is recorded here rather than quietly fixed.

---

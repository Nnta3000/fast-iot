# Plan — evaluation metric, then levers, then applications

**Status: PLAN, nothing run.** The user's wording is kept; the progress column is updated as things land.
The reasoning behind the ordering is in `reports/current/SANITY_CHECKS_REPORT.md` (checks 1–2 are done) and
`notes/APPLICATIONS.md` (the applications half).

---

## Step 0 — a metric whose noise is not the experiment

**The problem, measured.** `hit` is a hard 0/1 per row on a 400-pair pool, so one row is 0.25% and F2 swings
1–3 points between draws at the same cell. Checks A and B were read against that noise.

**What to report instead, side by side with `hit` (add, never replace):**

| metric | definition | why |
| --- | --- | --- |
| `R_test` | `⟨π̂_test, c_θ⟩ − OT_test(c_θ)` on the held-out plan | continuous, uses every pair, `0` exactly when the plan is optimal |
| `R_test / R_uniform` | the same, normalised by the uniform-`θ` reference | makes it comparable across `K` and instances; raw regret is scale-dependent |
| `recall` | fraction of rows whose assigned partner is the true one | the existing `hit`; keep it, it is the interpretable one |
| `R@1` | free per-row argmin | keep, but never as a substitute — it is hubness-capped (`REAL_PAIRS_REPORT.md` §5) |

**Where the code goes:** an added metric in `experiments/harness/records.py`'s `evaluate`, computed from
the EMD plan that `matching_hit` already solves — no second solve. `regret_objective` already exists in
`iot.relaxed.master_regret`.

**Progress:** `[ ]` not started.

---

## Steps 1–8 — the levers, in the user's order

| # | step | question | status |
| --- | --- | --- | --- |
| 1 | held-out regret + recall | is the evaluation reliable? | **[ ] step 0 above** |
| 2 | shuffled-pairs null at the real `(m, K)` | is there any signal? | **[x] DONE** — signal in 9/9 cells; the *label-specific* margin **decays** with `m`. `SANITY_CHECKS_REPORT.md` §2a |
| 3 | learning curve to `10³–10⁴` | sample-limited or feature-limited? | **[~] DONE to the pool ceiling** — `m ≤ 600`; still rising at `K=64` (**+13.2 pts**). `10³+` needs a different dataset, not a different solver. §2b |
| 4 | same-class baselines (softmax / entropic / exact / relaxed) | does OT structure help? | **[~] WRITTEN, NOT RUN** — `sanity/run.py --check baselines`. The `residual64` asymmetry (below) is the open question here |
| 5 | trust region around cosine, CV | can it be made **no worse** than cosine? | **[~] PARTLY EXISTS** — the `rel` box *is* `‖θ − θ_unif‖` bounded and F2 already CV's it. Open: treat `at_bound`, and the `R_ratio > 1` cells found in check B (`K=256, m≥100`), as the diagnostic |
| 6 | feature hygiene (double-centre, standardise) | is the scale confounding? | **[~] RUN, INCONCLUSIVE** — `sanity/ident.py`; no headroom on the instances tried. The "textbook" construction does load 0.60 on absorbed columns and fail to certify |
| 7 | multi-encoder dictionary, sparse `θ` | can fusion beat one cosine? | **[ ] NOT STARTED** — `K` in the hundreds–thousands with an `L1` penalty |
| 8 | many plans, shared `θ` | does diversity help? | **[ ] NOT STARTED** — and I would run this **before 7** |

**Why 8 before 7:** the dictionary changes the hypothesis class, many-plans changes the statistics — and it
makes every free-plan application well-posed (§`notes/APPLICATIONS.md` 3.3). A single frame's identity plan
is degenerate for this solver (`AGENTS.md` §4.7).

---

## Steps P1–P4 — learning the features

| # | what | status | note |
| --- | --- | --- | --- |
| P1 | frozen encoder + exact last layer | **[x] the current setup** | this is what every existing F2 number is |
| P4 | `R*` as a representation-quality score | **[ ] NOT STARTED** | free: same fit, reported differently. It is the direct test of step 7, so it should come before P2 |
| P3 | alternate: update `ψ` from `π̂ − π*`, refit `θ` | **[ ] NOT STARTED** | the cheap version of P2 |
| P2 | differentiable layer, envelope-theorem gradient `∂R*/∂φ_e = (π̂_e − π*_e)θ*` | **[ ] NOT STARTED** | non-convex in `ψ`; `θ*` must stay global. Only worth it at `n ≫ 10⁴` where minibatches lose structure |

---

## Applications — files, not experiments

Written as **notes and specs**, deliberately outside `reports/` (see `notes/APPLICATIONS.md` rationale):

- `experiments/notes/APPLICATIONS.md` — explanatory/audit uses, boundaries, picture projects, learning
  the features, tasks that fit better.
- **RETIRED 2026-11-09**: the picture line moved to `legacy/pictures/`. See `FAUST_CORRESPONDENCE_GUIDELINE.md`
  for what replaced it. `legacy/pictures/README.md` keeps the six pitfalls worth not repeating.

**Progress:** `[x] written`, `[ ] not run`.

---

## The observation that may matter more than any lever

`residual64` gains far faster than the OT arms in F2 (+25.5 points at `m = 300`). It is **not** an iOT arm:
a learned bilinear map with **step 0 exactly equal to cosine**, trained by gradient descent. So the
comparison is "flexible supervised metric, same features, same early stopping" vs "simplex-constrained OT
fit" — and the README records that residual64 selects the epoch-0 checkpoint (cosine) in 7 of 27 fits. If
that asymmetry is understood, most of steps 5–8 become moot. **This belongs in step 4's run, not after it.**

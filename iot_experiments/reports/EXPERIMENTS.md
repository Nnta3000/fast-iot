# Experiments — the list

A flat list of every experiment this project has attempted on the **inverse-OT side**, in the order
they were reached. One line each: what question it asks, how it was run, and where the result lives.

**Read the `status` column before citing anything.** Several entries are `NOT RUN` or `BLOCKED`;
those have no numbers and must not be quoted as if they did.

Vocabulary: `m` = labelled pairs, `n` = pool candidates, `d` = PCA width, `K` = `|θ|` (length of the
cost vector). Residual arms have **no θ** and are labelled `r = 64`.

---

## 1. Synthetic core (`studies/synthetic/`)

These established the solver on instances with known ground truth. All are **historical** on the
GPU-only target path (NumPy scan + HiGHS master + POT truth); kept for provenance.

| # | experiment | question | status | report |
|---|---|---|---|---|
| 1 | **scaling** | how does solve time grow in `K`, `m`, `n`, and where does it exceed 2 s? | historical | `superseded/SCALING_REPORT.md` |
| 2 | **marginals** | uniform vs Gaussian / GMM mass vectors — support structure and cost | historical | `superseded/MARGINAL_REPORT.md` |
| 3 | **families** | diagonal / quadratic / low-rank, plus tiny-`n` bilinear-PSD | historical | `superseded/FAMILY_REPORT.md` |
| 4 | **benchmark (exact)** | exact constraint generation on the frozen `PROTOCOL.md` instances | historical | `superseded/BENCHMARK_EXACT_REPORT.md` |
| 5 | **measurement notes** | measure the algorithm, not the scaffolding; two-phase RSS | historical | `superseded/MEASUREMENT_NOTES.md` |

## 2. Solver components (`src/iot`, measured on synthetic instances)

| # | experiment | question | status | report |
|---|---|---|---|---|
| 6 | **CuPDLPx master** | can the master LP run first-order on GPU and replace HiGHS? | partial (adapter current, measurements vs HiGHS) | `components/CUPDLPX_MASTER_REPORT.md` |
| 7 | **HELLO-OT separation scan** | is the GPU violation scan bit-identical to the NumPy one, and faster? | current, **diagonal only** | `components/HELLO_SCAN_REPORT.md` |
| 8 | **HELLO-OT inner solver** | HELLO as the exact-OT oracle for subgradient / cutting-plane arms | historical (other method) | `components/HELLO_SOLVER_REPORT.md` |
| 9 | **Cutting plane (Kelley)** | a different algorithm: cut pool + master, `n`-independent | **retired** to `legacy/reports/` | — |
| 10 | **Lex-generic tie-break** | the proof-covered two-solve rule vs `random` / `none` | retired | `legacy/reports/LEX_GENERIC_REPORT.md` |
| 11 | **GPU server tests** | consolidated A100 runs of the GPU-only path | current — cite for server results | `current/SERVER_TESTS_REPORT.md` |

## 3. Real data — T1/T2 pairs (`studies/real/t1t2/`)

1,000 subjects, DINOv2-pool4 PCA features, fixed 400-pair held-out pool, geometry `G2`.

| # | experiment | question | status | report / record |
|---|---|---|---|---|
| 12 | **metric learning** | does relaxed iOT beat ordinary metric learning, or only tie cosine? Arms: cosine, `infonce`, `sinkhorn`, `cca`, `iot`, `residual16`, `residual64`; `k ∈ {64,128,256}`, `m ∈ {50,100,300}`, 3 draws | current | `metric_learning/records.json`, F2 |
| 13 | **pool recovery** | given the pool, does the learned cost recover the true partner? P@25/50/75/100 | current (table only — generating JSON was never committed) | `current/REAL_PAIRS_REPORT.md` |
| 14 | **NMI + Hungarian** | registration-style pairing on all 1,000 pairs vs whitened-DINOv2 cosine / OT | current, one labelled reference row | `current/NMI_PAIRING_REPORT.md` |
| 15 | **geometry sweep** | does the choice of normalisation/geometry change the ranking? | **BLOCKED** — no geometry-sweep record exists anywhere; F1 was skipped | none |
| 16 | **E5 pseudo-pairs** | synthesise pseudo-pairs from the `N=1` trainer | **NOT RUN** — scope only | `current/E5_SCOPE.md` |
| 17 | **quadratic (Mahalanobis) iOT in F2's setup** | does the quadratic family `K = d(d+1)/2` beat diagonal iOT / cosine on F2's exact pool? `d=64`, same 9-cell grid, relaxed loop, `prune=False` | **RAN, then EXCLUDED** — 0/9 fits met `converged ∧ U−L ≤ 5 % R_uniform` | `out/t1t2_quad_f2/d64/` |
| 18 | **coupled family `(θ, η)`** | `c = Σ θ_a d_a² + Σ_{a<b} η_ab d_a d_b`, `‖η‖₁ ≤ δ`, `δ` by CV | **BLOCKED** — `theta_set` incompatible with the loop's uniform baseline; no result | none |

## 4. Real data — SynthRAD brain (`studies/real/synthrad/`)

Registered CT↔MR of the same patient; ground truth is `(patient, z_index)`. Split seed 2024,
stratified by centre: **FIT 105 / DEV 45 / FINAL TEST 30**.

| # | experiment | question | status | report / record |
|---|---|---|---|---|
| 19 | **Phase 1 probe** | are the three centres comparable, and are slices valid? | done | `STATE.md` §3 |
| 20 | **gate** | can registry/protocol compatibility be established at all? | superseded by 23 | `superseded/synthrad_gate.md` |
| 21 | **Phase A** | why did one `m=600` full LP take 178 s vs 0.4 s at `m=300`? | done | `STATE.md` |
| 22 | **Phase B** | does the loop reproduce the dense LP on the **same** instance? | done — bracket contains dense `R′` | `STATE.md` §3 |
| 23 | **Phase C** | scaling of the loop to `m = 2000` | done; `m ≥ 3000` OOMs in the σ scan | `STATE.md` §3 |
| 24 | **pixel baseline** | how much signal is in raw pixels, soft vs bone window? | done — features 2.3–2.9× on exact-slice, nothing on patient recovery | `STATE.md` §3.5 |
| 25 | **pair recovery** | exact-slice and same-patient recovery vs cosine and metadata baselines | current | `current/synthrad_recovery.md`, F4 |
| 26 | **protocol v2** | the agreed evaluation yardstick | current | `current/synthrad_plan_v2.md` |
| 27 | **quadratic iOT task test** | does quadratic iOT do the *task* (hit rate) rather than converge? `d ∈ {16, 32, 128}`, loop | done — positive at `m=300` on this pool | `out/synthrad_quad_task/d{16,32,128}/` |

## 5. Figures (`figures/make_figures.py`)

Every figure is built from a saved record, never recomputed.

| # | figure | question | status | source |
|---|---|---|---|---|
| F1 | geometry sweep | does geometry change the answer? | **SKIPPED** — no record exists (see #15) | — |
| F2 | learned costs vs cosine, three `k` | which arm's cost finds the true partner? | built | `metric_learning/records.json` |
| F3 | precision by confidence | does a learned cost decide better when it is sure? | built, **markdown source only** | `current/REAL_PAIRS_REPORT.md` |
| F4 | SynthRAD recovery, 7 methods | exact-slice vs same-patient recovery | built, **source is git-ignored** | `out/synthrad_v2`, `out/synthrad_cheap` |
| F5 | loop vs dense LP cost | is the loop cheap enough to use? | built | `STATE.md` §3.5–3.6 |

---

## Caveats that apply to the list as a whole

- **`fit_full_lp` is not an algorithm.** It solves one LP over every edge. It is a *reference* for
  checking the loop on small instances, and it is memory-bound (`lil` assembly is `O(|E|·K)` with a
  ~4× constant, which is why `m=300, K=2080` OOMs). **Production results use
  `solve_inverse_ot_regret`.** Every figure should state which one it used; as of this list, F2 and
  F4's iOT arms were produced by `fit_full_lp`, which is a defect to fix.
- **`rel` is inverse to box width.** `lo = rel/K`, `hi = 1/(rel·K)`: a *smaller* `rel` is a *wider*
  box (`rel=0.05` → `[0.00078, 0.3125]` at `K=64`; `rel=0.9` → `[0.014, 0.017]`).
- **`matching_hit` is draw-insensitive** by construction (square marginals ⇒ permutation plan, whose
  value is invariant to column order). Repeated draws on the same pool carry no information; the
  real source of variation is the pool.
- **A convergence verdict is not a task verdict.** Item 17's fits all stalled at the CuPDLPx floor
  with `U−L` at 34–84 % of `U`; under the acceptance rule they are solver-limited and excluded, so
  no conclusion about the quadratic family is currently supported.

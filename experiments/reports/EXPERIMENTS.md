# Experiments — the list

A flat list of every experiment this project has attempted on the **inverse-OT side** — 41 entries in
five groups — in the order they were reached. One line
each: what question it asks, how it was run, and where the result lives.

**Read the `status` column before citing anything.** Several entries are `NOT RUN` or `BLOCKED`;
those have no numbers and must not be quoted as if they did. For the reports themselves, the
one-stop table is `README.md` §2.

Vocabulary: `m` = labelled pairs, `n` = pool candidates, `d` = feature/output width, `K` = `|θ|` (length
of the cost vector). Residual arms have **no θ** and are labelled `r = 64`. `EOT` = the entropic
inverse-OT master (formerly "T3"). Full definitions: `GLOSSARY.md`.

---

## 1. Synthetic core (`studies/iot/synthetic/`)

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
| 6 | **CuPDLPx master** | can the master LP run first-order on GPU and replace HiGHS? | adapter current; **measurements historical** (vs HiGHS) | `components/CUPDLPX_MASTER_REPORT.md` |
| 7 | **HELLO-OT separation scan** | is the GPU violation scan bit-identical to the NumPy one, and faster? | current, **diagonal only** | `components/HELLO_SCAN_REPORT.md` |
| 8 | **HELLO-OT inner solver** | HELLO as the exact-OT oracle for subgradient / cutting-plane arms | historical (other method) | `components/HELLO_SOLVER_REPORT.md` |
| 9 | **Cutting plane (Kelley)** | a different algorithm: cut pool + master, `n`-independent | **retired** to `legacy/reports/` | — |
| 10 | **Lex-generic tie-break** | the proof-covered two-solve rule vs `random` / `none` | retired | `legacy/reports/LEX_GENERIC_REPORT.md` |
| 11 | **GPU server tests** | consolidated A100 runs of the GPU-only path | current — cite for server results (driver paths in it are stale, see `README.md` §4) | `current/SERVER_TESTS_REPORT.md` |

## 3. Real data — T1/T2 pairs (`studies/iot/real/t1t2/`)

1,000 subjects, DINOv2-pool4 PCA features, fixed 400-pair held-out pool, geometry `G2`.

| # | experiment | question | status | report / record |
|---|---|---|---|---|
| 12 | **metric learning** | does relaxed iOT beat ordinary metric learning, or only tie cosine? Arms: cosine, `infonce`, `sinkhorn`, `cca`, `iot`, `residual16`, `residual64`; `k ∈ {64,128,256}`, `m ∈ {50,100,300}`, 3 draws | current | `metric_learning/README.md`, F2, `out/records/metric_learning/` |
| 13 | **pool recovery** | given the pool, does the learned cost recover the true partner? P@25/50/75/100 | current (table only — generating JSON never committed) | `current/REAL_PAIRS_REPORT.md` |
| 14 | **NMI + Hungarian** | registration-style pairing on all 1,000 pairs vs whitened-DINOv2 cosine / OT | current, one labelled reference row; **record absent** | `current/NMI_PAIRING_REPORT.md` |
| 15 | **geometry sweep** | does the choice of normalisation/geometry change the ranking? | **BLOCKED** — no geometry-sweep record exists anywhere; F1 was skipped | none |
| 16 | **E5 pseudo-pairs** | synthesise pseudo-pairs from the `N=1` trainer | **NOT RUN** — scope only | `current/E5_SCOPE.md` |
| 17 | **quadratic (Mahalanobis) iOT in F2's setup** | does the quadratic family `K = d(d+1)/2` beat diagonal iOT / cosine on F2's exact pool? `d=64`, same 9-cell grid, relaxed loop, `prune=False` | **RAN, then EXCLUDED** — 0/9 fits met `converged ∧ U−L ≤ 5 % R_uniform` | `out/t1t2_quad_f2/d64/` — no report |
| 18 | **coupled family `(θ, η)`** | `c = Σ θ_a d_a² + Σ_{a<b} η_ab d_a d_b`, `‖η‖₁ ≤ δ`, `δ` by CV | **BLOCKED** — `theta_set` incompatible with the loop's uniform baseline; no result | `out/t1t2_coupled/` — one log |
| 28 | **entropic metric (EOT on T1/T2)** | does the EOT master learn useful weights on the real pool, vs cosine? `K ∈ {16,32}`, `m ∈ {50,100}`, 1 draw | **PARTIAL / exploratory** — one draw, and the cosine rows are not in the record | `studies/iot/real/t1t2/entropic_metric/README.md`, `out/entropic_metric/` |
| 29 | **EOT vs relaxed LP, head-to-head** | same features, same protocol: does the entropic objective beat the relaxed regret LP? | **PARTIAL — run stopped** after `K=64, m=50` and `m=100` | `out/entropic_vs_relaxed/` — no report |
| 30 | **EOT vs relaxed LP at m=300** | same at the largest `m` | **PARTIAL — run stopped** mid-grid (`K=64`, draw 1) | `out/entropic_vs_relaxed_m300/` — no report |
| 42 | **sanity — capacity null** (`sanity/run.py --check null`) | does the fit carry signal? fit on true vs permuted pairing at the same `(K, m)` | **DONE — SIGNAL in 9/9 cells; margin falls with `m`** | `out/sanity/null/records.json`; `current/SANITY_CHECKS_REPORT.md` §2a |
| 43 | **sanity — learning curve** (`--check curve`) | sample-limited or feature-limited? curve to the whole labelled pool (`m ≤ 600`) | **PARTIAL — `K=64` RISING (+13.2 pts at `m=600`); `K=128, 256` still running** | `out/sanity/curve/run.log`; §2b |
| 44 | **sanity — three losses** (`--check baselines`) | the same objective under row-softmax / Sinkhorn / exact iOT | **NOT RUN** (written) | `current/SANITY_CHECKS_REPORT.md` |
| 45 | **sanity — pool-size transfer** (`sanity/transfer.py`) | does a cost trained on `m` pairs transfer to pools of 50…1000? | **NOT RUN** (written) | `current/SANITY_CHECKS_REPORT.md` |
| 46 | **sanity — identifiability** (`sanity/ident.py`) | are row-only/column-only features absorbed, and does double-centring + standardisation fix the scale? | **RAN — INCONCLUSIVE on the main claim** (the admissible set had no headroom); the scale result and the "textbook" failure are recorded | `out/sanity_identifiability/records.json`; §3 |

## 4. Real data — SynthRAD brain (`studies/iot/real/synthrad/`)

Registered CT↔MR of the same patient; ground truth is `(patient, z_index)`. Split seed 2024,
stratified by centre: **FIT 105 / DEV 45 / FINAL TEST 30**.

| # | experiment | question | status | report / record |
|---|---|---|---|---|
| 19 | **Phase 1 probe** | are the three centres comparable, and are slices valid? | done | `STATE.md` §3 |
| 20 | **gate** | can registry/protocol compatibility be established at all? | superseded by 23 | `superseded/synthrad_gate.md` |
| 21 | **Phase A** | why did one `m=600` full LP take 178 s vs 0.4 s at `m=300`? | done | `STATE.md`, `out/synthrad_phase_a/` |
| 22 | **Phase B** | does the loop reproduce the dense LP on the **same** instance? | done — bracket contains dense `R′` | `STATE.md` §3, `out/synthrad_phase_b/` |
| 23 | **Phase C** | scaling of the loop to `m = 2000` | done; `m ≥ 3000` OOMs in the σ scan | `STATE.md` §3, `out/synthrad_phase_c/` |
| 24 | **pixel baseline** | how much signal is in raw pixels, soft vs bone window? | done — features 2.3–2.9× on exact-slice, nothing on patient recovery | `STATE.md` §3.5 |
| 25 | **pair recovery** | exact-slice and same-patient recovery vs cosine and metadata baselines | current | `current/synthrad_recovery.md`, F4 |
| 26 | **protocol v2** | the agreed evaluation yardstick | current | `current/synthrad_plan_v2.md` |
| 27 | **quadratic iOT task test** | does quadratic iOT do the *task* (hit rate) rather than converge? `d ∈ {16, 32, 128}`, loop | done — positive at `m=300` on this pool | `out/synthrad_quad_task/d{16,32,128}/` — no report |
| 31 | **Stage 0 / Stage 1 hierarchy** | does coarse-to-fine warming pay for inverse OT? | done, **negative** (relaxed variant; 1.6–1.9× slower than flat) | `current/hierarchy_warm_start.md`, `out/synthrad_stage0{,_1}/` |
| 32 | **quadratic iOT on SynthRAD** | quadratic family in the SynthRAD loop | records exist, **no report** | `out/synthrad_loop_quad/`, `out/synthrad_quad/` |

## 5. EOT — entropic inverse-OT master (`studies/iot/real/t1t2/metric_learning/eot_*.py`)

Formerly "T3". Index: `current/eot/README.md`. All rows are planted `diagonal` instances,
`point_dimension=4`, seed 0 unless stated; `ε = rel_eps × std(c(θ_uniform))`.

| # | experiment | question | status | report / record |
|---|---|---|---|---|
| 33 | **EOT bias** (`eot_bias.py`) | is `argmin_θ inf_{f,g} Ψ` an estimator of `θ*`? | done — **no**, even at small `ε`; L1 0.285→0.241→0.155 on `n=120`, and shrinking, not diverging | `current/eot/README.md` §1, `out/eot_bias/bias.json` |
| 34 | **EOT inner observations** (`eot_inner_observations.py`) | conditioning, plan sparsity, sweep counts, gradient propagation of the inner solve | **PARTIAL — measurement only** | `current/eot/eot_inner_observations.md`, `out/eot_inner_observations/` |
| 35 | **EOT planted loop** (`eot_planted.py`) | what `|F|` does the EOT loop reach on planted instances, and is `θ` recovered? | **NO RECORD** — driver exists, output was not kept | none |
| 36 | **EOT P1 full-grid reference** (`eot_p1_reference.py`) | the transport problem and the full-θ master, Newton-CG **and** Sinkhorn side by side | **NO RECORD** — driver exists, output was not kept | none |
| 37 | **EOT Gate B** (`eot_joint_reference.py`) | does the joint formulation have the right optimum? | done — **yes**, matches the independent reference to 1e-12 from 4 starts | `current/eot/eot_gate_b.md`, `out/eot_joint_reference/` |
| 38 | **EOT solver comparison** (`eot_solver_compare.py`) | SLSQP reference vs custom Newton-CG, `n ∈ {24,70,120,300}` | done — SLSQP 24/24 at the optimum, Newton-CG 16/24; neither certificate-grade | `current/eot/eot_solver_compare.md`, `out/eot_solver_compare/` |
| 39 | **EOT dense-SLSQP scaling** (`eot_scaling_slsqp.py`) | where does dense SLSQP stop being practical? | done to `n=1500` (two numbers retracted by #41) | `current/eot/eot_scaling_slsqp.md`, `out/eot_scaling_slsqp/` |
| 40 | **EOT `|F|` fraction vs `m`** (`eot_edge_fraction.py`) | does a 20 %-of-grid `|F|` stay 20 % as `m` grows? | **record exists, no report** | `out/eot_scaling/scaling.json` |
| 41 | **EOT restricted-`F`** (`eot_restricted_edges.py`) | does a restricted `F` stay cheap **and** approximate the full grid? | done — cheap yes, close **no**; 99 % of the mass needs ≈52 % of the grid at `n=300` | `current/eot/eot_restricted_edges.md`, `out/eot_restricted_edges/` |

## 6. Shape correspondence — FAUST (`studies/iot/real/faust/`)

Registered 3D human meshes, 6890 vertices sharing one ordering, so `π̂ = I/n` **is** the ground truth. Design:
`../notes/FAUST_CORRESPONDENCE_GUIDELINE.md`.

| # | experiment | question | status | report / record |
|---|---|---|---|---|
| 49 | **data check** (`data_check.py`) | do all 100 registrations share one topology and vertex order? | **DONE 2026-11-09 — PASS**: identical face arrays, area corr 0.965, edge corr 0.956 | `current/FAUST_PRELIMINARY_REPORT.md` §1, `out/faust_data_check/` |
| 50 | **pre-flight gates** (`gates.py`) | position-only fails? uniform leaves room? oracle θ beats uniform? | **DONE 2026-11-09 — all three PASS on 3 pairs**; HKS is load-bearing (gate 6.3 fails without it) | `current/FAUST_PRELIMINARY_REPORT.md` §2–§3, `out/faust_gates/` |
| 51 | **held-out θ, subject-disjoint** (`correspondence.py`) | learn θ on one pair, test on **unseen subjects**, 3 vertex subsamples | **DONE 2026-11-09 — transfer holds**: miss@5 % 0.399 → 0.248, 8/8 pairs vs every baseline, reproduced on the reversed split | `current/FAUST_CORRESPONDENCE_REPORT.md`, `out/faust_correspondence{,_{rev}}/` |
| 52 | **geodesic error** | the standard FAUST metric | **DONE** — `potpourri3d` heat method, `sqrt(area)` normalisation | same report §1 |
| 53 | **subject-disjoint split** | do seen subjects leak into the evaluation? | **FOUND: they did**; fixed, `check_split` now exits on overlap | same report §0 |
| 54 | **box width via held-out selection** (`--rels ... --simplex`) | is the trust region helping? choose it on validation pairs | **DONE — simplex-only wins**, monotone in box width; selected arm beats every baseline 8/8 | same report §2b, `out/faust_correspondence_box/` |
| 55 | **error heat maps + left/right flips** (`heatmaps.py`) | where are the errors, and how many are mirror confusions? | **DONE** — flip rate 39 % (position) / 19 % (uniform) / **8 %** (learned) / 6 % (oracle) | same report §4b, `out/faust_heatmaps/` |

## 7. Pictures — RETIRED (`legacy/pictures/`)

Retired 2026-11-09. The learned cost drove a forward OT step to assemble an image, which is a legitimate
use of the method, but both variants failed: the jigsaw is **degenerate** (the cost diagonal is exactly 0
for any `θ`, so restoring the shuffled image is free and every arm scored MAE 0), and the non-degenerate
variant needed a palette borrowed from a *different* image, which is arbitrary. On that variant the learned
weights matched the unweighted ones to three decimals, so there was nothing to report either. The six
defects found on the way — including two visible artefacts in the figures — are tabulated in
`legacy/pictures/README.md` and are worth reading before any image or geometry experiment.

| # | experiment | status |
|---|---|---|
| 47 | photomosaic | **RETIRED** — learned ≡ unweighted (1.014–1.026× optimal) |
| 48 | jigsaw | **RETIRED** — degenerate, MAE 0 for every arm |

**Superseded by** the shape-correspondence plan in `../notes/FAUST_CORRESPONDENCE_GUIDELINE.md`, where the
plan is free, the natural cost fails visibly, and a standard evaluation protocol already exists.

## 8. Figures (`figures/make_figures.py`)

Every figure is built from a saved record, never recomputed. **None has a `.caption.md` yet**, and
F3/F4's sources are missing or git-ignored, so under `AGENTS.md` §7.2 they are not citable.

| # | figure | question | status | source |
|---|---|---|---|---|
| F1 | geometry sweep | does geometry change the answer? | **SKIPPED** — no record exists (see #15) | — |
| F2 | learned costs vs cosine, three `k` | which arm's cost finds the true partner? | built, **no caption** | `out/records/metric_learning/records.json` |
| F3 | precision by confidence | does a learned cost decide better when it is sure? | built, **source missing** | `current/REAL_PAIRS_REPORT.md` |
| F4 | SynthRAD recovery, 7 methods | exact-slice vs same-patient recovery | built, **source git-ignored** | `out/synthrad_v2`, `out/synthrad_cheap` |
| F5 | loop vs dense LP cost | is the loop cheap enough to use? | built, **no caption** | `STATE.md` §3.5–3.6 |

### Where the renamed EOT drivers live

`experiments/studies/iot/real/t1t2/metric_learning/eot_{bias,inner_observations,planted,p1_reference,joint_reference,solver_compare,edge_fraction,scaling_slsqp,restricted_edges}.py`.
The old names `t3_*` appear in git history and in the output directories only.

---

## Caveats that apply to the list as a whole

- **`fit_full_lp` is not an algorithm.** It solves one LP over every edge. It is a *reference* for
  checking the loop on small instances, and it is memory-bound (`lil` assembly is
  `O(|E|·K)` with a ~4× constant, which is why `m=300, K=2080` OOMs). **Production results use
  `solve_inverse_ot_regret`.** Every figure should state which one it used; F2's iOT arm was produced by
  `fit_full_lp`, which is a defect to fix.
- **`rel` is inverse to box width.** `lo = rel/K`, `hi = 1/(rel·K)`: a *smaller* `rel` is a *wider*
  box (`rel=0.05` → `[0.00078, 0.3125]` at `K=64`; `rel=0.9` → `[0.014, 0.017]`). Do not confuse it
  with `rel_eps`, the EOT temperature.
- **`matching_hit` is draw-insensitive** by construction (square marginals ⇒ permutation plan, whose
  value is invariant to column order). Repeated draws on the same pool carry no information; the real
  source of variation is the pool. Cosine is label-free, so its `hit` is constant per `K` — any
  variation means the pool, the geometry or the shuffle was contaminated.
- **A convergence verdict is not a task verdict.** Item 17's fits all stalled at the CuPDLPx floor
  with `U−L` at 34–84 % of `U`; under the acceptance rule they are solver-limited and excluded, so
  no conclusion about the quadratic family is currently supported.
- **"EOT" replaced "T3" on 2026-11-05** (see `CHANGELOG.md`). Any note outside this folder that says
  "T3" means EOT.

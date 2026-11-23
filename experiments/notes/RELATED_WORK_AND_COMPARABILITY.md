# Related work: what the cited papers contain, and the protocol we use to place a number beside them

**Status: inventory + protocol (2026-11-19).** This note records what is in `paper/`, what the four cited
methods actually measure, and the protocol of the reproduction we run against them. It states facts and the
protocol; it does not rank the methods and does not say which of them we should pursue.

---

## 1. What is in `paper/`

The presentation (`paper/ours/presentation.typ`, slides 21–34) discusses four inverse-OT papers. Of those four,
**none has a PDF in this repository**; what `paper/additional_sources/` holds are Obsidian notes whose
`![[... .pdf]]` embeds point at files that are not in the tree.

| method | cited as | source file in the tree | what is in the tree |
| --- | --- | --- | --- |
| RIOT | Li, Ye, Zhou & Zha, JMLR 20 (2019), art. 18-700 | none | `additional_sources/Learning to Match via Inverse Optimal Transport.md` (note) |
| Ma et al. | Ma, Sun, Ye, Zha & Zhou (2021); arXiv 2002.09650 | none | `additional_sources/Learning Cost Functions for Optimal Transport.md` (note) |
| Stuart & Wolfram | Stuart & Wolfram (2020) | none | `additional_sources/Inverse Optimal Transport.md` (note) |
| M-IOT | Bao, Li & Sun (2026), arXiv 2609.37291 | none | `additional_sources/Learning Universal Costs via Multi-Observation Inverse Optimal Transport.md` (note) |
| — | — | **`paper/Bao et al. - 2026 - Well-Posedness and Efficient Algorithms for Inverse Optimal Transport with Bregman Regularization.pdf`** | a **different** Bao–Li paper: *Well-Posedness and Efficient Algorithms for IOT with Bregman Regularization*, Bao, Li **& Yang**, arXiv **2510.03803** (30 Apr 2026). Not cited on any slide. |
| URSSM | Cao et al., SIGGRAPH/TOG 2023 | none (slides only) | slides 19–20 of the deck |

The other PDFs in `paper/` (`DualGuided…`, `Xiong 2024 (BiDPM)`, `Zhu 2025 Phys. Med. Biol.`, `datasets/SynthRADReport.pdf`,
`discussions/One_Month_PoC…`) belong to the medical-imaging material and are not part of the inverse-OT review here.

The four PDFs cited on the slides were fetched from the links printed on those slides and read to verify the
numbers below (local copies: `/tmp/opencode/papers/{riot_li2019,ma2021,stuart2020,miot2026}.pdf`). They are scratch
copies, not added to the repository.

## 2. Verified numbers on the slides

Each entry was checked against the cited section of the paper, not against the note in `additional_sources`.

| slide claim | status | the paper's text |
| --- | --- | --- |
| RIOT, synthetic recovery at N = 200: empirical 1.269 / fixed-marginal IOT 0.464 / RIOT 0.219 (KL) | confirmed | `KL(π₀‖π̂) = 1.269`, `KL(π₀‖π_RIOT) = 0.219`, `KL(π₀‖π_IOT) = 0.464`, N = 200 |
| RIOT, Dutch Household Survey: RMSE / MAE 2.3 / 1.5 (×10⁻⁴) | confirmed | Table 1: Random 54.5/36.6, PMF 8.4/2.0, SVD 29.9/16.8, itemKNN 2.4/1.6, RIOT 2.3/1.5 |
| Ma et al., discrete recovery ≈ 10⁻⁴ after 500 iterations | confirmed | "relative error approximately 10⁻⁴ or lower after 500 iterations" |
| Ma et al., Algorithm 1 exact: RMSE 2.46×10⁻¹¹, MAE 1.90×10⁻¹¹ (rounded 0.0 in the table) | confirmed | Table 2 footnote text; RIOT 1.8/1.3, itemKNN 1.8/1.3, FM 7.4/5.6, SVD 109.8/62.4, PMF 114.5/64.9, Random 45.1/31.4 |
| Ma et al., non-uniqueness: relative **plan** error 0.026 | confirmed | Figure 4d caption: `|x − 2y|²`, relative error of π = 0.026 |
| Stuart & Wolfram, acceptance rates: exact LP 50.8 % (u/v/W 53.8/53.7/44.9), Sinkhorn 82.9 % (84.7/85.5/78.6) | confirmed | verbatim, ε = 0.04 |
| Stuart & Wolfram, synthetic graph 5 nodes / 12 edges / 4 % Gaussian noise; migration 9 countries / 30 edges | confirmed | "n = 5 nodes with m = 12 edges … add 4 % noise"; "n = 9 countries, connected by m = 30 edges … noise level is 4 %" |
| M-IOT, interpolated marginals: single-observation 0.147, averaged 0.022, M-IOT 0.012 | confirmed | verbatim; γ = 1, n = 20, 20 observations |
| M-IOT notation | minor | the paper calls the observation count `K`; the slide writes `L` |

## 3. The quantities each paper reports

Four axes, filled in from the papers. The FAUST row is our own experiment and is included for the same four
columns, so the difference in reported quantities is visible in one place.

| work | unknown being inferred | observation model | reported metric | data used |
| --- | --- | --- | --- | --- |
| RIOT (Li et al. 2019) | interaction matrix `A` (10×8); cost `(γuᵀAv + c₀)^q` | empirical plan `π̂` from `N` sampled pairs; both marginals relaxed | `KL(π₀‖π̂)`; RMSE / MAE on tabular prediction | synthetic 10×10; Florida 2012 (3×5); Dutch Household Survey |
| Ma et al. (2021) | free cost matrix `C` (discrete); `c_θ` with neural networks (continuous) | entropic plan, `ε ∈ {10, 1, 0.1, 0.01}` | relative Frobenius cost error `‖ĉ−c*‖/‖c*‖`; relative plan error; RMSE / MAE | synthetic `n = 100`; USPS-MNIST; colour transfer; Dutch Household Survey |
| Stuart & Wolfram (2020) | graph edge costs `W`, source marginal `u`, target marginal `v` | forward OT map + Gaussian noise (4 %) | posterior distributions, acceptance rates, data misfit | 5-node synthetic graph; Eurostat migration (9 countries, 30 edges) |
| M-IOT (Bao, Li & Sun 2026) | one shared free cost `C` across `K` plans | `K` entropic plans, γ = 1, multiplicative lognormal noise | relative Frobenius cost error | synthetic `n = 20`; urban mobility; e-commerce |
| **this repository, FAUST arms** | `θ` on the simplex over 13 per-vertex descriptors; cost `c_ij = Σ_k θ_k Φ_ij,k` | exact OT plan, `π̂ = I/n`, no noise | geodesic error on the target / `sqrt(area)`; miss@1 % and miss@5 % | MPI-FAUST training registrations, `n = 2000` subsample, 8 held-out pairs |

Two definitions used below, both from the papers and both kept for the reproduction:

- **relative cost error** (Ma eq. 22 region; M-IOT §6): `‖ĉ − c*‖_F / ‖c*‖_F`.
- **relative plan error** (Ma Figure 4d): `‖π̂ − π(ĉ)‖_F / ‖π̂‖_F`, with `π(ĉ)` the exact OT plan at cost `ĉ`.

## 4. Reproduction protocol (Ma et al. §5.1, discrete)

The benchmark among the four whose data-generating process is stated in full in §5.1 and whose score is an error
on the inferred cost.

**Generation** (as in the paper):

- `n = 100`; ground-truth cost `c*_ij = |(i − j)/n|^p` for `p ∈ {0.5, 1, 2, 3}`;
- `20` independent marginal pairs `(μ, ν)` — we draw `μ, ν ~ Dirichlet(1, …, 1)` on the 100-simplex and record the
  choice, because the paper says "20 random marginal pairs" without fixing the distribution;
- observed plan `π̂ = sinkhorn(μ, ν, c*, ε)` in the log domain (POT `sinkhorn_log`) for `ε ∈ {10, 1, 0.1, 0.01}`,
  which reproduces the paper's four regularization weights;
- the paper's structural prior (symmetric cost, zero diagonal, non-negative entries) is described in §5.1; the
  families below replace it by a stated feature family, which is what our `FeatureMap` interface takes.

**Estimation** — our **relaxed** arm (`solve_inverse_ot_regret`, `RegretMasterSolver`, `backend=highs|cupdlpx`),
not the entropic fit: it minimises the regret `R(θ) = ⟨c(θ), π̂⟩ − OT(c_θ)`, which is defined for any plan `π̂`
and needs no assumption that `π̂` be exactly optimal at some `θ`. `π̂` here is dense and positive, so the support
`S` is the whole grid and `R(θ̂)` is expected to be strictly positive; it is reported, not treated as a
certificate.

**Feature families** (the hypothesis class is the one variable we add, so it is swept, not assumed):

| id | `Φ` | `K` | note |
| --- | --- | --- | --- |
| `p_basis` | `Φ_k(i,j) = |i−j|/n|^{p_k}` over the same four `p` | 4 | the truth is a simplex vertex; measures recovery of the generating member and its `ε`-dependence |
| `asym_basis` | `|i/n − a·j/n|^q`, `a ∈ {1,2}`, `q ∈ {1,2}` | 4 | same truth as `p_basis` at `a = 1`, plus three distractors; `a = 2` is the paper's asymmetric example |
| `free_dense` | one indicator feature per entry, `K = n²` | 10 000 | the analogue of the paper's free `C`. Memory: one `(m·n, K)` block is `10⁴ × 10⁴ × 8 B = 800 MB` |
| `hat9` | 9 hat/ramp functions of `t = |i−j|/n ∈ [0,1]` | 9 | `c*` is smooth in `t`; no member equals the truth, so the residual is the basis's approximation error |

**Gauge.** `R(θ) = ⟨c, π̂⟩ − OT(c)` is positively homogeneous in the cost, so a family whose members differ in
magnitude is scored on magnitude as well as on shape. Every feature is therefore scaled to unit Frobenius norm
before the fit, and the recovered scale is reported (`best_scale`). A positive rescaling of a cost leaves the OT
plan unchanged, so this removes the part of `θ` that is not identifiable and keeps the recovery question.

`θ` lives on the simplex (`Σθ = 1`, `θ ≥ 0`). The cost error is reported raw and after the optimal positive
rescaling `min_{α>0} ‖αĉ − c*‖/‖c*‖`, alongside a plan metric that is immune to LP tie-breaking:
`plan_cost_rel` = the excess cost of the induced plan under the true cost, relative to the true optimum (0 exactly
when the induced plan is optimal). `plan_rel` (`‖EMD(ĉ) − EMD(c*)‖_F/‖EMD(c*)‖_F`) is also recorded, with the
caveat that costs of the form `|i−j|^p` have ties, so the LP has an optimal face and the solver may return a
different vertex for a rescaled cost.

**Reported per (family, p, ε, instance):**

- `cost_rel` and `cost_rel_scaled`;
- `plan_rel` (exact EMD at `ĉ`);
- `regret` `R(θ̂)`, and `R` at the generating `θ`, as a scale reference;
- iteration count, wall time, and the loop's `stall_status` / `converged` flags;
- `theta` itself, so any later analysis can re-score without re-solving.

Side by side with the paper's own numbers: relative cost error `≈ 10⁻⁴` after 500 iterations (their Algorithm 1,
free `C`, 20 instances), and relative plan error `0.026` in their non-uniqueness test.

### 4a. First measurement on that protocol

`n = 100`, 20 instances per cell, 320 rows per family, master = `highs` (exact CPU simplex), ~6 min per family.
Records: `out/related_cost_recovery_p_basis/`, `out/related_cost_recovery_asym_basis/`.

| generating member | `cost_rel_scaled`, `ε = 0.01 / 0.1 / 1 / 10` | member returned |
| --- | --- | --- |
| `p_basis`, truth `p = 3` | 0.000 / 0.000 / 0.006 / 0.017 | the truth |
| `p_basis`, truth `p = 2` | 0.199 / 0.218 / 0.215 / 0.208 | `p = 3` in 18–20 of 20 draws |
| `p_basis`, truth `p = 1` | 0.308 / 0.503 / 0.499 / 0.485 | `p = 3` in 18–19 of 20 |
| `p_basis`, truth `p = 0.5` | 0.090 / 0.602 / 0.665 / 0.655 | `p = 3` in 18–20 of 20 |
| `asym_basis`, truth `p = 1` | 0.323 / 0.605 / 0.584 / 0.581 | `(a = 2, q = 1)` in 12–19 of 20 |
| `asym_basis`, truth `p = 3` | 0.747 / 0.751 / 0.752 / 0.753 | `(a = 2, q = 1)` in 13–19 of 20 |

Five statements, each read off the records:

1. the LP's `θ` is the **global** minimiser of `R`: for `asym_basis`, `p = 1`, `ε = 1`, `n = 60` it matched a
   brute-force grid over the 4-simplex to `1e-16`, and `R` recomputed at the returned `θ` agrees to `1e-16`;
2. in 29 of the 32 cells the same member is returned for every generating member;
3. where the returned member is the generating one, `cost_rel_scaled` is 0.000 at `ε ≤ 0.1` and rises with `ε`;
4. at truth `p = 2` the recovered cost induces a plan identical to the truth's (`plan_rel ≈ 0`) with
   `cost_rel_scaled = 0.20–0.22`; at truth `p = 1` the recovered cost's plan is exactly optimal under the true
   cost (`plan_cost_rel = 0.000`) with `cost_rel_scaled = 0.49–0.50`;
5. `plan_rel` on its own is not a safe reading for these costs: `|i−j|^p` has ties, so two plans can both be
   optimal and differ as matrices (`plan_rel ≈ 1.1` with `plan_cost_rel = 0.000`).

The `free_dense` family (the paper's free cost matrix) is implemented and not yet run: its feature block is
`(m·n, K) = 10⁴ × 10⁴` doubles = 800 MB.

## 5. Facts about the other three benchmarks that shape which cells exist

These are statements about the problem definitions, not recommendations.

- **RIOT's synthetic cell** (`N = 200` sampled pairs out of `100 × 100`) yields an empirical plan with empty rows
  and columns. `InverseOTProblem` requires `a_i > 0` and `b_j > 0`; a plan with empty rows or columns is outside
  the object this code represents, so that cell needs zero-marginal handling before it can be run.
- **Ma et al.'s `ε = 10` cell** produces a plan close to the product of the marginals. The regret `R(θ)` is still
  defined there; the value of `R` at the generating `θ` is the natural reference for how much information the plan
  carries in that cell.
- **Stuart & Wolfram** report no scalar held-out error (acceptance rates and posterior plots), so no numeric cell
  of theirs can be matched directly.
- **M-IOT** needs `K ≥ 2` plans from the same cost. Our `InverseOTProblem` holds one plan; a multi-observation
  version is a different object (`experiments/studies/iot/synthetic/methods/*` and `benchmarks/` hold the
  single-plan studies).

## 6. Where the FAUST number sits

The FAUST arm comparison (`out/faust_arms_n2000/`, n = 2000, 8 held-out pairs) reports geodesic errors. None of
the four IOT papers reports a geodesic correspondence error, and none uses a 3-D shape dataset, so its number does
not correspond to a cell in their tables. The paper in the deck whose reported number *is* in our data and metric
is URSSM (FAUST geodesic error 1.6); our reproduction of it is 1.59, recorded in
`reports/current/NSR_ARM_REPORT.md` §5a.

---

**Provenance.** Paper PDFs fetched 2026-11-19 from the URLs printed on slides 21–24. Reproduction code:
`experiments/studies/iot/synthetic/inverse_cost_recovery/`. Numbers from this repository are cited with their
`out/<study>/records.json`.

# Kernel / dictionary axis for the cost family

**Status: REOPENED (2026-11-17, second revision).** The first verdict in this file — "the axis is closed" — was
an artefact of **three defects in the dictionaries I wrote**, not a property of richer dictionaries. With them
fixed, the oracle gate passes: `hinge8` **0.1041** vs `diag13` **0.1256** over 8 pairs × 3 seeds at n = 800
(paired mean −0.0215, median −0.0094, 14 wins / 8 losses / 2 ties, 6 of 8 pairs better). Sections §3a–§3c record
the wrong verdict and why it was wrong; §3d–§3e record the defects and the shared-θ result, which is the sharper
finding of the two.

**Read §3d before citing anything above it.** Two of the defects were in the *measurement* (a flatness tolerance
that counted a strictly-better diagonal as "tied"), and one was in the *parameterisation* (a null direction that
makes the cost constant).

**Why this file exists.** The cost family is the one place where the inverse-OT formulation can be made richer
without touching the solver, the scan, or the loop. This note fixes the question, the options, and the stop
rule *before* any code is written, so that a negative answer is recorded rather than argued away.

---

## 1. The observation

Everything in `src/iot` touches the data only through the feature tensor `Φ`, and the cost is linear in the
parameter: `c_ij(θ) = <φ_ij, θ>`. So **any cost that can be written as a finite linear combination of pairwise
functions is admissible with no change to the solver**.

That is exactly the shape of the kernel trick, restricted to the finite case:

- for a PSD kernel `k`, the squared feature-space distance is `k(x,x) + k(y,y) - 2k(x,y)`;
- the first two terms depend on **one** point each, so the OT potentials absorb them (adding `u_i + v_j` to a
  cost leaves the optimal plan unchanged under fixed marginals);
- therefore the pair-dependent part of a kernel cost is `-k(x_i, y_j)`.

So a kernel family enters as `φ_ij,k = -k_k(x_i, y_j)` and the solver is untouched.

**What is *not* claimed.** This is not the RKHS/representer version (§4, option D). It is a *finite dictionary*
of pairwise functions; "kernel" here means "a rich dictionary of PSD pairwise similarities", not an
infinite-dimensional feature map.

## 2. Options, cheapest first

| id | what | new `K` | cost per tile |
| --- | --- | --- | --- |
| **A** | Gaussian kernels at several bandwidths; cosine kernels on descriptors or layers; per-descriptor-group kernels | `K` = number of kernels | `O(n²(d + K))`, one distance matmul + `K` exponentials |
| **B** | additive per-coordinate profiles: `g_k(|a_ik - b_jk|) = Σ_l θ_kl h_l(·)` with a hat/spline basis | `K` = dims × basis size | `O(n² · dims · basis)` |
| **C** | explicit approximation of a big kernel (random Fourier features, Nyström) | `K ≈ 10³–10⁴` | `O(n² K)`; capacity then grows with `K`, and the capacity warnings in `SANITY_CHECKS_REPORT.md` apply |
| **D** | exact RKHS / structured-SVM version with a loss-augmented margin | — | needs a new master, not a new `FeatureMap` |

`A` and `B` are the same solver with a bigger dictionary: **zero risk to the existing path**, which is why they
come first. `C` is a capacity experiment. `D` is a different formulation and is explicitly out of scope here.

## 3. The gate: measure the oracle ceiling first

**Baseline to beat.** `θ` fitted on the *evaluated pair itself* (oracle, not transferable), 13 features
(`xyz` + log area + log curvature + 8 HKS), `n = 800`:

```
learned_selected   miss@5% = 0.1855      out/faust_correspondence_box/  (n=24)   ← transferred θ
oracle_theta       miss@5% = 0.1945      out/faust_correspondence_box/  (n=24)
uniform_theta      miss@5% = 0.4202      out/faust_correspondence_box/  (n=36)
nn_oracle_theta    miss@5% = 0.4951      out/faust_correspondence_box/  (n=24)
```

**Caveat that changes how the gate must be read.** The transferred `θ` (`learned_selected`, 0.1855) already
**beats** the oracle (0.1945) on this metric. So the oracle is *not* an upper bound on achievable quality — it is
the optimum of a **different objective** (the max-margin regret, which maximises `δ`), and `miss@5 %` is not what
it optimises. Two consequences:

- the gate must be stated against **0.1855**, the best number we have actually achieved, not against 0.1945;
- an "oracle ceiling" that a transferred fit already exceeds is a warning that the metric and the fitting
  objective are not aligned, and that a dictionary which merely improves the oracle may be optimising the wrong
  thing.

**The gate.** For each richer dictionary, fit `θ` on the test pair itself and report `miss@5 %` at `n = 800` on
the same pairs and the same vertex subsample.

- If the richer dictionary's oracle `miss@5 %` is **not clearly below 0.1855**, the dictionary is not the binding
  constraint and no kernel machinery will help the transferable case. **Stop.**
- If it **is** clearly lower, there is headroom, and only then ask whether the extra capacity survives *transfer*.

Because of the caveat above, also report the **oracle-versus-metric** gap: if a richer dictionary closes much of
the gap between the oracle objective and the metric, that is a statement about the fitting objective, not about
representation, and must be labelled as such.

This ordering matters: a dictionary that helps the oracle but not the transfer is a capacity increase with no
generalisation benefit, and we should know which of the two we have before spending more time.

**Reporting rule.** Oracle numbers are never reported as a method result. They are labelled `oracle` everywhere,
because they are fitted on the evaluated pair.

## 3a. First measurement (smoke: pair 050-066, seed 0, n = 800)

Run before any tuning, and it changed the picture twice.

| dictionary | K | miss@5 % | `L1(theta, uniform)` | note |
| --- | ---: | ---: | ---: | --- |
| `diag13` (baseline) | 13 | **0.1163** | 1.326 | reproduces the incumbent |
| `gauss` (5 bandwidths, one distance) | 5 | 0.5038 | **0.000** | θ unidentifiable |
| `gauss+cos` | 6 | 0.4875 | 1.667 | θ moves, but worse |
| `profile8` (hats) | 104 | 0.9925 | **0.000** | degenerate (see below) |
| `group_gauss` (per descriptor group) | 20 | 0.6850 | **0.000** | θ unidentifiable |
| `group+diag` | 33 | 0.4688 | **0.000** | θ unidentifiable |

### Finding 1 — several bandwidths of ONE distance cannot help (structural)

Gaussians at different `sigma` on the *same* Euclidean distance are **monotone transforms of one another**, so
they induce the **same ranking** of candidates. `theta` therefore cannot change the correspondence and the fit
leaves it at the simplex centroid: `L1(theta, uniform) = 0.000` exactly, in three of the six dictionaries.

This is a theoretical limitation of option A *as literally described* ("Gaussian kernels at several
bandwidths"), not a tuning problem. It predicts that a bandwidth mixture is *rank-equivalent* to a single
Gaussian, and the measurement matches: `miss@5 %` 0.50 against the baseline's 0.12.

### Finding 2 — the hat basis was degenerate at the uniform point (my bug, fixed)

A hat basis is a **partition of unity**: `sum_l h_l(t) = 1` for every pair. With uniform `theta` the cost is
therefore the constant `dim/K` for *every* candidate, `cost_std = 1.5e-17`, and the fit cannot start. Fixed by
adding a **linear ramp** per coordinate (`RadialProfileFeatures(include_ramp=True)`), after which
`cost_std = 3.5e-2` at the uniform point. Any radial-basis cost family needs such a term; this is now a
regression test.

### Finding 3 — kernels carry far less dynamic range than squared Euclidean

`cost_std` at uniform `theta` on the same pair: `diag13` 2.12, `gauss` 0.13, `group_gauss` 0.045. A Gaussian
similarity is bounded in `[-1, 0]`, whereas squared Euclidean is unbounded above; OT is driven by the *relative*
costs, so a bounded similarity gives a much flatter landscape. This is a plausible explanation for the size of
the gap and is independent of `theta` identifiability.

### The deciding run (8 pairs × 3 seeds = 24 rows per dictionary, n = 800, simplex)

| dictionary | K | miss@5 % | Δ vs baseline |
| --- | ---: | ---: | ---: |
| **`diag13` (incumbent)** | 13 | **0.1256** | — |
| `group+diag` (cross-distance + diagonal) | 33 | 0.2428 | +93 % |
| `profile8` (hats + ramp) | 117 | 0.2667 | +112 % |
| `gauss` (multi-bandwidth) | 5 | 0.4079 | +225 % |
| `gauss+cos` | 6 | 0.4111 | +227 % |
| `group_gauss` (cross-distance only) | 20 | 0.5780 | +360 % |

**Gate: FAIL.** The best richer dictionary is **2× worse** than the incumbent, and the cross-distance mixture —
the one family that is not rank-equivalent — is still 93 % worse. The 13-feature diagonal family is **not** the
binding constraint on FAUST at this scale.

**One correction to the gate statement.** `diag13` here scores 0.1256 under the **simplex** (`rel=None`), while the
0.1945 oracle in `faust_correspondence_box` was fitted with a **box** (`rel=0.25`). The box constrains more, so the
simplex oracle is stronger; 0.1256 > 0.1855 is therefore a *like-for-like* comparison only within this run (all
six dictionaries used `rel=None`). That internal comparison is the one the verdict rests on, and it is unaffected.

**Why the richer families lose, in one line each:**

- multi-bandwidth Gaussians on one distance are **rank-equivalent** to a single Gaussian (finding 1), so they add
  parameters without adding information, and they replace an unbounded cost with a bounded one (finding 3);
- the hat profile is a **smooth monotone** reparametrisation of per-coordinate absolute differences, so it is
  again close to rank-equivalent to a weighted L1, but with a much flatter scale;
- the cross-distance mixture *is* a genuine increase in expressivity, and it still loses — which is the strongest
  form of the negative result: **the ordering information the diagonal family already extracts is not the
  bottleneck.**

**What would be a different question.** Whether a richer dictionary helps *transfer* is now moot: it cannot help
the oracle, so it cannot help transfer. The bottleneck on FAUST is the descriptors and the decision rule, not the
cost's functional form.

## 3d. The three defects that produced the first verdict

Fixed in this revision; each has a regression test.

| # | defect | symptom | fix |
| --- | --- | --- | --- |
| 1 | a multi-σ kernel spec counted as **one** feature while only the first σ was evaluated | `n_features` disagreed with the cost | `_expand_kernel_specs` |
| 2 | the hat basis is a **partition of unity**, so uniform weights on one coordinate's hats give a *constant* cost; a linear weight profile plus the ramp is also constant | `cost_std = 1.5e-17` at uniform θ, the fit could not move | new `HingeProfileFeatures` (hinges vanish at `t = 0`, so no non-negative combination is constant) |
| 3 | the flatness diagnostic tested `gap < 1e-6`, so every row where the correct partner was *cheaper* (negative gap) counted as "tied" | `tied = 1.000` for any well-separated cost — i.e. the diagnostic fired on everything | test `abs(gap) < 1e-6`; `tests/iot_multi_pair_test.py` |

Defect 3 was caught by a test I wrote for defect 2's diagnostic, which is the only reason it is known: every
"tied" number in §3a–§3c is wrong and has been recomputed.

**The corrected oracle number.** Pair 050-066, seed 0, n = 800, simplex, identical subsample:

| dictionary | K | `miss@5 %` | regret | median identity gap |
| --- | ---: | ---: | ---: | ---: |
| `diag13` (incumbent) | 13 | 0.1163 | 1.9e-2 | +2.5e-4 |
| `profile8` (hats + ramp) | 117 | 0.0975 | 1.5e-3 | +2.5e-4 |
| **`hinge8`** (hinges, natural scale) | 104 | **0.0325** | 6.8e-5 | +6.2e-6 |
| `hinge8`, NN-residual scale | 104 | 0.0650 | 4.8e-4 | +6.8e-5 |

The hinge normalisation matters as much as the basis: with the nearest-neighbour residual spread as `extent`,
almost every candidate distance clips to 1 and the cost saturates (`tied = 1.000` at the joint level). The
99th percentile of `|a-b|` over **random** pairs spans the actual range of candidate distances.

**Full gate** (8 pairs × 3 seeds = 24 rows, n = 800, simplex, identical subsamples):

| dictionary | K | mean `miss@5 %` | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`hinge8`** | 104 | **0.1041** | **0.0969** | 0.0000 | 0.2162 |
| `diag13` | 13 | 0.1256 | 0.1206 | 0.0000 | 0.2425 |
| `profile8` | 117 | 0.2667 | 0.1656 | 0.0000 | 0.6937 |

Paired `hinge8 − diag13`: mean **−0.0215**, median −0.0094, **14 wins / 8 losses / 2 ties**. Per pair (3 seeds
averaged; seeds are repeated measures, so the effective sample is **8 pairs**, not 24 rows):

| pair | `hinge8` | `diag13` | diff |
| --- | ---: | ---: | ---: |
| 050-066 | 0.0413 | 0.1346 | **−0.0933** |
| 053-085 | 0.1125 | 0.1675 | **−0.0550** |
| 056-071 | 0.1933 | 0.1837 | +0.0096 |
| 059-099 | 0.0025 | 0.0017 | +0.0008 |
| 061-086 | 0.1029 | 0.1187 | −0.0158 |
| 064-072 | 0.0879 | 0.0862 | +0.0017 |
| 067-091 | 0.2025 | 0.2146 | −0.0121 |
| 072-085 | 0.0900 | 0.0979 | −0.0079 |

**How much this licenses.** 6 of 8 pairs improve, but two of them carry almost the whole effect, and with 8
pairs a 6/8 sign test is not significant (p ≈ 0.29). The honest statement is: *the corrected radial profile is
a modest, directionally consistent improvement in the oracle, not an established one.* It also has to survive
**transfer** before it means anything for a method — which is the next gate and has not been run.

## 3e. Shared θ across pairs: "satisfies all plans" is achievable and vacuous

Asked directly: *can one θ satisfy many pairs at once?* Yes — **and that is the problem.** Four training pairs,
n = 800, one shared θ, per-pair potentials, block LP, then the honest check on the **full edge set** with exact
POT (`R_p(θ) = <θ, φ̄_p> − OT_p(c(θ))`, never the restricted-master objective).

| family | K | joint full-edge regret (4 pairs) | median identity gap | tied | decoded `miss@5 %` | `single` | `uniform` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `diag13` | 13 | 0.0117 – 0.0265 | +2.0e-3 … +4.3e-3 | 0.000 | 0.3841 | 0.1694 | 0.4306 |
| `profile8` | 117 | **0.0000** on all four | **+1.6e-14** | **1.000** | **0.9600** | 0.3556 | 0.4131 |
| `hinge8` | 104 | **0.0000** on all four | **0.00e+00** | **1.000** | **0.9697** | 0.3203 | 0.3997 |

Both rich families **exactly** certify all four plans, and both do it by destroying the cost's information:

- `profile8`'s shared θ has 8 non-zeros, all on descriptor coordinate 6: hats 0–6 with weights `0.2 − l/35` plus
  the ramp at `0.2`. Because hats are a nodal basis, `Σ_l (0.2 − l/35) h_l(t) = 0.2 − 0.2t`, and the ramp adds
  `+0.2t`. The cost matrix is therefore **constant**: `std = 1.4e-14`, `max − min = 5e-14` over all 800×800
  entries.
- `hinge8`'s shared θ is a **single hinge** (coordinate 8, level 6, weight 1.0). It is zero on **93.6 %** of all
  pairs, so the cost is 0 almost everywhere.

A constant or (almost) zero cost gives regret 0 for **every** plan. The simplex bounds `Σθ = 1` but places no
constraint on the *dynamic range* of the resulting cost, so any family containing a feature that is ≈0 or
constant over the pair population has an escape direction, and a regret-minimising LP finds it first. `diag13`
has no such feature — its 13 columns all have large range (`cost std ≈ 3.2`) — which is exactly why it *cannot*
reach joint regret 0 (0.012–0.027), and exactly why its decoded map is merely mediocre (0.384) rather than
collapsed.

**The ordering is monotone in the achieved regret**: joint regret 0.02 → 0.0000, decoded `miss@5 %` 0.384 →
0.96. Satisfaction and usefulness are anti-correlated here, so "can expressivity find a plan satisfying all
pairs" cannot be the gate — the trivially degenerate cost satisfies everything. The gate has to be a **margin**.

**What the margin version needs, and why the naive version is broken.** Replacing the constraint with
`f_i + g_j + δ·ℓ_ij ≤ c_ij(θ)` and maximising δ is *unbounded* if every constraint stays an inequality: `f` and
`g` are free, so any δ can be absorbed by shifting them down. The observed support must be imposed as
**equalities** (`f_i + g_j = c_ij(θ)` on the plan's support, as in §2 of `AGENTS.md`), which is what pins `δ`.
This is the next experiment: a joint max-margin master with `ℓ_ij` the normalised geodesic loss (the task loss,
available on training pairs), one shared δ, and the same full-edge verification.

## 4. What option D would add, and why it is deferred

With `c_w(z) = <w, Φ(z)>`, a loss-augmented margin `ℓ_ij ≥ 0` (zero on the support, positive elsewhere) and
`μ > 0`, the primal is a structured SVM whose inference is an OT solve, and the dual is over transport plans:

```
max_{λ ∈ Π(a,b)}  <λ, ℓ> − (1/2μ) ‖E_λ Φ − Φ̄‖²_H,        w* = (1/μ)(E_{λ*} Φ − Φ̄)
```

Two things it would buy that the current formulation does not have:

1. **a unique `w`** (strict convexity), removing the simplex-corner behaviour we measured repeatedly
   (`θ` is not identifiable; the optimal set is a cone);
2. **a graded loss**, so a one-ring miss costs less than a far miss — which is exactly what the FAUST geodesic
   metric measures and what the exact formulation treats as all-or-nothing.

Two warnings already established by the proposer's own checks: without the margin `ℓ` the problem is degenerate
(`w = 0`, the observed plan is already feasible, because the regret is homogeneous), and **no convergence rate
is known and it has not been run on real data**.

**LP-compatible cousin, no kernels:** replace the constraint with `f_i + g_j + δ·ℓ_ij ≤ c_ij(θ)` and the scan
becomes `h_e = σ_e + δ·ℓ_e`. Everything else is unchanged. For FAUST this is likely more useful than the full
RKHS machinery, and it is a four-line change to the existing scan rather than a new formulation — so it is the
natural *second* step, after §3.

## 5. Order of work

1. `KernelMixtureFeatures` (A) and `RadialProfileFeatures` (B) in `src/iot/core/features.py`, each a `FeatureMap`
   with `features()` and `costs()`. No change to the solver, scan, loop or master.
2. Oracle-ceiling driver `studies/real/faust/oracle_ceiling.py`: same pairs, same `n = 800`, same subsample as
   `faust_correspondence_box`, so the comparison against 0.1945 is like-for-like.
3. **Gate** per §3. Record the verdict here, negative or positive.
4. Only if the gate passes: transfer evaluation (fit on pair A, evaluate on pair B), then the
   loss-augmented margin of §4.

## 6. Known risks

- **Capacity, not representation.** If the dictionary is rich enough to fit the training pair trivially, the
  oracle number improves for the wrong reason: it memorises. Check `θ` sparsity and the gap between oracle and
  transfer, not just the oracle alone.
- **The potentials absorb per-point terms.** Any kernel term that is a function of only `x_i` or only `y_j` is
  invisible under fixed marginals — do not add "confidence" or "region" features to the cost and expect an
  effect.
- **`K` is the simplex dimension.** `θ ∈ Δ` with `K = dims × basis` means the uniform point is `1/K`; a large
  basis dilutes every coordinate, and the learned `θ` may be numerically indistinguishable from uniform. Report
  the `L1` distance to uniform alongside every fit.

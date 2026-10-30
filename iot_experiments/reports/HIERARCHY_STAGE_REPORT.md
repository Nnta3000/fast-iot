# Stage 0 / Stage 1 report: does coarse-to-fine warming pay for inverse OT?

> **This report covers the RELAXED hierarchical variant only.**
>
> Every Stage 1 run used `solve_inverse_ot_regret` at **both** the coarse and the fine level: the
> relaxed regret master with an inequality on all of `F`. **The exact hierarchical algorithm was
> never run here.** That matters, because the Typst note (§5) is explicit that for the regret master
> there is **no analogue of Lemma 2 and no finite-termination theorem**, so the coarse levels are
> *warm starts only, with no bounds*.
>
> Consequently this report tests **none** of the hierarchy's proof:
>
> | claim | exercised here? |
> | --- | --- |
> | Lemma 2 (`δ*_{ℓ+1} ≥ δ*_ℓ`, restriction is a relaxation) | **no** — needs the exact master |
> | Corollary 3 (monotone bounds, infeasibility propagation) | **no** |
> | Theorem 6(b) (output identical to flat) | **no** — grow-only leaves θ unconstrained |
> | Lemma 5 (seed admissible, `B ⊆ F₀`) | yes |
>
> The exact variant is now implemented in `src/iot/hierarchical/` and tested on **synthetic,
> planted** instances in `tests/iot_hierarchical_test.py`, which is where Lemma 2, Corollary 3 and
> Theorem 6(b) are actually checked. The verdict below should be read as *"relaxed two-level
> warming is 1.6–1.9× slower than flat on SynthRAD"*, not as a verdict on the hierarchy as proved.

**Status: NEGATIVE (relaxed variant).** Two-level warming was built, run, and **fails the pre-registered criterion**
(`hier ≤ 0.70 × best cheap baseline`). The two cheap alternatives also gave far less than hoped
(γ tuning 21 %, warm start 4 %). Details and exact configurations below.

Everything here was measured on this machine (RTX 4060 Laptop, 15 GB RAM), `cuDPLPx` master,
`prune=False`, SynthRAD brain features, δ* ≈ 0 regime.

---

## 1. What was implemented

Six pieces of code changed or were added, in the order they were built.

### 1.1 `src/iot/core/scan.py` — shared separation scan (new)

The three-backend dispatch (`numpy` / `hello` / `torch`) was lifted out of `exact/loop.py`
into `scan_for_violations(...)`, plus `maximum_slack(...)` for blockwise `max σ`.

Reason: the dispatch existed **only** in the exact loop; the relaxed loop called the NumPy
implementation directly, so **a GPU scan was unreachable on the relaxed path** and nobody noticed.
This was a real defect caused by the two loops having diverged, not a style issue.

### 1.2 `src/iot/core/edges_advance.py` — shared edge-set advance (new)

`advance_edges(mandatory, pruned, candidates)` = `F ← B ∪ T ∪ V`, plus the two opposite-sign slack
helpers (`slack_toward_cost` = `c − f − g − δ`, `slack_toward_duals` = `−` that) and
`keep_by_tightness`. Same algebra in both loops; the prune *policy* stays local.

### 1.3 `src/iot/core/pruning.py` — shared hysteresis pruning (new)

`HysteresisPruner.step(candidates, slack) → (kept, next_pruner)`: drop an edge only when it was
loose **two consecutive rounds**, matched **by identity** (edge keys). Immutable, so both loops can
use it without sharing state.

Memory checked before choosing identity matching: the state is one `int64` array of the keys loose
*that round*, bounded by `|F|` (not `m·n`) and **replaced** every round. Measured: `|F|` grew
80 → 325 over 7 rounds at m=40, so the loose set is a subset of a few hundred keys — kilobytes.

### 1.4 Exact loop: `prune` default flipped to `False`

Pruning is a heuristic; only dropping strictly-positive-slack edges is proved safe end to end.
Keeping extra edges is always safe, it only costs size. Exactness by default, `prune=True` explicit.

### 1.5 Warm start for the master LP

`setInitSol` is exposed by the CuPDLPx binding but was not wired through. Added
`init_solution` on `solve_lp_cupdlpx`, `set_warm_start()` on `RegretMasterSolver`, and
`warm_start=True` on `solve_inverse_ot_regret`, which feeds the previous round's `(θ, f, g)` back.

Sound because `LpLayout` depends only on `(has_delta, n_features, n_source, n_target)` — **never on
`|F|`** — so the previous `x` is dimensionally valid for the next round; only the constraint rows
change. The binding requires `setInitSol(x0, y0)`; the dual is passed as `None`.

### 1.6 `src/iot/hierarchical/` — the library home (added after the report was first written)

The hierarchy originally lived **only** inside a study driver, which was the one-off-driver pattern
the project forbids. It now lives in the library and is reachable from both solvers by a keyword:

```python
from iot.hierarchical import HierarchyParameters
from iot.exact.loop import solve_inverse_ot

solve_inverse_ot(problem, hierarchical=HierarchyParameters(rho=0.25, kappa=3))
```

| file | contents |
| --- | --- |
| `hierarchical/parameters.py` | `HierarchyParameters(rho, max_levels, grid_budget, kappa, level_seed, carry)` |
| `hierarchical/levels.py` | `Level`, `full_level`, `coarsen`, `build_levels` (edge sampling; rejects a draw whose induced support fills the grid) |
| `hierarchical/transfer.py` | `prolong` (dual completion, `f` before `g`), `seed` (`B ∪ A_row ∪ A_col`) |
| `hierarchical/__init__.py` | `solve_hierarchical` — the driver, with the **level solver injected** so the exact and relaxed variants cannot diverge |

`solve_inverse_ot(..., hierarchical=...)` and `solve_inverse_ot_regret(..., hierarchical=...)` both
dispatch through the same code; the fine level calls the same solver with `_flat_only=True` to prevent
recursion. With `hierarchical=None` (the default) both are **bit-identical to before**.

### 1.7 Per-round timing and the Stage 0 / Stage 1 drivers

- `RegretIteration` gained `master_time` / `scan_time` (measurement only).
- `studies/real/synthrad/stage0_pivot.py` — master-vs-scan share, γ sweep, warm-start switch.
- `studies/real/synthrad/stage1_hierarchy.py` — two-level hierarchy:
  `BuildLevels` (edge sampling at ratio ρ) → coarse **dense** solve on `I₁ × J₁` with the induced
  plan → `Prolong` (retained nodes keep their potentials, new nodes min-completed, `f` before `g`)
  → `Seed` (`B ∪ A_row ∪ A_col`, κ per row/column by prolonged slack) → fine loop from `F₀`.

### What was deliberately **not** implemented

**`A_carry`** (the coarse level's final edge set). `RegretResult` does not expose it, so building it
needs a library change. Recorded as not implemented in the run's provenance rather than faked.
Omitting carry can only make the seed **worse**; it cannot affect correctness (Lemma 5 needs only
`B ⊆ F₀`). So the negative result below is not an artefact of the omission.

### Regression safety

- Both merged pieces keep the exact loop **byte-identical**: verified on a 6-configuration probe
  (m ∈ {12, 30} × prune ∈ {True, False} × hysteresis ∈ {True, False}), same δ* to 12 decimals.
- `warm_start` defaults to `False`, `prune` semantics unchanged for explicit callers.
- `tests/iot_shared_scan_test.py`: 9 tests (scan backends agree, slack sign conventions,
  `advance_edges` algebra, hysteresis two-round rule, identity matching across a changing
  population, exact `prune` default). iot suite: **223 passed, 3 skipped, 2 pre-existing failures**
  (optional `cvxpy` absent; a 5.6e-17 round-off).

---

## 2. Results, part by part

### 2.1 Stage 0 — is the master LP or the separation scan the bottleneck?

m = 2000, k = 64, γ = 1, `prune=False`, cuDPLPx.

| quantity | value |
|---|---|
| rounds | 12 |
| \|F\| final | 26,163 (0.65 % of grid) |
| master total | 35.9 s |
| scan total | 25.8 s |
| **`scan_share`** | **0.42** |
| total | 66 s |
| stall | `solver_floor_no_new_edges` |

Per round the split is flat: master 2.7–3.3 s, scan 2.12–2.21 s. The scan is nearly constant
regardless of `|F|`, as expected — it scans all `m·n` edges every round.

**Reading:** the master is the larger term (≥58 %, since the "scan" bucket also contains repair and
the tie-break re-solve), but not dominant. Rounds were therefore a plausible lever — which is why
Stage 1 went ahead.

**m = 5000 was OOM-killed** (exit 137) inside the loop before round 1. Peak at m=2000 was 6.32 GB;
the estimate I used for m=5000 came from m=2000 and was never measured at that size. Not retried.

### 2.2 Tuned γ — the cheapest lever, and the best one

m = 2000, cold start.

| γ | rounds | \|F\| | master | scan | **total** | vs γ=1 |
|---|---|---|---|---|---|---|
| 1 | 12 | 26,163 (0.65 %) | 33.2 s | 26.0 s | **63 s** | — |
| **2** | **9** | 32,688 (0.82 %) | 26.3 s | 19.5 s | **50 s** | **0.79×** |
| 4 | 9 | 43,402 (1.09 %) | 30.6 s | 19.4 s | 54 s | 0.86× |
| 8 | 8 | 61,021 (1.53 %) | 33.3 s | 17.4 s | 55 s | 0.87× |

**γ = 2 gives 21 %. Beyond that it reverses:** rounds drop 12 → 9 then stall, while per-round cost
rises because `|F|` grows faster than the rounds saved.

### 2.3 Warm-started master — 3–4 %

| config | rounds | master | scan | total |
|---|---|---|---|---|
| γ=1, cold | 12 | 33.2 s | 26.0 s | 63 s |
| γ=1, warm | 12 | 31.1 s | 25.5 s | **61 s** (−3 %) |
| γ=2, cold | 9 | 26.3 s | 19.5 s | 50 s |
| **γ=2, warm** | 9 | 25.3 s | 19.0 s | **48–49 s** (−4 %) |

The master costs ~2.9 s per round and is flat across rounds, so CuPDLPx is spending its time on final
convergence rather than on finding the neighbourhood — a closer start saves little. Warm start also
changes `|F|` slightly (25,887 vs 26,163): the start changes which vertex the first-order method
lands near, so the *path* differs even though the optimum does not.

**Best cheap baseline: γ=2 + warm = 49.1 s. The 0.70× bar is therefore 34.4 s.**

### 2.4 The hierarchy — 1.6–1.9× **slower**, criterion failed

m = 2000, γ = 2, warm, ρ = 0.25, κ = 3, coarse grid 500×500 (6.2 % of the fine grid).

| level seed | coarse | prolong+seed | fine | **total** | **vs flat** | rounds hier / flat |
|---|---|---|---|---|---|---|
| 0 | 4.9 s | 3.2 s | 83.1 s | **91.3 s** | **1.86×** | 14 / 9 |
| 1 | 4.4 s | 3.1 s | 70.8 s | **78.3 s** | **1.60×** | 11 / 9 |
| 2 | 4.5 s | 3.2 s | 76.6 s | **84.3 s** | **1.72×** | 12 / 9 |

`F₀/F_final = 0.31–0.35` (as instructed), `|B| = 4,000`, `|F₀| = 15,985`.

**Verdict against the pre-registered rule: FAIL.** 1.6–1.9× slower, not 0.70× faster.

### 2.5 Correctness gate

| check | flat | hierarchy |
|---|---|---|
| master feasibility (`max h ≤ 0`) | ✅ at m=200, m=2000 | ✅ all seeds |
| `is_optimal_plan` | ❌ **also flat** | ❌ |

The gate fails for **flat as well**, so it is not a hierarchy defect — it is the δ* ≈ 0 regime: the
relaxed loop certifies a *bracket*, not optimality, and POT's exact optimum disagrees. The gate as
specified is valid and remains in force; it simply reports that neither arm produces an exactly
optimal plan on this instance.

---

## 3. What this establishes

**1. The seed injected edges the fine level did not need.** Flat starts at `|B| = 4,000` and
converges in 9 rounds; the seeded run starts 4× larger (`|F₀| = 15,985`) and needs **11–14 rounds**.
The coarse solution ranks the bulk of the edges well and the handful that actually bind poorly — the
coordinator's first doubt, confirmed with numbers.

**2. `F₀/F_final` was the wrong acceptance measure.** It sat at the requested 0.31–0.35 and still
lost. What drives round count is **`|F₀|` relative to `|B|`**, not its ratio to the final size. That
criterion was mine and it was wrong.

**3. κ ≈ 3 is too large by construction.** `2κ(m+n) = 12,000` edges against a mandatory set of
4,000. Even κ = 1 would add 4,000 — the same size as `B` itself.

**4. The coordinator's 3–6-round estimate is too optimistic in this regime.** The realistic count
is 11–14, because the coarse problem's θ is not the fine θ that pins the binding edges.

---

## 4. What is left open

- **m = 5000 unmeasured.** The loop OOMs there; the blockwise scan covers the cost matrix but not
  the per-round index arrays or master bookkeeping.
- **`A_carry` unimplemented**, as stated. Implementing it could improve the seed but cannot fix a
  4× oversized `F₀`.
- **κ = 1 / ρ = 0.5 was not run.** It is the one variant that could change the answer by making
  `F₀` smaller, but κ = 1 still adds `2(m+n)`.
- **Planted δ* > 0 instances: now tested** (`tests/iot_hierarchical_test.py`): monotone bounds hold
  across levels and hierarchical δ equals flat δ on synthetic instances with the exact loop. That is
  the correctness evidence the proved algorithm needed, and it is **separate** from the SynthRAD
  timing verdict above.
- **Still not tested:** the exact hierarchy's *speed* on a synthetic instance at scale (only its
  correctness), and the decision regime at larger m.

## 5. Recommendation

**Drop the hierarchy.** By the pre-registered rule it is 1.6–1.9× slower. The useful findings are
the two cheap levers — **γ = 2 (−21 %)** and a small warm-start gain — and the diagnosis that a
coarse seed costs more in `|F₀|` than it saves in rounds. If anything is pursued it should be a
**pruning-aware** seed (so `F₀` can be smaller than `B ∪ 2κ(m+n)`), not more levels.

## 6. Reproduce

```bash
# Stage 0 pivot + gamma sweep
python -m iot_experiments.studies.real.synthrad.stage0_pivot --ms 2000 --gammas 1 2 4 8

# warm start comparison
python -m iot_experiments.studies.real.synthrad.stage0_pivot --ms 2000 --gammas 1 2 --warm

# two-level hierarchy, 3 level seeds
python -m iot_experiments.studies.real.synthrad.stage1_hierarchy --m 2000 --level-seeds 0 1 2
```

Records: `iot_experiments/out/synthrad_stage0/stage0.json`,
`iot_experiments/out/synthrad_stage1/stage1.json`, each with `run.log` alongside.

**Provenance of every table** (the JSON collides between runs, so each number is traced to the
artefact that still holds it):

| table | source that survives |
| --- | --- |
| §2.1 pivot (m=2000, γ=1 cold) | `out/_jobs/stage0_pivot_20261030_123510.log`; JSON overwritten by the warm rerun |
| §2.2 γ sweep (cold) | `out/_jobs/gamma_sweep_20261030_130900.log`; JSON overwritten |
| §2.3 warm vs cold | `out/synthrad_stage0/stage0.json` (current contents) + `out/_jobs/warm_vs_cold2_*.log` |
| §2.4–2.5 hierarchy | `out/synthrad_stage1/stage1.json` (current contents) + `out/_jobs/stage1_m2000_20261030_133742.log` |

**Known defect:** Stage 0 and Stage 1 write to one directory per study, so re-running at a different
`m` or `γ` set **overwrites** the previous JSON. Only the job logs keep the history. This is the same
collision that previously consumed the d=64 quadratic record. Fixing it means keying the run
directory by the config (as was done for `d<k>`), and has not been done.

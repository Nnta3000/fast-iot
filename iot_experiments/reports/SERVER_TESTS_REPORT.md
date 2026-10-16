# GPU server tests — consolidated results

**Status: CURRENT, empirical.** Every A100 run to date, in one place. Supersedes the ad-hoc tables
posted during development; **cite this report** for server results.

Backing data (all git-ignored, on the server and synced to `/srv/results/fast-iot/`):
- `iot_experiments/gpu_bench/out/queue/<timestamp>/{summary.tsv,queue.log,NN_*.log}` — the run records;
- `iot_experiments/gpu_bench/out/{no_prune,validated,e2e,e2e_gpu}/<run>/e2e.{json,csv}` — per-run records;
- `iot_experiments/core/out/planting/<tag>/planting_sweep.json` — planting sweep records.

## 0. Conventions

**Target path (binding, AGENTS.md framing).** All runs here are **GPU-only**: separation scan
`scan_backend="hello"` (HELLO-OT torch kernel), master `cupdlpx` (first-order GPU LP),
ground-truth plan `plan_backend="hello"` (HELLO-OT) or the **planting** generator. **No NumPy scan
and no HiGHS master produced any row in this report.**

**Symbols.**

| symbol | meaning |
| --- | --- |
| `n_eff`, `m_eff` | source / target counts **after pruning empty columns** (a planted/HELLO plan can leave targets unused; see §6) |
| `\|S\|` | support size of the observed plan `π̂` (= `n_eff` for Monge/planted; `≈2n_eff−1` fractional) |
| `\|F\|` | edge set the master is solved on (grows across iterations) |
| `δ` | max margin of the master (0 ⇒ `π̂` optimal with **no** slack — a tie, not a failure) |
| `maxviol` | loop's own OT-free certificate: `max_e widehatσ + δ` over edges `∉ F`; `≤ tol` ⇒ converged |
| `cert` | ground-truth dual certificate of `π̂` at `θ*` (`core.costs.dual_feasibility`) |
| `is_optimal` | **authoritative** verdict: exact OT of `c(θ_returned)` vs `⟨c, π̂⟩` (POT, dense) |
| `conv` / `cap` | converged / hit `max_iterations` without converging |

**Two regimes.** `gmm` / `gmm_floored` ⇒ **fractional** plans (`|S| ≈ 2n−1`). `uniform` ⇒
**permutation** plans (`|S| = n`). **Planted** instances ⇒ **Monge** plans (`|S| = n_eff`; one edge
per source, no mass splitting).

**A repeated trap.** `is_optimal=True` while `conv=False` is common: the returned `θ` can be right
even when the loop never closed. **`conv` and `is_optimal` must be read separately.**

---

## 1. `queue 2026-10-13_153639 / _153701 / _153717` — first GPU e2e (superseded)

`run_batch --batch e2e`, diagonal, `plan_backend=hello`, `cupdlpx`, cap 2–10 s, ≤20 iterations.
**Status: SUPERSEDED** — six of nine jobs failed on the pre-fix `gmm` marginal (zero-mass column
crash, see §6); the three that completed used the pre-fix GMM and a 10 s master cap.

| run | n | marginal | d | iters | conv | δ | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E3 | 12800 | gmm | 8 | 5 | ✅ | 2.36e-4 | ok |
| E4 | 6400 | gmm | 32 | 5 | ✅ | 0 | ok |
| E6 | 6400 | uniform | 8 | 11 | ✅ | 0 | ok |
| E1, E2, E5, E7, E8, E9 | 3200–25600 | gmm | 8–128 | — | ❌ | — | **fail(1): zero-mass column** |

**Takeaway:** the target path ran end-to-end on the GPU, but the GMM instance construction was
broken (§6) and the master cap (10 s) was too small to be informative. Do not cite these numbers.

---

## 2. `queue 2026-10-14_110632` — no-prune `|F|`/δ scaling (sufficient, but pre-GMM-fix)

`--scans hello --masters cupdlpx --plan-backend hello`, `prune=false`, `numpy` scan, cap 20 s,
≤30 iterations. **Generator: HELLO.** The GMM rows used the **pre-fix** GMM (stale; see §6).

| run | n | marginal | regime | iters | conv | δ | maxviol | \|F\|/(m+n) | master s/it | scan s/it | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | 3200 | gmm_floored | fractional | 3 | ✅ | **0** | −4.5e-4 | 2.45 | 14.6 | 0.42 | 45 s |
| N2 | 6400 | gmm_floored | fractional | 5 | ✅ | **0** | −4.7e-5 | 2.79 | 18.1 | 1.01 | 95 s |
| N3 | 12800 | gmm_floored | fractional | 6 | ✅ | **0** | −1.7e-5 | 2.84 | 19.9 | 3.32 | 140 s |
| N4 | 25600 | gmm_floored | fractional | 7 | ✅ | **0** | −4.4e-6 | 2.49 | 23.0 | 11.6 | 242 s |
| N5 | 12800 | uniform | permutation | 12 | ✅ | 8.3e-6 | −1.8e-5 | 5.16 | 22.9 | 2.01 | 298 s |

**Takeaways.**
- The **master dominates**: 14–23 s/iter vs 0.4–3 s/iter for the scan. The scan is not the wall.
- **δ = 0 in every fractional (`gmm_floored`) run**, at every n; the permutation run keeps a small
  positive δ. This is the observation that motivated §4 and the planting generator (§5).
- `cert=False` in all rows: HELLO's own duals are only ~1e-4 accurate against a 1e-6 tightness
  tolerance (§3) — a threshold artifact, not a failure.

---

## 3. `queue 2026-10-14_150534` — validated, uncapped (V1–V9)

Same path as §2 but cap **300 s** and `--validate` (authoritative POT check) where `m·n ≤ 5e7`
(n ≤ ~7000). This separated *"δ=0 is real"* from *"δ=0 was the 20 s cap"*.

| run | n | marginal | regime | seeds | iters | conv | δ | maxviol | `is_optimal` | rel_gap | \|F\|/(m+n) | master peak | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 3200 | gmm_floored | fractional | 0 | 3 | ✅ | 0 | −4.5e-4 | **True** | 1.3e-8 | 2.45 | 300.8 s (*cap*) | 623 s |
| V2 | 6400 | gmm_floored | fractional | 0 | 8 | ✅ | 0 | −3.3e-5 | **False** | 9.2e-7 | 4.49 | 302.9 s (*cap*) | 2348 s |
| V5 | 3200 | gmm_floored | fractional | 1,2,3 | 3–5 | ✅✅✅ | 0,0,0 | −3e-5…−4e-4 | **True**×3 | 1e-9…3e-8 | 2.1–2.6 | 32–49 s | 132–248 s |
| V3 | 3200 | uniform | permutation | 0 | 12 | ✅ | 5.5e-5 | −2.2e-4 | **True** | **1.1e-16** | 4.96 | 24.3 s | 296 s |
| V6 | 3200 | uniform | permutation | 1,2,3 | 11–12 | ✅✅✗ | 3.0e-5…9.9e-5 | ≤−7.7e-5 | **True**×3 | 0…8e-17 | 5.1–5.6 | 8–36 s | 89–432 s |
| V4 | 6400 | uniform | permutation | 0 | 12 | ✗ | 0 | +3.3e-4 | **False** | 2.4e-7 | 5.28 | 59.9 s | 366 s |
| V7 | 12800 | gmm_floored | fractional | 0 | 9 | ✅ | 0 | −1.7e-5 | — | — | 3.44 | 304.2 s (*cap*) | 2457 s |
| V8 | 12800 | uniform | permutation | 0 | 12 | ✗ | 8.8e-6 | +3.9e-4 | — | — | 5.24 | 301.3 s (*cap*) | 740 s |
| V9 | 25600 | gmm_floored | fractional | 0 | 9 | ✅ | 0 | −1.8e-5 | — | — | 3.66 | 309.1 s (*cap*) | 2560 s |

**Takeaways.**
- **n=3200 is validated optimal on both regimes** (V1, V3, V5×3, V6×3) — the strongest existing
  evidence that the GPU loop's `θ` genuinely makes `π̂` optimal.
- **n=6400 is NOT validated** (V2, V4 `is_optimal=False`, rel_gap ~1e-7) — consistent with the master
  being truncated (cap-bound in V2).
- The **master is cap-bound** in every `gmm`/fractional run and even in V8's first iteration (301 s).
  This is the central bottleneck.
- **δ=0 persists with the cap raised to 300 s** in the fractional regime (V1, V2, V5, V7, V9) —
  so δ=0 there is **not** simply a 20 s-cap artifact; it is the pre-fix-GMM/HELLO fractional plan.
- Validation is **impossible above n≈7000** (dense POT). Large-n rows can never be checked this way.
- **V6 is a `uniform` row with `|F|` up to 5.6·(m+n)** — the largest `|F|` growth seen on this path.

---

## 4. `queue 2026-10-15_154446` — conservative pruning on planted Monge instances

Driver `iot_experiments/core/planting_sweep.py`; generator **planting** (§5); `cupdlpx` + `hello`;
`n_eff=3200`, `d=8`, quadratic/diagonal, seeds 0–1, `g=0`; cap 100 iterations.
Each job records **both** a `prune=none` baseline **and** the requested strategy, on the same instances.

### 4.1 τ_prune and hysteresis (tie-break `none`)

| strategy | knob | converged | iterations: `none` → strategy | job wall |
| --- | --- | --- | --- | --- |
| tolerance | τ=1e-12 | 0/4 (cycles) | 8→cap 8→cap 10→cap 12→cap | 29 min |
| tolerance | τ=1e-6 | 0/4 (cycles) | 8→cap 8→cap 10→cap 12→cap | 26 min |
| tolerance | τ=1e-3 | 0/4 (cycles) | 8→cap 8→cap 10→cap 12→cap | 42 min |
| tolerance | τ=1e-1 | 0/4 (cycles) | 8→cap 8→cap 10→cap 12→cap | 29 min |
| hysteresis | τ=1e-6 | 0/4 (cycles) | 8→cap 8→cap 10→cap 11→cap | 107 min |
| hysteresis | τ=1e-3 | 0/4 (cycles) | 8→cap 8→cap 10→cap 12→cap | 25 min |
| hysteresis | τ=1e-1 | 2/4 (partial) | 8→cap **8→9** 10→cap **12→17** | 13 min |

### 4.2 regularized tie-break (`δ + ε⟨r,(θ,f,g)⟩`, one solve)

| strategy | ε | τ | converged | iterations: `none` → strategy | job wall |
| --- | --- | --- | --- | --- | --- |
| hysteresis + regularized | 1e-6 | 1e-3 | 0/4 (cycles) | 8→cap 8→cap 12→cap 12→cap | 120 min |
| **hysteresis + regularized** | **1e-4** | **1e-3** | **✅ 4/4** | 8→**13** 8→**9** 10→**30** 12→**24** | 14 min |

### 4.3 scale check (does `|F|` / iterations blow up?)

| n_eff | `prune=none` | `hysteresis` τ=1e-3 |
| --- | --- | --- |
| 6400 | ✅ 9 it | ❌ cap 100 |
| 12800 | ✅ 10 it | ❌ cap 100 |

**Takeaways.**
- **τ tuning is a dead end.** No τ_prune ∈ {1e-12 … 1e-1} makes `tolerance` or `hysteresis` converge
  (0/4 each). Consistent with `legacy/reports/ANALYSIS_REPORT.md` ("no τ/tolerance fix exists").
- **The tie-break is the deciding factor.** `hysteresis + regularized, ε=1e-4, τ=1e-3` is the **only**
  configuration that converges with pruning **on**: 4/4, 9–30 iterations. This mirrors the HiGHS
  result (`legacy/reports/LEX_GENERIC_REPORT.md`): the proof-covered tie-break works, pruning heuristics
  alone do not.
- **ε window is narrow**: 1e-4 works, 1e-6 does not (locally, 1e-2 hurt maxviol). ε must exceed the
  master's ~1e-4 accuracy to break ties without biasing δ.
- **`prune=none` does NOT explode here**: iterations stay flat at 8–10 from n_eff=3200 → 12800, and
  `|F|` is bounded by construction (`|S| = n_eff`). So on **planted Monge** instances pruning is not
  needed for growth — only for cost.
- **Cost profile is CPU-bound.** Jobs took 13–120 min with the GPU ≈3 % busy; all-time is spent in
  CPU-side master-LP assembly and edge bookkeeping, not the GPU solve.

### 4.4 `queue 2026-10-16_112747` — pruning-sweep grid (`jobs/pruning_sweep.txt`)

Driver `planting_sweep.py`; planted Monge, quadratic, `d=8`, seed 0, `g=0`; `cupdlpx` + `hello`.
**All 13 jobs completed.** Read-out: per-job `prune=none` baseline vs the strategy, on the same instance.

| job | n | strategy (knob) | conv | iters `none`→strat | **`\|F\|/(m+n)` `none`→strat** | wall | peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 3200 | `none` (baseline) | ✅ | 8 | 3.22 | 58 s | 3493 MB |
| A1 | 12800 | `none` (baseline) | ✅ | 10 | 3.82 | 196 s | 6077 MB |
| A2 | 3200 | hysteresis τ=1e-3, tie `none` | ❌ | 8→cap 60 | 3.22→2.77 | 207 s | 3515 MB |
| A2 | 12800 | hysteresis τ=1e-3, tie `none` | ❌ | 10→cap 60 | 3.82→2.88 | 647 s | 6088 MB |
| **A3** | 3200 | **hyst+reg ε=1e-4 τ=1e-3** | ✅ | 8→**13** | 3.22→**2.84** | 135 s | 3519 MB |
| **A3** | 12800 | **hyst+reg ε=1e-4 τ=1e-3** | ✅ | 10→**11** | 3.82→**2.86** | 184 s | 6093 MB |
| B1 | 3200 | hyst+reg **ε=3e-5** | ✅ | 8→15 | 3.22→2.82 | 106 s | 3532 MB |
| B2 | 3200 | hyst+reg **ε=3e-4** | ❌ | 9→cap 60 | 3.22→2.87 | 3654 s | 3517 MB |
| C1 | 3200 | hyst+reg **τ=1e-12** | ❌ | 8→cap 60 | 3.22→2.93 | 512 s | 3521 MB |
| **C2** | 3200 | hyst+reg **τ=1e-1** | ✅ | 8→**12** | 3.22→**2.84** | **62 s** | 3516 MB |
| D2 | 12800 | `none`, cap 300 | ✅ | 10 | 3.82 | 196 s | 3730 MB |
| E1 | 12800 | `none`, **tol_sep=1e-3** | ✅ | 11 | 3.81 | 247 s | 3725 MB |
| E2 | 12800 | `none`, **γ=2** | ✅ | 8 | **5.35** | 191 s | 3736 MB |
| **F1** | 51200 | `none` (baseline) | ✅ | 10 | **4.13** | 657 s | 10218 MB |
| F2 | 51200 | hyst+reg ε=1e-4 τ=1e-3 | ❌ | 10→cap 60 | 4.13→3.00 | 6968 s | 10216 MB |

**Takeaways.**
- **Pruning makes `|F|/(m+n)` size-independent; no-pruning drifts up.** `hyst+reg` gives
  **2.84 / 2.86 / 3.00** at n = 3200 / 12800 / 51200, while `prune=none` gives **3.22 / 3.82 / 4.13**.
  The pruned value is ~constant across a **16×** size range; the unpruned one rises monotonically.
  This is the strongest `O(1)` signal in the report — and it is a statement about `|F|`, not about
  iteration count.
- **`τ_prune = 1e-1` is the practical winner (C2).** Same `|F|` (2.84) and convergence as A3, but
  **62 s vs 135 s (2.2×)**, and fastest converging arm at every size. τ = 1e-12 fails (C1).
- **ε window is on the LOW side.** Server: **1e-4 and 3e-5 converge; 3e-4 FAILS** (cap, 3654 s).
  This **contradicts the local n=1000 result** (where 3e-4 was fastest) — the local ε guidance was
  wrong; use **ε ≤ 1e-4**.
- **`tol_sep = 1e-3` does not help on the server** (E1: `|F|` 3.81 vs 3.82, wall 247 s vs 196 s).
  The local n=1000 speed-up did **not** replicate — a small-`n` artifact.
- **`γ = 2` grows `|F|` ~40 %** (E2: 3.82 → 5.35), no convergence benefit. Confirms γ as the intake
  throttle.
- **At n = 51200 the regularized arm FAILED to converge** (cap 60, 3905 s) while the `prune=none`
  baseline converged in 10 it / 660 s. The n≤12800 pattern **reverses at the largest size** — the
  regularizer's advantage does not extend to 51200.
- **Peak memory 10.2 GB at n=51200** — but this includes the dense validation scaffold; the loop's own
  footprint is smaller (`peak_memory_mb` is process RSS, and `--validate` builds a dense `m·n` cost).
- **Cap-bound runs dominate wall-clock** (3654 s, 3905 s, 6968 s) — 60 iterations of a first-order
  master. The scan is never the cost.


---

## 5. Generators (ground truth) — what each instance is

| generator | plan type | `|S|` | needs | exact? | scope |
| --- | --- | --- | --- | --- | --- |
| **planting** (`core/planting.py`) | **Monge** | `= n_eff` | nothing (no OT solver) | **yes, by construction** (`cert=0`) | any linear-in-θ cost |
| HELLO (`plan_backend="hello"`) | fractional or permutation | `≈2n` / `n` | HELLO-OT | approx (`cert≈1e-5…1e-4`) | diagonal only |
| POT (`plan_backend="pot"`) | any | any | dense `m·n` | exact | `m·n ≤ 2e8` (n ≲ 14000) only |

Planted instances are the only ones that are **exactly optimal with no OT solver**, so their
`cert` is exactly 0 and δ is a genuine margin. They are **Monge** (no mass splitting ⇒ no fractional
support), so they do **not** exercise the `|S| ≈ 2n−1` fractional regime.

---

## 6. Known defects and corrections

1. **Old GMM construction was wrong (fixed).** The mixture was applied along the **sample generation
   order**, not the geometry, producing ~1e-8 mass tails; HELLO then dropped those rows/columns and
   `InverseOTProblem` rejected the plan ("every row and column must have positive mass"). This is why
   **6/9 jobs in §1 failed** and why the `gmm`/`gmm_floored` rows in §2–§3 are **stale**: the current
   GMM draws points from a d-dimensional mixture and keeps unit weights. **Any new `gmm` run must be
   regenerated.**
2. **`δ=0` in the fractional regime** is now attributed to the pre-fix fractional plan (non-tight
   support edges), **not** to a cap and not to a solver failure. See §5: the planting generator avoids
   this entirely.
3. **`cert=False`** is a tolerance artifact: HELLO duals ~1e-4 vs a 1e-6 threshold.
4. **`is_optimal` ≠ `conv`.** See §0's trap.
5. **Validation ceiling:** dense POT caps the authoritative check at n ≈ 7000, so **n ≥ 12800 rows
   have no exact verdict**.
6. **No `|F|` in the synced results.** The queue logs carry iterations and δ but not `|F|`; the sweep
   JSONs (which do) are not synced back. Some `|F|` claims are inferred from iteration counts.
   *(The `planting_sweep.py` stdout line now prints `F/(m+n)`, `t`, `mem` per arm — §4.4 uses those.)*
7. **Natural degeneracy makes `T` and `|F|/N` rise (local, not yet on the server).** Probe
   `core/degenerate_natural.py`, family quadratic, `cupdlpx`+`hello`, `prune=none`, γ=1:

   | kind | d | n | T | conv | \|F\|/(m+n) | δ |
   | --- | --- | --- | --- | --- | --- | --- |
   | duplicate | 1 | 200 / 1000 / 2000 / 4000 | 10 / 14 / 28 / 22 | ✅✅✅✅ | 5.67 / 7.35 / 8.50 / 8.63 | 0 |
   | lattice1d | 1 | 200 / 1000 / 2000 / 4000 | 7 / 15 / 23 / 16 | ✅✅✅✅ | 5.34 / 5.23 / 5.46 / 7.46 | 0 mostly |
   | grid | 2 | 1000 / 2000 | 14 / 12 | ✅✅ | 4.39 / 5.05 | >0 |

   Contrast: random-point planted Monge stays `T ≈ 8–10`, `|F|/(m+n) ≈ 2.8–3.6`. So **ties in the
   cloud itself** (duplicate points, or `d=1` where cost depends on one coordinate) are the first family
   that inflates both. All converge (no cycling), and `δ = 0` in 13/14 runs. The trend is **noisy**
   (T dips at n=4000) and there are only 4 points — an exponent fit is not yet possible. **This is the
   live risk for `Θ(N^c)`.**

---

## 7. Historical reports off the GPU/exact path — DO NOT CITE AS CURRENT

These predate the GPU reframing and/or are **not exact-OT** results. They are kept for provenance; a
row from them must never be mixed into a table above.

| file (in `reports/`, unless noted) | why it is off-path |
| --- | --- |
| `SCALING_REPORT.md` | shipped **HiGHS** path + **NumPy** scan; diagonal/quadratic CPU studies |
| `MARGINAL_REPORT.md` | **POT** ground truth; `p`/component framing; pre-GPU |
| `FAMILY_REPORT.md` | CPU solvers (regularized/SDP); `bilinear-PSD` is **not** exact-OT at scale |
| `BENCHMARK_EXACT_REPORT.md` | **HiGHS** exact constraint generation; not the GPU path |
| `MEASUREMENT_NOTES.md` | CPU/GPU arms of the **gradient** method; not constraint generation |
| `CUTTING_PLANE_REPORT.md` | a **different algorithm** (Kelley cutting plane), not this loop |
| `CUPDLPX_MASTER_REPORT.md` | *partially* current (the CuPDLPx adapter), but measured with **HiGHS comparisons** and the old tie-break framing |
| `HELLO_SCAN_REPORT.md` | current for the **diagonal** scan; scope-corrected banner added |
| `HELLO_SOLVER_REPORT.md` | HELLO as an **inner** solver for other methods, not the loop |
| `legacy/reports/*` | all retired (pruning/solver internals, geometry, cost-family survey) |

---

## 8. Summary of what is established

| claim | evidence | status |
| --- | --- | --- |
| GPU-only loop runs end-to-end on the target path | §2, §3, §4 | **established** |
| 3200-point instances are **validated optimal** (both regimes) | §3 (V1, V3, V5, V6) | **established** |
| The **master** (not the scan) is the bottleneck | §2 (14–23 vs 0.4–3 s/it) | **established** |
| First-order master is **cap-truncated at scale** | §3 (≈300 s/it), §4.4 (51200: 3905 s) | **established** |
| `δ=0` in the fractional regime | §2, §3 | **explained** (pre-fix GMM; see §6) |
| τ_prune can rescue pruning | §4.1, §4.4 (C1 τ=1e-12 fails) | **refuted** |
| `regularized` tie-break makes conservative pruning work | §4.2, §4.4 (A3/B1/C2) | **established up to n=12800**; **fails at n=51200** (F2) |
| Safe ε is on the **low** side (≤1e-4); 3e-4 fails | §4.4 (B1 ✅, A3 ✅, B2 ❌) | **established** (contradicts the local n=1000 run) |
| `tol_sep` largening speeds things up | §4.4 (E1) | **refuted** (local n=1000 artifact) |
| γ is the `|F|` intake throttle | §4.4 (E2: +40 %) | **established** |
| Pruning holds `\|F\|/(m+n)` ~constant in `n` on planted Monge | §4.4 (2.84→2.86→3.00 vs 3.22→3.82→4.13) | **established (planted Monge only)** |
| `\|F\|`/iterations explode on planted instances | §4.3, §4.4 | **refuted** (10 it at n=51200) |
| Pruning is needed for **growth** | §6.7 (natural degeneracy) | **open — new evidence that it IS needed** |

## 9. Open questions

1. **The `T(N)` growth question (highest priority).** The natural-degeneracy probe
   (`core/degenerate_natural.py`, *not* yet in a batched server run) found the **first** instances where
   both `T` and `|F|/(m+n)` rise with `n`: duplicate points and a 1-D lattice, at `d ∈ {1,2}`
   (`T` up to 28, `|F|/(m+n)` up to 8.6, all `δ=0`). If `T` keeps rising at scale,
   `|F|/N = Θ(N^c)` with `c>0` and the LP's `~N¹·⁹·|F|²` cost becomes `N⁴`. **This is the one
   mechanism that could invalidate the `O(1)` result below.**
2. **Fractional/Monge gap**: planting cannot produce `|S| ≈ 2n−1`. Mass splitting (a *separate,
   labelled* option) is the only way to exercise the regime the loop was designed for.
3. **Regenerate all `gmm` rows** with the fixed GMM (§6).
4. **Validation above n≈7000** — needs a HiGHS-free GPU certificate (not implemented).
5. **Why does `hyst+reg` fail at n=51200 but not ≤12800?** (F2 vs A3) — is it τ, ε, or size?


**Batch `jobs/pruning_sweep.txt` (13 jobs, staged)** addresses 1, 4 and two never-tested knobs:
- `hysteresis + regularized` vs `prune=none`, at n=3200/12800/51200, read-out `max|F|/(m+n)`,
  `peak_memory_mb`, mean master/scan time;
- **ε window** (3e-5, 1e-4, 3e-4) and **τ_prune** (1e-12, 1e-3, 1e-1) around the winner;
- **time budget t** (60 vs 300 s) at the previously cap-bound n=12800;
- **`tol_sep`** (1e-9 vs 1e-3) and **`gamma`** (1 vs 2) — the two knobs that were never exposed
  before this revision (now `RunSpec.tol_sep` / `RunSpec.gamma`), and the ones most likely to
  change `|F|` structurally;
- the **large run at n=51200** (§7's sizing report: ~15–20 min/run).

## 10. Newly exposed knobs (this revision)

`RunSpec` gained three fields (defaults preserve the old behaviour exactly):

| field | meaning | was |
| --- | --- | --- |
| `tol_sep` | separation tolerance (Separate: candidate iff `σ + δ > tol_sep`) | hardcoded `LP_TOL = 1e-9` |
| `gamma` | Select: top-γ violated edges per row/column | hardcoded `GAMMA = 1` |
| `tau_prune` | Pruning slack tolerance | was already reachable via the loop, now in the spec |

CLI: `run_batch --tol-sep/--gamma/--tau-prune`; `planting_sweep --tol-sep/--gamma/--tau-prune/
--epsilon/--master-time-limit`. All runs before this revision used the hardcoded defaults.


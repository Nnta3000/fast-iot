# GPU server tests — consolidated results

**Status: CURRENT, empirical.** Every A100 run to date. This revision is **tabulated only**: every
sentence here is either a configuration statement or a statement about a measured value. Interpretation,
hypotheses and recommendations are not in this file — see §9 for the list of open questions, and
`legacy/reports/SERVER_TESTS_REPORT_narrative.md` for the earlier prose revision (superseded).

**Changelog 2026-10-22.** Rewritten from the narrative revision: prose cut to tables, per-run columns
made uniform, and every cross-cell comparison that required an unmeasured quantity removed. No data
changed. Corrections carried over: §6.5 (`correlated d=8 n=6400` is uncertified) and §4.4/§4.5
(F2 retracted).

**Backing data** (git-ignored, on the server and synced to `/srv/results/fast-iot/`):

| path | content |
| --- | --- |
| `iot_experiments/out/queue/<ts>/{summary.tsv,queue.log,NN_*.log}` | queue console records |
| `iot_experiments/out/{no_prune,validated,e2e,e2e_gpu}/<run>/e2e.{json,csv}` | per-run records |
| `iot_experiments/core/out/planting/<tag>/planting_sweep.json` | planting sweep records |
| `iot_experiments/core/out/brenier_toy/brenier_toy.<tag>.json` | §6.5 records (full config + θ per run) |
| `iot_experiments/core/out/{fractional,degenerate}/...` | §4.5 A3/A4/A5 records |

## 0. Conventions

**Scope.** All rows are GPU-only: `scan_backend="hello"`, master `cupdlpx`, plan from `plant_backend`
or the planting generator. No NumPy-scan and no HiGHS-master row appears in this report.

**Symbols.**

| symbol | meaning |
| --- | --- |
| `n_eff`, `m_eff` | source / target counts after empty-column pruning |
| `\|S\|` | support size of the observed plan `π̂` |
| `\|F\|` | edge set the master is solved on |
| `δ` | max margin of the master (0 ⇒ `π̂` optimal with no slack) |
| `maxviol` | loop certificate: `max_e σ_e + δ` over `e ∉ F`; `≤ tol` ⇒ converged |
| `conv` / `cap` | converged / hit `max_iterations` |
| `is_optimal` | forward POT check of `c(θ_returned)` against `⟨c, π̂⟩` |
| `cert` | ground-truth dual certificate of `π̂` at `θ*` (`core.costs.dual_feasibility`) |

**`conv` and `is_optimal` measure different things.** Rows occur with `is_optimal=True` and
`conv=False`; both values are reported separately in every table.

---

## 1. `queue 2026-10-13_153639 / _153701 / _153717` — first GPU e2e (SUPERSEDED)

`run_batch --batch e2e`, diagonal, `plan_backend=hello`, `cupdlpx`, master cap 2–10 s, ≤20 iterations.

| run | n | marginal | d | iters | conv | δ | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E3 | 12800 | gmm | 8 | 5 | ✅ | 2.36e-4 | completed |
| E4 | 6400 | gmm | 32 | 5 | ✅ | 0 | completed |
| E6 | 6400 | uniform | 8 | 11 | ✅ | 0 | completed |
| E1, E2, E5, E7, E8, E9 | 3200–25600 | gmm | 8–128 | — | ❌ | — | failed: zero-mass column (§6.1) |

Superseded: 6 of 9 jobs used the pre-fix GMM (§6.1) and a 10 s master cap. Do not cite.

---

## 2. `queue 2026-10-14_110632` — no-prune `|F|`/δ scaling (pre-GMM-fix)

`--scans hello --masters cupdlpx --plan-backend hello`, `prune=false`, `numpy` scan, master cap 20 s,
≤30 iterations. Generator: HELLO. GMM rows use the **pre-fix** GMM (§6.1).

| run | n | marginal | regime | iters | conv | δ | maxviol | `\|F\|/(m+n)` | master s/it | scan s/it | wall | `cert` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | 3200 | gmm_floored | fractional | 3 | ✅ | 0 | −4.5e-4 | 2.45 | 14.6 | 0.42 | 45 s | False |
| N2 | 6400 | gmm_floored | fractional | 5 | ✅ | 0 | −4.7e-5 | 2.79 | 18.1 | 1.01 | 95 s | False |
| N3 | 12800 | gmm_floored | fractional | 6 | ✅ | 0 | −1.7e-5 | 2.84 | 19.9 | 3.32 | 140 s | False |
| N4 | 25600 | gmm_floored | fractional | 7 | ✅ | 0 | −4.4e-6 | 2.49 | 23.0 | 11.6 | 242 s | False |
| N5 | 12800 | uniform | permutation | 12 | ✅ | 8.3e-6 | −1.8e-5 | 5.16 | 22.9 | 2.01 | 298 s | False |

Master s/it is 14.6–23.0; scan s/it is 0.42–11.6 (§8).

---

## 3. `queue 2026-10-14_150534` — validated, uncapped (V1–V9)

Same path as §2, master cap **300 s**, `--validate` where `m·n ≤ 5e7` (n ≤ ~7000).

| run | n | marginal | regime | seeds | iters | conv | δ | maxviol | `is_optimal` | rel_gap | `\|F\|/(m+n)` | master peak | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 3200 | gmm_floored | fractional | 0 | 3 | ✅ | 0 | −4.5e-4 | True | 1.3e-8 | 2.45 | 300.8 s (cap) | 623 s |
| V2 | 6400 | gmm_floored | fractional | 0 | 8 | ✅ | 0 | −3.3e-5 | **False** | 9.2e-7 | 4.49 | 302.9 s (cap) | 2348 s |
| V5 | 3200 | gmm_floored | fractional | 1,2,3 | 3–5 | ✅✅✅ | 0,0,0 | −3e-5…−4e-4 | True ×3 | 1e-9…3e-8 | 2.1–2.6 | 32–49 s | 132–248 s |
| V3 | 3200 | uniform | permutation | 0 | 12 | ✅ | 5.5e-5 | −2.2e-4 | True | 1.1e-16 | 4.96 | 24.3 s | 296 s |
| V6 | 3200 | uniform | permutation | 1,2,3 | 11–12 | ✅✅✗ | 3.0e-5…9.9e-5 | ≤−7.7e-5 | True ×3 | 0…8e-17 | 5.1–5.6 | 8–36 s | 89–432 s |
| V4 | 6400 | uniform | permutation | 0 | 12 | ✗ | 0 | +3.3e-4 | **False** | 2.4e-7 | 5.28 | 59.9 s | 366 s |
| V7 | 12800 | gmm_floored | fractional | 0 | 9 | ✅ | 0 | −1.7e-5 | — | — | 3.44 | 304.2 s (cap) | 2457 s |
| V8 | 12800 | uniform | permutation | 0 | 12 | ✗ | 8.8e-6 | +3.9e-4 | — | — | 5.24 | 301.3 s (cap) | 740 s |
| V9 | 25600 | gmm_floored | fractional | 0 | 9 | ✅ | 0 | −1.8e-5 | — | — | 3.66 | 309.1 s (cap) | 2560 s |

n=3200 rows with a POT verdict: V1, V3, V5(×3), V6(×3) — all `is_optimal=True`. n=6400 rows:
V2 and V4 are `is_optimal=False`. Rows above n≈7000 carry no POT verdict (dense-POT ceiling, §6.5).

---

## 4. `queue 2026-10-15_154446` — conservative pruning on planted Monge instances

Driver `iot_experiments/core/planting_sweep.py`; generator planting; `cupdlpx` + `hello`;
`n_eff=3200`, `d=8`, quadratic/diagonal, seeds 0–1, `g=0`; `max_iterations=100` unless stated.
Each job records a `prune=none` baseline and the requested strategy on the same instances.

### 4.1 τ_prune and hysteresis (tie-break `none`)

| strategy | τ_prune | conv | iterations `none` → strategy | job wall |
| --- | --- | --- | --- | --- |
| tolerance | 1e-12 | 0/4 | 8→cap 8→cap 10→cap 12→cap | 29 min |
| tolerance | 1e-6 | 0/4 | 8→cap 8→cap 10→cap 12→cap | 26 min |
| tolerance | 1e-3 | 0/4 | 8→cap 8→cap 10→cap 12→cap | 42 min |
| tolerance | 1e-1 | 0/4 | 8→cap 8→cap 10→cap 12→cap | 29 min |
| hysteresis | 1e-6 | 0/4 | 8→cap 8→cap 10→cap 11→cap | 107 min |
| hysteresis | 1e-3 | 0/4 | 8→cap 8→cap 10→cap 12→cap | 25 min |
| hysteresis | 1e-1 | 2/4 | 8→cap, 8→9, 10→cap, 12→17 | 13 min |

### 4.2 regularized tie-break (one solve)

| strategy | ε | τ_prune | conv | iterations `none` → strategy | job wall |
| --- | --- | --- | --- | --- | --- |
| hysteresis + regularized | 1e-6 | 1e-3 | 0/4 | 8→cap 8→cap 12→cap 12→cap | 120 min |
| hysteresis + regularized | 1e-4 | 1e-3 | **4/4** | 8→13 8→9 10→30 12→24 | 14 min |

### 4.3 scale check

| n_eff | `prune=none` | `hysteresis` τ=1e-3 |
| --- | --- | --- |
| 6400 | conv, 9 it | not conv, cap 100 |
| 12800 | conv, 10 it | not conv, cap 100 |

### 4.4 `queue 2026-10-16_112747` — pruning-sweep grid (13 jobs, all completed)

Driver `planting_sweep.py`; planted Monge, quadratic, `d=8`, seed 0, `g=0`; `cupdlpx` + `hello`.

| job | n | strategy (knob) | conv | iters `none`→strat | `\|F\|/(m+n)` `none`→strat | wall | peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 3200 | `none` (baseline) | ✅ | 8 | 3.22 | 58 s | 3493 MB |
| A1 | 12800 | `none` (baseline) | ✅ | 10 | 3.82 | 196 s | 6077 MB |
| A2 | 3200 | hysteresis τ=1e-3, tie `none` | ❌ | 8→cap 60 | 3.22→2.77 | 207 s | 3515 MB |
| A2 | 12800 | hysteresis τ=1e-3, tie `none` | ❌ | 10→cap 60 | 3.82→2.88 | 647 s | 6088 MB |
| A3 | 3200 | hyst+reg ε=1e-4 τ=1e-3 | ✅ | 8→13 | 3.22→2.84 | 135 s | 3519 MB |
| A3 | 12800 | hyst+reg ε=1e-4 τ=1e-3 | ✅ | 10→11 | 3.82→2.86 | 184 s | 6093 MB |
| B1 | 3200 | hyst+reg ε=3e-5 | ✅ | 8→15 | 3.22→2.82 | 106 s | 3532 MB |
| B2 | 3200 | hyst+reg ε=3e-4 | ❌ | 9→cap 60 | 3.22→2.87 | 3654 s | 3517 MB |
| C1 | 3200 | hyst+reg τ=1e-12 | ❌ | 8→cap 60 | 3.22→2.93 | 512 s | 3521 MB |
| C2 | 3200 | hyst+reg τ=1e-1 | ✅ | 8→12 | 3.22→2.84 | 62 s | 3516 MB |
| D2 | 12800 | `none`, cap 300 | ✅ | 10 | 3.82 | 196 s | 3730 MB |
| E1 | 12800 | `none`, `tol_sep=1e-3` | ✅ | 11 | 3.81 | 247 s | 3725 MB |
| E2 | 12800 | `none`, γ=2 | ✅ | 8 | **5.35** | 191 s | 3736 MB |
| F1 | 51200 | `none` (baseline) | ✅ | 10 | **4.13** | 657 s | 10218 MB |
| F2 | 51200 | hyst+reg ε=1e-4 τ=1e-3 | ❌ | 10→cap 60 | 4.13→3.00 | 6968 s | 10216 MB |

F2 (`conv=❌`, cap 60, 6968 s) is **retracted** — see §4.5 B4: the identical configuration converges.
F1's peak memory includes the dense validation scaffold (`--validate`), not the loop alone.

### 4.5 `queue 2026-10-16_145725` — A/B/C: `T(N)` ladder, the 51200 re-run, fractional regime

Drivers `planting_sweep.py`, `fractional_sweep.py`, `degenerate_natural.py`. Quadratic `d=8`
(duplicate `d=1`), seed 0, `g=0`, `cupdlpx` + `hello`, `max_iterations=200`. **All 16 jobs completed**,
serial on one GPU (14:57 → 20:39).

#### A — `|F|/(m+n)` and `T` over a 16× size range

Cells are `|F|/(m+n)` with `T` in parentheses; slope is a log–log fit over the five sizes.

| arm | generator | prune | 3200 | 6400 | 12800 | 25600 | 51200 | slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | planted Monge | `none` | 3.22 (8) | 3.50 (9) | 3.82 (10) | 3.77 (9) | 4.13 (10) | +0.083 |
| A2 | planted Monge | `hyst+reg` | 2.84 (12) | 2.96 (10) | 2.89 (11) | 2.98 (11) | 3.00 (11) | +0.017 |
| A3 | fractional sf=1.0 | `none` | 2.63 (7) | 3.35 (10) | 3.45 (8) | 3.98 (7) | 4.15 (6) | +0.156 |
| A4 | fractional sf=1.0 | `hyst+reg` | 2.62 (8) | 2.92 (10) | 2.95 (16) | 3.38 (10) | 3.09 (7) | +0.069 |
| A5 | duplicate d=1 | `none` | 8.32 (17) | 10.46 (20) | — | — | — | +0.33 (2 pts) |

`T` slopes over the same range: A1 −0.10, A2 −0.05, A3 −0.06, A4 +0.06 (from the same fits).

#### A3/A4 — fractional regime

| n | `\|S\|/n` | split | skipped | `T` none/hyst | `\|F\|/(m+n)` none/hyst | `is_optimal` | rel_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3200 | 1.27 | 1751 | 1449 | 7 / 8 | 2.63 / 2.62 | True | 7e-9 |
| 6400 | 1.36 | 4538 | 1862 | 10 / 10 | 3.35 / 2.92 | True | 3e-10 |
| 12800 | 1.33 | 8700 | 4100 | 8 / 16 | 3.45 / 2.95 | True | 7e-11 |
| 25600 | 1.39 | 20090 | 5510 | 7 / 10 | 3.98 / 3.38 | True | 4e-10 |
| 51200 | 1.33 | 33260 | 17940 | 6 / 7 | 4.15 / 3.09 | True | 2e-11 |

`max_dual_violation = 0` in all five rows.

#### A5 — duplicate points, `prune=none`

| n | `T` | `\|F\|/(m+n)` | δ |
| --- | --- | --- | --- |
| 3200 | 17 | 8.32 | 0 |
| 6400 | 20 | 10.46 | 0 |
| ≥12800 | not run | — | — |

Not run at n≥12800 because the probe builds its ground-truth plan with dense POT (§9, item 1).

#### B — the §4.4 F2 re-run

| arm | n | τ_prune | ε | conv | it | `\|F\|/(m+n)` | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 25600 | 1e-3 | 1e-4 | ✅ | 10 | 2.98 | 301 s |
| B2 | 51200 | 1e-1 | 1e-4 | ✅ | 10 | 3.00 | 427 s |
| B3 | 51200 | 1e-3 | 3e-5 | ✅ | 11 | 3.00 | 459 s |
| B4 | 51200 | 1e-3 | 1e-4 | ✅ | 10 | 3.00 | 433 s |

B4 has the same settings as F2 (§4.4) and converges; F2 is retracted.

#### C — fractional knobs (n=12800)

| arm | family | split_fraction | split_ratio | slide_cap | `\|S\|/n` | split | skipped | `T` | `\|F\|/(m+n)` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | quadratic | 1.0 | 0.5 | 4.0 | 1.33 | 8700 | 4100 | 8 | 3.45 |
| C3 | quadratic | 0.5 | 0.5 | 4.0 | 1.16 | 4349 | 2051 | 6 | 3.43 |
| C4 | quadratic | 1.0 | 0.5 | 0.5 | 1.08 | 2141 | 10659 | 6 | 2.13 |
| C5 | quadratic | 1.0 | 0.9 | 4.0 | 1.33 | 8700 | 4100 | 8 | 3.45 |
| C6 | quadratic | 1.0 | 0.5 | 4.0 | 1.33 | 8700 | 4100 | 16 | 2.95 |
| C7 | diagonal | 1.0 | 0.5 | 4.0 | 1.51 | 12800 | 0 | 10 | 2.92 |

C2 and C5 (split_ratio 0.5 vs 0.9) give identical `|S|/n`, split count and `T`.
C2 and C6 differ only in prune strategy (none vs hyst+reg).

---

## 5. Generators (ground truth)

| generator | plan type | `\|S\|` | needs | exact? | scope |
| --- | --- | --- | --- | --- | --- |
| planting (`core/planting.py`) | Monge | `= n_eff` | no OT solver | yes by construction (`cert=0`) | any linear-in-θ cost |
| HELLO (`plan_backend="hello"`) | fractional or permutation | `≈2n` / `n` | HELLO-OT | approx (`cert≈1e-5…1e-4`) | diagonal only |
| POT (`plan_backend="pot"`) | any | any | dense `m·n` | exact | `m·n ≤ 2e8` (n ≲ 14000) |

---

## 6. Known defects and corrections

| # | defect | status |
| --- | --- | --- |
| 6.1 | pre-fix GMM built mixture ordering along sample index, not geometry, producing ~1e-8 mass tails; HELLO dropped rows/columns and `InverseOTProblem` rejected the plan | **fixed**; all `gmm`/`gmm_floored` rows in §1–§3 are stale and must be regenerated |
| 6.2 | `δ=0` in the fractional regime | attributed to the pre-fix fractional plan (non-tight support edges); planting avoids it |
| 6.3 | `cert=False` on HELLO plans | tolerance artifact: HELLO duals ~1e-4 vs a 1e-6 threshold |
| 6.4 | `is_optimal` ≠ `conv` | two different measurements (§0) |
| 6.5 | dense POT caps the authoritative check at n ≈ 7000 | n ≥ 12800 rows have no POT verdict |
| 6.6 | `is_optimal=True` can coexist with `conv=False` and `maxviol>0` | `is_optimal` is a forward-POT check, not a dual certificate; a row is certified only when `conv=True` **and** `maxviol ≤ tol` (§6.5) |

### 6.7 `core/degenerate_natural.py` (local, not server)

Quadratic, `cupdlpx`+`hello`, `prune=none`, γ=1.

| kind | d | n | T | conv | `\|F\|/(m+n)` | δ |
| --- | --- | --- | --- | --- | --- | --- |
| duplicate | 1 | 200 / 1000 / 2000 / 4000 | 10 / 14 / 28 / 22 | ✅✅✅✅ | 5.67 / 7.35 / 8.50 / 8.63 | 0 |
| lattice1d | 1 | 200 / 1000 / 2000 / 4000 | 7 / 15 / 23 / 16 | ✅✅✅✅ | 5.34 / 5.23 / 5.46 / 7.46 | 0 (13/14 rows) |
| grid | 2 | 1000 / 2000 | 14 / 12 | ✅✅ | 4.39 / 5.05 | >0 |

Four points per ladder; `T` is not monotone (4000 < 2000 in both d=1 rows).

---

## 6.5 `queue 2026-10-21_113400` — upstream Brenier / correlated ladder

Job file `iot_experiments/gpu_bench/jobs/brenier_ladder.txt`; driver `iot_experiments/core/brenier_toy.py`;
instances from `export_experiments/synthetic.py`. Plan = known permutation (`|S| = n`). Config for every
row: `scan=hello`, `solver=cupdlpx`, `tie_break=regularized` ε=1e-4, `prune=none`,
`max_iterations=200`, `master_time_limit=300 s`, `tol=1e-9`, `seed=0`. Records:
`core/out/brenier_toy/brenier_toy.<tag>.json`.

| tag | kind | d | n | T | `\|F\|/(m+n)` | δ | `is_optimal` | rel_gap | wall | conv | maxviol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brenier_d2 | brenier | 2 | 800 | 12 | 4.69 | 5.16e-6 | True | 0.0 | 42 s | ✅ | −5.21e-5 |
| brenier_d2 | brenier | 2 | 1600 | 11 | 4.95 | 2.70e-6 | True | 0.0 | 27 s | ✅ | −6.24e-6 |
| brenier_d2 | brenier | 2 | 3200 | 14 | 7.05 | 1.30e-7 | True | 0.0 | 195 s | ✅ | −6.25e-6 |
| brenier_d2 | brenier | 2 | 6400 | 19 | 7.84 | 2.18e-7 | True | 0.0 | 460 s | ✅ | −1.34e-6 |
| brenier_d8 | brenier | 8 | 800 | 5 | 3.92 | 6.82e-2 | True | 0.0 | 15 s | ✅ | −4.64e-3 |
| brenier_d8 | brenier | 8 | 1600 | 5 | 3.57 | 5.03e-2 | True | 0.0 | 33 s | ✅ | −2.52e-3 |
| brenier_d8 | brenier | 8 | 3200 | 9 | 4.74 | 4.00e-2 | True | 0.0 | 614 s | ✅ | −2.87e-3 |
| brenier_d8 | brenier | 8 | 6400 | 5 | 3.43 | 2.22e-2 | True | 0.0 | 284 s | ✅ | −2.24e-3 |
| correlated_d2 | correlated | 2 | 800 | 21 | 9.15 | 5.63e-6 | True | 0.0 | 645 s | ✅ | −1.34e-4 |
| correlated_d2 | correlated | 2 | 1600 | 19 | 9.50 | 0.0 | True | 0.0 | 727 s | ✅ | −5.40e-5 |
| correlated_d2 | correlated | 2 | 3200 | 28 | 9.91 | 3.73e-7 | True | 0.0 | 402 s | ✅ | −3.47e-6 |
| correlated_d2 | correlated | 2 | 6400 | 31 | 11.34 | 1.12e-7 | True | 0.0 | 1044 s | ✅ | −1.02e-5 |
| correlated_d8 | correlated | 8 | 800 | 42 | 9.06 | 7.74e-3 | True | 0.0 | 3612 s | ✅ | −6.02e-3 |
| correlated_d8 | correlated | 8 | 1600 | 29 | 10.11 | 3.69e-3 | True | 0.0 | 3983 s | ✅ | −3.73e-4 |
| correlated_d8 | correlated | 8 | 3200 | 33 | 11.12 | 2.79e-3 | True | 0.0 | 7668 s | ✅ | −2.80e-4 |
| correlated_d8 | correlated | 8 | 6400 | 200 (cap) | 27.20 | 7.07e-4 | True | 0.0 | 61288 s | ❌ | **+5.15e-3** |
| correlated_d2_seeds | correlated | 2 | 800 | 15 | 7.29 | 2.47e-5 | True | 0.0 | 133 s | ✅ | −2.89e-6 |
| correlated_d2_seeds | correlated | 2 | 800 | 15 | 7.20 | 1.71e-6 | True | 0.0 | 65 s | ✅ | −1.81e-4 |

Notes on this table:

1. **`correlated d=8 n=6400` is not certified.** `conv=False`, `maxviol=+5.15e-3` (positive ⇒ violated).
   Its `is_optimal=True` is the forward-POT check of §6.6, not a dual certificate. Its
   `|F|/(m+n)=27.20` is a value at the iteration cap, not a converged value.
2. **The two `correlated_d2_seeds` rows are seeds 1 and 2** (`seed` is not a field in these records;
   the assignment is from the job line `--seeds 1 2`). Job order, and hence row order, is not recorded
   in the JSON.
3. **`δ` is an absolute margin and is compared within a column only.** `tol=1e-9` is absolute, and the
   `brenier` and `correlated` costs have different scales; `median(c)` is not recorded in these JSONs.
4. **`theta_star` is the uniform vector `[0.125]*8` in every `correlated` record; `theta_returned` is
   not uniform in any of them.** `theta_returned` is not a recovery of `theta_star`.
5. `brenier`'s permutation is recovered by nearest-neighbour matching (upstream does not return it);
   `correlated`'s permutation is returned by upstream.

---

## 7. Historical reports off the GPU/exact path — DO NOT CITE AS CURRENT

| file (in `reports/`, unless noted) | why it is off-path |
| --- | --- |
| `SCALING_REPORT.md` | HiGHS master + NumPy scan; CPU cost studies |
| `MARGINAL_REPORT.md` | POT ground truth; `p`/component framing; pre-GPU |
| `FAMILY_REPORT.md` | CPU solvers (regularized/SDP); `bilinear-PSD` not exact-OT at scale |
| `BENCHMARK_EXACT_REPORT.md` | HiGHS exact constraint generation |
| `MEASUREMENT_NOTES.md` | gradient-method arms, not constraint generation |
| `CUTTING_PLANE_REPORT.md` | a different algorithm (Kelley cutting plane) |
| `CUPDLPX_MASTER_REPORT.md` | partially current; measured with HiGHS comparisons |
| `HELLO_SCAN_REPORT.md` | current for the diagonal scan only |
| `HELLO_SOLVER_REPORT.md` | HELLO as an inner solver for other methods |
| `SERVER_TESTS_REPORT_narrative.md` (this directory's `legacy/`) | the prose revision of this report |
| `legacy/reports/*` | all retired |

---

## 8. Summary of what is established

"Established" = the value was measured on the stated configuration. Interpretation is not included.

| claim | evidence | status |
| --- | --- | --- |
| GPU-only loop runs end-to-end | §2, §3, §4 | established |
| n=3200 rows with a POT verdict are `is_optimal=True` | §3 (V1, V3, V5×3, V6×3) | established |
| Master s/it (14.6–23.0) exceeds scan s/it (0.42–11.6) | §2 | established |
| Master is cap-bound at the 300 s limit in §3 fractional rows | §3 (V1, V2, V7, V8, V9) | established |
| τ_prune ∈ {1e-12 … 1e-1} does not make tolerance/hysteresis pruning converge | §4.1 (0/4 each) | established |
| hyst+reg ε=1e-4 τ=1e-3 converges with pruning on | §4.2 (4/4), §4.4 A3, §4.5 A2 | established to n=51200 |
| ε=3e-4 does not converge | §4.4 B2 | established (contradicts the local n=1000 run) |
| `tol_sep=1e-3` changes `\|F\|` by 0.01 and wall by +51 s at n=12800 | §4.4 E1 vs A1 | established |
| γ=2 raises `\|F\|/(m+n)` 3.82→5.35 at n=12800 | §4.4 E2 | established |
| `hyst+reg` holds `\|F\|/(m+n)` in 2.84–3.00 across n=3200–51200 | §4.4 A3, §4.5 A2 | established |
| `prune=none` raises `\|F\|/(m+n)` 3.22→4.13 across n=3200–51200 | §4.5 A1 | established |
| `\|F\|/N` slope is +0.017…+0.156 over 5 points (16× range) | §4.5 A | measured; not separable from `log N` |
| `T` slope is −0.10…+0.06 over the same range | §4.5 A | measured |
| Fractional instances converge with `is_optimal=True` to n=51200 | §4.5 A3/A4 | established |
| Duplicate points give `\|F\|/(m+n)` 8.32–10.46 at n=3200–6400 | §4.5 A5 | established, 2 sizes |
| F2 (§4.4) does not reproduce | §4.5 B4 | retracted |
| `correlated` `\|F\|/(m+n)` rises 9.15→11.34 (d=2) and 9.06→11.12 (d=8, n≤3200) | §6.5 | measured, 1 seed per cell |
| `brenier` d=8 `\|F\|/(m+n)` spans 3.43–4.74 with no monotone trend | §6.5 | measured |
| At n=800 `correlated` d=2, `\|F\|/(m+n)` = 9.15 / 7.29 / 7.20 for three seeds | §6.5 | measured |
| `correlated` d=8 n=6400 does not converge within 200 iterations | §6.5 | established (uncertified) |

---

## 9. Open questions

1. **Extend the `duplicate` ladder past n=6400** — blocked by the A5 probe's dense-POT ground truth
   (§6.5, A5 note). `plant_instance` needs no OT solver and would remove the ceiling.
2. **Separate `log N` from `N^c`** in §4.5 A — 5 points over 16×; needs more sizes.
3. **Reach `|S|/n → 2n`** in the fractional regime — `slide_cap` is the controlling knob (§4.5 C4
   gives 1.08, C2 gives 1.33); the cap sweep has not been done.
4. **Regenerate all `gmm` rows** with the fixed GMM (§6.1).
5. **POT verdict above n≈7000** — not implemented (§6.5).
6. **`median(c)` per run** — needed before `δ` can be compared across cells (§6.5 note 3).

---

## 10. Knobs exposed in this revision

`RunSpec` gained three fields (defaults preserve the previous behaviour):

| field | meaning | was |
| --- | --- | --- |
| `tol_sep` | separation tolerance (candidate iff `σ + δ > tol_sep`) | hardcoded `LP_TOL = 1e-9` |
| `gamma` | top-γ violated edges per row/column | hardcoded `GAMMA = 1` |
| `tau_prune` | pruning slack tolerance | reachable via the loop, not in the spec |

CLI: `run_batch --tol-sep/--gamma/--tau-prune`; `planting_sweep --tol-sep/--gamma/--tau-prune/
--epsilon/--master-time-limit`.

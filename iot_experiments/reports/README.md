# `reports/` — start here

**Last updated 2026-11-05.** Every report in this repository appears exactly once in the table in §2,
with its question, its state, its record path and what supersedes what. If a report is not in that
table, it is not current.

`reports/` is Markdown only. Machine-written records (`records.json`, `*.json`, run logs) live in
`../out/`; the committed ones are in `../out/records/` (see §5).

---

## 1. Read these first

| file | what it is |
| --- | --- |
| `STATE.md` | durable state: goal, SynthRAD split, verified results with paths, decisions with dates, resource rules. The most important document here |
| `EXPERIMENTS.md` | the flat list of every experiment, with a status column. Read the status before citing anything |
| `CHANGELOG.md` | what changed in this folder and in the tree it describes, one line each |
| `GLOSSARY.md` | the vocabulary: `EOT`, `relaxed iOT`, `exact iOT`, `F`, `S`, `B`, `rel`, `rel_eps`, `δ`, `Ψ` |
| `../notes/` | ideas and plans that are **not** results: the applications note, the levers plan. Nothing there may be cited as a number |
| `INVENTORY.md` | housekeeping view: file counts, records, remaining fixes |

## 2. The reports

Columns: **state** is one of `CURRENT` / `PARTIAL` / `HISTORICAL` / `SCOPE ONLY`.
Where a report corrects or replaces another, the correcting report's state column names it.

### 2.1 EOT — the entropic inverse-OT master (`current/eot/`)

Formerly "T3". The directory index (`current/eot/README.md`) carries the definitions, the P1/P2/P3
status and the "which verdict wins" order.

| report | data | question | state | record |
| --- | --- | --- | --- | --- |
| `current/eot/README.md` | synthetic | **index** for the family; what EOT is and what is open | CURRENT | — |
| `current/eot/eot_gate_b.md` | synthetic | does the joint formulation have the right optimum? | CURRENT (formulation only) | `out/eot_joint_reference/` |
| `current/eot/eot_solver_compare.md` | synthetic | SLSQP reference vs custom Newton-CG | CURRENT — recommendation later reversed by the scale study | `out/eot_solver_compare/` |
| `current/eot/eot_scaling_slsqp.md` | synthetic | where dense SLSQP stops being practical (`n ≤ 1500`) | CURRENT for the tables; two numbers retracted in `eot_restricted_edges.md` | `out/eot_scaling_slsqp/` |
| `current/eot/eot_restricted_edges.md` | synthetic | does a restricted `F` stay cheap **and** approximate the full grid? | CURRENT — read last | `out/eot_restricted_edges/` |
| `current/eot/eot_inner_observations.md` | synthetic | conditioning, sparsity, sweep counts of the inner solve | PARTIAL — measurement only | `out/eot_inner_observations/` |
| `current/eot/PLAN.md` | — | the user's binding P1/P2/P3 order, with progress markers | plan | — |

**Not written up** (drivers exist, records do or do not): `eot_planted.py` (no record),
`eot_p1_reference.py` (no record), `eot_edge_fraction.py` (record in `out/eot_scaling/`, no report),
`eot_bias.py` (record in `out/eot_bias/bias.json`, surfaced here for the first time — the verdict is in
`current/eot/README.md` §1 row 5).

### 2.2 Real data — T1/T2 pairs (`current/`)

| report | data | question | state | record |
| --- | --- | --- | --- | --- |
| `current/REAL_PAIRS_REPORT.md` | real | does a learned cost recover hidden partners on a 700-pair pool? | CURRENT, but its generating JSON is **missing** (see §3) | `out/pool_emd/` — absent |
| `current/NMI_PAIRING_REPORT.md` | real | NMI + Hungarian vs cosine / OT vs the learned cost, 1000 pairs | CURRENT — a reference row only, the benchmark is saturated | `out/nmi/` — **absent** |
| `current/synthrad_recovery.md` | real | SynthRAD pair recovery, 15/15 patients | CURRENT — solver later noted as `fit_full_lp` | `out/records/synthrad_probe/` |
| `current/synthrad_plan_v2.md` | real | the agreed evaluation protocol (yardstick = `(patient, z)`) | CURRENT protocol | — |
| `current/SANITY_CHECKS_REPORT.md` | real + synthetic | the five sanity hypotheses: what existed, what is now run, and where | CURRENT — checks A, B, E run (B recovered from log) | `out/sanity*/` |
| `current/FAUST_PRELIMINARY_REPORT.md` | real (3D) | FAUST shape correspondence: do the pre-flight gates pass? | CURRENT — gates 6.0–6.3 **pass**; preliminary only | `out/faust_data_check/`, `out/faust_gates/` |
| `current/FAUST_CORRESPONDENCE_REPORT.md` | real (3D) | does a `θ` learned on one pair transfer to held-out pairs? | CURRENT — **yes, 24/24 pair-wins**; learned captures ≈75 % of the oracle gain | `out/faust_correspondence{,_b,_c}/` |
| `current/PAIRING_GATE_REPORT.md` | real (3D) | does a balanced transport plan beat per-point `argmin` on a given cost? | CURRENT — **yes, 15–71 %**, with the mechanism measured | `out/faust_pairing_gate{,_learned}/` |
| `current/NSR_ARM_REPORT.md` | real (3D) | why we used FAUST, and why the NSR/NICP vehicle was abandoned | CURRENT — **negative result**; records the structural blocker and the unreproducible submission path | `out/nsr_gate/`, server-side |
| `current/E5_SCOPE.md` | real | pseudo-pair synthesis with the N=1 trainer | SCOPE ONLY — never run | — |

### 2.3 Other current reports

| report | data | question | state | record |
| --- | --- | --- | --- | --- |
| `current/SERVER_TESTS_REPORT.md` | synthetic | consolidated A100 runs of the GPU-only path | CURRENT, tabulated only | server-side, patched paths below |
| `current/hierarchy_warm_start.md` | real | does coarse-to-fine warming pay for inverse OT? | CURRENT, **negative** (relaxed variant) | `out/synthrad_stage0/`, `out/synthrad_stage1/` |

### 2.4 Components — one backend or algorithm each

| report | state |
| --- | --- |
| `components/CUPDLPX_MASTER_REPORT.md` | PARTIAL — the adapter is current, the measurements were vs HiGHS |
| `components/HELLO_SCAN_REPORT.md` | CURRENT for diagonal features only; other families route to the generic torch scan |
| `components/HELLO_SOLVER_REPORT.md` | HISTORICAL — HELLO as an inner OT solver, a different method |

### 2.5 Superseded — kept for provenance, never cited

Every file carries a `Status: HISTORICAL` banner naming what replaced it.

`superseded/SCALING_REPORT.md`, `MARGINAL_REPORT.md`, `FAMILY_REPORT.md`, `BENCHMARK_EXACT_REPORT.md`,
`MEASUREMENT_NOTES.md` (synthetic, retired CPU path: NumPy scan + HiGHS master + POT truth),
`superseded/synthrad_gate.md` (replaced by `current/synthrad_recovery.md`).

### 2.6 Runs that exist as data but not as a report

These are in `../out/` and nowhere in this folder. They are listed so they are not mistaken for never
having run.

| path | what it is | state |
| --- | --- | --- |
| `out/t1t2_quad_f2/` | quadratic (Mahalanobis) iOT in F2's grid | RAN, EXCLUDED — 0/9 fits met the acceptance rule (`EXPERIMENTS.md` #17) |
| `out/t1t2_coupled/` | coupled family `(θ, η)` | BLOCKED, one log only (`EXPERIMENTS.md` #18) |
| `out/entropic_metric/` | entropic EOT on T1/T2, K ∈ {16,32} | PARTIAL — see `studies/iot/real/t1t2/entropic_metric/README.md` §Corrections |
| `out/entropic_metric_matched/`, `out/entropic_metric_loweps/` | variants of the above | no write-up |
| `out/entropic_vs_relaxed/` | EOT vs relaxed LP head-to-head, K=64 | **PARTIAL — stopped after m=50 and m=100** |
| `out/entropic_vs_relaxed_m300/` | same at m=300 | **PARTIAL — stopped mid-grid (K=64, draw 1)** |
| `out/synthrad_loop_quad/`, `out/synthrad_quad/` | quadratic iOT on SynthRAD | records exist, no report |
| `out/synthrad_phase_a/`, `..._b/`, `..._c/` | loop vs dense, loop scaling | folded into `STATE.md` §3 |
| `out/synthrad_stage0/`, `..._stage1/` | hierarchy warming | folded into `current/hierarchy_warm_start.md` |

## 3. Numbers with no record behind them

A number may be cited only if its record exists. These are the known gaps:

| report | cited data | problem |
| --- | --- | --- |
| `current/REAL_PAIRS_REPORT.md` | `out/pool_emd/pool_emd2.json` | **absent**; the report says the generating JSON "was never committed" |
| `current/NMI_PAIRING_REPORT.md` | `out/nmi/records.json` | **absent** |
| `components/HELLO_SCAN_REPORT.md` | `out/hello_scan/{scan_speedup,hello_solver}.json` | **absent** |
| `current/synthrad_recovery.md` (F4) | `out/synthrad_v2/`, `out/synthrad_cheap/` | present locally but **git-ignored** |
| `current/REAL_PAIRS_REPORT.md` (F3) | — | markdown only, no record at all |

## 4. Stale paths in `current/SERVER_TESTS_REPORT.md` — read with this mapping

The report records the paths **as they were at run time**. Those modules have since moved, so a reader
searching for them will not find them. Mapping (all confirmed absent from `iot_experiments/`):

| recorded | now |
| --- | --- |
| `iot_experiments/core/{scan,pruning,edges_advance}.py` | `src/iot/core/…` |
| `iot_experiments/core/{planting,planting_sweep,brenier_toy,degenerate_natural}.py` | `iot_experiments/harness/lib/planting.py` + `iot_experiments/legacy/…` (drivers retired) |
| `iot_experiments/gpu_bench/jobs/brenier_ladder.txt` | `iot_experiments/benchmarks/jobs/brenier_ladder.txt` |
| `iot_experiments/out/queue/<ts>/…` | server-side only, synced to `/srv/results/fast-iot/` |

The same mapping applies to the driver paths quoted inside `current/hierarchy_warm_start.md` §1;
its body is otherwise unchanged.

## 5. Records

Committed, in `../out/records/` (a `.gitignore` exception):

| path | contents | cited by |
| --- | --- | --- |
| `../out/records/metric_learning/` | `records.json` (189 rows), `run.log` | `EXPERIMENTS.md` #12 |
| `../out/records/synthrad_probe/` | `records.json` (168 rows), `run.log`, `split_manifest.json`, `metric_gate_{soft,bone}.json` | `STATE.md`, `current/synthrad_recovery.md` |

Everything else quoted in this folder is in `../out/` and **git-ignored** — reproducible only on the
machine that ran it, unless a report says otherwise.

## 6. Regenerating

There is no single regeneration script. Each study owns a driver:

| study | command (from the repository root) |
| --- | --- |
| `metric_learning` (#12, F2) | `PYTHONPATH=. .venv/bin/python -m iot_experiments.studies.iot.real.t1t2.metric_learning.run --ks 64 128 256 --ms 50 100 300` |
| EOT studies | `PYTHONPATH=. .venv/bin/python -m iot_experiments.studies.iot.real.t1t2.metric_learning.eot_<name>` (see `current/eot/README.md`) |
| SynthRAD phases | `iot_experiments/studies/iot/real/synthrad/phase_{b,c}.py` (see `STATE.md` §7) |
| entropic metric | `iot_experiments/studies/iot/real/t1t2/entropic_metric/run.py` (its README) |
| hierarchy | `iot_experiments/studies/iot/real/synthrad/stage{0,1}_*.py` |

## 7. Figures

`../figures/make_figures.py` builds F2–F5 from saved records, never by recomputation. **No figure
currently has its `<name>.caption.md`** and F3/F4 are built from records that are missing or
git-ignored, so under `AGENTS.md` §7.2 they are not yet citable. Treat the figures as a to-do, not as
published results.

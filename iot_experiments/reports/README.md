# reports

Written results for the inverse-OT work. **This folder is Markdown only.** Machine-written records
(`records.json`, run logs, manifests) live in `../out/records/`, which is a *committed* exception to
the otherwise git-ignored `out/` tree — see `.gitignore`.

The `data` column says whether a result is about **real images** (`studies/real/`) or a **synthetic**
instance (`studies/synthetic/`). The two are never mixed in one table.

## Start here

| file | what it is |
| --- | --- |
| `STATE.md` | goal, split, verified results with paths, decisions, backlog, resource rules |
| `EXPERIMENTS.md` | the flat list of every experiment attempted, with its status |
| `INVENTORY.md` | what is in this folder and how it is organised |

## `current/` — cite these

| report | data | question |
| --- | --- | --- |
| `SERVER_TESTS_REPORT.md` | synthetic | consolidated A100 runs of the GPU-only path |
| `REAL_PAIRS_REPORT.md` | real | T1/T2 pool recovery, controls, baselines (source of F3) |
| `NMI_PAIRING_REPORT.md` | real | NMI + Hungarian vs cosine / OT vs the learned cost, all 1,000 pairs |
| `synthrad_recovery.md` | real | SynthRAD pair recovery against ground truth (source of F4) |
| `synthrad_plan_v2.md` | real | the agreed evaluation protocol |
| `E5_SCOPE.md` | real | scope only — the experiment was never run |

## `components/` — one backend or algorithm each

| report | status |
| --- | --- |
| `CUPDLPX_MASTER_REPORT.md` | partial — adapter current, measurements were against HiGHS |
| `HELLO_SCAN_REPORT.md` | current for diagonal features; any other family routes to the generic scan |
| `HELLO_SOLVER_REPORT.md` | historical — HELLO as an inner exact-OT solver, a different method |

## `superseded/` — kept for provenance, never cited

Every file here carries a `Status: HISTORICAL` banner. A row from any of them must never be mixed
into a table from `current/`.

`SCALING_REPORT.md`, `MARGINAL_REPORT.md`, `FAMILY_REPORT.md`, `BENCHMARK_EXACT_REPORT.md`,
`MEASUREMENT_NOTES.md` — synthetic studies on the retired CPU path (NumPy scan + HiGHS master + POT
truth). `synthrad_gate.md` — the earlier compatibility test, replaced by `synthrad_recovery.md`.

## Records

All cited machine-written data is in `../out/records/`:

| path | contents | cited by |
| --- | --- | --- |
| `../out/records/metric_learning/` | `records.json` (189 rows), `run.log` | F2 |
| `../out/records/synthrad_probe/` | `records.json` (168 rows), `run.log`, `split_manifest.json`, `metric_gate_{soft,bone}.json` | `STATE.md`, `synthrad_recovery.md` |

Two figures still fail the "reproducible from a clone" test and are tracked in `INVENTORY.md` §5:
**F3** (its generating JSON was never committed) and **F4** (its records are still in git-ignored
`out/synthrad_v2/` and `out/synthrad_cheap/`).

## Regenerating

Current studies own their own drivers under `iot_experiments/studies/<axis>/<study>/`. Each study's
`README.md` states its exact command; there is no single regeneration script, and an earlier version
of this file listed stale paths that no longer exist.

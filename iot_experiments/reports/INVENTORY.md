# What is under `reports/`

An inventory of `iot_experiments/reports/` — what each file is, what it is for, and what is wrong
with the current layout. Read this together with `EXPERIMENTS.md` (the list of experiments).

The rule this folder exists to enforce: **a number belongs in a report only if its config
(instance, `K`, seed, budget, code path) is stated in the same report.** Raw data belongs in
`iot_experiments/out/` (git-ignored); what is here is what we are prepared to defend.

---

## 1. Current count

| kind | count | where |
|---|---|---|
| Orientation | 4 | `reports/{README,STATE,EXPERIMENTS,INVENTORY}.md` |
| Current result reports | 6 | `reports/current/` |
| Component reports | 3 | `reports/components/` |
| Superseded reports | 6 | `reports/superseded/` |
| **Committed data records** | 7 files, ~540 kB | `iot_experiments/out/records/` (**not** this folder) |
| Retired reports | 12+ | `iot_experiments/legacy/reports/` |

`reports/` is Markdown only. Data was moved out on 2026-10-30.

## 2. The files, grouped by role

### 2.1 Orientation — read these first

| file | role |
|---|---|
| `README.md` | the index: which reports are current, which are historical, and why. Written before the latest work; **does not yet list `EXPERIMENTS.md`** |
| `STATE.md` | durable state: goal, split, verified results with paths, decisions, backlog, resource rules. The most important file in the folder |
| `EXPERIMENTS.md` | **new**: the flat list of every experiment, with status |
| `INVENTORY.md` | **new**: this file |

### 2.2 Current result reports

| file | data | question |
|---|---|---|
| `current/SERVER_TESTS_REPORT.md` (377 lines) | synthetic | consolidated A100 runs of the GPU-only path — the largest report here |
| `current/REAL_PAIRS_REPORT.md` (177) | real | T1/T2 pool recovery, controls, baselines. **Source of F3, and it is Markdown-only: the generating JSON was never committed** |
| `current/NMI_PAIRING_REPORT.md` (144) | real | NMI + Hungarian vs cosine/OT vs the learned cost, all 1,000 pairs |
| `current/synthrad_recovery.md` (128) | real | SynthRAD pair recovery against ground truth — **source of F4's report side** |
| `current/synthrad_plan_v2.md` (88) | real | the agreed evaluation protocol (yardstick = ground-truth `(patient, z)`) |
| `current/E5_SCOPE.md` (77) | real | scope only; the experiment was never run |

### 2.3 Component / backend reports

`components/CUPDLPX_MASTER_REPORT.md` (master LP on GPU, partial — measurements were vs HiGHS),
`components/HELLO_SCAN_REPORT.md` (GPU separation scan, diagonal-only scope), `components/HELLO_SOLVER_REPORT.md`
(HELLO as inner OT solver, historical), `superseded/BENCHMARK_EXACT_REPORT.md` (frozen `PROTOCOL.md`
instances, historical), `superseded/MEASUREMENT_NOTES.md` (RSS hygiene, historical).

### 2.4 Superseded but retained

`superseded/synthrad_gate.md` — the earlier compatibility test, superseded by `current/synthrad_recovery.md`.
`superseded/SCALING_REPORT.md`, `superseded/MARGINAL_REPORT.md`, `superseded/FAMILY_REPORT.md` — synthetic studies on the retired
CPU path (NumPy scan + HiGHS master + POT truth). Each carries a `Status:` banner.

### 2.5 Committed data records

Users often assume `reports/` is Markdown only. It is not:

| path | contents | used by |
|---|---|---|
| `out/records/metric_learning/records.json` | **189 rows**, arms `cosine/infonce/sinkhorn/cca/iot/residual16/residual64`, `k ∈ {64,128,256}`, `m ∈ {50,100,300}`, 3 draws | **F2** |
| `out/records/metric_learning/run.log` | the run's own log | provenance |
| `out/records/synthrad_probe/records.json` | **168 rows**, arms `cosine_ot/learned/nmi/pixel/random/oracle/mask_iou/z_only` | `STATE.md`, `current/synthrad_recovery.md` |
| `out/records/synthrad_probe/split_manifest.json` | the FIT/DEV/TEST split | the SynthRAD studies |
| `out/records/synthrad_probe/metric_gate_{soft,bone}.json` | pixel-baseline gate readings | `STATE.md` §3.5 |
| `out/records/synthrad_probe/run.log` | probe log | provenance |

## 3. What is wrong with the current layout

1. **Two kinds of file in one folder.** Reports (prose, hand-written) and data records
   (`records.json`, machine-written) sit side by side. Nothing distinguishes them except the
   extension, and `README.md` does not mention the data files at all.
2. **`README.md` is stale.** It lists `metric_learning/` and `synthrad_probe/` as if they were
   reports; it does not list `EXPERIMENTS.md`, `STATE.md` is described only as "START HERE", and
   the "Regenerating" block references drivers that have since moved.
3. **Mixed lifecycle in one flat namespace.** Current, historical and superseded `.md` files are
   interleaved. The `Status:` banner convention exists but is applied inconsistently — e.g.
   `current/REAL_PAIRS_REPORT.md` has no banner but its source JSON is missing.
4. **A figure can depend on a file that is not here.** F4's records live in `iot_experiments/out/`
   and are **git-ignored**; F3's generating JSON is **gone entirely**. A fresh clone can regenerate
   F2 but not F3 or F4.
5. **`synthrad_probe/records.json` contains dropped arms.** It still holds `mask_iou`, `z_only` and
   `oracle`, which were retired as tautological or non-competitors. Nothing marks them as retired
   inside the file, so a reader can still pick them up.

## 4. Organisation

**Done on 2026-10-30.** `reports/` is now Markdown only, split into `current/`, `components/`,
`superseded/`; the data moved to `iot_experiments/out/records/`.

```
reports/
├── README.md              # index only: what is current, what is historical, one line each
├── EXPERIMENTS.md         # the experiment list (this folder's table of contents by question)
├── INVENTORY.md           # this file
├── current/               # reports we will cite
│   ├── SERVER_TESTS_REPORT.md
│   ├── REAL_PAIRS_REPORT.md
│   ├── NMI_PAIRING_REPORT.md
│   ├── synthrad_recovery.md
│   ├── synthrad_plan_v2.md
│   └── E5_SCOPE.md
├── components/            # one backend / one algorithm each
│   ├── CUPDLPX_MASTER_REPORT.md
│   ├── HELLO_SCAN_REPORT.md
│   └── HELLO_SOLVER_REPORT.md
├── superseded/            # kept, banner required, never cited
│   ├── synthrad_gate.md
│   ├── SCALING_REPORT.md
│   ├── MARGINAL_REPORT.md
│   ├── FAMILY_REPORT.md
│   ├── BENCHMARK_EXACT_REPORT.md
│   └── MEASUREMENT_NOTES.md
```

**Data does not live here.** Machine-written records are in `iot_experiments/out/records/`, which
`.gitignore` tracks as an exception to the otherwise ignored `out/` tree:

```
iot_experiments/out/records/
├── metric_learning/{records.json,run.log}
└── synthrad_probe/{records.json,run.log,split_manifest.json,metric_gate_*.json}
```

Launchers are documented in `iot_experiments/scripts/README.md` (local `run_job.sh` / `safe_run.sh` /
`wipe_cache.sh`, and the A100 `legacy/gpu_bench/` set, **which does not currently import**).

**Rules that go with it:**

- `README.md` becomes a pure index: one line per current report, no tables of results, no stale
  regeneration commands.
- Every file under `current/` states its config, its source record path, and how to regenerate it.
- Every file under `superseded/` carries a `Status: HISTORICAL` banner and the name of the current
  report that replaced it.
- `data/` holds only records that are committed and cited. Records that a figure needs must be
  copied here from `out/` before the figure is built — otherwise the figure is not reproducible
  from a clone (F3 and F4 both fail this today).
- Retiring an arm inside a record means recording the retirement next to the data, not only in prose.

## 5. Remaining actions, in order of value

1. **Restore `legacy/gpu_bench/` to a runnable state.** It is the only path to the A100 results and
   it cannot be imported at all today (`ModuleNotFoundError: iot_experiments.gpu_bench`). See
   `../scripts/README.md` §2.
2. **Commit a `records.json` for F3** (or regenerate it) — the only figure whose numbers cannot be
   reproduced at all.
3. **Copy F4's records** out of `out/synthrad_v2/` and `out/synthrad_cheap/` into `out/records/`.
4. **Relabel F2 and F4's iOT arms** as `fit_full_lp`-computed until loop versions exist (see
   `EXPERIMENTS.md`, caveat 1).
5. **Add `Status: HISTORICAL` banners** to the six files now in `superseded/` that lack one.

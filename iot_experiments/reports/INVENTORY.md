# What is under `reports/`

A housekeeping view: what files exist, what each one is for, and what is still wrong. The **single
source of truth for "what is where"** is `README.md` §2 — this file does not repeat the table, it
counts and audits it.

The rule this folder exists to enforce: **a number belongs in a report only if its config
(instance, `K`, seed, budget, code path) is stated in the same report, and its record exists.**
Raw data belongs in `iot_experiments/out/` (git-ignored) — what is here is what we are prepared to
defend.

**Last audited 2026-11-05.**

---

## 1. Current count

| kind | count | where |
|---|---|---|
| Orientation | 5 | `reports/{README,STATE,EXPERIMENTS,INVENTORY,CHANGELOG}.md` + `GLOSSARY.md` |
| Current result reports | 3 | `reports/current/*.md` (T1/T2 + SynthRAD) |
| **EOT reports** | 6 + plan | `reports/current/eot/` |
| Other current | 1 | `reports/current/hierarchy_warm_start.md` |
| Component reports | 3 | `reports/components/` |
| Superseded reports | 6 | `reports/superseded/` (all banner-carrying) |
| **Committed data records** | 7 files, ~540 kB | `iot_experiments/out/records/` (**not** this folder) |
| Retired reports | 12+ | `iot_experiments/legacy/reports/` |

`reports/` is Markdown only. Data was moved out on 2026-10-30. The five `T3_*.md` / `HIERARCHY_*`
files that used to sit at the top of this folder were moved into `current/` on 2026-11-05
(`CHANGELOG.md`).

## 2. The files, grouped by role

### 2.1 Orientation

| file | role |
|---|---|
| `README.md` | **the index** — every report once, with question, state, record, supersession. Start here |
| `STATE.md` | durable state: goal, split, verified results with paths, decisions, backlog, resource rules |
| `EXPERIMENTS.md` | the flat list of every experiment (41 items), with its status |
| `CHANGELOG.md` | organisational changes, newest first |
| `GLOSSARY.md` | definitions of `EOT`, `relaxed iOT`, `rel`, `rel_eps`, `F`, `S`, `B`, `hit`, … |
| `INVENTORY.md` | this file |

### 2.2 Current result reports

| file | data | question |
|---|---|---|
| `current/SERVER_TESTS_REPORT.md` | synthetic | consolidated A100 runs of the GPU-only path. Driver paths inside are stale — `README.md` §4 |
| `current/REAL_PAIRS_REPORT.md` | real | T1/T2 pool recovery, controls, baselines — **markdown only, no record** |
| `current/NMI_PAIRING_REPORT.md` | real | NMI + Hungarian vs cosine/OT vs the learned cost — **record absent** |
| `current/synthrad_recovery.md` | real | SynthRAD pair recovery against ground truth |
| `current/synthrad_plan_v2.md` | real | the agreed evaluation protocol |
| `current/E5_SCOPE.md` | real | scope only; never run |
| `current/hierarchy_warm_start.md` | real | coarse-to-fine warming — negative for the relaxed variant |
| `current/eot/*.md` | synthetic | the EOT family — index in `current/eot/README.md` |

### 2.3 Component / backend reports

`components/CUPDLPX_MASTER_REPORT.md` (adapter current, measurements historical vs HiGHS),
`components/HELLO_SCAN_REPORT.md` (diagonal-only scope), `components/HELLO_SOLVER_REPORT.md`
(historical).

### 2.4 Superseded but retained

`superseded/SCALING_REPORT.md`, `MARGINAL_REPORT.md`, `FAMILY_REPORT.md`, `BENCHMARK_EXACT_REPORT.md`,
`MEASUREMENT_NOTES.md` (retired CPU path), `synthrad_gate.md` (replaced by `current/synthrad_recovery.md`).
All six now carry a `Status: HISTORICAL` banner.

### 2.5 Committed data records

| path | contents | used by |
|---|---|---|
| `out/records/metric_learning/records.json` | 189 rows, 7 arms, `k ∈ {64,128,256}`, `m ∈ {50,100,300}`, 3 draws | F2 |
| `out/records/metric_learning/run.log` | provenance | — |
| `out/records/synthrad_probe/records.json` | 168 rows | `STATE.md`, `current/synthrad_recovery.md` |
| `out/records/synthrad_probe/split_manifest.json` | the FIT/DEV/TEST split | SynthRAD studies |
| `out/records/synthrad_probe/metric_gate_{soft,bone}.json` | pixel-baseline readings | `STATE.md` §3.5 |
| `out/records/synthrad_probe/run.log` | provenance | — |

**The EOT records are not committed.** `out/eot_*/` is git-ignored, so every EOT number reproduces only
on the machine that ran it. Committing them is the single highest-value fix below.

## 3. What is still wrong

1. **Two reports have no record at all** (`current/REAL_PAIRS_REPORT.md`, `current/NMI_PAIRING_REPORT.md`)
   and one has a git-ignored one (`current/synthrad_recovery.md`). Their numbers cannot be reproduced
   from a clone.
2. **No EOT record is committed**, so the newest results are the least reproducible.
3. **Three EOT drivers have no output and no report**: `eot_planted.py`, `eot_p1_reference.py`,
   and `eot_edge_fraction.py` (record exists, report does not). P1 in `current/eot/PLAN.md` depends on
   the first two.
4. **Two entropic-vs-relaxed runs stopped mid-grid** (`out/entropic_vs_relaxed*`); no report.
5. **No figure has a `.caption.md`** (§`AGENTS.md` 7.2), and F3/F4's sources are missing or ignored.
6. **`synthrad_probe/records.json` still holds retired arms** (`mask_iou`, `z_only`, `oracle`); nothing
   inside the file marks them as retired.
7. **`STATE.md` is stale on the EOT side** — it does not mention EOT at all (it predates it). It is
   still the best document for SynthRAD/T1-T2 state.

## 4. Organisation

```
reports/
├── README.md              # the index: every report once
├── STATE.md  EXPERIMENTS.md  CHANGELOG.md  GLOSSARY.md  INVENTORY.md
├── current/
│   ├── SERVER_TESTS_REPORT.md  REAL_PAIRS_REPORT.md  NMI_PAIRING_REPORT.md
│   ├── synthrad_recovery.md  synthrad_plan_v2.md  E5_SCOPE.md
│   ├── hierarchy_warm_start.md
│   └── eot/               # EOT family (formerly "T3")
│       ├── README.md      #   index + P1/P2/P3 status + order of authority
│       ├── PLAN.md        #   the user's binding P1/P2/P3 order
│       ├── eot_gate_b.md  eot_solver_compare.md  eot_scaling_slsqp.md
│       ├── eot_restricted_edges.md  eot_inner_observations.md
├── components/            # one backend / one algorithm each
└── superseded/            # banner required, never cited
```

**Rules that go with it:**

- `README.md` is a pure index: one row per report, no tables of results, no regeneration recipes
  beyond one line.
- Every report states `Status:`, its driver, its record path, and `Supersedes` / `superseded by`.
- Every file under `superseded/` carries the banner and the name of its replacement.
- A record a figure needs must be copied to `out/records/` before the figure is built.
- Retiring an arm inside a record means recording the retirement next to the data, not only in prose.

## 5. Remaining actions, in order of value

1. **Commit the EOT records** (`out/eot_*/*.json`, ~0.3 MB total) into `out/records/eot_*/`.
2. **Commit a `records.json` for F3** (or regenerate it) and for `NMI_PAIRING_REPORT.md`.
3. **Write `eot_p1_reference.md` and `eot_edge_fraction.md`**, or mark them "run, not kept" in
   `current/eot/README.md` (already done there) and delete the dead drivers.
4. **Copy F4's records** out of `out/synthrad_v2/` and `out/synthrad_cheap/` into `out/records/`.
5. **Add the four figure captions**, or drop F3/F4.
6. **Add an EOT section to `STATE.md`** (or an explicit pointer that EOT state lives in
   `current/eot/README.md`).

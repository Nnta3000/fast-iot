# iot_experiments

Experiments for inverse OT and for the OT solvers it uses. Run from the repository root after
`pip install -e .`.

**This file is the index.** If you are looking for "the current code", it is
`harness/` (shared machinery) plus `studies/` (the science).

## The split that matters: real data vs synthetic

Experiments answer two different kinds of question, and the directories say which:

| directory | data | question |
|---|---|---|
| **`studies/iot/real/`** | **real data** (T1/T2 MRI, SynthRAD MR/CT, FAUST, NYC taxi) | does inverse OT recover the right partner or cost? |
| **`studies/iot/synthetic/`** | **generated in-process** | does the solver solve it, how fast, does it certify? |
| **`studies/ot/`** | **generated point clouds** | HELLO-OT component cost: scan, inner solver, CuPDLPx master |
| **`studies/flow_matching/`** | toy generative data | does an exact/global coupling beat minibatch OT at equal steps? |
| `benchmarks/` | frozen instances | cross-method comparison under a fixed protocol |
| `harness/` | — | shared machinery: config, preprocessing, splits, records, metrics, gates |

Everything else is support or history: `reports/` (what to cite), `theory/` (proofs), `out/`
(run results, git-ignored), `legacy/` (retired code, git-ignored).

## Directory map

```
iot_experiments/
├── harness/                     shared machinery — the only place preprocessing/splits/records live
├── studies/
│   ├── iot/
│   │   ├── real/                t1t2/, synthrad/, faust/, nyc_taxi/
│   │   └── synthetic/           scaling/, marginals/, families/, methods/,
│   │                            multi_observation_sparse/, planted_shared_theta/,
│   │                            inverse_cost_recovery/
│   ├── ot/                      hello_scan_benchmark, hello_solver_benchmark,
│   │                            cupdlpx_master_benchmark
│   └── flow_matching/           reserved — no run yet
├── benchmarks/                  frozen cross-method contract (PROTOCOL.md)
├── theory/                      HierarchicalIOT.typ
├── reports/                     citable results; reports/README.md is the index
├── out/                         ALL run output (git-ignored) — studies write iot_experiments/out/<study>/
└── legacy/                      retired: the frozen core/ tree, gpu_bench/, external/  (git-ignored)
```

**`out/` is the only place results are written.** Drivers do not carry their own `out/`, so you
always know where to look. Each run also writes its own `run.log` (timestamped, per-step elapsed)
next to its `records.json` — a long job that is launched with plain `python -u` still leaves a record.

## What each study answers

### `studies/iot/real/`

| study | question | report |
|---|---|---|
| `t1t2/geometry` | which post-PCA transform makes retrieval work (G0–G3)? | `current/REAL_PAIRS_REPORT.md` |
| `t1t2/pool` | which of the selected pairs can be trusted (precision at coverage)? | `current/REAL_PAIRS_REPORT.md` |
| `t1t2/nmi` | NMI + Hungarian vs cosine + OT vs the learned cost | `current/NMI_PAIRING_REPORT.md` |
| `t1t2/metric_learning` | InfoNCE / Sinkhorn-InfoNCE / CCA / residual vs relaxed iOT | `studies/iot/real/t1t2/metric_learning/README.md` |

### `studies/iot/synthetic/`

| study | question | report |
|---|---|---|
| `iot/synthetic/scaling` | where does the solve exceed 2 s in `(m, n, K)`? | `superseded/SCALING_REPORT.md` |
| `iot/synthetic/marginals` | uniform vs Gaussian / GMM marginals: structure and cost | `superseded/MARGINAL_REPORT.md` |
| `iot/synthetic/families` | diagonal / quadratic / low-rank (+ tiny-`n` PSD) | `superseded/FAMILY_REPORT.md` |
| `iot/synthetic/multi_observation_sparse` | one shared cost from many sparse plans, memory-linear | `notes/MULTI_OBSERVATION_SPARSE_REPORT.md` |
| `iot/synthetic/planted_shared_theta` | shared-cost recovery with a planted same-zone/offset control | `reports/current/` |
| `iot/synthetic/inverse_cost_recovery` | relaxed arm on Ma et al.'s cost-recovery benchmark | `reports/current/` |

### `studies/ot/`

| study | question | report |
|---|---|---|
| `hello_scan_benchmark.py` | is the torch scan bit-identical and how much faster? | `reports/components/HELLO_SCAN_REPORT.md` |
| `hello_solver_benchmark.py` | HELLO vs POT as the inner exact-OT solver | `reports/components/HELLO_SOLVER_REPORT.md` |
| `cupdlpx_master_benchmark.py` | CuPDLPx vs HiGHS on the master LP | `reports/components/CUPDLPX_MASTER_REPORT.md` |

### `studies/flow_matching/`

Reserved for the flow-matching coupling study; no run exists yet, so no number may be cited.

These three are driven by `harness/lib/` generators (`planting`, `marginals`, `families`) and
`harness/lib/runner.py`; every record carries a full `RunSpec`, so results from different studies
merge and cite directly.

## Running

```bash
# real-data study (needs data/, see data/README.md)
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot \
  .venv/bin/python -u -m iot_experiments.studies.iot.real.t1t2.nmi \
    --ks 256 --ms 100 300 --draws 0 1 2 --solver cupdlpx

# synthetic study
PYTHONPATH=/home/agent/Documents/dpm-inverse-ot \
  .venv/bin/python -u -m iot_experiments.studies.iot.synthetic.scaling \
    --family quadratic --marginal gmm --ms 200 800 3200
```

Output lands in `iot_experiments/out/<study>/` with its `run.log`.

## The rule: no one-off drivers

**Every experiment is a study, and every study has the same shape:**

```
studies/<axis>/<study>/
├── README.md     what question it answers, the exact command, the result
├── run.py        the driver: config + one fit call + records. Nothing else.
└── analyse.py    optional: turns records into tables/figures
```

A driver may be **short**. It must not contain:

- its own data preamble (load split, discover, stream, reduce, subset) — that is
  `harness/datasets.py`;
- its own solver assembly (`plan`, `theta_set`, `RegretMasterSolver`, `fit_full_lp`) — that is
  `harness/fit.py`;
- its own JSON schema — that is `harness/records.py`.

If answering a question would need a new file, first ask whether it is a **new axis value** for an
existing study (`--family`, `--marginal`, `--k`, `--engine`). It usually is.

**Never write a second file to re-ask a question an existing file already asks.** That is how
`quadratic.py`, `loop_quadratic.py`, `quad_task.py` and `quadratic_f2.py` came to exist — four
files, one question. Files that do this are retired to `legacy/`, not extended.

## Conventions

- **A number belongs in a report only with its config** (instance, `K`, seed, budget, code path,
  backend). If you cannot state the config, the number is not usable (`AGENTS.md` §7.1).
- **Backend is stated in every table**: `cupdlpx` (the production path) or `highs` (reference only).
- **A pairing metric must pass `harness/discriminates.py`** before it is used to evaluate anything:
  it must beat the **same-centre** control, not just the random floor.
- **A learned θ is applied in the space it was fitted in**, asserted in code, not by comment.
- **No silent fallbacks**: a failed fit prints `FAILED` and is excluded; it is never replaced by
  cosine or by uniform θ.
- Negative results are reported plainly.

See `AGENTS.md` for the full contract and `reports/README.md` for what to cite.

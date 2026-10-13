# iot_experiments

Experiments for the inverse-OT work. Run everything from the repository root after
`pip install -e .`. This file is the **index**: what each directory answers and which
driver/report to cite.

## The design: three axes on one record

Every current experiment is a point in a three-axis design, described by one `RunSpec` and
recorded as one `RunRecord` (`core/records.py`):

1. **scale** — growth in `K` (point dimension / feature count), `m`, `n`;
2. **marginal** — `uniform` vs non-uniform (`gaussian`, `gmm`, `gaussian:<c>:<w>`);
3. **cost family** — `diagonal`, `quadratic`, `lowrank` (+ `bilinear-psd` at tiny `n`).

A study is just a grid over these axes plus an analysis. `RunSpec`/`RunRecord` live in
`core/`; studies import them, so results from different studies merge and cite directly
(`AGENTS.md` §7.1: a number is usable only with its config).

| path | what it is |
| --- | --- |
| `core/` | shared axes + `RunSpec`/`RunRecord` + the single `run_spec`/`run_grid` entry point |
| `benchmark/` | **frozen** cross-method contract (`PROTOCOL.md`), generators, evaluation, gradient runner |
| `scaling/` | **study 1** — growth in `K`, `m`, `n` |
| `marginals/` | **study 2** — uniform vs Gaussian / GMM marginals (and a peakedness sweep) |
| `families/` | **study 3** — diagonal / quadratic / low-rank (+ tiny-`n` bilinear-PSD) |
| `external/` | **study 4** — adapters to external benchmarks (placeholder) |
| `reports/` | current written results (see `reports/README.md`) |
| `legacy/` | retired experiments and superseded reports (see `legacy/README.md`) |
| `theory/` | the convergence proof source (`CONVERGENCE_PROOF.typ`) |

## Question → study → driver → report

| question | study | driver | report |
| --- | --- | --- | --- |
| where does the solve exceed 2 s in `(m,n,K)`? | `scaling/` | `scaling/run.py` + `scaling/analyse.py` | `reports/SCALING_REPORT.md` |
| how do marginals change structure / cost? | `marginals/` | `marginals/run.py` + `marginals/analyse.py` | `reports/MARGINAL_REPORT.md` |
| how do cost families differ? | `families/` | `families/run.py` + `families/analyse.py` | `reports/FAMILY_REPORT.md` |
| external datasets | `external/` | (adapter, later) | (later) |
| cross-method comparison (exact vs gradient) | `benchmark/` | `benchmark/runner.py` | `reports/BENCHMARK_EXACT_REPORT.md` |

## Running

```bash
# study 1: scale
python iot_experiments/scaling/run.py --family quadratic --marginal gmm \
  --sizes 200 400 800 1600 3200 --dims 2 4 8 16
python iot_experiments/scaling/analyse.py iot_experiments/scaling/out/quadratic_gmm.json

# study 2: marginals
python iot_experiments/marginals/run.py --families diagonal quadratic lowrank \
  --marginals uniform gaussian gmm --n 800
python iot_experiments/marginals/analyse.py iot_experiments/marginals/out/marginals_n800.json

# study 3: cost families (LP) + tiny-n PSD
python iot_experiments/families/run.py --families diagonal quadratic lowrank \
  --marginal gmm --sizes 200 800 1600
python iot_experiments/families/run.py --families bilinear-psd --marginal gmm --psd-ns 8 16 32
python iot_experiments/families/analyse.py iot_experiments/families/out/families_gmm.json
```

Generated outputs land in each study's `out/` (git-ignored) as `<name>.json` + `<name>.csv`.

## Reading the results

- Every record carries its `RunSpec`; `is_optimal` comes from `iot.validation.is_optimal_plan`
  (exact POT), never from `theta == theta_star`. Discrete data is generically
  non-identifiable, so min/max ranges and distances are diagnostics, never equalities.
- Timing is split into `scaffold_seconds` (POT ground truth, **not** the algorithm's budget)
  and `solve_seconds`, with the per-iteration `mean_master_seconds` / `mean_oracle_seconds`
  split — the bottleneck flips from oracle to master as `K` grows.
- `certified`/`cert_support_residual` flag the known certificate edge case on degenerate
  plans (a spurious `~1e-19` support entry); `is_optimal` is the verdict.
- On noisy/off-support plans the exact method is out of scope (`AGENTS.md` §4.6); do not
  threshold silently.

## Conventions

- One report per question; if superseded, add a `Status: HISTORICAL` banner (or move it to
  `legacy/reports/`) rather than deleting.
- Figures: instance in the title, labelled axes, legend, sibling `<name>.caption.md` + CSV
  (`AGENTS.md` §7.2). Generated outputs go under the study's `out/` (git-ignored).
- `legacy/` is import-only-for-provenance; new code must not import from it.

# WORK IN PROGRESS — harness refactor (temporary file, delete when done)

Written 2026-10-26 so the plan survives context compression. **Delete this file once the
refactor is committed.**

## Task

`iot_experiments/core/` has 41 files, 32 with their own `ArgumentParser`, 264 `add_argument`
calls, ~7,800 lines. Nothing guarantees two drivers share a setting — and this has produced
wrong numbers repeatedly:
- `--out` written to `core/out` in some files and `gpu_bench/out` in others (results never pulled);
- "cosine" meaning two different pipelines (L2-only = R@1 0.270 vs standardise-then-L2 = 0.768);
- `tol_eff` differing between drivers.

Replace with a shared `harness/` + thin `studies/`. Freeze the old tree.

## Decisions (user-confirmed)

1. **Freeze** `core/` wholesale into `iot_experiments/legacy/core/` with a `HISTORICAL` banner.
   Nothing may import from it.
2. **Promote the 9 libraries as-is** (move, do not rewrite), fixing only import paths:
   `costs.py, families.py, marginals.py, planting.py, fractional_planting.py, records.py,
   runner.py, plot_pruning_report.py, __init__.py`.
   The user will read these later once the code is readable again.
3. **Reimplement `geometry` first** as the reference study, with **toy settings** for the diff
   (K ∈ {16,32}, m ∈ {20,40}, 2 draws — seconds not minutes), and **diff its numbers against the
   frozen `geometry_sweep.py`** on identical inputs. Report any mismatch before claiming
   faithfulness.
4. **Strict harness.** Study flag colliding with a base flag → **raise at import**. Unknown
   geometry/solver name → **raise**. `uniform θ ≡ cosine` asserted inside `registry`.
   `matching_hit` asserts the EMD oracle returns 1.0 before emitting any number.
5. Skip draws 3–5 of the geometry study (ties already established; no need to confirm).

## Target layout

```
iot_experiments/
  harness/
    config.py     RunConfig + strict parser (--out/--solver/--inner-solver/--scan/--device/
                  --draws/--seed/--json/--resume), collision guard
    registry.py   features(name) + geometry(name) -> arrays   ← ONLY place preprocessing lives
    solver.py     make_master(name) for cupdlpx/highs, limits, timeouts
    driver.py     run(config) -> list[RunRecord]
    metrics.py    R@1/R@5/median, matching_hit (oracle-asserted), precision_at_coverage
    records.py    RunRecord embedding its config; table writer; verdict evaluator
    io.py         run dirs under gpu_bench/out/<study>/, resume
    lib/          the promoted libraries (costs, families, marginals, planting, ...)
  studies/
    geometry/{config.py, run.py, analyse.py, README.md, out/}
    (later: pool/, encoder/, regret_loop/, baselines/, generators/)
  legacy/core/    the frozen 41 files
```

Study drivers should be ≤ ~60 lines: build config → call harness.

## jaxtyping conventions (user-confirmed)

- **Use `Float[np.ndarray, "..."]` / `Shaped[np.ndarray, "..."]`, NOT bare `Array`.**
  Bare `Array` resolves through **jax, which is not installed** → `ModuleNotFoundError` at
  annotation time. Do **not** add jax as a dependency.
- **Symbolic axes must be meaningful.** If an integer variable of that name is already in scope,
  write `Float[np.ndarray, "K m n"]` (do not rename it in the annotation).
- Only when no variable exists, use an **explicit word**: `"draws m n"` rather than `"t m n"`.
- **If a single letter would not be self-evident, spell it out** even if verbose.
- Keep these axes consistent project-wide:

| axis | meaning |
|---|---|
| `m` | source points / labelled pairs |
| `n` | target points / pool candidates |
| `d` | point dimension |
| `k` | number of features (length of θ) |
| `K` | PCA components |
| `draws` | split draws (explicit) |
| `cycles` | 2×2 block grid (explicit) |
| `blocks` | `grid × grid` pooled patches (explicit) |

Note: repeating a name asserts equality (`"m m"` = square); do not write it by accident.

- `beartype` is installed but **not yet enforced**; do not add `@beartype` during this refactor.

## Order of work

1. Freeze `core/` → `legacy/core/` (+ banner).
2. Promote the 9 libraries into `harness/lib/` (import paths only).
3. Write `harness/` modules, annotated.
4. Write `studies/geometry` with toy settings.
5. Diff against the frozen driver; report.
6. **Ask the user to review `harness/registry.py`** before building further studies —
   that file is where preprocessing lives, i.e. the source of today's contradictions.

## Repo invariants (do not violate)

- `src/hello_ot/**`, `setup.py`, `MANIFEST.in`, `export_experiments/**`, `scripts/**`,
  `docs/**`, `paper_experiments/**` are **read-only**.
- `pyproject.toml`: only additive `"iot*"` include and the `psd` extra.
- Do not touch `AGENTS.md` (personal, git-ignored) unless the user explicitly asks in-session;
  the user asked to ADD the jaxtyping section to it — that is the one permitted edit.
- `pytest tests/iot_*_test.py` must stay green (currently 171 passed, 1 skipped).
- Outputs go under a **pulled** directory: `iot_experiments/gpu_bench/out/<study>/`.

# Cost families — diagonal, quadratic, low-rank (+ tiny-n PSD)

**Status: CURRENT, empirical.** Study 3. Backing data:
`../families/out/families_gmm.{json,csv}`, `../families/out/psd_gmm.{json,csv}`,
`../families/out/psd_uniform.{json,csv}` (git-ignored), from `../families/run.py`.

`bilinear-signed` is **excluded**: `0∈Θ` makes feasibility trivial, so only max-margin is
meaningful. It is a documented structural trap, not a test of the solver.

## LP families (`marginal=gmm`, `point_dimension=4`)

| family | K | n=200 | n=800 | n=1600 | verdict |
|---|---|---|---|---|---|
| diagonal | 4 | 0.10–0.13 s | 0.63–0.84 s | 1.9–2.3 s | all converge |
| quadratic | 10 | 0.17–0.19 s | 0.96–1.00 s | 2.7–2.9 s | all converge |
| lowrank | 8 | 0.13 s | 0.80–0.81 s | 2.2–2.6 s | all converge |

Seeds {0,1}. Every LP run reaches `converged=True` with the generic `gmm` structure
(`|S|≈2n−1`, `p=1–2`, 4–7 iterations). The cost family changes the per-iteration constant
(`K`) but not the qualitative behaviour: the `p=1` fractional regime is what makes the loop
cheap (`MARGINAL_REPORT.md`).

## bilinear-PSD (spectrahedron, `SdpMasterSolver`, `point_dimension=2` → K=4)

| marginal | n=8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| **gmm** (`p=1–2`) | 3 it ✅ | 3 it ✅ | 3 it ✅ | 3–5 it ✅ | 4–10 it ✅ |
| **uniform** (`p=n`) | ✗ 100 it | ✗ 100 it | ✗ 100 it | — | — |

`✅` = `converged=True` and `certified=True`; `✗` = hit the 100-iteration cap (`cert=True`,
`converged=False`). Seeds {0,1}.

**This is the sharpest family result.** The widely reported "bilinear-PSD is a tiny-`n`
prototype" brittleness is, at least here, a **marginal-structure artifact**: under uniform
marginals the plan is a permutation (`p=n`) and the SDP master never certifies; under a
non-uniform marginal the plan is fractional (`p=1`) and the same master converges in a handful
of iterations up to `n=128`. The driver of the difference is the support-graph component count
`p`, not the PSD cost itself (see also `MARGINAL_REPORT.md`).

## Caveats
Two seeds, tiny `n` for PSD (the SDP master is only run to `n=128` here); `is_optimal` not run
in this table — the verdict is `converged` + `certified`, which is weaker than the exact-POT
`is_optimal` check. Regenerate with `../families/run.py`.

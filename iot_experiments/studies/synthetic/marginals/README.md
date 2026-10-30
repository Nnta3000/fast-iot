# marginals — uniform vs non-uniform mass vectors

**Question.** How does the marginal family change the *structure* of the problem and the
solver's behaviour, at fixed cost family and scale?

## Axes held fixed
- cost family (swept here), scale (`--n`), solver config.

## What varies
- the mass vectors `a, b`, defined on the latent axis `z_i=(i+0.5)/n`:
  - `uniform` — the current benchmark default;
  - `gaussian` — a single Gaussian `N(z; 0.5, 0.15²)`;
  - `gmm` — a two-component mixture `0.5·N(0.30,0.07²) + 0.5·N(0.72,0.09²)`;
  - `gaussian:<center>:<width>` — a parameterised Gaussian, for the **peakedness sweep**.

## Run
```bash
python iot_experiments/marginals/run.py --families diagonal quadratic lowrank \
  --marginals uniform gaussian gmm --n 800
python iot_experiments/marginals/analyse.py iot_experiments/marginals/out/marginals_n800.json
```

Outputs land in `out/` (git-ignored).

## What to look for
`n_components` (`p`) is the structural tell: under uniform marginals a generic plan is a
**permutation** (`|S|=n`, `p=n`), while a non-uniform marginal gives a fractional basic plan
(`|S|≈2n-1`, `p=1`). Iterations and wall time track `p`. `cert_support_residual` flags the
known certificate edge case on degenerate plans (a spurious `~1e-19` support entry), not a
solver failure — `is_optimal` is the verdict. See `../reports/superseded/MARGINAL_REPORT.md`.

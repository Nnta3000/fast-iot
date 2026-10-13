# Marginals — uniform vs Gaussian / GMM mass vectors

**Status: CURRENT, empirical.** Study 2. Backing data:
`../marginals/out/marginals_n800.{json,csv}` (git-ignored), from `../marginals/run.py`.

## Config
- families `diagonal` (K=4), `quadratic` (K=10), `lowrank` (K=8);
- `m = n = 800`, seeds {0,1};
- marginals defined on the latent axis `z_i=(i+0.5)/n`:
  `uniform`; `gaussian` = `N(0.5, 0.15²)`; `gmm` = `0.5·N(0.30,0.07²)+0.5·N(0.72,0.09²)`;
- shipped solver path as in `SCALING_REPORT.md`.

## Result (medians over 2 seeds)

| family | marginal | \|S\| | components p | iters | solve s |
|---|---|---|---|---|---|
| diagonal | uniform | 800 | **800** | 17–21 | 4.7–5.9 |
| diagonal | gaussian | 1597–1599 | 1–3 | 4–6 | 0.6–0.9 |
| diagonal | gmm | 1598–1599 | 1–2 | 4–5 | 0.6–0.7 |
| quadratic | uniform | 800 | **800** | 19 | 6.9–7.9 |
| quadratic | gaussian | 1599 | 1 | 5 | 0.8–1.0 |
| quadratic | gmm | 1599 | 1 | 5–6 | 1.1 |
| lowrank | uniform | 800 | **800** | 16–17 | 5.7–5.8 |
| lowrank | gaussian | 1598–1599 | 1–2 | 5–8 | 0.8–1.1 |
| lowrank | gmm | 1598–1599 | 1–2 | 5 | 0.8 |

## What the marginals change
1. **Structure.** Uniform + a generic cost gives a **permutation** plan (`|S|=n`, `p=n`
   components). Any non-uniform marginal gives a **fractional basic** plan (`|S|≈2n−1`, `p=1`).
2. **Cost.** The solver runs `4–8` iterations on the fractional plans vs `16–21` on the
   permutation plans, i.e. **~6–8× faster**. Iterations track `p` (the component count = the
   gauge freedom the connector set must pin), not "Gaussian-ness" per se.
3. **Identifiability is not implied.** `delta` is the same order in both regimes; these runs
   show structure/behaviour, not that one `theta` set is larger.

> Interpretation: the uniform/permutation corner is a **degenerate** instance family; the
> fractional regime is the generic one. "Non-uniform is easier" is therefore a statement about
> this solver's conditioning, not about the inverse problem being easier.

## Certificate edge case
`lowrank` with `gaussian`/`gmm` (seed 1) show `certified=False`: POT returns an exactly
optimal plan with a spurious `~1e-19` support entry, so `dual_optimality_certificate`'s
`support_residual` fires (`4.4e-2`, `6.8e-3`) while dual feasibility is `~1e-12` and
`is_optimal_plan` passes. It is a certificate-hygiene issue on degenerate plans, not a solver
failure (`AGENTS.md` §7.4 candidate fix: ignore numerically-zero `Pi` entries).

## Caveats
Two seeds; `is_optimal` not run here (verdict is `converged` + `certificate`); a peakedness
sweep (`gaussian:<c>:<w>`) is supported by the driver but not tabulated in this revision.

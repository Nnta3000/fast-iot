# families — diagonal, quadratic, low-rank (+ tiny-n PSD)

**Question.** How does the cost family change the solver's cost and verdict, at fixed marginal
and scale?

## Axes held fixed
- marginal (default `gmm`), scale (`--sizes` / `--psd-ns`), solver config.

## What varies
- `diagonal` — `Σ_k θ_k (x_ik − y_jk)²`, `K=d`;
- `quadratic` — `Σ_{a≤b} θ_ab (x_ia−y_ja)(x_ib−y_jb)`, `K=d(d+1)/2`;
- `lowrank` — `Σ_k θ_k U_ik V_jk`, `K` independent of `d`;
- `bilinear-psd` — `−x_iᵀ A y_j`, `A⪰0`, `Tr A=1` (spectrahedron → `SdpMasterSolver`, tiny `n`).

`bilinear-signed` is **excluded on purpose**: `0∈Θ` makes feasibility trivial, so only
max-margin is meaningful. It is a documented structural trap, not a test of the solver.

## Run
```bash
python iot_experiments/families/run.py --families diagonal quadratic lowrank \
  --marginal gmm --sizes 200 800 1600 --point-dimension 4
python iot_experiments/families/run.py --families bilinear-psd --marginal gmm --psd-ns 8 16 32
python iot_experiments/families/analyse.py iot_experiments/families/out/families_gmm.json
```

Outputs land in `out/` (git-ignored).

## Takeaway to date
The `gmm` structure (fractional plan, `p=1`, few iterations) carries over to quadratic and
low-rank. On `bilinear-psd`, the marginals matter qualitatively: under uniform marginals
(`p=n`) the SDP master does not converge at `n≥8`, while under non-uniform marginals (`p=1`)
it converges in a few iterations up to `n=128`. See `../reports/FAMILY_REPORT.md`.

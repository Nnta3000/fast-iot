# EOT inner-solve observations (measurement only)

**Status: PARTIAL — no conclusions drawn.** A measurement log for `eot_inner_observations.py`.
Record: `experiments/out/eot_inner_observations/observations.json` (`m=40`, `K=5`, one run).

**What it measures.** Four things about the EOT inner `(f, g)` solve on a sparse edge set `F = B ∪ R ∪`
(per-row cheapest `k` edges), none of which is visible from a final objective value:

1. **Conditioning** — the eigenvalue spectrum of the `g`-block Hessian after `f` is eliminated, as a
   function of `|F|`, expressed as `range/eps` and `|F|/(m+n)`.
2. **Plan sparsity** — the fraction of `pi` entries below `1e-6 · max(pi)`.
3. **Sweep count** — the Sinkhorn sweeps needed to reach a given marginal error, against the "expected
   number of ties" heuristic `m · range / eps`.
4. **Gradient propagation** — how an inner marginal error propagates to the outer `theta` gradient, i.e.
   how accurate the inner solve must be before `theta` is meaningful.

**Why it exists.** The EOT reports kept saying "the inner solve is fine" without a number that would
predict when it stops being fine. This file is the number. Nothing here is cited as a result in
`eot_*.md`; if a claim needs it, it must be re-derived and pointed at `observations.json` directly.

**Honest limitations.**

- One instance size (`m=40`), `K=5`, one seed. This is a probe, not a sweep.
- It measures the *inner* problem only; the outer master is not exercised.
- `|F|` is built by a fixed heuristic (support + connectors + `k` cheapest per row), not by the adaptive
  loop, so it says nothing about which edges the loop would actually pick.

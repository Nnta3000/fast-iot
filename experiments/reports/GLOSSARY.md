# Glossary

One line per term. If a term is missing, the report using it is responsible for defining it there.

| term | meaning |
| --- | --- |
| **exact iOT** | the constraint-generation solver of `AGENTS.md` §2: maximise the dual margin `delta` on an edge set `F`, add violated edges, `src/iot/exact/`. Production scale path uses the GPU master + GPU scan |
| **relaxed iOT** | `src/iot/relaxed/`: minimise the **regret** `R(θ) = <θ, ĝ(π̂)> − R_uniform`. This is the `iot` arm of `metric_learning` and the solver named `loop_regret` / `fit_full_lp` in older reports. It is **not** max-margin |
| **EOT** | the **entropic** inverse-OT master: minimise `Ψ_F(θ,f,g)` (convex in all three) on an edge set `F`. Formerly called **T3**. Joint Newton-CG is EOT proper; the nested SLSQP form is a reference |
| **`Ψ_F`** | `Ψ_F = <φ̄, θ> − a·f − b·g + ε Σ_{e∈F} exp(σ_e/ε)`, `σ_e = f_i + g_j − θ·φ_e` |
| **`σ`** | the dual score `σ_e = f_i + g_j − c_e(θ)`. In the exact loop an edge is violated when `σ_e + δ > tol` |
| **`δ`** | the max-margin variable of the exact master (`δ ≥ 0`, `δ = 0` means `π̂` is exactly optimal) |
| **`S`** | the support of the observed plan: `S = {(i,j) : π̂_ij > 0}` |
| **`B`** | the mandatory edge set `B = S ∪ R`, where `R` connects the components of `S`. Never pruned |
| **`F`** | the edge set the master is solved on. `|F|/(m+n)` is the usual size statistic |
| **`K`** | `|θ|`, the number of cost features (PCA width times the family) |
| **`m` / `n`** | labelled pairs / pool candidates |
| **`m_eff` / `n_eff`** | source / target counts after empty-row and empty-column pruning |
| **`rel`** | the **box width knob** of the relaxed arm. **Inverse to width**: `θ ∈ [rel/K, 1/(rel·K)]`, so a *smaller* `rel` is a *wider* box (`rel=0.05` is the widest). Chosen by CV on labelled pairs only |
| **`rel_eps` / `ε`** | the **entropic** temperature, scaled: `ε = rel_eps × std(c(θ_uniform))`. Chosen by CV on labelled pairs only, ties to the smaller value |
| **`γ`** | top-`γ` violated edges per row and per column added each round |
| **`τ_prune`** | pruning slack tolerance: an edge may be dropped only with slack above it (pruning is a heuristic; `prune=False` is the exact default) |
| **`hyst` / `hyst+reg`** | the hysteresis pruning strategy and its regularised one-solve variant (`ε=1e-4`, the only pruning setting measured to converge) |
| **`hit`** | `matching_hit`: fraction of pool rows whose exact-EMD partner is the true one. Square pool, uniform marginals, columns shuffled (so it is draw-insensitive) |
| **`R@1`** | top-1 retrieval rate per row, identity target. **Capped by hubness** and not interchangeable with `hit` |
| **`exact_hit` / `patient_hit`** | SynthRAD scoring: same `(patient, z)` / same `patient` |
| **`F2`, `F3`, `F4`, `F5`** | figure numbers in `EXPERIMENTS.md` §5, not studies |
| **`excluded`** | a run that failed an acceptance rule; its numbers must not be used |

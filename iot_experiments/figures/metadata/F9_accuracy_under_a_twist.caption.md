# F9_accuracy_under_a_twist

**axes:** x = twist amplitude σ as a fraction of π applied to the target; y = miss@5 %, the share of vertices whose geodesic error exceeds 5 % of √area.
**instance:** MPI-FAUST, n = 1500, one pair, one seed, K = 13, σ ∈ {0, 0.25, 0.5, 0.75, 1}
**methods:** learned θ (fitted once at σ = 0), equal weights, late-HKS only, position only
**config:** MPI-FAUST, n = 1500 vertices, K = 13, a single pair and one seed; σ = 0, 0.25, 0.5, 0.75, 1 times π scaled by the normalised height; the θ is not refitted at any σ.
**sources:** out/faust_deformation/records.json (60 rows: 12 arms × 5 levels, the nn_ rows are nearest-neighbour counterparts not plotted here)
**conventions:** a twist is not an isometry, so the curve bounds descriptor degradation under a smooth distortion, not tolerance to real pose change

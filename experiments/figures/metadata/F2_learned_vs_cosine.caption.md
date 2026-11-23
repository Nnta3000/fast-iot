# F2_learned_vs_cosine

**axes:** x = number of labelled pairs m; y = held-out matching hit minus the cosine hit at the same k, m and draw, in percentage points (0 = cosine); one panel per PCA width k.
**instance:** k ∈ {64, 128, 256}, m ∈ {50, 100, 300}, 3 draws, geometry G2, one fixed 400-pair held-out pool, chance 1/400
**methods:** cosine (baseline, the zero line); iOT with diagonal θ; InfoNCE with diagonal θ; InfoNCE with a low-rank r=64 residual
**config:** k ∈ {64, 128, 256} · m ∈ {50, 100, 300} · one fixed 400-pair held-out pool · 3 draws · chance 1/400
**sources:** experiments/out/records/metric_learning/records.json (study t1t2/metric_learning); cosine is label-free and constant per k, asserted in the study
**conventions:** hollow marker = solver-limited iOT fit (R'/R_uniform > 1); numbers beside a marker are the m=300 mean, the full set of points is in the CSV

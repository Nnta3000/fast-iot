# F5_constraint_generation

**axes:** left: final edge set |F| as a percentage of all m·n edges; right: wall time in seconds on a log scale; both against the number of labelled slices m (log scale).
**instance:** m ∈ (300, 600, 2000), K=64, geometry G3, soft window, one draw, GPU master and scan
**methods:** the constraint-generation loop at every m; one dense LP over all m·n edges at m=300, drawn as a hollow diamond
**config:** m=300 from Phase B (loop 2.5 s, dense LP 14.13 s) · m=600 and m=2000 from Phase C · 1 draw
**sources:** reports/STATE.md §3.5 (Phase B: loop vs dense LP at m=300) and §3.6 (Phase C: loop scaling at m=600 and m=2000); the values are parsed from those prose tables, the generating records were not preserved
**conventions:** the diamond is a single dense-LP measurement, not a curve; m=300 on the right panel is the Phase B loop

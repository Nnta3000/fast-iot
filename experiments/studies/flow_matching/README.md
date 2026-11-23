# flow_matching — exact and global OT couplings for flow matching

**Status: reserved (2026-11-23). No run exists; no number here may be cited.** The planned question is whether
couplings from HELLO-OT — a global exact coupling, or the semi-discrete target duals from
`hello_ot.solve_semidiscrete` — improve a small flow-matching model over independent pairing and minibatch OT at
equal step count. The planned instance is a 2-D toy; the planned readings are sample quality (energy distance /
2-Wasserstein), mode coverage, and wall time and peak GPU memory per coupling. See
`notes/` for the design discussion once the study is written.

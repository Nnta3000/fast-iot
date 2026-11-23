# F4_pair_recovery

**axes:** left: exact-slice recovery, i.e. the paired slice has the right patient and the right z index (log scale); right: same-patient recovery (linear scale); one row per method.
**instance:** 630 slices / 45 patients, soft CT window, one draw, 60 FIT patients for the fits, DEV split for evaluation
**methods:** pixel MSE and correlation (no learning); cosine NN and cosine + OT; the learned θ; pixel NMI as a reference; random
**config:** Costs are fitted on 60 other FIT patients, in one draw. Any of the 630 slices could be named, so blind guessing is right 1 time in 630 for the slice and 1 in 45 for the patient.
**sources:** experiments/out/synthrad_v2/records.json (feature arms) and experiments/out/synthrad_cheap/records.json (pixel arms, item A)
**conventions:** left and right share one categorical y axis; the dashed line is chance; pixel NMI is a reference on registered pairs, drawn hollow

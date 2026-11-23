# F8_confidence_scores

**axes:** left: x = confidence score, y = AUROC (0.5 is chance, higher means correct vertices outrank wrong ones); right: x = miss@5 % among the kept vertices, filled dot = all vertices kept, hollow dot = only the most confident half kept.
**instance:** MPI-FAUST, n = 4000 vertices, eight held-out pairs, K = 13, τ = 3e-3
**methods:** cost margin, OT dual margin, relative cost margin, softmax margin; three θ (inverse-OT, row-softmax, equal weights). Keeping a subset and scoring it only on the kept vertices is what the right panel measures.
**config:** MPI-FAUST, n = 4000 vertices per pair, eight held-out pairs, K = 13. AUROC here is the probability that a correctly matched vertex is ranked more confident than a wrongly matched one, so 0.5 is chance; the softmax temperature is 3e-3; only two coverage levels (100 % and 50 %) were recorded.
**sources:** out/faust_abstention/records.json (288 rows: 3 θ × 4 scores × 3 readouts × 8 pairs)
**conventions:** AUROC is per pair and averaged; the dual-margin AUC moves by 0.002–0.006 between reruns because the spectral basis is not bit-reproducible

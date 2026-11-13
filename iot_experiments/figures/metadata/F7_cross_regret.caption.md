# F7_cross_regret

**axes:** left: rows are the plan being scored, columns the θ that was fitted, colour and number the regret normalised by that plan's uniform regret; right: held-out miss@5 % on the eight held-out pairs.
**instance:** MPI-FAUST, n = 1200 vertices, K = 13; Pair 1 is subject 0 in pose 0 against subject 1 in pose 3; Pair 2 is subject 0 in pose 2 against subject 4 in pose 1; Pair 3 is subject 0 in pose 5 against subject 8 in pose 8; Pair 4 is subject 0 in pose 7 against subject 5 in pose 2
**methods:** the first fourθs are fitted on one pair each, the last θ is shared by all four; the green boxes mark the diagonal, where a θ is scored on the plan it was fitted to
**config:** MPI-FAUST, n = 1200 vertices, K = 13 descriptors; regret is divided by that pair's uniform regret, so 1.0 means no better than equal weights; Pairs 3 and 4 use a second person who also appears among the evaluation subjects.
**sources:** out/faust_multi_pair/records.json; the matrix is the 'cross_regret' row, the held-out values are the per-pair rows of the same record
**conventions:** the shared θ is the last column; colour saturates at 0.35 and the two values above it are printed in full; green boxes mark the diagonal

# F10_selection

**axes:** left: x = miss@5 % on the held-out pairs, y = pair; right: x = miss@5 % difference from the shipped θ on that pair, zero marked, with the mean and its 95 % interval as the two bottom rows.
**instance:** MPI-FAUST, n = 4000, eight held-out pairs, K = 13, 71 admitted candidates
**methods:** equal weights, median of the candidate set, the shipped max-margin θ, and the validation-selected θ
**config:** MPI-FAUST, n = 4000 vertices, eight held-out pairs, K = 13; candidates admitted within 1.25× the reference regret; selection uses six validation pairs only, never the held-out pairs.
**sources:** out/faust_select/records.json (8 pairs × 4 rules plus four contrast rows)
**conventions:** the contrast rows come from the same record; wins counts pairs where the left side is lower; selection uses validation pairs from training subjects only

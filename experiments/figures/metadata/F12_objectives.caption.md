# F12_objectives

**axes:** x = objective, y = share of evaluated vertices whose matched partner is farther than the threshold from the true one (left 5 %, right 1 %). A hollow dot is one held-out pair, the horizontal bar their mean.
**instance:** MPI-FAUST, n = 1500, K = 13; two held-out pairs, 050-066 and 056-071
**methods:** uniform, entropic iOT, one-sided softmax, Sinkhorn softmax, relaxed regret LP
**config:** MPI-FAUST, n = 1500 vertices, K = 13, pairs 050-066 and 056-071 (the two the record kept of eight); each hollow dot is one pair's value for that θ, the bar is the arithmetic mean over the two pairs; hyperparameters were chosen on validation pairs from the training subjects.
**sources:** out/faust_arms/records.json (21 rows: five objectives over two pairs plus the hyperparameter-selection row)
**conventions:** the mean is taken over the two pairs' results for the same θ; it is not a θ fitted on both pairs. The source report tabulates eight pairs, the record kept two, so these values are not the reported 8-pair means

# F13_regional_theta

**axes:** left: rows are spatial regions, columns descriptors, colour and number the fitted weight θ; right: L1 distance from each regional θ to the global θ, plus the held-out result.
**instance:** MPI-FAUST pair 0-13, n = 2000 vertices, K = 13, six k-means regions, one seed
**methods:** one simplex fit per region on that region's sub-block of the identity plan; the dashed line is the mean pairwise regional distance
**config:** MPI-FAUST training pair 0-13, n = 2000 vertices, K = 13, simplex only, one seed; region sizes 279–493 vertices; the naive regional field is compared with the global θ on eight held-out pairs.
**sources:** out/faust_regional/records.json (six region rows with their θ, one spread row, and the global/field rows)
**conventions:** weights below 0.01 are left unlabelled to keep the grid readable; the regional field is preliminary and has no regional supervised control

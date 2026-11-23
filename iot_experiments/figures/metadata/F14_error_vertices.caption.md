# F14_error_vertices

**axes:** no axes: an orthographic front view of the target surface; each dot is one evaluated vertex, coloured by the geodesic distance from its matched partner to the true one, in units of √area.
**instance:** MPI-FAUST pair 050-066, n = 600 of 6890 vertices, K = 13 descriptors, one draw
**methods:** position_only, uniform_theta, learned (simplex fit on pair 000-013, exact CPU master)
**config:** MPI-FAUST pair 050-066, n = 600 evaluated vertices of 6890, K = 13 descriptors; the learned weights come from pair 000-013 fitted with an exact CPU master; dots are drawn for the front-facing part of the body only, so counts include vertices that are not visible in this view.
**sources:** out/faust_error_points/records.json and errors_pair050-066.npz, written by studies/iot/real/faust/error_points.py; both are git-ignored and regenerated in ~1 min
**conventions:** dots are drawn only for the front-facing 65 % of the evaluated vertices, so the counts in the titles exceed what is visible; grey is the rest of the surface, not a data value

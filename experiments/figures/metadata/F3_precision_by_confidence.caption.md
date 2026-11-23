# F3_precision_by_confidence

**axes:** x = share of matches kept, ranked from most to least confident by margin (%); y = precision among those kept; two arms.
**instance:** K=256, m ∈ {100,300}, 3 draws, one fixed 700-pair pool, geometry G2
**methods:** cosine (hollow squares) and the learned diagonal θ (filled circles); every row is matched to exactly one column and a row is a hit when the id matches
**config:** K=256 · m ∈ {100,300} · 3 draws · P@c = hit rate of the top c·n rows; no pair discarded from the matching
**sources:** reports/current/REAL_PAIRS_REPORT.md §5, table of P@25..P@100; the generating records were never committed, so these values are parsed from the report table
**conventions:** margin = min(row best−second gap, column best−second gap); coverage selects which subset is evaluated, no pair is removed from the matching

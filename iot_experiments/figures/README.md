# figures

Meeting figures. **Every plotted value is parsed from a saved record at run time** — none is typed into
the code. The `EXPECTED` dict in `make_figures.py` holds the values from the brief as a **check** only: the
run stops if a parsed value differs beyond tolerance, so a number cannot reach a figure unless it came from
a record *and* matches the brief.

```bash
.venv/bin/python -m iot_experiments.figures.make_figures
```

Outputs land in `iot_experiments/figures/out/` as PNG and PDF at 300 dpi.

| file | contents | status |
|---|---|---|
| `style.py` | palette, markers, title guard, footer, save | — |
| `make_figures.py` | all figures, one launch | — |
| `F2_learned_vs_cosine` | small multiples, Δ hit vs cosine | ✅ |
| `F3_precision_by_confidence` | precision at coverage, 2 arms | ✅ |
| `F4_pair_recovery` | 7 methods × 2 metrics | ✅ |
| `F5_constraint_generation` | edge share and time vs m | ✅ |
| F1 | **not built — no record exists** | ❌ |

## Vocabulary used in the figures

- **`k`** (lowercase) — the **PCA width**: how many components the DINOv2 pool4 features were reduced to,
  before any cost is applied.
- **`K`** (capital) — reserved in this project for **|θ|**, the number of features in `c_ij = θᵀφ_ij`. It is
  **not** used in these figures. For the diagonal arms |θ| = `k`, which is why the two were easy to confuse.
- **`r`** — the rank of the residual bilinear correction `xᵀ(I + UVᵀ)y`. Those arms have **no θ** at all, so
  quoting a `K` for them would assert a θ size that does not exist.

## F2 — learned costs and cosine

**Source:** `iot_experiments/reports/metric_learning/records.json` (committed).
**Config:** geometry G2, one fixed 400-pair held-out pool, 3 draws, arms cosine / iOT / InfoNCE /
residual r=64.

Plotted (Δ hit in points at m=300, parsed from the record):

| k | iOT | InfoNCE | residual r=64 |
|---|---|---|---|
| 64 | +6.6 | +11.1 | +25.5 |
| 128 | +2.7 | +9.3 | +14.6 |
| 256 | +1.7 | +4.6 | +2.5 |

**Caveats:** hollow marker = solver-limited iOT fit (`R'/R_uniform > 1`), which happens in 3 of 9 iOT cells
(mostly k=256). Pre-registered counts (higher on all 3 draws by ≥ 3 points): iOT 1/9, InfoNCE 6/9,
residual64 4/9.

## F3 — matching precision by confidence

**Source:** `iot_experiments/reports/REAL_PAIRS_REPORT.md`, table lines 115–118.
**⚠️ The generating JSON was never committed.** `studies/real/t1t2/pool/run.py` still exists and asserts
`P@100 == hit`, which is why the `hit` and `P@100` columns agree, but the raw records are gone. These numbers
are therefore **parsed from a report table, not from primary data.**

| arm | P@25 | P@50 | P@75 | P@100 |
|---|---|---|---|---|
| cosine | 1.0000 | 0.9996 | 0.9456 | 0.8428 |
| learned | 1.0000 | 0.9988 | 0.9503 | 0.8518 |

**How precision is defined:** every row is matched to exactly one column; a row is a hit if the id matches.
Rows are ranked by **margin** = `min(row best−second gap, column best−second gap)`. `P@c` is the hit rate
among the top `c·n` rows. **No pairs are discarded from the matching** — coverage only selects which subset
is evaluated. Margin is a property of the cost matrix, not of correctness.

**Config:** k=256, G2, m ∈ {100,300}, fixed 400-pair pool, 3 draws. Standardisation and PCA fitted on pool
images (transductive, label-free).

## F4 — pair recovery on CT and MR

**Sources:** `iot_experiments/out/synthrad_v2/records.json` (feature arms) and
`iot_experiments/out/synthrad_cheap/records.json` (pixel arms). **Both are git-ignored and not durable.**
**Config:** 630 slices / 45 patients, soft window, 1 draw, 15 fit / 15 evaluation patients.

| method | exact-slice | same-patient |
|---|---|---|
| random | 0.006 | 0.027 |
| pixel MSE | 0.019 | 0.146 |
| pixel correlation | 0.049 | 0.190 |
| cosine NN | 0.111 | 0.189 |
| cosine + OT | 0.114 | 0.178 |
| learned cost | 0.167 | 0.233 |
| pixel NMI (reference) | 0.764 | 0.824 |

**Caveats:** preliminary — one draw, small fit/eval split. NMI is a **reference**, not a competitor: on
registered pairs it is close to an oracle.

## F5 — constraint generation at scale

**Source:** `iot_experiments/reports/STATE.md` §3.5–3.6 (prose table, parsed at run time).
**Config:** k=64, G3, soft window, GPU master and scan.

| m | final `|F|` as % of m·n | loop time | peak RSS |
|---|---|---|---|
| 300 | 3.40 % | 2.5 s | — |
| 600 | 1.86 % | 7.8 s | 5.2 GB |
| 2000 | 0.65 % | 185 s | 5.9 GB |

Dense LP at m=300: **14.13 s**, shown as a hollow marker. **Caveats:** the time step between m=600 and
m=2000 (**7.8 s → 185 s**, ×24 while |F| only grew ×3.9) is **not yet explained**; m ≥ 3000 is not runnable
on this machine because the loop evaluates σ over all `m·n` edges.

## F1 — not built

`F1` ("Choice of preprocessing") has **no record**: no geometry-sweep JSON exists anywhere in the repository
(`find` for `*geometr*.json` returns nothing), and `iot_experiments/out/` has no geometry directory. The
values quoted in the brief are not in any saved file. **Skipped rather than retyped.**

## Title guard

`style.add_title()` raises `StyleError` if any title or subtitle matches
`beats|better|worse|improv|helps|matters|wins|fails|outperform|than|knows|finds|touches|best|only|clearly|strong|weak|gain|lose|saturat`
(case-insensitive). It caught a real case during development: the default F2 title "Learned costs **against**
cosine" contains "gain" and was rejected; it is now "Learned costs and cosine".

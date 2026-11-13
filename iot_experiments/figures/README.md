# figures

**The style contract is `FIGURE_STYLE.md`.** It is the specification (canvas, type, colour, footer,
metadata, review checklist) and `F4_pair_recovery` is its worked example. Read it before adding a figure.

Meeting figures. **Every plotted value is parsed from a saved record at run time** — none is typed into
the code. The `EXPECTED` dict in `make_figures.py` holds the values from the brief as a **check** only: the
run stops if a parsed value differs beyond tolerance, so a number cannot reach a figure unless it came from
a record *and* matches the brief.

```bash
MPLBACKEND=Agg .venv/bin/python -m iot_experiments.figures.make_figures        # F2-F5
MPLBACKEND=Agg .venv/bin/python -m iot_experiments.figures.make_study_figures  # F6-F14
```

Layout:

| path | contents |
|---|---|
| `figures/out/` | **PNG only** (300 dpi), git-ignored, regenerable: `F2_learned_vs_cosine.png`, … |
| `figures/metadata/` | `<name>.csv` (one row per plotted point) and `<name>.caption.md`; **tracked**, because `out/` is ignored and "what is plotted" must survive a clone |
| `style.py` | palette, markers, title guard, footer, save |
| `metadata.py` | csv + caption writer, record loader, the `EXPECTED` check |
| `make_figures.py` | F2–F5 (T1/T2 metric learning, SynthRAD, constraint generation) |
| `make_study_figures.py` | F6–F14 (FAUST correspondence) |

No PDFs are produced. The brief's convention of a sibling `<name>.caption.md` is kept in content but placed
in `metadata/`, so `out/` holds nothing but PNG files.

## Figure index

| figure | what it shows | source |
|---|---|---|
| F2 | learned costs vs cosine on T1/T2, three PCA widths | `out/records/metric_learning/` |
| F3 | precision by confidence, cosine vs learned θ | `current/REAL_PAIRS_REPORT.md` (no record) |
| F4 | SynthRAD pair recovery, seven methods | `out/synthrad_v2`, `out/synthrad_cheap` |
| F5 | constraint-generation loop vs dense LP | `reports/STATE.md` §3.5–3.6 |
| F6 | FAUST per-vertex error heat maps, four weightings — **not built**: needs `errors_pair*.npz` from a completed heat-map run | `out/faust_heatmaps/` |
| F7 | cross-regret matrix and the cost of sharing one θ | `out/faust_multi_pair/` |
| F8 | label-free confidence and abstention | `out/faust_abstention/` |
| F9 | accuracy under a controlled twist | `out/faust_deformation/` |
| F10 | selecting inside the near-optimal set | `out/faust_select/` |
| F12 | five objectives on one hypothesis class | `out/faust_arms/` |
| F13 | regional θ: do body parts want different weights? | `out/faust_regional/` |
| F14 | wrong vertices drawn on the body of a held-out FAUST pair | `out/faust_error_points/` |

## The footer contract

Every figure carries a three-field footer, and `style.add_footer` **enforces** the fields:

| field | question it answers |
|---|---|
| `experiment` | which run is this? (study, data, features, geometry) |
| `comparing` | what is put side by side in the plot? |
| `config` | instance and protocol, including what the chance level means |

Each field is a **natural sentence, without em dashes**, and says something the other two do not
(`labelled=False` in `add_footer`). Each wraps to the plot width and the block grows **downwards** — the
number of printed lines follows the content, so a footer is never squeezed or clipped. Rules, enforced in
code:

- all three fields are **required** and must be single-line;
- `add_footer` **rejects any file path** (`.md`, `.json`, `.csv`, `.py`, `.pdf`, `.log`, `/out/`): the footer
  states the experiment, it never cites a record. Sources live in `metadata/<name>.caption.md`;
- no caveats and no conclusions on the figure. The footer states facts; interpretation is the reader's job.

`add_title` separately rejects concluding words, so titles stay noun phrases.

## Vocabulary used in the figures

- **`k`** (lowercase) — the **PCA width**: how many components the DINOv2 pool4 features were reduced to,
  before any cost is applied.
- **`K`** (capital) — reserved in this project for **|θ|**, the number of features in `c_ij = θᵀφ_ij`. For
  the diagonal arms |θ| = `k`, which is why the two were easy to confuse.
- **`r`** — the rank of the residual bilinear correction `xᵀ(I + UVᵀ)y`. Those arms have **no θ** at all, so
  quoting a `K` for them would assert a θ size that does not exist.

## F2 — learned costs and cosine

**Experiment:** `t1t2/metric_learning` — T1↔T2 MRI pairing, DINOv2-pool4 features, PCA to k, geometry G2.
**Comparing:** iOT (diagonal θ), InfoNCE (diagonal θ), InfoNCE low-rank r=64; y = hit − cosine at the same
k, m and draw.
**Config:** k ∈ {64,128,256}, m ∈ {50,100,300}, one fixed 400-pair held-out pool, 3 draws, chance 1/400.
**Source:** `iot_experiments/out/records/metric_learning/records.json` (tracked).

Plotted (Δ hit in points at m=300, parsed from the record):

| k | iOT | InfoNCE | residual r=64 |
|---|---|---|---|
| 64 | +6.6 | +11.1 | +25.5 |
| 128 | +2.7 | +9.3 | +14.6 |
| 256 | +1.7 | +4.6 | +2.5 |

Hollow marker = solver-limited iOT fit (`R'/R_uniform > 1`), in 3 of 9 iOT cells (mostly k=256).

## F3 — matching precision by confidence

**Experiment:** `t1t2/pool` — T1↔T2 matching on one fixed 700-pair pool, geometry G2, standardisation and
PCA fitted on the pool images.
**Comparing:** cosine (label-free) vs learned diagonal θ, the same cost family with two θ; precision among
the top 25/50/75/100 % of rows ranked by margin.
**Config:** K=256, m ∈ {100,300}, 3 draws; P@c = hit rate of the top c·n rows, no pair discarded.
**Source:** `iot_experiments/reports/current/REAL_PAIRS_REPORT.md` §5 table. ⚠️ The generating records were
never committed, so these values are **parsed from a report table, not from primary data**.

| arm | P@25 | P@50 | P@75 | P@100 |
|---|---|---|---|---|
| cosine | 1.0000 | 0.9996 | 0.9456 | 0.8428 |
| learned | 1.0000 | 0.9988 | 0.9503 | 0.8518 |

## F4 — pair recovery on CT and MR

**Experiment:** SynthRAD brain MR→CT pair recovery against the ground-truth `(patient, z)` pairing;
630 slices / 45 patients; soft window; DEV split.
**Comparing:** pixel baselines (MSE, correlation) vs feature methods (cosine NN, cosine+OT, learned θ) vs
the pixel-NMI reference; exact-slice and same-patient recovery.
**Config:** K=64, 1 draw, 60 FIT patients for the fits, 45 DEV patients (≤14 slices each, ≥8 apart);
chance 1/630 (slice) and 1/45 (patient).
**Sources:** `iot_experiments/out/synthrad_v2/records.json` (feature arms) and
`iot_experiments/out/synthrad_cheap/records.json` (pixel arms, item A). Both are git-ignored and not durable.

| method | exact-slice | same-patient |
|---|---|---|
| random | 0.006 | 0.027 |
| pixel MSE | 0.019 | 0.146 |
| pixel correlation | 0.049 | 0.190 |
| cosine NN | 0.111 | 0.189 |
| cosine + OT | 0.114 | 0.178 |
| learned cost | 0.167 | 0.233 |
| pixel NMI (reference) | 0.763 | 0.824 |

## F5 — constraint generation at scale

**Experiment:** constraint-generation loop on real SynthRAD features — K=64, G3, soft window, GPU master
and GPU scan.
**Comparing:** the relaxed loop vs one dense LP over all m·n edges (m=300 only); |F|/(m·n) and wall time vs m.
**Config:** m=300 from Phase B (loop 2.5 s, dense LP 14.13 s); m=600 and m=2000 from Phase C; 1 draw.
**Source:** `iot_experiments/reports/STATE.md` §3.5–3.6 (prose tables, parsed at run time).

| m | final `|F|` as % of m·n | loop time | peak RSS |
|---|---|---|---|
| 300 | 3.40 % | 2.5 s | — |
| 600 | 1.86 % | 7.8 s | 5.2 GB |
| 2000 | 0.65 % | 185 s | 5.9 GB |

Dense LP at m=300: **14.13 s**, drawn as a hollow diamond (a single point, not a curve). m ≥ 3000 is not
runnable on this machine: the loop evaluates σ over all `m·n` edges.

## F1 — not built

`F1` ("Choice of preprocessing") has **no record**: no geometry-sweep JSON exists anywhere in the repository
(`find` for `*geometr*.json` returns nothing), and `iot_experiments/out/` has no geometry directory. **Skipped
rather than retyped.**

## Title guard

`style.add_title()` raises `StyleError` if any title or subtitle matches
`beats|better|worse|improv|helps|matters|wins|fails|outperform|than|knows|finds|touches|best|only|clearly|strong|weak|gain|lose|saturat`
(case-insensitive). It caught a real case during development: the default F2 title "Learned costs **against**
cosine" contains "gain" and was rejected; it is now "Learned costs and cosine".

`style.add_footer()` separately raises `StyleError` on a path, so a figure cannot fall back to citing a
record file instead of naming its experiment.

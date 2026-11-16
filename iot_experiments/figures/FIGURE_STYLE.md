# Figure style — the contract, with F4 as the worked example

**Status: CURRENT (2026-11-12).** This file is the specification. `style.py` is its executable form;
`make_figures.py` is the reference implementation; `F4_pair_recovery` is the example every new figure is
compared against. When the two disagree, this file and `style.py` win over any existing script.

A figure is a **statement about an experiment**, not a rendering of a file. Everything below follows from
that: the data comes from a record, the text names the experiment, and interpretation stays outside the
picture.

---

## 0. Output contract

| path | contents | tracked? |
| --- | --- | --- |
| `figures/out/` | **PNG only**, 300 dpi, white, `bbox_inches="tight"` | no (git-ignored, regenerable) |
| `figures/metadata/` | `<name>.csv` and `<name>.caption.md`, one pair per figure | yes |

- **No PDF, no SVG, no intermediate artefact in `out/`.** One figure is one `.png`.
- `metadata/` sits beside `out/`, not inside it, because `out/` is git-ignored and "what is plotted here"
  must survive a clone.
- One launch builds a family of figures: `.venv/bin/python -m iot_experiments.figures.make_figures` (F2–F5)
  or `.venv/bin/python -m iot_experiments.figures.make_study_figures` (F6–F13).
- Each build prints the figure's footer text to stdout (`print_caption`), so a log records what the figure
  claims without opening the PNG.

## 1. Data discipline

1. **Every plotted value is parsed from a saved record at run time.** No number is typed into a plotting
   script, and no number is copied from a report by hand.
2. The script keeps an `EXPECTED` table of the brief's values; a parsed value outside tolerance **stops the
   run** (`FigureError`). A number therefore cannot reach a figure unless it came from a record *and*
   matches the brief.
3. If a value has no machine-readable record (F3, F5), that is stated in `metadata/<name>.caption.md`. It is
   not hidden and not presented as primary data.
4. Instance parameters that the record carries (`k`, `K`, `m`, `draws`, `holdout_pairs`, `n_side`,
   `n_patients`, `window`, `slices_per_patient`, `fit_patients_per_draw`) are **read from the record**, never
   retyped. Retyped config drifts: F4's footer claimed "15 fit / 15 evaluation patients" while the record
   said 60 FIT / 45 DEV.

## 2. Canvas and geometry

| element | rule |
| --- | --- |
| multi-panel figure | `figsize=(9.0, 3.6)`, `subplots_adjust(left≈0.09–0.12, right=0.985, top=0.78, bottom≈0.32)` |
| single-panel figure | `figsize=(6.5, 4.4)` (taller than 4.0: the three-footer needs the room) |
| resolution | 300 dpi PNG, white face, tight bbox |
| left margin | set the same fraction in `subplots_adjust` and in `add_title`/`add_footer` (`axes_left`) so the title, axes and footer share one left edge |
| bottom margin | large enough that the three footer lines sit **below** the x-axis label with no overlap |

Footer line spacing is computed **in inches** (9.5 pt) inside `add_footer`, because a fixed fraction of the
figure height makes 7 pt text collide on a 3.6 in canvas.

## 3. Typography

| role | size | colour | notes |
| --- | --- | --- | --- |
| title | 13 pt bold | `#1a1a1a` | one line, ≤ 6 words, left-aligned |
| subtitle | 10 pt | `#666666` | one line: what is on the axes |
| axis labels / ticks | 9 pt | default | units or symbol always present |
| annotations | 7.5 pt | the series colour | values beside a marker |
| footer | 7 pt | `#8c8c8c` | three lines, see §6 |

- Font family is DejaVu Sans; PDF/PS font type 42 (irrelevant now that no PDF is emitted, kept for safety).
- **Titles are noun phrases, never conclusions.** `add_title` rejects
  `beats|better|worse|improv|helps|matters|wins|fails|outperform|than|knows|finds|touches|best|only|clearly|strong|weak|gain|lose|saturat`
  and any title longer than six words, and any subtitle containing a newline.
- No `\n` inside a title or subtitle; a two-line title is a sign the wording is too long.

## 4. Colour, markers and encodings

- One fixed colour per method, shared by every figure (`ARM_COLORS`), taken from the project palette
  `PALETTE` in `style.py`: the 14 colours of `orange-palette-q3.png`, read pixel by pixel. The pale end
  (white, pale yellow, pink) is for fills and heat maps only, never for a line on white.
- Baselines and references are neutral grey (`cosine`, `random`, `pixel_*`, `nmi`).
- **Marker shape varies with the method** as well, so the figure survives greyscale printing.
- `colour()` and `marker()` honour the `ALIASES` table, so `learned` and `iot` are one series everywhere.
- Encodings beyond colour:
  - **hollow** = solver-limited fit (F2) or an explicit reference row (F4's NMI);
  - a **dashed grey line** = chance, labelled `chance`;
  - **filled** = a measurement that is the subject of the figure.

## 5. Axes

- No grid. Top and right spines off. Bottom/left spines 0.8 pt. Tick length 3, minor ticks off.
- **Exception — surface panels (F6).** A rendered surface has no cartesian axes at all: axes are off, the
  aspect is equal, and the quantity is carried by a continuous colour field on `SEQUENTIAL_CMAP`
  (the palette's monotone-lightness segment: ink → crimson → orange → yellow → pale yellow)
  colormap with one labelled colourbar per comparable group. The same rules still apply to titles,
  annotations and the footer.
- Explicit ticks: `set_xticks` / `set_yticks` with the values that appear in the data, and explicit tick
  labels where matplotlib would otherwise print an exponent (`2`, `10`, `100`, never `4×10²`).
- `log` scale when the range spans decades (F3's exact-slice axis, F5's m and time axes), and the axis label
  says so: `exact-slice recovery (log)`.
- Every panel with a categorical axis fixes its tick positions and labels; a shared categorical axis is
  enforced with `set_ylim`/`set_yticks`, never left to autoscaling.
- Data labels sit next to the point and are offset in **points** (`xytext=(dx, dy), textcoords="offset
  points"`), never in data units, so they survive a scale change.
- Values are labelled where a reader would otherwise estimate: the last point of a series, the extremes, a
  single point that is not a curve.

## 6. The footer — one block that wraps, natural language

`add_footer(fig, *, experiment, comparing, config, axes_left=..., axes_right=..., labelled=False)`.

The footer is **one block of as many lines as the text needs**, not three fixed lines. Each field is wrapped
to the width of the plot (`axes_left`..`axes_right`, measured with the renderer, so a wide glyph cannot push
a line past the edge), the fields are separated by half a line, and the block runs **downwards** from just
below the axis labels. When it passes the bottom of the canvas, `bbox_inches="tight"` extends the saved
image instead of the footer eating into the axes. Horizontal space is the constraint; vertical space is not.

| line | answers | must contain |
| --- | --- | --- |
| 1 | *which run is this?* | the study, the data, the representation, the geometry |
| 2 | *what is being compared?* | the arms, and which panel shows which metric |
| 3 | *what is the instance and protocol?* | sizes, draws, the fitting/evaluation split, **and what chance means** |

Hard rules, enforced in code and in review:

1. **Natural sentences. No em dashes, no `·`-separated fragments.** Write "Each method must pair MR slices
   with CT slices from one 630-slice pool…", not "data: x | commit y | geometry G2".
2. **No file paths.** `add_footer` rejects `.md`, `.json`, `.csv`, `.py`, `.pdf`, `.log` and `/out/`.
   A footer that cites a record has told the reader nothing about what they are looking at.
3. **No commit hash.** It dates the plotting, not the data, and is stale by the next commit.
4. **No caveats and no conclusions.** "The time step is not yet explained", "preliminary", "NMI is only a
   reference" are interpretation or self-assessment. Facts that a reader needs to decode an encoding go in
   `metadata/<name>.caption.md`, not on the figure.
5. **The three fields must not restate each other.** If `config` repeats the pool size from `experiment` and
   the arms from `comparing`, it is empty. Check by deleting each field in turn: every deletion must lose
   information. Wrapping does not change this — a field that says nothing is still a paragraph of nothing.
6. **A bare number that a reader would misread must be explained.** Write "blind guessing is right 1 time in
   630 for the slice and 1 in 45 for the patient", not "chance 1/630 and 1/45".
7. All three fields are required, non-empty and single-line; an empty field raises.

## 7. Metadata

`metadata/<name>.caption.md` — one bolded field per line:

| field | content |
| --- | --- |
| `axes` | one sentence per panel: quantity, units, scale |
| `instance` | the instance string, with every parameter that identifies it |
| `methods` | the arms, including which are references |
| `config` | the protocol: sizes, draws, fitting/evaluation split |
| `sources` | the record/report the values came from, and whether the record is durable |
| `conventions` | what hollow/hatched/dashed means, if the figure uses them |

**No takeaway.** The caption records what the figure is; what it means is the reader's job.

`metadata/<name>.csv` — one row per plotted point, long format, first column the instance axis:

| figure | header |
| --- | --- |
| F2 | `k, m, arm, stat, value` (per-draw rows and a `mean` row) |
| F3 | `arm, coverage_percent, precision` |
| F4 | `method, metric, value` |
| F5 | `m, quantity, value` |

A CSV row exists for **every** plotted point, including per-draw points that are not drawn as markers.

## 8. Naming and provenance

- Descriptive file names: `<figure>_<subject>.png` (`F4_pair_recovery.png`), never `fig1.png`.
- The figure number identifies the slot in the deck; the rest of the name says what it shows.
- A figure whose source record is missing or git-ignored is still built (all data comes from a record), but
  `sources` in the caption says so plainly.

---

## 9. Worked example — `F4_pair_recovery`

The reference figure, panel by panel.

**Canvas.** `figsize=(9.0, 3.6)`, two panels sharing one categorical y axis, `left=0.115` everywhere
(title, axes, footer), `bottom=0.32` to clear the three-line footer.

**Panels.** Left: exact-slice recovery on a log x axis (`0.004 … 3`), because the arms span two decades.
Right: same-patient recovery on a linear axis (`0 … 1.08`). Both carry a dashed grey `chance` line
(`1/630` and `1/45`) drawn from the record's instance, not from the report.

**Encodings.** One row per method, in a fixed order (weakest to strongest), each with its own marker and
colour; `pixel NMI` is hollow because it is a reference rather than a competitor. Every point carries its
value as a 7.5 pt annotation offset in points, so the numbers can be read without the CSV.

**Title / subtitle.**

```
Pair recovery on CT and MR
Exact-slice and same-patient recovery for pixel, feature and learned methods
```

Noun phrase, six words, no conclusion; the subtitle names both axes and the three method families.

**Footer** (`labelled=False`, i.e. printed verbatim; wrapped here by hand only for this document — the code
wraps at the plot width):

```
Each method must pair MR slices with CT slices from one 630-slice SynthRAD pool (45 patients, soft
window, DEV split), and is scored against the true patient and slice.
Rows compare pixel distance, cosine, cosine plus OT and the learned cost, plus NMI as a reference; the
left panel needs the exact slice, the right one only the patient.
Costs are fitted on 60 other FIT patients, in one draw. Any of the 630 slices could be named, so blind
guessing is right 1 time in 630 for the slice and 1 in 45 for the patient.
```

Why it passes:

1. Line 1 answers *which run*: SynthRAD brain, MR→CT, the pool size, the split, the window.
2. Line 2 answers *what is compared*: the four families plus the reference, and which panel is which metric.
3. Line 3 answers *instance and protocol*: the fitting split, the number of draws, and the meaning of both
   chance levels. Deleting it loses the fit/eval split and the reading of `1/630`.
4. No path, no commit, no caveat, no conclusion, no em dash.
5. Every number in it (`630`, `45`, `60`, `1`) is read from the record at run time.

**Metadata.**

- `metadata/F4_pair_recovery.csv` — `method,metric,value`, 14 rows (7 methods × 2 metrics).
- `metadata/F4_pair_recovery.caption.md` — axes, instance, methods, config, sources (a git-ignored record,
  stated), conventions (shared categorical axis, dashed chance line, hollow NMI).

---

## 10. Review checklist

Before a figure is called done:

- [ ] every value parses from a record, and `EXPECTED` still matches;
- [ ] title ≤ 6 words, noun phrase, subtitle one line;
- [ ] colours and markers come from `style.py`, not literals in the script;
- [ ] top/right spines off, no grid, explicit ticks, no minor tick labels on log axes;
- [ ] chance or reference lines labelled and their meaning explained in the footer;
- [ ] footer: three wrapped paragraphs, no em dash, no path, no commit, no caveat, no conclusion;
- [ ] no footer line runs past the right edge, and the block does not overlap the axis labels;
- [ ] deleting any footer field loses information;
- [ ] `metadata/<name>.csv` has a row for every plotted point;
- [ ] `metadata/<name>.caption.md` has all six fields and no takeaway;
- [ ] `out/` gained exactly one PNG and nothing else;
- [ ] `ruff check --select F401,F811,F841,E9 iot_experiments/figures` is clean;
- [ ] the figure regenerates from a clean checkout given the record.

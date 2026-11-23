# notes — ideas, plans, and specs that are not results

**Why this directory exists.** `reports/` is Markdown about **results** (with a record) or **plans** with a
progress marker. Speculative material does not belong there: it would be the first file in that folder with
no record behind it. `notes/` is the home for ideas that are being thought about but not yet tested.

**The rule that keeps this honest:** nothing in `notes/` may be cited as a result. If an idea becomes an
experiment, it moves to `studies/<axis>/<name>/` with a driver and a record, and the result goes to
`reports/current/`. The note then points at the report instead of carrying the number.

| file | what it is | status |
| --- | --- | --- |
| `PLAN_EVALUATION_AND_APPLICATIONS.md` | metric change, the user's levers 1–8 and P1–P4, ordered, with what is done | PLAN |
| `APPLICATIONS.md` | what iOT is for beyond paired data → cost → pair; boundaries; picture projects | NOTES |
| **`FAUST_CORRESPONDENCE_GUIDELINE.md`** | **the shape-correspondence plan that replaces the photomosaic: data, instance, descriptor family, solver use, evaluation, baselines, pre-flight gates** | **DESIGN ONLY** |
| `FAUST_NEXT_STEPS.md` | what to try next after the current FAUST results: two go/no-go diagnostics, four ranked directions, and what not to do; plus the VRAM/resource note | PLAN |
| `RELATED_WORK_AND_COMPARABILITY.md` | the four inverse-OT papers cited in `paper/ours/presentation.typ`: what is in the tree, their verified numbers, the four axes each reports, and the protocol of the reproduction in `../studies/iot/synthetic/inverse_cost_recovery/` | PROTOCOL (data in the study's `records.json`) |
| `../legacy/pictures/README.md` | the **retired** photomosaic study: why it was retired, and six reusable pitfalls | RETIRED |

**Current direction.** Image mosaics are retired (`../legacy/pictures/`). The active design is
**shape correspondence on FAUST**: the plan is free and exact, the natural cost fails visibly, and a
standard evaluation protocol already exists. See `FAUST_CORRESPONDENCE_GUIDELINE.md`; it is design only,
and its first blocker is that FAUST needs a licence accepted by the user.

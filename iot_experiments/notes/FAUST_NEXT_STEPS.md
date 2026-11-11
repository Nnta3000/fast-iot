# FAUST — what to try next (plan, not results)

**Status: PLAN. Nothing in this file has been run, and no number here may be cited as a result.**
It records the decision made on 2026-11-11 after the current results in
`../reports/current/FAUST_CORRESPONDENCE_REPORT.md`, so the next step does not have to be re-derived
from the conversation.

---

## 0. Where the study actually stands

The measured picture, one line each (all records under `../out/records/faust_*`, sections in the report):

| finding | report section | number |
| --- | --- | --- |
| learned beats fixed weightings | §2, §4c, §4d | `miss@5 %` 0.200 vs 0.419 at n = 6890 |
| but a *supervised* reweighting of the same 13 features gets most of it | §4f | LP − `row_softmax` = −0.035; LP − `rowcol_softmax` = −0.026 with a CI that includes 0 |
| the advantage collapses under deformation | §4g | gap −0.341 at σ = 0 → −0.067 at σ = 1 |
| a shared θ over many pairs fails, and the cross-regret matrix says why | §4h | `multi` 0.467 vs `single` 0.197; low training regret does not transfer |
| the OT dual certificate does **not** locate errors | §4i | AUROC 0.502 (chance); the probability margin does (0.63) |
| inverse OT as a hard-negative miner makes the cheap arm worse | §4j | +0.022 over `plain`, 0/8 pairs |

**The diagnosis.** None of these are objective problems. Every objective swap (four arms), every
teacher trick, every confidence trick lands in the same place because the binding constraint is the
**hypothesis class**: 13 global weights over fixed hand-made descriptors, one cost for the whole body.
That is why θ is not identified (the corner analysis), why the LP is worth ~0.03 over a plain
reweighting, and why the advantage evaporates under deformation.

So the things worth trying are the ones that change the hypothesis class or stop pretending the fit
returns a point. Adding another objective is not on the list.

---

## 1. Go/no-go diagnostics (run these first, ~45 min together)

Both are cheap measurements that kill a direction before it costs a day.

### D1 — is the admissible set worth more than one point? (feeds A)

θ is identified only up to a cone, and the shipped solver returns the max-margin vertex of it. Sample
several θ admissible for the *same* training plan (the machinery exists: `selection="central"` /
`"max-margin"` in `src/iot/relaxed/selection.py`, and the `random` tie-break), evaluate **all** of them
on the 8 held-out pairs, and report the spread.

- **Go if** the spread of `miss@5 %` across the sampled face is materially larger than the validation
  noise (~0.01). Then selection inside the face is a real lever.
- **No-go if** every face point scores within noise — then the max-margin convention is harmless and A
  is dead.

### D2 — does the body want different weights in different places? (feeds B)

Fit θ separately on 6–8 regions of the training pair (geodesic k-means on one mesh, or height/limb
splits), and report the pairwise L1 distance between the regional θ's.

- **Go if** the regional θ's are far apart (comparable to the 0.5–1.5 L1 distances already measured
  between *pairs* in §4h, which showed how different two plans' demands can be).
- **No-go if** they are all within ~0.1 of each other — then a regional cost has nothing to exploit.

---

## 2. Candidate directions, in the order I would spend time

### A. Return the admissible set, not a point

Use iOT's **cone** as a candidate generator, then select inside it on held-out data. This turns the
identifiability weakness (§4.5 of `AGENTS.md`, the corner analysis) into a feature, and needs no new
hypothesis class and no new solver — only sampling plus validation.

- Cost: ~1 h after D1 goes.
- Risk: validation noise; the face may contain mostly bad points.
- Falsifiable: the selected θ must beat the max-margin θ on the held-out pairs, with a paired CI.

### B. A piecewise-constant cost field (regional θ)

Different body parts plausibly want different descriptors: hands need curvature, the torso needs
position. Each region's plan is a sub-permutation, so the existing LP machinery applies unchanged per
region; nothing new is needed in `src/iot`.

- Cost: ~2–3 h (region construction + per-region fits + evaluation).
- **Mandatory control: regional InfoNCE.** Without it we cannot separate "regions helped" from
  "inverse OT helped", which is exactly the mistake §4f was designed to catch.
- Falsifiable: a global θ must lose to the regional field by more than the seed spread.

### C. Enlarge the hypothesis class *inside* iOT

Replace the diagonal weights by a low-rank metric `‖A(a_i − b_j)‖²`, `A = I + UVᵀ`, fitted by descending
the regret (`∂R*/∂φ = (π̂ − π*)·θ*` by the envelope theorem — P2 in `notes/APPLICATIONS.md` §4.2).

- Cost: days. This is the only direction that could plausibly reach a competitive correspondence
  number, but it is really "a metric-learning paper with iOT as the loss", not an iOT paper.
- Not recommended before D1/D2 say the cheaper levers are exhausted.

### D. Write the characterisation paper

The results already exist and are coherent: identifiability, transfer failure (cross-regret matrix),
deformation collapse, certificate-at-chance, multi-pair failure. That is a defensible
workshop-level contribution — *"what inverse OT does and does not give you for shape correspondence"* —
and it is honest about the 0.03.

**Fallback if both diagnostics are negative: take D and stop trying to make the correspondence number
competitive.**

---

## 3. What not to do

- **No more objectives.** Four objectives on one hypothesis class have been run (§4f); a fifth lands in
  the same place.
- **No two-sided (Sinkhorn) arm at n = 6890.** It is not scalable, and §4f already showed it is worth
  ~0.01 over the one-sided loss. The driver exists (`finalists.py`) and the run is deliberately
  abandoned.
- **No comparison to published FAUST errors.** This is not a state-of-the-art claim, and the withheld
  200 test scans are not on disk.
- **No new `*_REPORT.md`.** Extend `FAUST_CORRESPONDENCE_REPORT.md`; ideas stay in `notes/`.

---

## 4. Resource note (why some runs are blocked)

A containerised `qwen3-tts-api` service runs on this machine and, when started, holds ~4.2 GB of the
8 GB GPU. It was killed once and came straight back (a restart policy restarted it) — the container is
`ee1cf0c34f56`, so stopping it needs `sudo docker stop ee1cf0c34f56`, not a `kill` on the PID. Our user
is not in the `docker` group and `sudo` requires a password, so no experiment script can stop it.

Consequences for this study:

- the one-sided and LP paths fit in ~1.2 GB of VRAM and run either way;
- the two-sided Sinkhorn arm at n = 6890 needs ~3 GB and is what took the GPU down twice;
- the host-RAM ceiling (`MemAvailable` ≤ ~10.7 GB) was never the binding constraint.

If the TTS service is needed regularly, make it on-demand: `docker update --restart=no` plus a wrapper
that starts it explicitly, and an idle-stop timer (`docker stop` after N minutes without a request).
`docker pause` does **not** free the VRAM — only `stop` does.

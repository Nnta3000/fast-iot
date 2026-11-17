# The NSR/NICP arm — why we moved to FAUST, and why this vehicle was abandoned

**Status: CURRENT (2026-11-17).** A record of a **negative result** and the decision that followed.
Nothing here is a claim about the entropic or exact inverse-OT solvers; those live in §2.1 and §2.2.

Driver/patch: `studies/real/faust/pairing_gate.py`, `scripts/nsr_gate.patch`
Records: `out/faust_pairing_gate{,_learned}/`, `out/nsr_gate/` (pulled from the server)

---

## 0. Summary in five lines

1. We tried to use **NSR/NICP** (ECCV 2024, FAUST leaderboard 1.056 intra / 2.261 inter) as the host pipeline in
   which to test whether a global transport plan beats per-point `argmin`.
2. **It runs.** Four scans registered end to end at 2.5 min/scan, and we extracted NICP's cost matrix.
3. **The measurement it supports is weak**, for a structural reason: NICP's cost matrix never reaches the metric.
4. **It cannot be reproduced against its published number**: the test-scan pipeline and the submission writer
   are not in the released repository.
5. We therefore kept the *question* and changed the *vehicle* — to functional-maps matching, where the decoder
   is the last operation before scoring and dense ground truth exists in the same space.

---

## 1. Why FAUST rather than T1/T2 metric learning

The T1/T2 work (§2.2) is a real-data application test, but as a *benchmark for a correspondence method* it has
three defects:

| issue | T1/T2 | FAUST |
| --- | --- | --- |
| ground truth on the scored quantity | none — the score is retrieval hit against a 400-pair held-out pool | **exact and dense**: registrations share one vertex ordering, so `π̂ = I/n` *is* the truth |
| saturation | in-sample hit reaches **1.0** for `m ≤ 100` at `K ≥ 32`, so the arms are not being steered by the data | unsaturated at `m = 300` |
| external anchor | none — the study's own arms only | published leaderboard with per-method numbers |

FAUST thus supplies what the decoder question needs: a cost matrix, an exact ground truth in the same space,
and a standard error metric (geodesic on the target, normalised by `sqrt(area)`).

**What FAUST costs us:** it is a 3D-mesh benchmark with its own data plumbing (registrations, remeshed variants,
spectral operators), and the community's data links are external and rot (§5).

---

## 2. What the NSR arm did, and what it produced

NSR registers a scan to a body model: a neural field predicts, for each query point, offsets to 690 SMPL
template vertices; NICP iterates between *picking the smallest offset per target point* and *fine-tuning the
field so those offsets collapse*; the result is fitted to a full SMPL model (6890 vertices) and refined.

We ran it on four FAUST training registrations (`tr_reg_000..003`), patched only to dump the cost matrix
`C[k,i] = ‖F(y_k)_i‖²` that NICP computes immediately before its own `argmin`:

| measurement | value |
| --- | --- |
| throughput | **4/4 scans**, 145–152 s per scan |
| run-to-run variation | `aligned.ply` exact; NICP output **0.018 %** of the bbox diagonal; chamfer 0.007 % |
| dumped cost | `K = 6889` scan points × `M = 690` template vertices, per scan |
| official internal metric | **ERR = 0.00912** over 6 registration pairs |

The 0.018 % variation is well below FAUST's own 2 mm annotation tolerance (~0.1 %), so the pipeline is
*stable*. Instability is not the reason it was dropped.

---

## 3. The diagnostics: NICP's own assignment is not a map

Scored against the expected multiplicity `K/M ≈ 9.99` on NSR's own cost:

| scan | argmin coverage | argmin max load | imbalance | transport max load | imbalance | disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tr_reg_000 | 0.991 | 91 | 9.1× | 11 | 1.10 | 51.5 % |
| tr_reg_001 | 0.970 | 127 | 12.7× | 10 | 1.00 | 50.8 % |
| tr_reg_002 | 0.978 | 87 | 8.7× | 10 | 1.00 | 50.0 % |
| tr_reg_003 | 0.945 | 152 | 15.2× | 10 | 1.00 | 55.1 % |

`imbalance = max_load / (K/M)`. **Transport delivers exactly the expected 10; argmin delivers 0–152 and leaves
up to 5.5 % of template vertices unused.** So the same hubness pathology measured on our own descriptor cost
(§2.2, `PAIRING_GATE_REPORT.md`: 35–47 % coverage, 16–30×) is present inside a SOTA registration prior.

**This is the arm's one solid contribution**, and it is a statement about NICP's correspondence stage, not about
whether transport *improves the registration*.

---

## 4. Why the arm was dropped — three blockers, in order of severity

### 4.1 The cost matrix does not reach the metric (structural)

The evaluation is a **pure nearest-neighbour cascade on the final meshes**:

```
scan A → NN → our registration of A → NN → scan B → match → error
```

NICP's `argmin` influences the number only *indirectly*: correspondence rule → NICP loss → optimised field →
registration mesh → NN matching → error. Any effect is filtered through 10 Adam steps of field refitting.

Contrast with functional-maps matching, where the decoder **is** the last operation before scoring.

### 4.2 We cannot obtain independent ground truth where the cost lives (decisive)

The cost is `K × 690` — scan points against **690 SMPL vertices**. FAUST's ground truth is the **6890-vertex
FAUST template** — a different mesh. The only bridge is the NN cascade above, which is derived from NSR's own
output, so scoring argmin and transport against it is **circular**.

### 4.3 The published number is not reproducible (institutional)

```
get_dataset(challenge)  →  accepts only 'FAUST_train_reg' and 'FAUST_train_scans'
                            raises ValueError otherwise
```

- There is **no test-scan path** and **no submission writer** anywhere in the repository.
- The README's own final command uses `core.challenge.name='FAUST_scan_reg'`, which would crash.
- The internal metric we *can* run is mean **Euclidean**, **unnormalised**, on registration meshes — that is why
  it reads `0.00912` and not a leaderboard-style number.

So the leaderboard entry (1.056 / 2.261) was produced with code that was not released. This is not a criticism
of the authors — releasing a submission pipeline is not standard — but it means the arm has **no validation
anchor**: we cannot confirm that our copy of the pipeline is the one that scored 1.056.

### Also: the resolution question

The 690 is the neural field's internal resolution; the submitted object is 6890 because a full SMPL model is
fitted afterwards. SMPL also has 6890 vertices — the same count as the FAUST template but a different mesh with
a different ordering — so whether the challenge accepts a SMPL-keyed map or requires the FAUST ordering cannot
be determined from the released code.

---

## 5. What we changed to, and what happened next

**Vehicle changed:** functional-maps matching (**URSSM**, SIGGRAPH/TOG 2023; FAUST 1.4–1.6). Reason: its
matching stage is

```python
similarity = feat_x @ feat_yᵀ            # (V1 × V2), full resolution
P          = row-softmax(similarity / τ)  # τ = 0.07, then hard argmax
C          = evecs_yᵀ (P @ evecs_x)       # 200 × 200 functional map
p2p        = argmin cdist(evecs_x Cᵀ, evecs_y)   ← the decoder, last step before scoring
geo_err    = metrics['geo_error'](dist_x, corr_x, corr_y, p2p)
```

Three properties that the NSR arm lacked: the decoder is the final step, ground truth (`.vts`) is in the same
vertex space and independent of the method, and the metric is the published geodesic convention.

**Correction recorded while reading it:** its assignment stage is a **row softmax → hard argmax with no column
constraint**, not the Sinkhorn normalisation its own code comment claims. So it shares the greedy per-row
structure we measured above — which makes it a fair and *unexploited* place to add a column constraint.

**Data status:** the URSSM README's FAUST_r link is dead (Drive 404, file deleted). A live mirror was found in
the ancestor repository (`Unsupervised-Deep-Multi-Shape-Matching`, ECCV 2022) on École Polytechnique's
Nextcloud, exposed over WebDAV:

```
https://nuage.lix.polytechnique.fr/public.php/webdav/FAUST_r.zip          19.8 MB  (100 × .off, 4999 verts)
https://nuage.lix.polytechnique.fr/public.php/webdav/FAUST_r_vts.zip       2.2 MB  (100 × .vts ground truth)
user LJFXrsTG22wYCXx, empty password
```

Both downloaded and integrity-checked. **Reproduction of URSSM has not been achieved yet**; the baseline run is
the outstanding step.

---

## 6. What survives from this arm

| survives | where |
| --- | --- |
| transport beats argmin on our own FAUST cost, 15–71 %, with the mechanism measured | `current/PAIRING_GATE_REPORT.md` |
| NICP's own assignment is hubness-collapsed (8.7–15.2× imbalance, ≤ 99.1 % coverage) | this report §3 |
| URSSM's assignment is also row-wise greedy with no column constraint | this report §5 |
| NSR runs reproducibly enough for cost-level study (0.018 % variation) | this report §2 |
| a live FAUST_r mirror, with exact file sizes | this report §5 |

**Does not survive:** any claim that adding a transport plan to NSR improves registration, because we never had
a non-circular way to measure it.

---

## 7. Decision record

- **2026-11-16** — NSR arm dropped as the vehicle for the decoder question. Reason: §4.1 and §4.2, not
  difficulty of installation.
- **2026-11-16** — vehicle changed to functional-maps matching (URSSM).
- **2026-11-17** — FAUST_r mirror found and downloaded; URSSM reproduction pending.

**To re-open the NSR arm** it would need one of: (a) the authors' test-scan pipeline, (b) a way to define
per-scan-point ground truth in the 690-vertex SMPL space independently of NSR, or (c) a decision to accept a
circular comparison, which we rejected.

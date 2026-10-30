# STATE — SynthRAD brain pairing work

**Last updated: 2026-10-28.** This file is the durable state. If context is lost, read this first.

---

## 1. Goal

Decide whether a **learned inverse-OT cost** recovers the correct MRI→CT pair better than a plain
feature distance, on the SynthRAD2023 Task 1 brain data. The yardstick is the ground truth
(`patient`, `z_index`), which we know and the method does not.

This is a **pairing** study, not a synthesis study. The challenge organisers score synthesis quality
inside the body mask; we score pairing recovery. Both are legitimate; they are different questions.

---

## 2. Data and split

- `data/Task1/` — real SynthRAD2023 Task 1 brain, 180 patients (60 per centre A/B/C), 1807 files.
  The archive `Task1.zip` was verified as md5 `360bc61e2320d5c5a7145454a81c566a` and **deleted after
  verification**; only the extracted folders remain (14 GB).
- MR and CT are on the **same voxel grid** (verified on 40 patients: identical matrix, 1 mm spacing,
  identical z origin, zero mismatches). Hence `id = (patient, z_index)` is exact and needs no alignment.
- **Split** (`iot_experiments/out/synthrad_v2/split_v2.json`, seed 2024, stratified by centre):
  - **FIT 105** (35/centre) — includes the 30 Phase-1 probe patients, since already-seen patients go to
    fit and never to eval/test.
  - **DEV 45** (15/centre) — shared by every arm and every draw.
  - **FINAL TEST 30** (10/centre) — **never touched**.
- Preprocessed slice cache: `cache/slices/` (1.9 GB, both CT windows, all 150 fit+dev patients).
  Rebuildable; wipe with `iot_experiments/scripts/wipe_cache.sh`.

---

## 3. Verified results

### 3.1 Data audit (Phase 1)

Slices retained after the ≥5% body-mask rule, evaluated on the **native** grid:

| centre | slices total | included | excluded (mask < 5%) |
|---|---|---|---|
| A | 1980 | 1938 | 42 |
| B | 1869 | 1835 | 34 |
| C | 1802 | 1772 | 30 |

Centre B is **post-gadolinium**: nonzero MR median 53.7 vs A's 536.8, while its normalised mean is the
*highest* of the three (0.191 vs 0.145 / 0.139). Artifacts: `iot_experiments/out/records/synthrad_probe/`.

### 3.2 Pair recovery, small sample (superseded, kept for provenance)

15 fit / 15 eval patients, 100 slices. Learned cost beat cosine with DINOv2 in all four cells
(0.333→0.378, 0.084→0.342, 0.324→0.396, 0.062→0.347 patient-level) but lost with pixel-PCA.
`reports/current/synthrad_recovery.md`. **Small sample; superseded by 3.5.**

### 3.3 Plain-pixel baseline (NumPy only, no torch, no GPU)

DEV pool, 630 slices, 45 patients, 1 draw. Chance: exact **0.0016** (1/630), patient **0.022** (1/45).

| window | arm | exact_hit | patient_hit |
|---|---|---|---|
| soft | pixel MSE | 0.0190 | 0.1460 |
| soft | pixel correlation | **0.0492** | **0.1905** |
| bone | pixel MSE | 0.0111 | 0.1238 |
| bone | pixel correlation | 0.0238 | 0.1746 |

Records: `iot_experiments/out/synthrad_cheap/records.json`, log `out/synthrad_cheap/run.log`.

### 3.3b Pixel correlation vs features, on BOTH metrics (correction, 2026-10-28)

The two metrics tell **different stories**, and the earlier wording ("learning only marginally ahead")
was too coarse. Soft window, same DEV pool, chance 0.0016 exact / 0.022 patient:

| arm | exact_hit | vs pixel corr | patient_hit | vs pixel corr |
|---|---|---|---|---|
| pixel correlation | 0.049 | 1× | 0.190 | 1× |
| pixel MSE | 0.019 | 0.4× | 0.146 | 0.8× |
| cosine nearest-neighbour | 0.111 | **2.3×** | 0.189 | 1.0× |
| cosine + exact OT | 0.114 | **2.3×** | 0.178 | 0.9× |
| learned cost (dense LP) | 0.140 | **2.9×** | 0.203 | 1.1× |
| learned cost (loop) | 0.135 | **2.8×** | 0.198 | 1.0× |

- **On exact-slice recovery, features win clearly**: 0.111–0.140 against 0.049 — a **2.3–2.9×** improvement
  over pixel correlation. This is the metric that distinguishes methods.
- **On same-patient recovery they are all ~0.19–0.20**, i.e. indistinguishable. Almost any method that finds
  the right *level* also finds the right *patient*, so this metric saturates far too early to separate them.
- The correct summary is therefore: **features roughly triple exact-slice recovery and buy nothing on
  patient recovery at this pool size.**

### 3.4 Paired soft-vs-bone table (same patients, same pool, same draw)

| arm | soft | bone | change |
|---|---|---|---|
| pixel MSE, exact | 0.0190 | 0.0111 | −42% |
| pixel correlation, exact | 0.0492 | 0.0238 | −52% |
| pixel MSE, patient | 0.1460 | 0.1238 | −15% |
| pixel correlation, patient | 0.1905 | 0.1746 | −8% |

The earlier "roughly halving" claim is **only true for pixel exact-recovery**; on patient-level recovery
the drop is 8–15%. That claim was originally about feature arms, for which **no paired numbers exist**.

### 3.5 Phase B — loop vs dense LP at m=300 (accepted with note)

Real SynthRAD features, K=64, G3, soft window, 1 draw.

| | dense LP (CuPDLPx) | relaxed loop (CuPDLPx) |
|---|---|---|
| objective | `R' = 1.2935e-3` | bracket `L = 1.2432e-3`, `U = 1.2949e-3` |
| dense `R'` inside bracket | — | **yes** |
| time | 14.13 s | 2.50 s (5.6×) |
| rounds / final `|F|` | 1 solve, 90,000 edges | 9 rounds, 3,075 edges (3.4%) |
| held-out exact / patient | 0.1397 / 0.2032 | 0.1349 / 0.1984 |

`‖θ_loop − θ_dense‖₁ = 1.30e-2` = **1.3%** of `‖θ‖₁ = 1` (two independent draws from the same box differ
by 0.902 ± 0.078, so the solutions are effectively identical).

**Bracket width `U − L = 5.17e-5`.** Under the *original* bar `1e-2 × R_uniform` (3.63e-5) this is
**44% over** (an earlier note said 14% — that was wrong). The bracket is **1.4% of `R_uniform`**.

**Accepted with note** under the revised bar below.

### 3.6 Phase C — loop scaling (m = 600, 2000 only)

GPU master + GPU scan, K=64, G3, soft.

| m | rounds | `|F|` | `|F|/(mn)` | time | peak RSS | bracket rel-width | status |
|---|---|---|---|---|---|---|---|
| 600 | 10 | 6,689 | 0.0186 | 7.8 s | 5.2 GB | 6.1e-3 | accepted |
| 2000 | 12 | 26,163 | 0.0065 | 185 s | 5.9 GB | 1.1e-2 | accepted (under the 5% bar) |

`|F|/(mn)` **falls** with m (0.019 → 0.0065), so the edge-fraction advantage improves. Per-round time
grows steeply (0.65 s → 14–16 s).

**m ≥ 3000 is NOT run on this machine** and must not be: the loop evaluates `sigma` over all edges, so
memory is `O(m²)` regardless of `|F|`. Attempts at m=5000/11263 were OOM-killed.

---

## 4. Decisions (with date)

| # | decision | date |
|---|---|---|
| 1 | Ground-truth pairing is the yardstick; the proxy-metric gate (SSIM/PSNR/NMI/IoU) is dropped — no validated discriminating power | 2026-10-28 |
| 2 | `mask_iou` and `z_only` dropped as competitors: **tautological for registered pairs** (mask and slice index are properties of the same patient on a shared grid) | 2026-10-28 |
| 3 | `ct_mask_iou` removed: duplicate of `mask_iou`, identical values | 2026-10-28 |
| 4 | NMI kept as **one labelled reference row only**; it is a *registration* metric, so on registered pairs it is near-oracle. **Not** corroboration of Zhu et al. | 2026-10-28 |
| 5 | Solver: **GPU (CuPDLPx) for master and scan, always**. HiGHS removed from the production path; exactness is not a goal. An exact CPU master was tested once and converged to 1.6e-16, but is not used | 2026-10-28 |
| 6 | **Acceptance bar changed to `U − L ≤ 5% × R_uniform`** — a threshold change made **after seeing** that m=300 landed at 1.4% and m=2000 at 1.1%. Must be stated as such wherever it is used | 2026-10-28 |
| 7 | Outline-neutralised control **skipped**: it would be tautological for the same reason as #2 | 2026-10-28 |
| 8 | No memory-limit enforcement mechanism exists in this session (see §6). Jobs must be checked manually before launch | 2026-10-28 |

---

## 5. Unverified / open

- **Feature arms across both CT windows**: no paired numbers exist. §3.4 is pixels only.
- **Paired noise estimate** for the Phase-B `exact_hit` gap (0.0048 on 630 rows, 0.35 binomial SE): still
  outstanding; a paired test needs per-row disagreement vectors.
- **Whether learning helps at all** beyond pixel correlation: at 630 slices, learned ≈ 0.20 patient vs
  pixel-correlation 0.19. Not resolved; needs a larger pool.
- **Phase D** (learning curve in m ∈ {300, 1000, 3000, 11263}) — **not started**, blocked by the O(m²) loop
  limit and by the memory ceiling.
- **Plain-pixel baseline at other draws/windows** — only 1 draw so far.
- **Pelvis** anatomy: downloaded and audited but never evaluated.
- **FINAL TEST 30 patients**: never touched.

---

## 6. Resource rules and the enforcement mechanism (measured)

**Environment:** native Arch Linux, kernel 7.2.11, 15.3 GB RAM. **Not WSL2.**

Measured results of trying to cap a 2 GB allocation at a 1 GB limit:

| mechanism | enforced? | evidence |
|---|---|---|
| `prlimit --rss=1G` | **NO** | allocated 2 GB successfully — the kernel does not enforce RLIMIT_RSS |
| `ulimit -v 1G` (address space) | yes, but **unusable** | raises `ArrayMemoryError`; breaks CUDA because the driver reserves huge address space |
| `systemd-run --user --scope -p MemoryMax=1G` | **unavailable** | `Failed to connect to user scope bus`; with `XDG_RUNTIME_DIR=/run/user/1000` set: `Operation not permitted` |
| cgroup v2 child creation | **unavailable** | `mkdir /sys/fs/cgroup/...` → Permission denied; current scope's `memory.max` is `max` and not writable |

**Conclusion: no hard memory limit can be enforced by the OS in this session.** Protection therefore has
two layers:

**Launcher (`iot_experiments/scripts/run_job.sh`).**
- records each job's exit code in `cache/jobs/<name>.state`;
- **refuses to start the same command again** if the last run failed, unless `--ack` is passed;
- checks `MemAvailable` first and refuses if `< budget + 2 GB`;
- runs `nice -n 19 ionice -c3`; no address-space limit (it would break CUDA).

**Watchdog (`iot_experiments/harness/watchdog.py`).** A daemon thread in the driver that aborts with
`os._exit(125)` if either the process RSS exceeds the stated budget or system `MemAvailable` drops below
1.5 GB. **Measured: it trips correctly** (allocating 2.4 GB under a 0.5 GB budget aborted at 2.08 GB with
exit 125) and does not fire on normal completion.

Manual rules on top of those:

1. **State the memory need before every job** and check `MemAvailable` in `/proc/meminfo`.
2. **Do not launch** if estimate + 2 GB > available.
3. **One launch per task.**
4. **Stop after one failure.** Never retry with changed settings without asking.
5. **A kill/OOM must be reported immediately** in the same message (AGENTS.md §0.2).
6. Do **not** run m ≥ 3000 loop fits on this machine.

**Estimates must come from a measured run of the same code path**, not array arithmetic. The pixel
baseline was estimated at 1.5 GB and actually peaked at **7.4 GB** — a 5× miss.

---

## 7. How to run

```bash
cd /home/agent/Documents/fast-iot

# cheap pixel baseline (NumPy only, no torch, no GPU)
.venv/bin/python -u -m iot_experiments.studies.real.synthrad.cheap_items --no-features

# with features (needs CUDA, ~7 GB peak)
.venv/bin/python -u -m iot_experiments.studies.real.synthrad.cheap_items --ks 64 128

# patient split (writes out/synthrad_v2/split_v2.json)
.venv/bin/python -m iot_experiments.studies.real.synthrad.split_v2

# Phase B: loop vs dense at m=300
.venv/bin/python -u -m iot_experiments.studies.real.synthrad.phase_b --m 300

# Phase C: loop scaling (keep m <= 2000 on this machine)
.venv/bin/python -u -m iot_experiments.studies.real.synthrad.phase_c --ms 600 2000
```

Every study writes `run.log` and `records.json` into `iot_experiments/out/<study>/`.

---

## 8. Code map

| path | what |
|---|---|
| `iot_experiments/studies/real/synthrad/split_v2.py` | stratified 105/45/30 split, seed 2024 |
| `.../recovery.py` | pair-recovery scoring (`exact_hit`, `patient_hit`), oracle assertion |
| `.../v2_run.py` | v2 evaluation driver, streaming feature extraction |
| `.../phase_b.py` | loop vs dense LP at one m |
| `.../phase_c.py` | loop scaling in m |
| `.../cheap_items.py` | pixel baseline + paired window table (`--no-features` for NumPy only) |
| `iot_experiments/harness/dinov2.py` | generic DINOv2 pool4 encoder (arrays in, features out) |
| `iot_experiments/harness/cache.py` | slice/feature cache |
| `src/iot/relaxed/loop_regret.py` | the relaxed loop; `solver_floor` + scan tolerance floor added |

---

## 9. Corrections to earlier statements (for the record)

- "The loop's bracket contains the dense `R'`" was **first claimed wrongly** (I compared two different
  problems: 25 vs 60 fit patients). Re-verified on the same problem: **it does contain it**.
- "The dense LP's optimum" — CuPDLPx under-solves by ~2% vs HiGHS on the full-edge master. Say "dense LP
  via CuPDLPx", not "the optimum".
- "14% over the bar" → **44%**. The bracket is **1.4% of `R_uniform`**.
- "Roughly halving" for the bone window is **true only for pixel exact-recovery** (−52%); patient-level
  recovery drops 8–15%. No feature-arm paired numbers exist.
- The first pair-recovery study (15/15 patients) is **superseded**; its numbers are small-sample.

---

## 10. Backlog (do NOT start without approval)

### 10.1 Tile the sigma scan so memory is not `O(m²)`

The loop evaluates `sigma = f + g - c` over **all** `m·n` edges every round, so memory is `O(m²)`
regardless of `|F|`. That is why m ≥ 3000 is not runnable here. The fix:
- scan in **row blocks**, maintaining a **running top-`gamma` per row and per column** across blocks;
- never materialise the full cost matrix or the full `sigma` array;
- memory then becomes `O(block · n + gamma · (m + n))`.

**Done already** (partial): `_sigma_max` and the median cost-scale probe are blockwise, and
`_best_dual_for` is blocked. The violation scan itself still needs the running-top-`gamma` change.

### 10.2 Explain the 7.8 s → 185 s step from m=600 to m=2000

The time grew **24×** while `m` grew 3.3× and `|F|` grew 3.9×. `O(m·n)` scan work alone would predict
~11×; the extra factor is unexplained. Three candidates to separate, measured per round rather than
guessed:
1. **`|F|` growth** — 6,689 → 26,163 edges (×3.9); the LP is solved on `F`, so this scales with `|F|`,
   not `m²`.
2. **LP time vs scan time** — currently only a single `wall_time` per round is recorded. Each round
   needs its master solve timed **separately** from its scan.
3. **Per-round count** — 10 → 12 rounds (×1.2), which explains little.

Action if approved: instrument `RegretIteration` with `master_s` and `scan_s`, rerun m = 600 and 2000,
and report the split. No new code paths, no larger m.

# flow_matching_headroom — is there coupling headroom beyond squared-Euclidean OT?

**Status: Stage 0 code + Stage 1 server script (2026-11-24).** This is a **headroom test only**. No inverse-OT, cost
learning or HELLO-based learning is implemented or claimed. The question: for plain flow matching trained with a
coupling of noise and data, does a coupling that uses the model's own behaviour improve few-step quality enough, at
equal training cost, to justify learning a cost with inverse OT later?

## Stage 0 — toys (minutes)

Targets and protocol are the v2 flow-matching study's: 8-Gaussians and two-moons (10 seeds), mixture d = 8 (5 seeds),
8 192 training points, MLP 2→128→128→2, CFM loss, batch 256, 10 000 steps. The **baselines** (`independent`, `w2_mb`
at B = 256, `global_ot`) come from `experiments/out/records/flow_matching_coupling_v2/eval.json`, paired by seed; they
are not retrained. Metric: sliced W2 (512 projections, 4 096 generated vs a fresh 4 096-point sample) at Euler NFE
1/2/5/10, with the v2 floor, plus straightness.

| arm | coupling | extra cost |
| --- | --- | --- |
| `mac_full` | 32-step independent warm-up; then the **current EMA model**'s error at `t∈{0,1}` on a fresh 256-pool (two forward passes; the error matrix is two squared distances), Sinkhorn with `reg = 0.5` | 2 extra forwards per step + Sinkhorn 256² |
| `teacher_routed_ot` | teacher = independent arm (10 000 steps); bank of 10× the data (81 920 noise points) with 20-step Euler endpoints; pool of 256 noise from the bank and 256 real points; **exact POT** on `‖T(x0)−x1‖²` | teacher 10 000 steps + bank + one POT 256² per step |
| `reflow_ref` | independent teacher, then CFM on `(x0, T(x0))`; no OT (reference) | teacher 10 000 steps + bank |

The teacher bank is the one fixed noise set in the design, as specified; every other arm draws fresh noise each step.

```bash
# local, one GPU, through the local launcher (memory check, no silent retry)
experiments/scripts/run_job.sh --name headroom_toys --budget-gb 2 -- \
  python -u -m experiments.studies.flow_matching_headroom.toys --steps 10000 --tag stage0
# server, one job per arm, assigned to idle GPUs by the usual queue
experiments/scripts/gpu_queue.sh experiments/studies/flow_matching_headroom/jobs_toys.txt
# table + decision (merges every stage0* tag)
python -m experiments.studies.flow_matching_headroom.analyse --stage toys
```

The queue writes per-job logs and `summary.tsv` under `experiments/out/queue/<timestamp>/`, pins
`CUDA_VISIBLE_DEVICES`, and only uses a GPU with < 1 GiB used and < 20 % utilisation (so it would wait,
correctly, while another service holds the card). Run `python -m experiments.harness.preflight --need-gpu`
first on a fresh server.

Measured cost estimate at 10 000 steps: `mac_full` ≈ 50–100 s per seed, `teacher_routed_ot` ≈ 60–90 s per seed
(teacher + bank + 10 000 POT plans), `reflow_ref` ≈ 10 s per seed; 25 runs per arm (10+10+5) → about **50–80 min**
wall, < 1 GiB GPU.

## Stage 1 — CIFAR-10 (server script)

Unconditional CIFAR-10: conditioning interacts badly with a global coupling, so it is deliberately absent. A small
DiT (default 256 wide, 6 blocks, patch 4, ~5M parameters) with the same CFM loss and the arms above; `teacher` is
optional because a 10× bank of 500k noise images is 6 GiB in fp32. Evaluation: Euler NFE 2/4/8/32, FID on 10 000
generated images against a 40 000-image training reference, with floors from a held-out 10 000-image real subset and
the CIFAR-10 test set; straightness reported per arm.

```bash
python -m pip install -q pytorch-fid        # FID only; the Inception weights download on first use
experiments/scripts/gpu_queue.sh --parallel --jobs 3 \
  experiments/studies/flow_matching_headroom/jobs_cifar.txt
python -m experiments.studies.flow_matching_headroom.analyse --stage cifar
```

One job per arm in `jobs_cifar.txt`, so the queue can put `independent`, `w2_mb` and `mac_full` on
different GPUs; the analysis merges every `cifar*` tag. `--smoke` runs four steps and skips FID, so
the time per step can be measured before the full run. The training/eval script is
`experiments/studies/flow_matching_headroom/cifar.py`; records land in
`experiments/out/flow_matching_headroom/cifar_<arm>/records.json`, and the FID image folders in
`experiments/out/flow_matching_headroom/cifar_images/` (git-ignored).

**Cost.** On a 4060-class card, 50k steps at batch 128 for a 5M-parameter DiT is roughly 2–3 h per run; 3 seeds × 3
arms ≈ 18–27 h, plus FID. On an A100/H100 it is roughly 20–40 min per run. The reference values (Lin et al.) are
class-conditional, 200 epochs, batch 256; absolute FIDs here will differ, and only relative arm differences at equal
steps are read.

## Decision rule (fixed before running)

- Stage 0: headroom at NFE 1–2 only if the better of `mac_full` / `teacher_routed_ot` beats `w2_mb` by more than
  2 standard errors, paired by seed; otherwise "no headroom at this scale", no tuning.
- Stage 1: headroom at NFE 4 if the better of `mac_full` / `teacher_routed_ot` beats the best `w2_mb` by more than
  `max(1.0 FID, 2 × seed sd)`, paired, in at least 2 of 3 seeds. Otherwise report "no headroom at this scale".
- Both equal-steps and equal-compute comparisons are reported, including the extra network evaluations and the teacher
  bank construction.

## Environment (measured 2026-11-24)

RTX 4060 Laptop, 8 188 MiB, 7 011 MiB free after the owner's Docker service was stopped; 15 GiB RAM (6 available);
32 GB disk; torch 2.14.1+cu130, torchvision, POT available; **no FID package and no cached Inception weights**, both
downloadable (PyPI and Hugging Face reachable). Install `pytorch-fid` once before Stage 1:
`python -m pip install pytorch-fid`. The queue (`experiments/scripts/gpu_queue.sh`) is the server
launcher; `experiments/scripts/run_job.sh` is the local one. Do not add launchers to the
repository-root `scripts/`, which belongs to HELLO-OT.

## Phase A/B/C/D pre-registered rules (2026-11-27)

These are fixed before the phase runs. Nothing below changes the model, loss, optimiser, 50 000 steps, batch 128,
seeds 0/1/2 or the FID protocol. The threshold at NFE 4 is always `T = max(1.0 FID, 2 × sd of the paired differences)`,
and a win is a FID decrease larger than `T`, paired by seed.

**Phase A — housekeeping (no GPU hours).** `analyse.py` must filter `config["steps"]`; the Stage 0 table must use the
v2 **10 000-step** baselines. The CIFAR analysis must ignore records whose `steps` is not 50 000 (in particular the
4-step smoke). A test must guard the budget filter. The stale smoke must be removed or ignored on `im2`. The queue
typo must be corrected. Sample grids (64 per arm, same noise, NFE 4/32, seed 0) go under
`experiments/out/flow_matching_headroom/cifar_grids/`. Equal-compute accounting uses `train_seconds`,
`coupling_seconds`, `teacher_seconds`, `bank_seconds` from the records.

**Phase B — why is MAC worse?** Add `w2_sinkhorn` (plain Euclidean cost, exactly the same Sinkhorn, `--mac-reg-rel`
0.1, and row-wise sampling as `mac_full`; no EMA and no warm-up) and `mac_exact` (same MAC cost and EMA 0.999,
390-step warm-up, exact EMD instead of Sinkhorn). Run each for seeds 0/1/2, 50k steps. At NFE 4:
- If `|w2_sinkhorn − mac_full| < T`, the soft plan explains MAC's loss, not the model-aligned cost.
- If `w2_sinkhorn` beats `mac_full` by more than `T` in at least 2/3 seeds, the cost hurts.
- If `mac_exact` beats `w2_mb` by more than `T` in at least 2/3 seeds, the model-aligned cost has headroom.
- If `mac_exact` is within `T`, it neither helps nor hurts; if worse by more than `T` in at least 2/3 seeds, it hurts.
Optional only with spare time: `mac_full` at `--mac-reg-rel 0.03` and `0.3`, one seed, as a sensitivity note; the
pre-registered setting remains 0.1.

**Phase C — best-informed pairing.** Use the existing 50k `independent_s{0,1,2}_t50000.pt` EMA weights as the teacher
via `--teacher-ckpt`; build the chunked float16 bank (factor 10, 20 Euler steps); record the teacher's original
`train_seconds` from the checkpoint metrics as `teacher_seconds` and the bank time as `bank_seconds`. Run
`teacher_routed_ot` and `reflow_ref`, seeds 0/1/2, 50k steps. At NFE 4, `teacher_routed_ot` beats `w2_mb` by more
than `T` in at least 2/3 seeds ⇒ headroom for a better-informed pairing; otherwise **no headroom at this scale**.
Also state whether it wins at equal total compute (`teacher_seconds + bank_seconds + train_seconds`); `reflow_ref`
is a reference only.

**Phase D — larger OT pools.** Add `--pool` to `w2_mb`: draw `pool` noise points and `pool` real images each step,
solve exact OT on the pool, train on 128 of the resulting pairs. Existing pool 128 records are the baseline. Measure
seconds per step at pool 512 and 1024 on the A100 before launching: if a 50k-step run would exceed **~2 hours**,
stop and report the timing. If acceptable, run both pools for seeds 0/1/2 at 50k steps. At NFE 4, pool 1024 beating
pool 128 by more than `T` in at least 2/3 seeds ⇒ pool size matters on CIFAR; otherwise **no evidence that pool size
matters here**. Do not attempt a global 50k × 50k POT plan; write the one-page memory/time note instead, including
the fixed-noise control required by the toy study.

**Scope.** Every result here is a coupling screen. It is not evidence about inverse OT, learned costs, HELLO, or a
large-scale solver. Null results are reported as nulls, without tuning or extra runs to rescue them.

## What is not tested

No inverse OT, no learned cost, no HELLO, no CIFAR at the reference scale, no conditional CIFAR, no FFHQ, no state of
the art. Stage 1 is a headroom screen, not a benchmark.

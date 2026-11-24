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
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching_headroom.toys --smoke   # ~10 s
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching_headroom.toys --steps 10000 --tag stage0
PYTHONPATH=. .venv/bin/python -m experiments.studies.flow_matching_headroom.analyse --stage toys
```

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
bash experiments/studies/flow_matching_headroom/server.sh stage0   # Stage 0 on the server
bash experiments/studies/flow_matching_headroom/server.sh cifar    # 50k steps, 3 seeds, 3 arms
```

`server.sh` installs `pytorch-fid` only (for FID); nothing else is added. `--smoke` runs four steps and skips FID, so
the time per step can be measured before the full run. The training/eval script is
`experiments/studies/flow_matching_headroom/cifar.py`; records land in
`experiments/out/flow_matching_headroom/cifar/records.json`, and the FID image folders in
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
downloadable (PyPI and Hugging Face reachable). `pytorch-fid` is installed by `server.sh`.

## What is not tested

No inverse OT, no learned cost, no HELLO, no CIFAR at the reference scale, no conditional CIFAR, no FFHQ, no state of
the art. Stage 1 is a headroom screen, not a benchmark.

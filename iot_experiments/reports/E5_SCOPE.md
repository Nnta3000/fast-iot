# E5 scope (T1/T2) — pseudo-pair synthesis with the existing N=1 trainer

**Status: scope only. Nothing has been run. Waiting for go-ahead.**

## What the trainer is (verified read-only, `DPM/mmd_train_new.py`)

It is the Bi-DPM trainer **at N = 1**, which is the paper's best setting on T1/T2. With `N = 1` the
intermediate-state loss `Σ_n w_n d(X^f_{t_n}, X^b_{t_n})` collapses to the two-point form the code
implements:

```
x1_fake = x0 + u(x0, t0)      t0 = eps
x0_fake = x1 - u(x1, t1)      t1 = 1
paird   = LPIPS(x0_fake, x0_pair) + LPIPS(x1_fake, x1_pair)
unpaird = MMD / exact-OT / Sinkhorn between fake and real marginals
loss    = paird + unpaired_weight * unpaird
```

`utils/ode.py` implements multi-step Euler (`t_n = n/N·(1−eps)+eps`) but **`ODESolver` is imported and
never instantiated**, so `N ≥ 2` intermediate matching is not wired up. Multi-step is therefore *not*
available without code changes; N=1 is what we would use.

## Arms

| arm | pseudo-pair source |
|---|---|
| `random` | uniform random T1→T2 assignment (floor) |
| `paper` | NMI + Hungarian on full-resolution pixels |
| `cosine_ot` | whitened-DINOv2 cosine + exact OT |
| `learned` | best metric-learning arm's cost + exact OT |
| `true` | the real pairs (upper bound) |

**Split:** T1 pool from half **A** of the subjects, T2 pool from half **B**, so every pseudo-pair joins
two *different* patients. Evaluation uses held-out **true** pairs (same-subject T1/T2) never seen in
training.

## Adapter spec — the CSV the trainer consumes

`FakePairFrequencyDataset` (`utils/datasets.py`) requires **`source_file`, `target_file`, `frequency`**;
`weight` and `is_native_pair` are optional. The sibling `ot_fake_pairs` runner writes different names, so
an adapter is required:

| `ot_fake_pairs.csv` (produced) | → | trainer CSV |
|---|---|---|
| `source_file` | → | `source_file` |
| `fake_target_file` | → | `target_file` |
| `transport_weight` | → | `weight` |
| — | → | `frequency` (**must be synthesised**: per-pair occurrence count over the OT runs / draws; the dataset filters on `frequency >= --fake_pair_min_frequency`) |
| `is_native_hit` | → | `is_native_pair` (bool) |

Paths must be **absolute** PNG paths on disk (the dataset calls `Image.open(...).convert('L')` and
resizes to `--img_size`). One row per (source, target) pair; duplicate pairs should be merged with summed
`frequency`.

Minimal invocation:

```bash
python mmd_train_new.py --task_name T1T2_pseudo --root <PNG_ROOT> \
  --img_type_1 T1 --img_type_2 T2 --img_size 192 \
  --is_pair --only_pair --loss_type lpips \
  --pair_ratio 1.0 --fake_pair_sampling_csv <pairs.csv> \
  --velocity_type unet --num_channels 128 --batch_size 2 --n_epochs 30 --lr_init 1e-3
```

## Per-epoch cost — to be measured, not guessed

I have **not** measured this. The measurement I owe you before any run is one epoch at
`--img_size 192 --batch_size 1` and `2`, recording wall time and peak GPU memory
(`torch.cuda.max_memory_allocated`), because the 8 GB laptop GPU is the binding constraint and
`LPIPS(net='vgg')` is loaded unconditionally alongside the UNet. I will report seconds/epoch and GB,
then extrapolate to the five arms (one training run per arm, ≥3 seeds each), so the total is stated
before we commit.

## Evaluation

Held-out true pairs only: MAE, PSNR, SSIM, mean ± std over ≥ 3 seeds. The `random` arm is the floor and
`true` the ceiling; a pseudo-pair arm is only interesting if it sits strictly between them.

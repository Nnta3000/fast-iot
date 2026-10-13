# gpu_bench batches

Run everything from the repo root. Each command prints a device line and writes to
`iot_experiments/gpu_bench/out/<batch>/`. Use `--gpu auto` (first idle A100) or a fixed
`--gpu <id>`.

## Batch 0 — `env` (seconds, run first)

```bash
python -m iot_experiments.gpu_bench.run_batch --batch env --gpu auto
```

Checks: GPU/VRAM, torch+CUDA, POT, `pycupdlpx` loadable, HELLO GPU scan supported, one tiny
`is_optimal` run. **Abort the session if `pycupdlpx` is false** (the `cupdlpx` arms need it) —
tell me and I'll adapt.

## Batch 1 — `gpu_master_k` (the premise test; minutes)

Isolated master LP, same `F` for both backends, `K` from the family (default `lowrank`, so `K`
is a direct knob). Default grid: `K ∈ {64,128,256,512,1024}`, `n=800`, 1 seed, 3 repeats,
`F = B` (mandatory).

```bash
python -m iot_experiments.gpu_bench.run_batch --batch gpu_master_k --gpu auto \
  --family lowrank --marginal gmm --n 800 --ks 64 128 256 512 1024 \
  --repeats 3 --backends highs cupdlpx
```

Decision: the crossover `K` where CuPDLPx beats HiGHS, and the scaling exponent of each. Local
A100-alternative (RTX 4060) showed CuPDLPx/HiGHS ratio 8.6× at `K=8` → 1.3× at `K=256`; we
expect the crossover to move much lower on an A100. If it does not appear by `K=1024`, the
premise fails for the master and we stop there.

Optional second call to separate `K` from `|F|` (add `|F|` pressure):

```bash
python -m iot_experiments.gpu_bench.run_batch --batch gpu_master_k --gpu auto \
  --family lowrank --marginal gmm --n 800 --ks 512 --extra-edges 8000 --repeats 3
```

## Batch 2 — `e2e` (only if Batch 1 wins; tens of minutes)

Full loop comparing scan (`numpy` vs `hello`) and master (`highs` vs `cupdlpx`). `cupdlpx`
runs with `prune=False`. Default sizes `{800,3200}`, `gmm`, `K=8` (lowrank) / `d=8` (diagonal).

```bash
python -m iot_experiments.gpu_bench.run_batch --batch e2e --gpu auto \
  --family lowrank --marginal gmm --sizes 800 3200 --lowrank-k 8 \
  --scans numpy hello --masters highs cupdlpx
```

Note: the HELLO GPU scan only supports `DiagonalWeightFeatures`; for `lowrank`/`quadratic` the
`hello` scan falls back to NumPy unless we add a GPU feature map (follow-up). Use
`--family diagonal` if you want to see the GPU scan actually engage.

## Budget discipline

- One A100, one batch at a time; start with Batch 0 then Batch 1.
- Add `--repeats 5` only if Batch 1's timing looks noisy.
- Do **not** raise `--ks` past 1024 or `n` past 3200 without telling me first — the master LP
  grows ~`K^2.4` and a single solve can take minutes.

## What to send back

`iot_experiments/gpu_bench/out/{env,master,e2e}/` — the JSON/CSV plus the captured stdout.

# nyc_taxi — does the relaxed formulation generalise across time windows?

**Question.** M-IOT (Bao, Li & Sun 2026, §6.3) learns one time-invariant cost from six NYC taxi flow windows. This
study asks the same thing of **this repository's relaxed (regret) formulation** — exact OT, `θ` on the simplex, no
entropic regularisation — and measures how much of the answer is the distance imputation rather than the method.

## Protocol

| step | choice |
| --- | --- |
| data | TLC `yellow_tripdata_2016-01.parquet`, 10.9 M trips, official 263 zone ids (`PULocationID`/`DOLocationID`; the 2016 files have no coordinates) |
| windows | six, by pickup time: `weekday_am/midday/pm/night`, `weekend_day/night` (`instance.WINDOWS`) |
| flows | per window, `263 × 263` trip counts; **80/20 split within each window**, seeded, so the held-out flow is a sample of the same demand |
| folds | leave-one-window-out, six folds |
| distance | mean reported `trip_distance` per zone pair, from training trips; **`NaN` where no training trip** |
| scope | `other_windows` (default): the five training windows only, so the held-out window contributes no trips. `all_train` (sensitivity): all six, which leaks the held-out window's *support* |
| unsupported pairs | `median` (default): filled with the median observed distance (7.66 mi). `mask` (sensitivity): kept at 0 and given one indicator column |
| edges | `--restrict-edges`: pairs with no training trip are forbidden (large finite cost) — the edge-set form, and the analogue of Stuart & Wolfram's graph restriction |
| metrics | RMSE / Pearson / CPC on the held-out flow (M-IOT Table 1's columns), and the formulation's own `regret_ratio = R_test / R_uniform` |

`θ` is on the simplex over non-negative features. Two columns are deliberately excluded, each after measuring the
degeneracy: a **constant** (a constant cost makes every plan optimal, so `min_θ R` reaches 0 on any observation) and,
as the main configuration, the **mask** (with `θ ≥ 0` it can only raise the cost of unexplored pairs; it also admits a
near-trivial `θ = e_mask`, `F_train ≈ 3` on this instance).

## Results

**Status: the first version of this table was scale-biased and is superseded.** `R(αc) = αR(c)`, so over
unnormalised features the simplex can lower the regret by selecting the column with the smallest entries. Every
feature column is now scaled to unit Frobenius norm, and the metric reported is a **share**, not a ratio:

- `share = R_test / R_independent`: 0 means the held-out flow is as cheap as the forward optimum, 1 means it is as
  wasteful as the "pickup and drop-off are unrelated" plan (a product of the marginals under the *same* cost);
- `fraction = R_test / ⟨c, π̂⟩`: the regret as a share of the flow's own transport cost.

Both are invariant to `c → αc`, so neither can be lowered by shrinking the cost.

**The control that settles it** (`--geometries shuffled`): permute the distance *values* among the pairs that have a
training trip, keeping the support and the mask exactly as they are. The flow keeps its real labels, so the only
thing destroyed is the assignment of distances to geography.

Six leave-one-window-out folds, strict scope, `out/related_nyc_taxi_shuffle2/records.json`:

| arm | RMSE | Pearson | CPC | share (mean) |
| --- | ---: | ---: | ---: | ---: |
| gravity | 13.90 | 0.9215 | 0.8257 | — |
| independence (`IPF(ones)`) | 20.07 | 0.8207 | 0.6816 | — |
| relaxed, median, **real** geometry | 122.21 | 0.3541 | 0.1107 | 1.4568 |
| relaxed, median, **shuffled** distance | 126.56 | 0.1678 | 0.0471 | 0.9000 |
| relaxed, mask, real geometry | 122.38 | 0.3526 | 0.1109 | 0.2693 |
| relaxed, mask, shuffled distance | 126.13 | 0.1719 | 0.0485 | 0.2699 |

Per fold, median arm (real → shuffled): am 0.340 → 1.119, midday 6.801 → 1.198, pm 0.534 → 1.743,
night 0.535 → 0.629, weekend_day 0.201 → 0.300, weekend_night 0.330 → 0.411. `θ` is the pure `s²` column on
every fold; `fraction ≈ 0.99–1.00` on every fold.

## What the table shows, read carefully

1. **The old ratio was scale.** Normalising the features moved the main configuration from `ratio 0.0045` to
   `share 1.46` — i.e. *worse* than the unrelated plan — and changed the selected column from `s` to `s²`. The column
   `s` had simply been the one with the smallest entries.
2. **One fold dominates the mean.** Real geometry beats the shuffled control on **5 of 6** folds (0.34 vs 1.12, 0.53
   vs 1.74, 0.53 vs 0.63, 0.20 vs 0.30, 0.33 vs 0.41) but reverses on `weekday_midday` (6.80 vs 1.20), which carries
   the mean. That fold is also the one where the strict distance scope changed the training objective from 172 to
   1419: removing its train trips turns many pairs into `NaN`, the median imputation places them at 7.5 mi, and the
   forward OT uses them as new routes, so `OT(c)` falls and `R` rises.
3. **The mask arm carries no geographic information at all**: real 0.2693 vs shuffled 0.2699, identical per fold to
   three decimals. Its number is set by the support, not by the map.
4. **`fraction ≈ 1`**: the fitted cost admits a nearly free forward plan (`OT(c) ≈ 0`, because `θ = e_{s²}` makes
   staying near one's own zone almost free) while the observed flow pays close to its full cost. The observation does
   not look efficient under the fitted cost.
5. **The flow columns do not move** across the four cells (122.2–126.6 RMSE, correlation 0.17–0.35): they are
   decoder-limited, an exact-OT plan having at most 525 nonzeros against ~9 000 occupied cells. Gravity wins them.
6. M-IOT is **not implemented**; its published `4.45 / 0.9989 / 0.9566` is the regime, not a reproduction.

## What this cannot answer

A dense trip-count matrix is not an optimal-transport plan, so it cannot test whether the relaxed formulation
recovers a *cost*. The clean test of generalisation is a planted instance: several marginals whose observed plans are
exactly optimal at one shared `θ*`, some windows held out, scored by `R` on the held-out windows and by whether the
fitted cost's plan matches theirs. That is not in this study yet.

## Reproduce

```bash
python -m iot_experiments.studies.related.nyc_taxi.instance --months 2016-01   # downloads 145 MB, writes the npz
python -m iot_experiments.studies.related.nyc_taxi.run --study related_nyc_taxi_imputation --restrict-edges
```

The instance builder needs `pyarrow==16.1.0` (numpy 1.x compatible); it is not a package dependency, because
`pyproject.toml` is kept to the `iot*` include and the `psd` extra.

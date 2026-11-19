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

## Results (6 folds, `out/related_nyc_taxi_imputation/records.json`)

| arm | RMSE | Pearson | CPC | regret ratio |
| --- | ---: | ---: | ---: | ---: |
| gravity (strict scope) | 13.90 | 0.9215 | 0.8257 | — |
| gravity (all-train, sensitivity) | 13.87 | 0.9215 | 0.8258 | — |
| independence (`IPF(ones)`) | 20.07 | 0.8207 | 0.6816 | — |
| relaxed, **median + strict** (main) | 122.28 | 0.3537 | 0.1107 | **0.0045** |
| relaxed, median + all-train | 122.66 | 0.3517 | 0.1094 | 0.0024 |
| relaxed, mask + strict | 122.38 | 0.3526 | 0.1109 | 0.1043 |
| relaxed, mask + all-train | 123.03 | 0.3501 | 0.1102 | 0.0743 |
| relaxed, edges + strict | 122.28 | 0.3537 | 0.1107 | 0.0045 |
| relaxed, edges + all-train | 122.66 | 0.3517 | 0.1094 | 0.0024 |

Per-fold, the strict-scope median arm gives `θ = (0, 1, 0, 0)` every fold — the pure `s = d / max(d)` column — with
ratios 0.0007, 0.0014, 0.0016, 0.0034, 0.0063, 0.0138. The mask arm always picks `θ = e_mask`.

## What the table shows

1. **The imputation moves the regret metric by 20–40×** (0.0045 → 0.1043 for median → mask, strict scope). It does not
   move the flow metrics at all (122.3 → 122.4 RMSE), because those are set by the decoder: an exact-OT plan has at
   most `m + n − 1 = 525` nonzeros against ~9 000 occupied cells.
2. **The scope moves it by ~1.9× on average and by 15× on one fold** (`weekday_midday`: 0.0138 strict vs 0.00088
   all-train, `F_train` 1443 vs 172). The leak was not uniform across windows.
3. **The edge restriction is inert here**: bit-identical to `median`, with zero mass placed on forbidden pairs, and
   183–613 held-out trips (0.06–0.18 %) that the predicted plan cannot carry. Median imputation already places those
   pairs at 7.66 mi, so forbidding them changes nothing.
4. **Gravity is robust to both knobs** (13.87 vs 13.90 RMSE, identical correlation and CPC) and improved from 14.46
   once the distance matrix stopped using a zero sentinel.
5. M-IOT itself is **not implemented**; its published `4.45 / 0.9989 / 0.9566` is quoted as the regime, not
   reproduced.

## Reproduce

```bash
python -m iot_experiments.studies.related.nyc_taxi.instance --months 2016-01   # downloads 145 MB, writes the npz
python -m iot_experiments.studies.related.nyc_taxi.run --study related_nyc_taxi_imputation --restrict-edges
```

The instance builder needs `pyarrow==16.1.0` (numpy 1.x compatible); it is not a package dependency, because
`pyproject.toml` is kept to the `iot*` include and the `psd` extra.

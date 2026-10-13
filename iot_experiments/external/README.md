# external — run the solver on an external benchmark

**Status: placeholder.** To be designed once the external benchmark(s) are chosen.

## Intended contract
An external benchmark plugs into the same core, so results are directly comparable to the
synthetic studies. Each benchmark provides one adapter:

```python
def load(name: str, seed: int) -> ExternalInstance:
    """Return (features, plan, source_mass, target_mass, meta)."""
```

`ExternalInstance` is then converted to an `iot.InverseOTProblem`. Requirements:

- `plan` must be **exactly optimal** for some `theta` (clean), or the run is reported as
  out-of-scope for the exact method (`AGENTS.md` §4.6) — never thresholded silently;
- every record still carries a `core.records.RunSpec` (with `family="external:<name>"`), so
  `is_optimal`, timings, and the master/oracle split are reported the same way as the
  synthetic studies;
- figures/logs follow `AGENTS.md` §7.2 (caption + CSV next to every figure).

## Open questions
- Which benchmark(s)? domain, size range, whether marginals are given or estimated;
- how to define the feature family `Phi` for a real dataset (learned embeddings, distances,
  or an explicit tensor);
- whether the observed plan is clean, sampled, or noisy.

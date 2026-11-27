"""
EN: A small parametric cost-family screen on the two 2-D toys.  Diagonal weights `(1, r)` around L2 (`r=1`) and
    RBF bandwidth scales around the median-bandwidth choice (`scale=1`).  This is a low-dimensional black-box
    search, not a high-capacity learned cost.

CN: 两个二维玩具上的小型参数化 cost family 筛选。对角权重 `(1, r)`（L2 为 `r=1`）与 RBF 带宽缩放（中位带宽为
    `scale=1`）。这是低维 black-box 搜索，不是高容量学习代价。

    python -m experiments.studies.flow_matching_cost_learning.parametric
"""

from __future__ import annotations

import argparse

import torch

from experiments.harness.records import RunLog, RunRecord, write_records
from experiments.studies.flow_matching_cost_learning.run import OUT, _Features, _commit, evaluate, train
from experiments.studies.flow_matching.data import dataset
from experiments.studies.flow_matching.run import BATCH

TARGETS = ("eight_gaussians", "two_moons")
DIAG = (0.25, 0.5, 2.0, 4.0)
RBF = (0.25, 0.5, 2.0, 4.0)


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = argparse.ArgumentParser(description="parametric toy cost screen")
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--tag", default="parametric")
    args = parser.parse_args(argv)
    out_dir = OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for name in args.targets:
        data = dataset(name, 8192, 10001)
        specs: list[tuple[str, _Features, dict[str, object]]] = []
        for value in DIAG:
            weights = (1.0, float(value)) if data.shape[1] == 2 else (1.0,) + (float(value),) * (data.shape[1] - 1)
            specs.append((f"diag_r{value:g}", _Features("diag", data, device, weights=weights),
                          {"family": "diag", "value": value}))
        for value in RBF:
            specs.append((f"rbf_s{value:g}", _Features("rbf", data, device, scale=value),
                          {"family": "rbf", "value": value}))
        for label, features, config_extra in specs:
            for seed in args.seeds:
                model, info = train(name, seed, cost_name=label, steps=args.steps, device=device, log=log,
                                    features=features)
                for row in evaluate(model, name=name, seed=seed, device=device):
                    records.append(RunRecord(
                        study="flow_matching_cost_learning", geometry=name, arm=f"cost_{label}",
                        k=int(model.dim), m=BATCH, draw=seed,
                        metrics={**row, **info},
                        config={"stage": "parametric_screen", "dataset": name, "cost": label,
                                **config_extra, "seed": seed, "steps": args.steps, "batch": BATCH,
                                "sampler": "euler", "nfe": int(row["nfe"]), "device": device,
                                "commit": _commit()},
                        extra={}))
    write_records(out_dir / "records.json", records,
                  provenance={"question": "does a low-dimensional cost family improve toy OT-CFM?",
                              "coupling": "exact OT on a fresh 256-point pool, as w2_mb",
                              "diag": list(DIAG), "rbf_scale": list(RBF), "seeds": args.seeds,
                              "steps": args.steps, "note": "parametric screen; no high-capacity learner",
                              "commit": _commit()})
    log.line(f"parametric screen done: {len(records)} records")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

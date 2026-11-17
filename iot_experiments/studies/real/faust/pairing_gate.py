"""
EN: **Does a globally constrained transport plan beat NICP's independent argmin -- on a cost matrix we already
    have, before NSR is available?**

    This is the gate for the OT-on-NICP idea, run on the one cost matrix this machine can build today:
    the diagonal descriptor cost on two FAUST registrations. It does NOT test NSR. It answers the prior
    question -- *when a pairwise cost is given, does replacing per-row argmin with a balanced transport plan
    change the correspondence, and does it help?* -- on real meshes with real ground truth.

    Ground truth is exact here: registrations share one vertex ordering, so the true correspondence is the
    identity (gate 6.0 in `FAUST_PRELIMINARY_REPORT.md`). Error is geodesic on the target, normalised by
    `sqrt(surface area)`.

    **What is isolated.** `theta` is held FIXED across both arms, so the only difference is the pairing rule.
    Any difference in error is attributable to argmin vs transport, not to the weights.

    **Why this pair of arms is the right control for NSR.** NICP's cost is `C[k,i] = ||F(y_k)_i||^2` with a
    rectangular `K x M` support; ours is a square `n x n`. The decoders are the same code and the square case
    is the special case where a balanced plan is a permutation. If transport does not beat argmin here, the
    argument that it will beat it inside NSR is weak and should not be assumed.

CN: **在拿到 NSR 之前，用我们已有的代价矩阵检验"全局运输是否胜过逐行 argmin"。** 真值精确（配准共享顶点序，
    恒等映射即真值），误差为目标上的测地距离除以 `sqrt(area)`。θ 在两个臂中**保持不变**，因此差异只能来自
    配对规则。NICP 的代价是矩形 `K x M`，这里是方阵 `n x n`，解码器相同，方阵只是平衡方案退化为置换的特例。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import (  # noqa: E402
    default_registrations_root,
    geodesic_matrix,
    read_ply,
    shared_index,
    surface_area,
)
from iot_experiments.harness.faust_match import cost_matrix, geodesic_errors  # noqa: E402
from iot_experiments.harness.nicp_pairing import (  # noqa: E402
    decode_argmin,
    decode_transport,
    diagnose,
)
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import RunLog, RunRecord, write_records  # noqa: E402
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    descriptors,
    fit_theta,
    standardise,
)
from iot.core.features import SquaredDifferenceFeatures  # noqa: E402

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "does a balanced transport plan beat per-row argmin on a given pairwise cost matrix?",
    "ground_truth": "identity -- registrations share one vertex ordering (gate 6.0)",
    "isolated": "theta is FIXED across arms, so the only difference is the pairing rule",
    "error": "geodesic on the target divided by sqrt(surface area)",
    "not_nsr": "this uses OUR descriptor cost, not NSR's offset cost; it is the control for the NSR gate",
    "arms": ["argmin (NICP's rule)", "balanced transport (uniform marginals, exact LP)"],
}


@dataclasses.dataclass(frozen=True)
class GateConfig(RunConfig):
    """EN: knobs; real dataclass fields so `from_namespace` cannot drop them. CN: 旋钮。"""

    section = "faust_pairing_gate"
    study = "faust_pairing_gate"
    root: str = DEFAULT_ROOT
    pairs: tuple[int, ...] = (0, 66)
    n_sub: int = 800
    theta_mode: str = "uniform"

    def __post_init__(self) -> None:
        """EN: normalise and validate. CN: 归一化与校验。"""
        object.__setattr__(self, "pairs", tuple(int(v) for v in self.pairs))
        if not self.pairs or len(self.pairs) % 2 != 0:
            raise ValueError("--pairs needs an even, non-empty list of ids")
        if int(self.n_sub) < 2:
            raise ValueError("--n-sub must be at least 2")
        if self.theta_mode not in ("uniform", "learned"):
            raise ValueError(f"unknown theta_mode {self.theta_mode!r}")

    @classmethod
    def add_study_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """EN: only this study's flags. CN: 只加本研究特有的参数。"""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--pairs", type=int, nargs="+", default=[0, 66])
        parser.add_argument("--n-sub", type=int, default=800)
        parser.add_argument("--theta-mode", default="uniform", choices=["uniform", "learned"])


def main(argv: list[str] | None = None) -> int:
    """EN: entry point. CN: 入口。"""
    parser = build_parser(GateConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = GateConfig.from_namespace(args, parser)
    for flag in ("pairs", "n_sub"):
        parsed = getattr(args, flag, None)
        carried = getattr(config, flag, None)
        if parsed is not None and carried is not None:
            left = tuple(int(v) for v in parsed) if isinstance(parsed, (list, tuple)) else int(parsed)
            right = tuple(int(v) for v in carried) if isinstance(carried, (list, tuple)) else int(carried)
            if left != right:
                raise SystemExit(f"flag --{flag}={parsed!r} did not reach the config ({carried!r})")

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    log.line(f"pairing gate: theta_mode={config.theta_mode} n_sub={config.n_sub} pairs={config.pairs}")
    preflight = preflight_check(run_tiny=False, need_gpu=False)
    preflight.write(run_dir / "preflight.json")
    log.line(f"preflight ok={preflight.ok} mem={preflight.mem_available_gb:.1f} GB")

    root = Path(config.root)
    records: list[RunRecord] = []
    for start in range(0, len(config.pairs), 2):
        source_id, target_id = int(config.pairs[start]), int(config.pairs[start + 1])
        geometry = f"pair{source_id:03d}-{target_id:03d}"
        log.line(f"[{geometry}] start")
        started = time.perf_counter()
        a_mesh = read_ply(root / f"tr_reg_{source_id:03d}.ply")
        b_mesh = read_ply(root / f"tr_reg_{target_id:03d}.ply")
        index = shared_index(a_mesh, int(config.n_sub), seed=0)
        da, db = standardise(descriptors(a_mesh)[index], descriptors(b_mesh)[index])
        n = int(da.shape[0])
        k = int(da.shape[1])
        if config.theta_mode == "uniform":
            # EN: fixed weights isolate the pairing rule. CN: 固定权重以隔离配对规则。
            theta = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            fit = fit_theta(SquaredDifferenceFeatures(da, db), rel=None, max_iterations=5)
            if fit["theta"] is None:
                raise SystemExit(f"[{geometry}] theta fit failed: {fit}")
            theta = np.asarray(fit["theta"], dtype=np.float64)
        log.line(f"[{geometry}] n={n} K={k} theta_mass_hks={theta[5:].sum():.3f}")

        cost = cost_matrix(da, db, theta)
        geo = geodesic_matrix(b_mesh.vertices, b_mesh.faces, index.astype(np.int64))
        norm = float(np.sqrt(surface_area(b_mesh.vertices, b_mesh.faces)))

        for rule, pairing in (("argmin", decode_argmin(cost)),
                              ("transport", decode_transport(cost))):
            pred = np.asarray(pairing.assignment, dtype=np.int64)
            errors = geodesic_errors(pred, geo, norm)
            stats = diagnose(pairing, n_template=n)
            records.append(RunRecord(
                study=str(config.study), geometry=geometry, arm=rule, k=k, m=n, draw=0,
                metrics={"mean_error": errors["mean"], "median_error": errors["median"],
                         "miss_1pct": errors["miss_1pct"], "miss_5pct": errors["miss_5pct"],
                         "exact": float(np.mean(pred == np.arange(n)))},
                config=config.to_dict(),
                extra={"theta_mode": config.theta_mode,
                       "template_coverage": stats.template_coverage,
                       "max_load": stats.max_load, "mean_load": stats.mean_load,
                       "load_imbalance": stats.load_imbalance,
                       "excess_repeat_fraction": stats.excess_repeat_fraction,
                       "repeat_rate": stats.repeat_rate,
                       "unique_templates": stats.unique_templates,
                       "transport_cost": stats.transport_cost,
                       "n_vertices_in_plan": n}))
            log.line(f"[{geometry}] {rule:>9} mean={errors['mean']:.4f} "
                     f"miss5%={errors['miss_5pct']:.3f} exact={np.mean(pred == np.arange(n)):.3f} "
                     f"coverage={stats.template_coverage:.3f} "
                     f"imbalance={stats.load_imbalance:.2f} "
                     f"excess_repeat={stats.excess_repeat_fraction:.3f}")
        log.line(f"[{geometry}] done in {time.perf_counter() - started:.1f}s")

    write_records(run_dir / "records.json", records, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

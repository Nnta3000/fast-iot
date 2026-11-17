"""
EN: **Deformation sweep -- accuracy against a controlled, non-isometric distortion.**

    The FAUST scans vary pose, but pose is only one axis of difficulty. This module applies a **smooth,
    deterministic deformation** to a registered mesh with a single strength knob, so the relationship between
    deformation and accuracy can be plotted instead of inferred.

    The deformation is a **twist about the vertical axis**: with `y` normalised to `[0, 1]` over the mesh's
    extent,

        x' = cos(a) x - sin(a) z,   z' = sin(a) x + cos(a) z,   a = sigma * pi * y_norm

    so `sigma = 0` is the original mesh and `sigma = 1` twists the top of the body by half a turn relative to
    the feet. **Vertex identity is preserved exactly** — this is a point-map, not a remeshing — so the true
    correspondence stays `π̂ = I/n` and no new ground truth is needed.

    **What it is and is not.** A twist is *not* an isometry, so intrinsic descriptors (HKS, curvature) are not
    invariant to it; that is the point — the sweep measures how fast they degrade. It is *not* an articulation
    (no bending at joints), so it should not be read as "the method tolerates real pose change". The FAUST
    pose axis (real scans, §2) remains the evidence for that.

    CN: **形变扫描**：对注册网格施加一个**平滑、确定性**的扭转（绕竖轴），强度 `sigma` 可调；顶点身份保持，
        真对应仍是 `π̂ = I/n`。扭转**不是等距**，所以内在描述子会退化——这正是要测的；但它**不是关节运动**，
        不能当作"方法能容忍真实姿态变化"的证据。

    Run:
        .venv/bin/python -u -m iot_experiments.studies.real.faust.deformation --n-sub 2000
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from iot_experiments.harness.config import RunConfig, build_parser  # noqa: E402
from iot_experiments.harness.faust import default_registrations_root, FaustMesh, read_ply  # noqa: E402
from iot_experiments.harness.preflight import check as preflight_check  # noqa: E402
from iot_experiments.harness.records import (  # noqa: E402
    RunLog,
    RunRecord,
    format_table,
    summarise,
    write_records,
)
from iot_experiments.studies.real.faust.correspondence import (  # noqa: E402
    _evaluate_pair,
    _sample_index,
    descriptors,
    fit_theta,
    standardise,
)

DEFAULT_ROOT = str(default_registrations_root())

PROVENANCE: dict[str, object] = {
    "question": "how fast does correspondence accuracy degrade under a controlled non-isometric deformation?",
    "deformation": "twist about the vertical axis, angle a = sigma*pi*y_norm, applied to the TARGET only; "
                   "sigma in the swept set, sigma=0 = identity",
    "vertex_identity": "the deformation is a point map, so vertex i still corresponds to vertex i and pi_hat = "
                       "I/n is unchanged -- no new ground truth is needed",
    "not_an_articulation": "a twist is not isometric and is not joint bending; it measures descriptor "
                           "degradation, not tolerance to real pose change (the FAUST pose axis covers that)",
    "theta": "fitted ONCE on the undeformed training pair, then applied at every sigma -- so the curve is "
             "transfer, not refitting",
    "metric": "geodesic error on the deformed target, normalised by sqrt(surface area), as everywhere else",
}


@dataclasses.dataclass(frozen=True)
class DeformationConfig(RunConfig):
    """CN: 旋钮。EN: knobs."""

    study: str = "faust_deformation"
    section: str = "faust_deformation"
    root: str = DEFAULT_ROOT
    train_pair: tuple[int, ...] = (0, 13)
    pair: tuple[int, ...] = (50, 66)
    n_sub: int = 2000
    sigmas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    max_iterations: int = 30

    @classmethod
    def add_study_arguments(cls, parser) -> None:
        """CN: 只加本研究特有的参数。EN: only this study's flags."""
        parser.add_argument("--root", default=DEFAULT_ROOT)
        parser.add_argument("--train-pair", type=int, nargs=2, default=[0, 13])
        parser.add_argument("--pair", type=int, nargs=2, default=[50, 66])
        parser.add_argument("--n-sub", type=int, default=2000)
        parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
        parser.add_argument("--max-iterations", type=int, default=30)


def twist(mesh: FaustMesh, sigma: float) -> FaustMesh:
    """
    CN: 绕竖轴扭转，角度随高度线性增长：`a = sigma*pi*y_norm`。只改坐标，顶点顺序与面都不动。
    EN: twist about the vertical axis with the angle linear in height: `a = sigma*pi*y_norm`. Coordinates only;
        vertex order and faces are untouched.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    y = v[:, 1]
    span = float(y.max() - y.min())
    y_norm = (y - y.min()) / span if span > 1e-12 else np.zeros_like(y)
    angle = float(sigma) * np.pi * y_norm
    c, s = np.cos(angle), np.sin(angle)
    x, z = v[:, 0].copy(), v[:, 2].copy()
    out = v.copy()
    out[:, 0] = c * x - s * z
    out[:, 2] = s * x + c * z
    return FaustMesh(vertices=np.ascontiguousarray(out), faces=mesh.faces, path=mesh.path)


def main(argv: list[str] | None = None) -> int:
    """CN: 入口。EN: entry point."""
    parser = build_parser(DeformationConfig)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = DeformationConfig.from_namespace(args, parser)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "run.log")
    pre = preflight_check(run_tiny=False, need_gpu=False)
    pre.write(run_dir / "preflight.json")
    log.line(f"preflight ok={pre.ok} mem={pre.mem_available_gb:.1f} GB")

    root = Path(config.root)
    a_tr = read_ply(root / f"tr_reg_{int(config.train_pair[0]):03d}.ply")
    b_tr = read_ply(root / f"tr_reg_{int(config.train_pair[1]):03d}.ply")
    idx = _sample_index(a_tr, config, 0)
    da, db = standardise(descriptors(a_tr)[idx], descriptors(b_tr)[idx])
    fit = fit_theta(features=_features(da, db), rel=None,
                    max_iterations=int(config.max_iterations))
    theta = np.asarray(fit["theta"], dtype=np.float64)
    log.line(f"theta fitted once at sigma=0: regret={fit['regret']:.4g} "
             f"|F|={fit['edges']} theta={np.round(theta, 4).tolist()}")

    a_mesh = read_ply(root / f"tr_reg_{int(config.pair[0]):03d}.ply")
    b_mesh = read_ply(root / f"tr_reg_{int(config.pair[1]):03d}.ply")

    rows: list[RunRecord] = []
    for sigma in [float(v) for v in config.sigmas]:
        deformed = twist(b_mesh, sigma)
        log.line(f"sigma={sigma:g}: target deformed (vertex identity preserved)")
        these = {"learned": theta}
        for arm, th in (
            ("position_only", _one_hot(da.shape[1], 3)),
            ("uniform_theta", np.full(da.shape[1], 1.0 / da.shape[1])),
            ("uniform_late_hks", _late_hks(da.shape[1])),
        ):
            these[arm] = th
        pair_rows = _evaluate_pair(a_mesh, deformed, config=config, thetas=these, seed=0)
        rows.extend(dataclasses.replace(r, geometry=f"sigma{sigma:g}") for r in pair_rows)
        learn = next(r for r in pair_rows if r.arm == "learned")
        unif = next(r for r in pair_rows if r.arm == "uniform_theta")
        log.line(f"  sigma={sigma:g}  learned miss@5%={learn.metrics['miss_5pct']:.3f}  "
                 f"uniform={unif.metrics['miss_5pct']:.3f}  "
                 f"position={next(r for r in pair_rows if r.arm == 'position_only').metrics['miss_5pct']:.3f}")

    table = summarise(rows, group=("geometry", "arm"), metrics=("mean", "miss_5pct"))
    log.line("\n" + format_table(table, ("geometry", "arm", "n_draws", "mean_mean", "miss_5pct_mean")))
    write_records(run_dir / "records.json", rows, provenance=PROVENANCE)
    log.line(f"wrote {run_dir / 'records.json'} ({len(rows)} rows)")
    log.close()
    return 0


def _features(da: Float[np.ndarray, "n k"], db: Float[np.ndarray, "n k"]):
    """CN: 按需特征族。EN: the on-demand feature family."""
    from iot.core.features import SquaredDifferenceFeatures  # noqa: PLC0415

    return SquaredDifferenceFeatures(da, db)


def _one_hot(k: int, take: int) -> Float[np.ndarray, "k"]:
    """CN: 只给前 `take` 个坐标等权的 θ。EN: uniform over the first `take` coordinates."""
    theta = np.zeros(int(k), dtype=np.float64)
    theta[: int(take)] = 1.0 / float(take)
    return theta


def _late_hks(k: int) -> Float[np.ndarray, "k"]:
    """CN: 只给最后三列（t=32/64/128）等权的 θ。EN: uniform over the last three (late-HKS) columns."""
    theta = np.zeros(int(k), dtype=np.float64)
    theta[-3:] = 1.0 / 3.0
    return theta


if __name__ == "__main__":
    raise SystemExit(main())

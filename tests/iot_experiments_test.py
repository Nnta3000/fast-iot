from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "iot_experiments"


def _run(script: str, args: list[str], out: Path) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(EXPERIMENTS / script), *args, "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_e1_full_master_script(tmp_path: Path) -> None:
    payload = _run("studies/synthetic/methods/exact/e1_full_master.py", ["--sizes", "4", "6", "--features", "3"], tmp_path / "e1.json")
    assert payload["experiment"] == "e1_full_master"
    rows = payload["rows"]
    assert isinstance(rows, list) and len(rows) == 2
    assert all(row["all_optimal"] for row in rows)


def test_e2_loop_vs_full_script(tmp_path: Path) -> None:
    payload = _run("studies/synthetic/methods/exact/e2_loop_vs_full.py", ["--sizes", "6", "--features", "3"], tmp_path / "e2.json")
    assert payload["experiment"] == "e2_loop_vs_full"
    row = payload["rows"][0]
    assert float(row["delta_difference"]) < 1e-5
    assert float(row["theta_distance"]) < 1e-6
    assert float(row["dual_distance"]) < 1e-6
    assert bool(row["is_optimal"])
    assert int(row["support_loop"]) < int(row["support_full"])


def test_e3_scaling_script(tmp_path: Path) -> None:
    payload = _run(
        "studies/synthetic/methods/exact/e3_scaling.py",
        ["--sizes", "20", "--features", "3", "--max-iterations", "20"],
        tmp_path / "e3.json",
    )
    assert payload["experiment"] == "e3_scaling"
    row = payload["rows"][0]
    assert float(row["iterations"]) >= 1.0
    assert float(row["final_support"]) < int(row["n_source"]) * int(row["n_target"])
    assert float(row["certificate_fraction"]) == 1.0


def test_cutting_plane_benchmark_script(tmp_path: Path) -> None:
    payload = _run(
        "studies/synthetic/methods/exact/cutting_plane_benchmark.py",
        ["--sizes", "30", "--features", "3", "--seeds", "0", "--methods", "cutting-plane"],
        tmp_path / "cutting_plane.json",
    )
    assert payload["experiment"] == "cutting_plane_benchmark"
    rows = payload["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "cutting-plane"
    assert bool(row["converged"])
    assert int(row["oracle_calls"]) >= 1
    assert int(row["oracle_calls"]) == int(row["iterations"])
    assert bool(row["is_optimal"])


def test_entropic_psd_moment_fit_script(tmp_path: Path) -> None:
    payload = _run(
        "studies/synthetic/methods/entropic/entropic_psd_moment_fit.py",
        ["--n", "40", "--dim", "2", "--num-outer", "15", "--samples", "2000"],
        tmp_path / "entropic.json",
    )
    assert payload["experiment"] == "entropic_psd_moment_fit"
    exact = next(row for row in payload["rows"] if row["scenario"] == "exact")
    assert float(exact["exact_gap"]) <= float(exact["exact_gap_initial"]) + 1e-9
    assert float(exact["scale_error"]) < 0.5

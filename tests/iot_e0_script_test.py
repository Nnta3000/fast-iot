from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
E0_SCRIPT = (
    REPO_ROOT / "iot_experiments" / "legacy" / "methods" / "exact" / "e0_fixed_fg.py"
)


def test_e0_script_writes_valid_report(tmp_path: Path) -> None:
    out = tmp_path / "e0.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(E0_SCRIPT),
            "--m",
            "10",
            "--n",
            "10",
            "--features",
            "3",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["theta_star_in_range"] is True
    assert payload["is_optimal_plan"] is True
    assert len(payload["lower"]) == 3
    assert len(payload["width"]) == 3
    assert payload["support_size"] > 0
    assert payload["relative_gap"] <= 1e-9

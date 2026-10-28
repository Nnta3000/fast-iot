"""
CN: 资源看门狗的断言。看门狗是**进程自保**机制，因此必须证明它真的会中止，而不是只打印一行。

    CN: 中止用 `os._exit(125)`，所以测试必须在**子进程**里跑，否则会把 pytest 自己杀掉。
EN: Assertions for the resource watchdog. It is a self-protection mechanism, so it must be proven to actually
    abort rather than merely print.

    The abort is `os._exit(125)`, so each test runs in a **subprocess**; otherwise it would kill pytest itself.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_EXIT_CODE = 125


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """CN: 在子进程里运行一段代码。EN: run a snippet in a subprocess."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          cwd=str(REPO_ROOT))


def test_watchdog_aborts_when_rss_exceeds_budget() -> None:
    """CN: RSS 超过预算必须中止（退出码 125）。EN: exceeding the RSS budget must abort with exit 125."""
    code = (
        "import numpy as np\n"
        "from iot_experiments.harness.watchdog import Watchdog\n"
        "Watchdog(budget_gb=0.5, name='t', min_available_gb=0.0, interval_s=0.1).start()\n"
        "a = np.ones((300_000_000,), dtype=np.float64)\n"
        "a[:] = 1.0\n"
        "import time; time.sleep(3)\n"
        "print('NOT ABORTED')\n"
    )
    result = _run(code)
    assert result.returncode == WATCHDOG_EXIT_CODE, (result.returncode, result.stdout[-300:])
    assert "exceeds budget" in result.stdout
    assert "NOT ABORTED" not in result.stdout


def test_watchdog_aborts_when_system_memory_is_low() -> None:
    """
    CN: 系统可用内存低于下限也必须中止。用一个不可能满足的下限（1e9 GB）来触发，避免真的耗尽内存。
    EN: an impossible floor (1e9 GB available required) triggers the low-memory path without actually
        exhausting memory.
    """
    code = (
        "from iot_experiments.harness.watchdog import Watchdog\n"
        "Watchdog(budget_gb=100.0, name='t', min_available_gb=1e9, interval_s=0.1).start()\n"
        "import time; time.sleep(3)\n"
        "print('NOT ABORTED')\n"
    )
    result = _run(code)
    assert result.returncode == WATCHDOG_EXIT_CODE, (result.returncode, result.stdout[-300:])
    assert "MemAvailable" in result.stdout


def test_watchdog_does_not_fire_on_normal_completion() -> None:
    """CN: 正常结束（内存远低于预算）不得中止。EN: normal completion under budget must not abort."""
    code = (
        "import numpy as np\n"
        "from iot_experiments.harness.watchdog import Watchdog\n"
        "w = Watchdog(budget_gb=4.0, name='t', min_available_gb=0.0, interval_s=0.1).start()\n"
        "a = np.ones((10_000_000,), dtype=np.float64)\n"
        "a[:] = 1.0\n"
        "w.disarm()\n"
        "print('COMPLETED')\n"
    )
    result = _run(code)
    assert result.returncode == 0, (result.returncode, result.stdout[-300:], result.stderr[-300:])
    assert "COMPLETED" in result.stdout

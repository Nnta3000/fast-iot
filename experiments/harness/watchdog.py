"""
CN: **资源看门狗** —— 每个驱动都可以装的守护线程，超过预算就**主动中止**，而不是等内核杀进程。

    CN: 为什么需要：本机**没有**可用的硬内存上限机制（实测：`prlimit --rss` 不生效；`ulimit -v`
        会破坏 CUDA；`systemd-run --user` 无 user bus；cgroup 子组无权限）。因此只能靠进程自己检查并退出。

    CN: 两条规则：
        1. 本进程 RSS 超过 `budget_gb` ⇒ 中止；
        2. 系统 `MemAvailable` 低于 `min_available_gb`（默认 1.5 GB）⇒ 中止。
        中止方式是 `os._exit(WATCHDOG_EXIT)`（**不是** SIGKILL）：进程立刻消失、退出码可辨认，
        不会因为 Python 层的清理而再多占内存。

    CN: 线程是 daemon，每 `interval_s` 秒检查一次；`Watchdog.disarm()` 在正常结束时关闭它。

EN: **Resource watchdog** -- a daemon thread any driver can install; it aborts **itself** on budget overrun
    instead of waiting for the kernel.

    Why: this machine has **no** usable hard memory cap (measured: `prlimit --rss` does not enforce;
    `ulimit -v` breaks CUDA; `systemd-run --user` has no user bus; cgroup child creation is denied). So the
    process must police itself and exit.

    Two rules: (1) this process's RSS above `budget_gb` aborts; (2) system `MemAvailable` below
    `min_available_gb` (1.5 GB default) aborts. Abort is `os._exit(WATCHDOG_EXIT)`, not SIGKILL: the process
    vanishes at once with a recognisable exit code and does not spend more memory on interpreter cleanup.

    The thread is a daemon checking every `interval_s`; `Watchdog.disarm()` stops it on normal completion.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

# CN: 看门狗中止时的退出码（125 = 与常见 shell 错误码区分）。EN: exit code used when the watchdog aborts.
WATCHDOG_EXIT: int = 125


class WatchdogTripped(RuntimeError):
    """CN: 中止原因（在主线程里读取时使用）。EN: the abort reason, for the main thread to read."""


def _rss_gb() -> float:
    """CN: 本进程当前 RSS（GB）。EN: this process's current RSS in GB."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:  # pragma: no cover - /proc always present on Linux
        pass
    return 0.0


def _available_gb() -> float:
    """CN: 系统 `MemAvailable`（GB）。EN: system `MemAvailable` in GB."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:  # pragma: no cover
        pass
    return float("inf")


@dataclass
class Watchdog:
    """
    CN: 看门狗。`budget_gb` 是本作业允许的最大 RSS；`min_available_gb` 是系统必须保留的最小可用内存。
        `on_trip` 可选，在**中止前**调用（例如写一行日志），但不要在其中做重活。
    EN: the watchdog. `budget_gb` is the maximum RSS this job may use; `min_available_gb` is the least system
        memory that must remain free. `on_trip` is optional and runs **before** the abort (e.g. to log one
        line); keep it cheap.
    """

    budget_gb: float
    name: str = "job"
    min_available_gb: float = 1.5
    interval_s: float = 1.0
    on_trip: Callable[[str], None] | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    tripped: str | None = field(default=None, init=False)

    def start(self) -> "Watchdog":
        """CN: 启动守护线程。EN: start the daemon thread."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name=f"watchdog-{self.name}", daemon=True)
        self._thread.start()
        return self

    def disarm(self) -> None:
        """CN: 正常结束时关闭。EN: disarm on normal completion."""
        self._stop.set()

    def _trip(self, reason: str) -> None:
        """CN: 记录原因、回调、然后**立即退出进程**。EN: record, call back, then **exit the process now**."""
        self.tripped = reason
        if self.on_trip is not None:
            try:
                self.on_trip(reason)
            except Exception:  # noqa: BLE001 - 回调失败不能阻止中止 / a failing callback must not block the abort
                pass
        print(f"[watchdog] ABORT: {reason}", flush=True)
        os._exit(WATCHDOG_EXIT)

    def _loop(self) -> None:
        """CN: 每隔 `interval_s` 检查一次。EN: check every `interval_s`."""
        while not self._stop.wait(self.interval_s):
            rss = _rss_gb()
            if rss > float(self.budget_gb):
                self._trip(f"{self.name}: RSS {rss:.2f} GB exceeds budget {self.budget_gb:.2f} GB")
                return
            avail = _available_gb()
            if avail < float(self.min_available_gb):
                self._trip(f"{self.name}: system MemAvailable {avail:.2f} GB below the "
                           f"{self.min_available_gb:.2f} GB floor")
                return


__all__ = ["WATCHDOG_EXIT", "Watchdog", "WatchdogTripped"]

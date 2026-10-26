"""
CN: `harness/discriminates.py` 的断言 —— 判别力检查必须能**接受**有信号的指标、**拒绝**噪声指标，
    并且**同中心对照**必须比随机对照更严格（这才是它存在的理由）。

    CN: 项目教训：SSIM/PSNR 在 T1/T2 上区分真配对与跨受试者只有 0.00–0.11 SD，却被当作评估指标用过。
        这些测试把"用前先检查"变成不可绕过的不变量。

EN: Assertions for `harness/discriminates.py` -- the check must **accept** a metric with signal, **reject**
    a noise metric, and the **same-centre control** must be stricter than the random floor (that is the
    whole point of it).

    Lesson being encoded: SSIM/PSNR separated true from cross-subject pairs by only 0.00-0.11 SD on T1/T2
    yet were used as evaluation metrics. These tests make "check before use" a non-bypassable invariant.
"""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float

from iot_experiments.harness.discriminates import (
    DiscriminatesError,
    discrimination_check,
    format_check,
)


def planted(n: int = 60, dims: int = 8, *, seed: int = 0, noise: float = 0.1
            ) -> tuple[Float[np.ndarray, "n dims"], Float[np.ndarray, "n dims"]]:
    """CN: 有真实配对的合成数据：同一样本共享潜码。EN: synthetic data with a real pairing signal."""
    rng = np.random.default_rng(seed)
    z: Float[np.ndarray, "n dims"] = rng.normal(size=(n, dims))
    a: Float[np.ndarray, "n dims"] = np.hstack([z + rng.normal(scale=noise, size=(n, dims)),
                                                rng.normal(size=(n, dims))])
    b: Float[np.ndarray, "n dims"] = np.hstack([z + rng.normal(scale=noise, size=(n, dims)),
                                                rng.normal(size=(n, dims))])
    return a, b


def cosine(a: Float[np.ndarray, "dims"], b: Float[np.ndarray, "dims"]) -> float:
    """CN: 余弦相似度（越大越好）。EN: cosine similarity (higher is better)."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_accepts_a_metric_with_signal_and_rejects_noise() -> None:
    """CN: 有信号的指标通过，纯噪声被拒。EN: a signal metric passes and noise is rejected."""
    a, b = planted()
    n = int(a.shape[0])
    rng = np.random.default_rng(5)
    centres = np.zeros(n, dtype=np.int64)

    def pure_noise(_a: Float[np.ndarray, "dims"], _b: Float[np.ndarray, "dims"]) -> float:
        return float(rng.random())

    result = discrimination_check({"cosine": cosine, "noise": pure_noise},
                                  [a[i] for i in range(n)], [b[i] for i in range(n)],
                                  centres=centres, draws=2, seed=1)
    assert "cosine" in result.usable_metrics()
    assert "noise" not in result.usable_metrics()
    assert result.rejection_summary(), "a rejected metric must explain itself"


def test_same_centre_control_is_stricter_than_random() -> None:
    """
    CN: 构造一个只编码"中心"的捷径：所有样本都等于自己的中心向量，因此同中心配对与真配对一样相似。
        随机对照会认为该指标极好，同中心对照必须把它拦下。
    EN: build a metric that only encodes **centre identity**: every sample equals its centre vector, so a
        same-centre pair is as similar as a true pair. The random floor would call it excellent; the
        same-centre control must reject it.
    """
    n = 60
    centres = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    signal: Float[np.ndarray, "centres 2"] = np.eye(2) * 2.0
    a: Float[np.ndarray, "n 2"] = np.array([signal[c] for c in centres])
    b: Float[np.ndarray, "n 2"] = np.array([signal[c] for c in centres])

    result = discrimination_check({"centre_only": cosine},
                                  [a[i] for i in range(n)], [b[i] for i in range(n)],
                                  centres=centres, draws=1, seed=2)
    report = result.reports[0]
    # CN: 对随机下界看起来完美，对同中心对照毫无优势。
    # EN: perfect against the random floor, no advantage against the same-centre control.
    assert report.effects["random"] > 0.5, report.effects
    assert report.effects["same"] < 0.5, report.effects
    assert not report.usable, report.effects


def test_mismatched_inputs_raise() -> None:
    """CN: 输入长度不一致或样本太少时必须报错，而不是静默截断。"""
    a, b = planted(n=10)
    with pytest.raises(DiscriminatesError):
        discrimination_check({"cos": cosine}, [a[i] for i in range(10)], [b[i] for i in range(9)])
    with pytest.raises(DiscriminatesError):
        discrimination_check({"cos": cosine}, [a[0]] * 3, [b[0]] * 3)


def test_format_check_lists_every_metric() -> None:
    """CN: 表格包含每个指标与判定列。EN: the table lists every metric and a verdict column."""
    a, b = planted(n=20)
    result = discrimination_check({"cosine": cosine}, [a[i] for i in range(20)],
                                  [b[i] for i in range(20)], draws=1)
    table = format_check(result)
    assert "cosine" in table
    assert "usable" in table

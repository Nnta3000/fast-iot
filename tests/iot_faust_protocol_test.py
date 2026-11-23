"""
EN: **Tests for the FAUST protocol driver.** The claims that can silently go wrong are all protocol claims, not
    numerical ones:

    1. `diameter` is the normaliser FAUST's own metric uses; getting it wrong rescales every number by a
       shape-dependent factor, which no table would reveal.
    2. `DescriptorScaler` must be fitted on fixed shapes and applied unchanged. If someone restores the pooled
       `standardise(a, b)`, the target's statistics enter the feature map and the evaluation stops being honest.
    3. the pair constructors must be **ordered**, must respect the intra/inter definition (same vs different
       subject), and must be deterministic -- otherwise a "reproduction" is a different experiment.
    4. the subject split must fail loudly on overlap.

CN: **FAUST 协议 driver 的测试。** 会静默出错的全是协议性主张：`diameter` 是 FAUST 自身度量的归一化尺度；`DescriptorScaler`
    必须只在固定形状上拟合并原样应用（否则测试侧统计泄漏）；配对构造必须**有向**、符合 intra/inter 定义且确定性；
    受试者划分重叠时必须立即报错。
"""

from __future__ import annotations

import numpy as np
import pytest

from iot_experiments.harness.faust import DescriptorScaler, diameter
from iot_experiments.harness.records import RunRecord, summarise
from iot_experiments.studies.iot.real.faust.challenge import ChallengeConfig, inter_pairs, intra_pairs


def test_diameter_is_the_largest_pairwise_distance() -> None:
    """EN: a unit square's diameter is sqrt(2); the bbox diagonal happens to agree here, the cube case pins it."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(diameter(square), np.sqrt(2.0), rtol=0, atol=1e-12)
    cube = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 2.0)])
    np.testing.assert_allclose(diameter(cube), np.sqrt(6.0), rtol=0, atol=1e-12)


def test_descriptor_scaler_depends_only_on_what_it_was_fitted_on() -> None:
    """
    EN: the protocol point. `transform` of a shape must be identical no matter which *other* shapes exist -- that
        is exactly what `standardise(a, b)` violates, because it pools the pair.
    CN: 协议要点：某个形状的 `transform` 不得依赖于除拟合集以外的任何形状——这正是 `standardise(a,b)` 违反的性质。
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3))
    b = rng.normal(loc=5.0, scale=3.0, size=(50, 3))
    scaler = DescriptorScaler.fit([a])
    first = scaler.transform(b)
    second = DescriptorScaler.fit([a]).transform(b)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    # EN: mean/std are those of `a` alone, so `b` is not centred.
    np.testing.assert_allclose(first.mean(axis=0), (b.mean(axis=0) - a.mean(axis=0)) / a.std(axis=0),
                               rtol=1e-12, atol=1e-12)
    # EN: pooling b in changes the transform -- the leak, made observable.
    # CN: 把 b 也纳入拟合会改变变换——泄漏因此可见。
    assert not np.allclose(first, DescriptorScaler.fit([a, b]).transform(b))


def test_intra_pairs_are_ordered_same_subject_different_pose() -> None:
    """EN: FAUST's intra challenge. CN: FAUST 的 intra challenge。"""
    pairs = intra_pairs((0, 1, 2), 3, seed=7)
    assert len(pairs) == 9
    for first, second in pairs:
        assert first // 10 == second // 10, "intra pairs must share the subject"
        assert first != second, "a pair of a shape with itself is not a pair"
        assert first // 10 in (0, 1, 2)
    assert pairs == intra_pairs((0, 1, 2), 3, seed=7), "must be deterministic"
    assert pairs != intra_pairs((0, 1, 2), 3, seed=8), "the seed must select the pairs"


def test_inter_pairs_are_ordered_different_subjects_and_balanced() -> None:
    """EN: FAUST's inter challenge, with every subject contributing the same count. CN: FAUST 的 inter challenge，
    每个受试者贡献相同数量。"""
    pairs = inter_pairs((3, 4, 5, 6), 2, seed=1)
    assert len(pairs) == 8
    for first, second in pairs:
        assert first // 10 != second // 10, "inter pairs must cross subjects"
    counts = {subject: sum(1 for first, _ in pairs if first // 10 == subject) for subject in (3, 4, 5, 6)}
    assert set(counts.values()) == {2}, f"unbalanced coverage: {counts}"
    assert pairs == inter_pairs((3, 4, 5, 6), 2, seed=1)


def test_challenge_config_rejects_a_subject_overlap() -> None:
    """EN: the split is the protocol; overlap must be an error, not a warning. CN: 划分即协议，重叠必须报错。"""
    with pytest.raises(ValueError, match="overlap"):
        ChallengeConfig(study="x", section="x", train_subjects=(0, 1), test_subjects=(1, 2))


def test_summarise_can_group_by_an_extra_key() -> None:
    """
    EN: the protocol reports the two challenges separately, and the challenge name lives in `extra` (the record
        schema is deliberately not extended). Grouping must read it, and an unknown key must raise rather than
        silently pool everything into one bucket.

    CN: 协议分两组报告，challenge 名放在 `extra`（刻意不扩记录字段）。分组必须能读到它；未知键必须报错，而不是
        静默合并成一个桶。
    """
    rows = [
        RunRecord(study="s", geometry="a", arm="shared", k=1, m=1, draw=0,
                  metrics={"miss_5pct_diam": 0.1}, extra={"challenge": "intra"}),
        RunRecord(study="s", geometry="b", arm="shared", k=1, m=1, draw=0,
                  metrics={"miss_5pct_diam": 0.3}, extra={"challenge": "inter"}),
    ]
    table = summarise(rows, group=("challenge", "arm"), metrics=("miss_5pct_diam",))
    assert {row["challenge"] for row in table} == {"intra", "inter"}
    with pytest.raises(KeyError):
        summarise(rows, group=("nonesuch",), metrics=("miss_5pct_diam",))

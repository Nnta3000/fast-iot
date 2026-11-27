"""EN: regression tests for the Stage 0 budget filter in the headroom analysis.
CN: headroom Stage 0 分析中训练预算过滤的回归测试。
"""

from __future__ import annotations

from experiments.studies.flow_matching_headroom.analyse import _sw2_table


def _record(*, steps: int, sw2: float, nfe: int = 1) -> dict:
    """EN: one minimal v2-shaped Stage 0 record. CN: 一条最小化的 v2 形状 Stage 0 记录。"""
    return {
        "geometry": "two_moons",
        "arm": "ot_pool_B256",
        "draw": 0,
        "config": {"sampler": "euler", "nfe": nfe, "steps": steps},
        "metrics": {"sw2": sw2},
    }


def test_sw2_table_uses_only_the_requested_budget() -> None:
    """EN: the 1k record must not overwrite the 10k record (the bug that produced the wrong Stage 0 table).
    CN: 1k 记录不得覆盖 10k 记录（正是该缺陷生成了错误的 Stage 0 表）。"""
    table = _sw2_table([_record(steps=1000, sw2=9.0), _record(steps=10000, sw2=1.0)], steps=10000)
    assert table[("two_moons", "ot_pool_B256", 0)][1] == 1.0


def test_sw2_table_refuses_a_missing_budget() -> None:
    """EN: a table requested at a budget absent from the records is empty, not a silent mix.
    CN: 请求的记录中不存在的预算时，表为空，而不是静默混合。"""
    table = _sw2_table([_record(steps=1000, sw2=9.0)], steps=10000)
    assert table == {}

"""
EN: **Baselines from the comparison papers.** Each module implements one method exactly as its paper states it, and
    documents every choice the paper leaves open. A method takes arrays and returns arrays or a small frozen record;
    none of them imports from `iot_experiments`, downloads data, or writes files.
CN: **比较论文中的基线。** 每个模块按其论文所述实现一种方法，并把论文未规定的选择写明。方法只接收数组、返回数组或小的
    冻结记录；都不从 `iot_experiments` 导入、不下载数据、不写文件。
"""

from __future__ import annotations

from .gravity import GravityFit, fit_gravity

__all__ = ["GravityFit", "fit_gravity"]

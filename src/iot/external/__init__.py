"""
EN: **External comparisons** — the code that implements or loads something that is not this repository's: other
    papers' instances, other papers' baselines, and the metrics those papers report.

    Why a separate subpackage. Everything else in `src/iot` is our formulation (a plan, a linear cost family,
    `theta` in a convex set, a master and a scan). This package is the opposite side of a comparison: it builds an
    observed plan the way a given paper builds it, runs a baseline the way that paper runs it, and scores both with
    that paper's metric. Keeping it here means the comparison is importable and testable from the library, and the
    drivers under `experiments/` stay orchestration (flags, run directory, records) with no computation of their
    own.

    Layout:
      `metrics`     RMSE, Pearson correlation, CPC, relative Frobenius, and the IPF rescaling a flow comparison
                    needs (a predicted flow is compared in absolute units, so it must be matched to the observed
                    marginals first)
      `methods`     the baselines. Each one states the objective it minimises and the hyperparameters it fixes

    Data building stays in `experiments/` (`studies/iot/{real,synthetic}/<instance>/`): downloading a dataset and turning it
    into flow matrices is I/O and bookkeeping, not an algorithm. What lives here is what a comparison *computes*.

    Rule for this package: a method never imports from `experiments`, and never downloads. It is given an array
    (or an explicit path) and returns an array or a small record.

CN: **外部比较** —— 实现或加载"不属于本仓库"的东西：其它论文的实例、基线与指标。`src/iot` 其余部分都是我们自己的形式
    （计划、线性代价族、凸集内的 `theta`、master 与扫描）；本包是比较的另一侧：按某篇论文的方式构造观测计划、按该论文的
    方式运行基线、用该论文的指标打分。放在这里意味着比较代码可导入、可测试，而 `experiments/` 下的驱动只做编排
    （参数、运行目录、记录），不再自己承担计算。

    布局：`metrics`（RMSE、Pearson、CPC、相对 Frobenius、以及流量比较所需的 IPF 重标定）、`methods`（基线，每个都写明
    它最小化的目标与固定的超参数）。

    **数据构建留在 `experiments/`**（`studies/iot/{real,synthetic}/<instance>/`）：下载数据集并转成流量矩阵是 I/O 与簿记，不是
    算法。放在这里的是比较**计算**的东西。本包规则：方法不从 `experiments` 导入，也不下载；只接收数组（或显式路径），
    返回数组或小的记录。
"""

from __future__ import annotations

from .metrics import (
    FlowScores,
    align_to_marginals,
    cpc,
    flow_scores,
    pearson,
    relative_frobenius,
    rmse,
)

__all__ = [
    "FlowScores",
    "align_to_marginals",
    "cpc",
    "flow_scores",
    "pearson",
    "relative_frobenius",
    "rmse",
]

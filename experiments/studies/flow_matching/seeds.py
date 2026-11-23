"""
EN: **Deterministic integer seeds from mixed string/int parts.**

    NumPy's `default_rng` accepts ints and sequences of ints, not arbitrary strings. `seed("eight_gaussians", 3)` maps
    every part through `zlib.crc32` and returns a fixed list of uint32-sized ints, so a run is reproducible from names
    and indices alone and hash randomisation cannot leak in.

CN: **由字符串/整数混合部分生成确定性整数种子。** NumPy 的 `default_rng` 只接受整数或整数序列，不接受任意字符串。
    `seed("eight_gaussians", 3)` 把每一部分过一遍 `zlib.crc32`，返回固定的 uint32 范围整数列表，故运行可由名称与下标复现，
    不受哈希随机化影响。
"""

from __future__ import annotations

import zlib

__all__ = ["seed", "seed_int"]


def seed(*parts: object) -> list[int]:
    """EN: a stable list of ints usable as `np.random.default_rng` entropy. CN: 可作为 `default_rng` 熵的稳定整数列表。"""
    return [int(zlib.crc32(str(part).encode("utf-8")) & 0x7FFFFFFF) for part in parts]


def seed_int(*parts: object) -> int:
    """EN: a single stable 31-bit int, for `torch.manual_seed`. CN: 单个稳定的 31 位整数，供 `torch.manual_seed` 使用。"""
    value = 0
    for part in parts:
        value = (value * 1_000_003) ^ (int(zlib.crc32(str(part).encode("utf-8"))) & 0x7FFFFFFF)
    return value & 0x7FFFFFFF

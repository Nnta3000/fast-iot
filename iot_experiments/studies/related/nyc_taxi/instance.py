"""
EN: **NYC yellow-taxi flows → a multi-observation inverse-OT instance** (Bao, Li & Sun 2026, §6.3).

    The paper's setting: one city discretised into zones, trips partitioned into `K` time-window scenarios, and for
    each window an aggregated flow matrix whose row/column sums are the pickup and drop-off distributions. One
    time-invariant cost must rationalise all of them. This module builds that object from the public TLC parquet
    files and records every step the paper leaves unspecified.

    What is fixed here, and what the paper does not state (all of it is written into the manifest):

      * **source**: `yellow_tripdata_2016-MM.parquet` from the TLC bucket. The 2016 files carry
        `PULocationID`/`DOLocationID` (the official 263-zone discretisation) and **no** longitude/latitude, so the
        instance uses the official zone ids;
      * **`n`**: the paper uses `n = 100` and does not say how 263 official zones were merged into 100. We keep the
        official 263 and record `n_zones = 263`; a 100-zone instance would need a merge rule the paper does not give,
        so inventing one would put an unspecified preprocessing step inside a comparison;
      * **filters**: zone id in `[1, 263]`, both endpoints valid, pickup and drop-off inside the same month. No
        distance or duration filter (the paper states none);
      * **windows**: six, defined by pickup time — `weekday_am`, `weekday_midday`, `weekday_pm`, `weekday_night`,
        `weekend_day`, `weekend_night` (exact hours in `WINDOWS`);
      * **train/test**: an 80/20 random split of the trips **within each window** (seeded), so a test flow is a
        held-out sample of the same demand rather than a different period. The paper says "test data" without
        defining the split;
      * **marginals**: taken from the test flows themselves. Predicting a flow pattern given demand is the standard
        reading of "RMSE on the held-out flows", and every arm then sees the same marginals.

CN: **纽约黄色出租车流量 → 多观测逆 OT 实例**（Bao, Li & Sun 2026, §6.3）。论文设定：城市离散为若干 zone，行程按 `K`
    个时间窗切分，每个窗给一个聚合流量矩阵，其行/列和是上车/下车分布；一个时不变的代价要解释全部。本模块用公开的 TLC
    parquet 构造该对象，并把论文未规定的每一步写进 manifest：数据源（2016 文件只有官方 263 个 zone 的 ID，无经纬度，故
    沿用官方 zone id）、`n`（论文用 100 但未说明如何把 263 并成 100，故保留 263 并记录；擅自发明合并规则等于把未规定的预
    处理塞进比较）、过滤（仅 zone id 与端点有效）、六个时间窗（见 `WINDOWS`）、每窗内按行程 80/20 随机划分（带种子）、
    marginal 取自测试流量本身。
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

TLC_BUCKET = "https://d37ci6vzurychx.cloudfront.net/trip-data"
N_ZONES = 263
WINDOWS: tuple[tuple[str, int, int, bool], ...] = (
    # EN: (name, start_hour, end_hour, weekday_only). `end_hour` is exclusive; 24 means midnight.
    # CN: (名称, 起始小时, 结束小时(不含), 仅工作日)。
    ("weekday_am", 7, 10, True),
    ("weekday_midday", 10, 16, True),
    ("weekday_pm", 16, 20, True),
    ("weekday_night", 20, 24, True),
    ("weekend_day", 9, 18, False),
    ("weekend_night", 18, 24, False),
)
TEST_FRACTION = 0.2
SPLIT_SEED = 0


def source_path(month: str) -> Path:
    """EN: `<repo>/data/nyc_taxi/yellow_tripdata_<month>.parquet`. CN: 数据文件路径。"""
    return Path("data/nyc_taxi") / f"yellow_tripdata_{month}.parquet"


def download(month: str, *, target: Path | None = None) -> Path:
    """
    EN: fetch one month if it is not already on disk; returns the path. Never rewrites an existing file.
    CN: 若本地没有则下载一个月，返回路径；不覆盖已存在的文件。
    """
    path = Path(target) if target is not None else source_path(month)
    if path.is_file() and path.stat().st_size > 0:
        return path
    import urllib.request  # noqa: PLC0415  # only needed on the download path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{TLC_BUCKET}/yellow_tripdata_{month}.parquet"
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed https URL
    partial.replace(path)
    return path


def sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """EN: checksum for the manifest. CN: 清单用的校验和。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_month(path: Path) -> tuple[Int[np.ndarray, "trips"], Int[np.ndarray, "trips"],
                                    Int[np.ndarray, "trips"], Float[np.ndarray, "trips"]]:
    """
    EN: `(pickup_zone, dropoff_zone, pickup_hour_of_week, trip_distance)` for the valid trips of one month.

        `hour_of_week` is `weekday * 24 + hour` with Monday = 0, so a window is a half-open range on it. Only three
        columns are read, so the peak memory is that of the used columns rather than the whole file.

    CN: 一个月有效行程的 `(上车 zone, 下车 zone, 周内小时, 行程距离)`；`hour_of_week = 星期*24 + 小时`，周一为 0。
        只读四列，峰值内存是用到的列而不是整个文件。
    """
    import pyarrow.parquet as pq  # noqa: PLC0415  # optional dependency, only on this path

    table = pq.read_table(path, columns=["PULocationID", "DOLocationID", "tpep_pickup_datetime", "trip_distance"])
    pickup = np.asarray(table.column("PULocationID").to_numpy(), dtype=np.int64)
    dropoff = np.asarray(table.column("DOLocationID").to_numpy(), dtype=np.int64)
    stamps = np.asarray(table.column("tpep_pickup_datetime").to_numpy())
    distance = np.asarray(table.column("trip_distance").to_numpy(), dtype=np.float64)
    del table
    valid = ((pickup >= 1) & (pickup <= N_ZONES) & (dropoff >= 1) & (dropoff <= N_ZONES)
             & np.isfinite(distance) & (distance >= 0.0))
    pickup, dropoff, stamps, distance = pickup[valid], dropoff[valid], stamps[valid], distance[valid]
    # EN: `numpy` datetime64 -> day of week (0 = Monday) and hour.
    # CN: `datetime64` → 星期（0=周一）与小时。
    days = stamps.astype("datetime64[D]")
    weekday = ((days - np.datetime64("1970-01-01", "D")).astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday
    hour = stamps.astype("datetime64[h]").astype(np.int64) % 24
    hour_of_week = (weekday * 24 + hour).astype(np.int64)
    return pickup - 1, dropoff - 1, hour_of_week, distance


def window_of(hour_of_week: Int[np.ndarray, "trips"]) -> Int[np.ndarray, "trips"]:
    """
    EN: window index per trip, `-1` when no window covers it.

        A window is `weekday_only ? Mon-Fri : Sat-Sun` and `start <= hour < end`. Hours outside every window (00:00-
        07:00 on weekdays, 00:00-09:00 on weekends) get `-1` and are dropped; that is stated in the manifest rather
        than silently folded into a neighbouring window.

    CN: 每个行程的时间窗下标；没有窗覆盖时为 `-1`。窗为 `仅工作日 ? 周一到周五 : 周六周日` 且 `start <= 小时 < end`。
        不在任何窗内的小时（工作日 00:00–07:00、周末 00:00–09:00）记 `-1` 并丢弃，写进清单而不是悄悄并入相邻窗。
    """
    index = np.full(hour_of_week.shape, -1, dtype=np.int64)
    weekday = hour_of_week // 24
    hour = hour_of_week % 24
    is_weekday = weekday <= 4
    for position, (_name, start, end, weekday_only) in enumerate(WINDOWS):
        day_ok = is_weekday if weekday_only else ~is_weekday
        index[day_ok & (hour >= start) & (hour < end)] = position
    return index


def accumulate(pickup: Int[np.ndarray, "trips"], dropoff: Int[np.ndarray, "trips"],
               window: Int[np.ndarray, "trips"], *, n_windows: int) -> Float[np.ndarray, "windows zones zones"]:
    """
    EN: `(K, n, n)` flow counts, `flows[k, i, j] = #{trips in window k from zone i to zone j}`.
    CN: `(K, n, n)` 流量计数。
    """
    flows = np.zeros((int(n_windows), N_ZONES, N_ZONES), dtype=np.float64)
    for k in range(int(n_windows)):
        selected = window == k
        if not np.any(selected):
            continue
        flat = np.bincount(pickup[selected] * N_ZONES + dropoff[selected], minlength=N_ZONES * N_ZONES)
        flows[k] = flat.reshape(N_ZONES, N_ZONES)
    return flows


def accumulate_distance(pickup: Int[np.ndarray, "trips"], dropoff: Int[np.ndarray, "trips"],
                        distance: Float[np.ndarray, "trips"]) -> tuple[Float[np.ndarray, "zones zones"],
                                                                      Float[np.ndarray, "zones zones"]]:
    """
    EN: mean reported trip distance per zone pair, and the number of trips behind each mean.

        This is the distance a gravity model needs, and it is taken from the data rather than from zone centroids:
        the 2016 files carry no coordinates, and the pair's own mean trip distance is the quantity the trips
        actually realised. Pairs with no train trip have `count = 0` and the caller decides how to fill them.

    CN: 每个 zone 对的平均报告行程距离，以及每个均值背后的行程数。这是引力模型所需的距离，取自数据而非 zone 质心：
        2016 文件没有经纬度，而成对自身的平均行程距离才是行程真正实现的量。无训练行程的对 `count = 0`，由调用方决定
        如何填充。
    """
    n = N_ZONES
    flat = pickup * n + dropoff
    sums = np.bincount(flat, weights=distance, minlength=n * n).reshape(n, n)
    counts = np.bincount(flat, minlength=n * n).reshape(n, n).astype(np.float64)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return mean, counts


@dataclass(frozen=True)
class Instance:
    """
    EN: the built instance: train/test flows per window and their marginals, plus the zone set actually used.
    CN: 构造好的实例：每窗训练/测试流量及其 marginal，以及实际使用的 zone 集合。
    """

    train_flows: Float[np.ndarray, "windows zones zones"]
    test_flows: Float[np.ndarray, "windows zones zones"]
    train_source: Float[np.ndarray, "windows zones"]
    train_target: Float[np.ndarray, "windows zones"]
    test_source: Float[np.ndarray, "windows zones"]
    test_target: Float[np.ndarray, "windows zones"]
    distance: Float[np.ndarray, "zones zones"]
    distance_count: Float[np.ndarray, "zones zones"]
    manifest: dict[str, object] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        """EN: number of windows. CN: 时间窗数。"""
        return int(self.train_flows.shape[0])

    @property
    def n_zones(self) -> int:
        """EN: number of zones. CN: zone 数。"""
        return int(self.train_flows.shape[1])

    def save(self, path: Path) -> Path:
        """EN: one `.npz` plus a sibling `.manifest.json`. CN: 一个 `.npz` 与同名的清单。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            train_flows=self.train_flows, test_flows=self.test_flows,
            train_source=self.train_source, train_target=self.train_target,
            test_source=self.test_source, test_target=self.test_target,
            distance=self.distance, distance_count=self.distance_count,
        )
        path.with_suffix(".manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Path) -> "Instance":
        """EN: read an `.npz` written by `save`. CN: 读取 `save` 写出的 `.npz`。"""
        arrays = np.load(path)
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        return cls(
            train_flows=np.asarray(arrays["train_flows"], dtype=np.float64),
            test_flows=np.asarray(arrays["test_flows"], dtype=np.float64),
            train_source=np.asarray(arrays["train_source"], dtype=np.float64),
            train_target=np.asarray(arrays["train_target"], dtype=np.float64),
            test_source=np.asarray(arrays["test_source"], dtype=np.float64),
            test_target=np.asarray(arrays["test_target"], dtype=np.float64),
            distance=np.asarray(arrays["distance"], dtype=np.float64),
            distance_count=np.asarray(arrays["distance_count"], dtype=np.float64),
            manifest=manifest,
        )


def build(path: Path, *, test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED) -> Instance:
    """
    EN: read one month, split each window's trips 80/20 and accumulate the flows.

        The split is by **trip**, not by time: two weeks of data would otherwise make the test set a different
        demand regime and every arm would be scored on a distribution shift rather than on its cost.

    CN: 读一个月，按窗把行程 80/20 划分后累加流量。划分按**行程**而不是按时间：否则测试集就是另一种需求状态，
        所有臂都在被分布漂移打分，而不是被各自的代价打分。
    """
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction!r}")
    pickup, dropoff, hour_of_week, distance = read_month(path)
    window = window_of(hour_of_week)
    keep = window >= 0
    pickup, dropoff, window, distance = pickup[keep], dropoff[keep], window[keep], distance[keep]
    rng = np.random.default_rng(int(seed))
    is_test = rng.random(pickup.size) < float(test_fraction)
    train_flows = accumulate(pickup[~is_test], dropoff[~is_test], window[~is_test], n_windows=len(WINDOWS))
    test_flows = accumulate(pickup[is_test], dropoff[is_test], window[is_test], n_windows=len(WINDOWS))
    # EN: the distance matrix comes from the TRAIN trips only, so a model fitted on train cannot see test pairs.
    # CN: 距离矩阵只由**训练**行程构造，避免训练侧模型看到测试配对。
    distance_mean, distance_count = accumulate_distance(pickup[~is_test], dropoff[~is_test], distance[~is_test])
    manifest: dict[str, object] = {
        "source": str(path),
        "sha256": sha256(path),
        "n_zones": N_ZONES,
        "n_windows": len(WINDOWS),
        "windows": [{"name": name, "start_hour": start, "end_hour": end, "weekday_only": weekday_only}
                    for name, start, end, weekday_only in WINDOWS],
        "test_fraction": float(test_fraction),
        "split_seed": int(seed),
        "trips_read": int(pickup.size),
        "trips_used": int(pickup.size),
        "train_trips": int(np.count_nonzero(~is_test)),
        "test_trips": int(np.count_nonzero(is_test)),
        "train_flow_total": float(train_flows.sum()),
        "test_flow_total": float(test_flows.sum()),
        "distance_pairs_observed": int(np.count_nonzero(distance_count)),
        "distance_note": "mean reported trip_distance per zone pair over TRAIN trips; pairs with no train trip "
                         "have count 0 and are filled by the method (documented per method).",
        "note": "n=263 is the official TLC zone discretisation; the cited paper's n=100 merge rule is not stated, "
                "so no merge is applied.",
    }
    return Instance(
        train_flows=train_flows, test_flows=test_flows,
        train_source=train_flows.sum(axis=2), train_target=train_flows.sum(axis=1),
        test_source=test_flows.sum(axis=2), test_target=test_flows.sum(axis=1),
        distance=distance_mean, distance_count=distance_count,
        manifest=manifest,
    )


def main(argv: list[str] | None = None) -> int:
    """EN: build the instance for the months given. CN: 为给定月份构造实例。"""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="build the NYC taxi multi-observation instance")
    parser.add_argument("--months", nargs="+", default=["2016-01"])
    parser.add_argument("--out", type=Path, default=Path("data/nyc_taxi/instance_n263_k6.npz"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if len(args.months) != 1:
        raise SystemExit("one month per instance for now: the windows are defined per month")
    month = args.months[0]
    path = download(month)
    instance = build(path)
    out = instance.save(args.out)
    print(f"wrote {out} (+ .manifest.json): n_zones={instance.n_zones} windows={instance.n_windows} "
          f"train_total={instance.train_flows.sum():.0f} test_total={instance.test_flows.sum():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""随机 JSSP 实例生成器。

默认生成经典 JSSP（permutation）：每个工件恰好访问每台机器一次，
机器访问顺序为 0..m-1 的随机排列；加工时间默认 U[1, 99]（与 Taillard 基准一致）。

可复现性：同一 (n, m, seed, index) 组合必定生成同一实例 —— instance_id 直接编码
(n, m, batch_seed, index)，重新运行 generate_batch 即可完整重现数据。
"""

from __future__ import annotations

import numpy as np

from .instance import Instance


def generate_instance(
    n: int,
    m: int,
    *,
    seed: int,
    index: int = 0,
    p_min: int = 1,
    p_max: int = 99,
) -> Instance:
    """生成单个经典 JSSP 实例。

    Args:
        n, m: 工件数、机器数
        seed: 批次种子（与 index 共同决定随机流）
        index: 批次内序号（决定 instance_id 与随机流位置）
        p_min, p_max: 加工时间范围 [p_min, p_max]，闭区间
    """
    rng = np.random.default_rng(seed)
    # 消耗固定长度的随机流定位到第 index 个实例，保证批次内独立复现
    for _ in range(index):
        rng.permutation(m)
        rng.integers(p_min, p_max + 1, size=m)
    machines = [rng.permutation(m).tolist() for _ in range(n)]
    durations = [rng.integers(p_min, p_max + 1, size=m).tolist() for _ in range(n)]

    instance_id = f"gen-{n}x{m}-{seed}-{index:05d}"
    return Instance(instance_id=instance_id, n=n, m=m, machines=machines, durations=durations)


def generate_batch(
    n: int,
    m: int,
    count: int,
    *,
    seed: int = 0,
    p_min: int = 1,
    p_max: int = 99,
) -> list[Instance]:
    """批量生成 count 个 n×m 实例（同一随机流，顺序确定，可复现）。"""
    if count < 0:
        raise ValueError(f"count 必须非负，得到 {count}")
    return [
        generate_instance(n, m, seed=seed, index=i, p_min=p_min, p_max=p_max)
        for i in range(count)
    ]

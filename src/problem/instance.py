"""JSSP 实例数据结构与序列化（docs/jssp_definition.md §1）。

约定：
- 机器编号统一 0-based；解析 OR-Library / Taillard 原始文件时在 src/data/parsers.py 归一化
- instance_id 是实例全局唯一标识，贯穿数据 / 监督 / 划分 / 实验记录全程
- 经典 JSSP：每个工件恰好访问每台机器一次（machines[j] 是 0..m-1 的排列）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any


def validate_structure(n: int, m: int, machines: list[list[int]], durations: list[list[int]]) -> list[str]:
    """结构合法性检查，返回错误列表（空 = 合法）。

    检查项：维度为正、矩阵形状一致、机器编号在 [0, m) 内、加工时间为正整数。
    """
    errors: list[str] = []
    if n <= 0:
        errors.append(f"n 必须为正整数，得到 {n}")
    if m <= 0:
        errors.append(f"m 必须为正整数，得到 {m}")
    if len(machines) != n:
        errors.append(f"machines 行数 {len(machines)} != n={n}")
    if len(durations) != n:
        errors.append(f"durations 行数 {len(durations)} != n={n}")
    if errors:
        return errors
    for j in range(n):
        if len(machines[j]) != m:
            errors.append(f"machines[{j}] 长度 {len(machines[j])} != m={m}")
        if len(durations[j]) != m:
            errors.append(f"durations[{j}] 长度 {len(durations[j])} != m={m}")
        for k in range(m):
            mu = machines[j][k]
            if not (0 <= mu < m):
                errors.append(f"machines[{j}][{k}]={mu} 超出 [0, {m})")
            p = durations[j][k]
            if not isinstance(p, int) or p <= 0:
                errors.append(f"durations[{j}][{k}]={p} 必须为正整数")
    return errors


@dataclass(frozen=True)
class Instance:
    """静态 JSSP 实例。

    n 个工件 × m 台机器；工序 (j,k) 在机器 machines[j][k] 上加工 durations[j][k] 时间。
    """

    instance_id: str
    n: int
    m: int
    machines: list[list[int]]   # μ[j][k]，0-based
    durations: list[list[int]]  # p[j][k]，正整数

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id 不能为空")
        errors = validate_structure(self.n, self.m, self.machines, self.durations)
        if errors:
            raise ValueError(f"非法实例结构（{self.instance_id}）: " + "; ".join(errors))

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "n": self.n,
            "m": self.m,
            "machines": self.machines,
            "durations": self.durations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        missing = [f.name for f in fields(cls) if f.name not in data]
        if missing:
            raise ValueError(f"实例 JSON 缺少字段: {missing}")
        return cls(
            instance_id=data["instance_id"],
            n=int(data["n"]),
            m=int(data["m"]),
            machines=[list(map(int, row)) for row in data["machines"]],
            durations=[list(map(int, row)) for row in data["durations"]],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Instance":
        return cls.from_dict(json.loads(s))

    # ---- 文本表示（阶段 2 TAI 输入的基础） ----

    def to_text(self) -> str:
        """人类/LLM 可读的实例描述。"""
        lines = [f"Instance {self.instance_id}: {self.n} jobs x {self.m} machines"]
        for j in range(self.n):
            ops = " ".join(f"(m{self.machines[j][k]},{self.durations[j][k]})" for k in range(self.m))
            lines.append(f"Job {j}: {ops}")
        return "\n".join(lines)


def from_raw(instance_id: str, n: int, m: int, pairs: list[list[tuple[int, int]]], machine_offset: int = 0) -> Instance:
    """从 (机器编号, 加工时间) 对序列构建实例。

    Args:
        instance_id: 实例标识（如 'ft06' / 'la01' / 'gen-...'）
        n, m: 工件数、机器数
        pairs: pairs[j][k] = (μ[j][k], p[j][k])
        machine_offset: 机器编号归一化偏移（1-based 输入传 1，0-based 传 0）
    """
    if len(pairs) != n:
        raise ValueError(f"pairs 行数 {len(pairs)} != n={n}")
    machines: list[list[int]] = []
    durations: list[list[int]] = []
    for j in range(n):
        if len(pairs[j]) != m:
            raise ValueError(f"pairs[{j}] 长度 {len(pairs[j])} != m={m}")
        machines.append([int(mu) - machine_offset for mu, _ in pairs[j]])
        durations.append([int(p) for _, p in pairs[j]])
    return Instance(instance_id=instance_id, n=n, m=m, machines=machines, durations=durations)

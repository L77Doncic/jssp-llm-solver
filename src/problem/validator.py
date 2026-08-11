"""排程可行性验证器（docs/jssp_definition.md §1.5）。

三条硬约束：
1. 工序顺序：同一工件的工序必须按序串行
2. 机器独占：同一台机器同一时刻最多加工一道工序
3. 不可抢占：工序开始后连续加工（由 starts + durations 定义隐含）

验证器接受与 LLM 输出对齐的结构化排程格式：
starts[j][k] 为工序 (j,k) 的开始时间；-1 表示未调度（非法）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .instance import Instance
from .makespan import makespan_from_starts


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    makespan: int | None = None

    def __bool__(self) -> bool:
        return self.valid


def validate(instance: Instance, starts: list[list[int]]) -> ValidationResult:
    """完整校验一个排程。

    校验顺序：形状 → 非负 → 工序顺序 → 机器独占。错误逐条记录、不提前退出，
    便于训练时给出细粒度反馈。
    """
    n, m = instance.n, instance.m
    errors: list[str] = []

    # 1. 形状
    if len(starts) != n:
        return ValidationResult(False, [f"starts 行数 {len(starts)} != n={n}"])
    for j in range(n):
        if len(starts[j]) != m:
            return ValidationResult(False, [f"starts[{j}] 长度 {len(starts[j])} != m={m}"])

    # 2. 非负（-1 视为未调度）
    for j in range(n):
        for k in range(m):
            if starts[j][k] < 0:
                errors.append(f"工序 ({j},{k}) 未调度（start={starts[j][k]}）")

    # 3. 工序顺序约束
    for j in range(n):
        prev_end = None
        for k in range(m):
            s = starts[j][k]
            if s < 0:
                continue
            end = s + instance.durations[j][k]
            if prev_end is not None and s < prev_end:
                errors.append(f"工件 {j} 工序 {k} 开始时间 {s} 早于工序 {k-1} 完成时间 {prev_end}")
            prev_end = end

    # 4. 机器独占约束：每台机器按开始时间排序后检查区间重叠
    for machine in range(m):
        intervals = []
        for j in range(n):
            for k in range(m):
                if instance.machines[j][k] == machine and starts[j][k] >= 0:
                    intervals.append((starts[j][k], starts[j][k] + instance.durations[j][k], j, k))
        intervals.sort()
        for i in range(1, len(intervals)):
            prev_end = intervals[i - 1][1]
            cur_start, _, j, k = intervals[i]
            if cur_start < prev_end:
                errors.append(
                    f"机器 {machine} 上工序 ({j},{k}) 在 {cur_start} 开工，"
                    f"与 {prev_end} 前仍未完工的工序冲突"
                )

    valid = not errors
    makespan = makespan_from_starts(instance, starts) if valid else None
    return ValidationResult(valid=valid, errors=errors, makespan=makespan)


def validate_machine_order(instance: Instance, machine_order: list[list[tuple[int, int]]]) -> ValidationResult:
    """校验机器排列编码：每台机器上的工序必须属于该机器、不重复、且全部覆盖。

    最左调度对任意合法排列都满足三条硬约束，因此这里只需做编码层校验。
    """
    n, m = instance.n, instance.m
    errors: list[str] = []
    if len(machine_order) != m:
        return ValidationResult(False, [f"machine_order 长度 {len(machine_order)} != m={m}"])
    for machine in range(m):
        seen: set[tuple[int, int]] = set()
        for j, k in machine_order[machine]:
            if not (0 <= j < n and 0 <= k < m):
                errors.append(f"机器 {machine} 上出现越界工序 ({j},{k})")
                continue
            if instance.machines[j][k] != machine:
                errors.append(f"工序 ({j},{k}) 的机器是 {instance.machines[j][k]}，不在机器 {machine} 上")
                continue
            if (j, k) in seen:
                errors.append(f"机器 {machine} 上工序 ({j},{k}) 重复")
            seen.add((j, k))
        for j in range(n):
            for k in range(m):
                if instance.machines[j][k] == machine and (j, k) not in seen:
                    errors.append(f"机器 {machine} 漏掉工序 ({j},{k})")
    return ValidationResult(valid=not errors, errors=errors)

"""启发式基线：Giffler-Thompson 主动调度 + 优先分配规则（SPT/LPT/MOR/MWR）。

GT 算法保证生成的调度是主动调度（active schedule），且必然满足三条硬约束：
- 工件顺序：只调度每个工件当前待执行的工序（next_op 指针）
- 机器独占：机器就绪时间单调递增
- 不可抢占：区间连续

用途：① 与 LLM 输出对比的 makespan 参考；② 廉价启发式特征（阶段 2 TAI）。
"""

from __future__ import annotations

import random

from problem.instance import Instance
from problem.makespan import makespan_from_starts

RULE_CHOICES = ("spt", "lpt", "mor", "mwr", "random")


def _select(candidates: list[dict], rule: str, rng: random.Random) -> dict:
    """从候选工序中按规则选出一个（tie-break：工件号小者优先，保证确定性）。"""
    if rule == "random":
        return rng.choice(candidates)
    if rule == "spt":
        return min(candidates, key=lambda c: (c["p"], c["j"]))
    if rule == "lpt":
        return max(candidates, key=lambda c: (c["p"], -c["j"]))
    if rule == "mor":
        # 剩余工序最多 = op 序号最小（每工件固定 m 道工序）
        return min(candidates, key=lambda c: (c["k"], c["j"]))
    if rule == "mwr":
        # 剩余加工时间最长 = 该工件尚未调度工序的加工时间之和最大
        return max(candidates, key=lambda c: (c["remaining"], -c["j"]))
    raise ValueError(f"未知规则 {rule!r}，可选: {RULE_CHOICES}")


def gt_schedule(instance: Instance, rule: str = "spt", seed: int | None = None) -> list[list[int]]:
    """Giffler-Thompson 主动调度生成。

    Args:
        instance: JSSP 实例
        rule: 候选工序选择规则（spt/lpt/mor/mwr/random）
        seed: 仅 rule="random" 时使用

    Returns:
        starts: n×m 开始时间矩阵（满足全部约束）
    """
    if rule not in RULE_CHOICES:
        raise ValueError(f"未知规则 {rule!r}，可选: {RULE_CHOICES}")
    n, m = instance.n, instance.m
    rng = random.Random(seed)

    starts = [[-1] * m for _ in range(n)]
    next_op = [0] * n                # 工件 j 的下一道待调度工序
    job_ready = [0] * n              # 工件 j 当前工序可开工的最早时刻
    machine_ready = [0] * m          # 机器 i 空闲的最早时刻
    remaining_time = [sum(instance.durations[j]) for j in range(n)]

    for _ in range(n * m):
        # 候选集合：每个工件的下一道工序（若还有）
        candidates = []
        for j in range(n):
            k = next_op[j]
            if k >= m:
                continue
            machine = instance.machines[j][k]
            ready = max(job_ready[j], machine_ready[machine])
            candidates.append({
                "j": j, "k": k,
                "machine": machine,
                "p": instance.durations[j][k],
                "ready": ready,
                "finish": ready + instance.durations[j][k],
                "remaining": remaining_time[j],
            })

        # t* = 最早可完成时刻；只考虑 ready < t* 的工序（GT 主动调度条件）
        t_star = min(c["finish"] for c in candidates)
        eligible = [c for c in candidates if c["ready"] < t_star]
        chosen = _select(eligible, rule, rng)

        j, k, machine = chosen["j"], chosen["k"], chosen["machine"]
        starts[j][k] = chosen["ready"]
        finish = chosen["finish"]
        job_ready[j] = finish
        machine_ready[machine] = finish
        next_op[j] = k + 1
        remaining_time[j] -= chosen["p"]

    return starts


def solve_all_rules(instance: Instance) -> dict[str, tuple[list[list[int]], int]]:
    """用全部确定性规则求解，返回 {rule: (starts, makespan)}（不含 random）。"""
    results = {}
    for rule in ("spt", "lpt", "mor", "mwr"):
        starts = gt_schedule(instance, rule=rule)
        results[rule] = (starts, makespan_from_starts(instance, starts))
    return results

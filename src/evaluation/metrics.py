"""评估指标：可行性、最优性 gap、汇总统计（阶段 5 评估协议的基础）。

gap 定义（与 LLMCoSolver 一致）：
    gap% = (makespan - reference) / reference × 100
reference 取已知最优（小实例）或参考求解器近优解（大实例）。
"""

from __future__ import annotations

from dataclasses import dataclass

from problem.instance import Instance
from problem.validator import validate


@dataclass
class ScheduleEval:
    instance_id: str
    feasible: bool
    makespan: int | None
    gap_pct: float | None       # 相对 reference 的 gap%；不可行或无 reference 时为 None
    errors: list[str] = None

    def __bool__(self) -> bool:
        return self.feasible


def evaluate_schedule(
    instance: Instance,
    starts: list[list[int]],
    reference_makespan: int | None = None,
) -> ScheduleEval:
    """评估单个排程：可行性 + 相对 reference 的 gap。

    Args:
        instance: JSSP 实例
        starts: n×m 开始时间矩阵
        reference_makespan: 参考最优/近优 makespan（如 CP-SAT 结果）

    Returns:
        ScheduleEval
    """
    result = validate(instance, starts)
    if not result.valid:
        return ScheduleEval(
            instance_id=instance.instance_id,
            feasible=False,
            makespan=None,
            gap_pct=None,
            errors=result.errors,
        )
    gap_pct = None
    if reference_makespan is not None and reference_makespan > 0:
        gap_pct = (result.makespan - reference_makespan) / reference_makespan * 100.0
    return ScheduleEval(
        instance_id=instance.instance_id,
        feasible=True,
        makespan=result.makespan,
        gap_pct=gap_pct,
    )


def summarize(evals: list[ScheduleEval]) -> dict:
    """汇总一批评估结果：可行性率、平均/最优 gap、makespan 统计。

    Returns:
        {"n", "feasible_rate", "avg_gap_pct", "best_gap_pct", "avg_makespan",
         "min_makespan", "max_makespan"}
        无可行排程时 gap 相关字段为 None。
    """
    n = len(evals)
    feasible = [e for e in evals if e.feasible]
    gaps = [e.gap_pct for e in feasible if e.gap_pct is not None]
    makespans = [e.makespan for e in feasible if e.makespan is not None]

    stats: dict = {
        "n": n,
        "feasible_rate": len(feasible) / n if n else None,
        "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None,
        "best_gap_pct": min(gaps) if gaps else None,
        "avg_makespan": sum(makespans) / len(makespans) if makespans else None,
        "min_makespan": min(makespans) if makespans else None,
        "max_makespan": max(makespans) if makespans else None,
    }
    return stats

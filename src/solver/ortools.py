"""OR-Tools CP-SAT 参考求解器封装。

用途：为 SFT 数据集生成最优/近似最优监督信号（中小规模），并在评估阶段作对比基线。
建模：disjunctive 区间模型 —— 每工序一个固定时长区间变量，
工件顺序约束 + 每机器 AddNoOverlap + 最小化 makespan。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from problem.instance import Instance


@dataclass
class SolveResult:
    """一次 CP-SAT 求解的结果。

    status: OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN
    makespan: 目标值（无解为 None）
    starts: n×m 开始时间矩阵（无解为 None）
    wall_time: 求解耗时（秒）
    solver_gap: CP-SAT 报告的目标上界 gap（OPTIMAL 为 0.0；UNKNOWN/INFEASIBLE 为 None）
    """

    status: str
    makespan: int | None
    starts: list[list[int]] | None
    wall_time: float
    solver_gap: float | None


def solve_cp_sat(instance: Instance, time_limit: float = 60.0, num_workers: int = 8) -> SolveResult:
    """CP-SAT 求解静态 JSSP，最小化 makespan。

    Args:
        instance: JSSP 实例
        time_limit: 求解时间上限（秒）
        num_workers: CP-SAT 并行线程数

    Returns:
        SolveResult；模型不可行或超时无解时 status 相应为 INFEASIBLE / UNKNOWN
    """
    from ortools.sat.python import cp_model  # 延迟导入：避免模块导入期依赖 OR-Tools

    n, m = instance.n, instance.m
    horizon = sum(sum(row) for row in instance.durations)  # 所有加工时间之和是安全上界

    model = cp_model.CpModel()
    starts_var: dict[tuple[int, int], cp_model.IntVar] = {}
    intervals: dict[tuple[int, int], cp_model.IntervalVar] = {}
    for j in range(n):
        for k in range(m):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            d = instance.durations[j][k]
            starts_var[(j, k)] = s
            intervals[(j, k)] = model.NewFixedSizeIntervalVar(s, d, f"iv_{j}_{k}")

    # ① 工件顺序约束：O(j,k) 完成后才能开始 O(j,k+1)
    for j in range(n):
        for k in range(m - 1):
            model.Add(starts_var[(j, k)] + instance.durations[j][k] <= starts_var[(j, k + 1)])

    # ② 机器独占约束：每台机器上的工序区间两两不重叠
    by_machine: dict[int, list[tuple[int, int]]] = {i: [] for i in range(m)}
    for j in range(n):
        for k in range(m):
            by_machine[instance.machines[j][k]].append((j, k))
    for machine in range(m):
        model.AddNoOverlap([intervals[op] for op in by_machine[machine]])

    # 目标：最小化 makespan
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    for j in range(n):
        for k in range(m):
            model.Add(makespan_var >= starts_var[(j, k)] + instance.durations[j][k])
    model.Minimize(makespan_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = False

    wall_start = time.perf_counter()
    status = solver.Solve(model)
    wall_time = time.perf_counter() - wall_start

    status_name = solver.StatusName(status)
    if status_name in ("OPTIMAL", "FEASIBLE"):
        starts = [
            [solver.Value(starts_var[(j, k)]) for k in range(m)]
            for j in range(n)
        ]
        makespan = int(solver.ObjectiveValue())
        bound = solver.BestObjectiveBound()
        gap = 0.0 if status_name == "OPTIMAL" else (float(makespan - bound) / makespan if makespan > 0 else None)
        return SolveResult(status=status_name, makespan=makespan, starts=starts, wall_time=wall_time, solver_gap=gap)
    return SolveResult(status=status_name, makespan=None, starts=None, wall_time=wall_time, solver_gap=None)

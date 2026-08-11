"""CP-SAT 封装测试：已知最优、与启发式一致性、防御性校验。"""

import pytest

from problem.generator import generate_batch
from problem.instance import Instance
from problem.validator import validate
from solver.heuristics import solve_all_rules
from solver.ortools import solve_cp_sat


def test_two_by_two_optimal():
    """n=2, m=2 全 1 时间：最优 makespan = 2。"""
    inst = Instance(
        instance_id="manual-2x2",
        n=2, m=2,
        machines=[[0, 1], [1, 0]],
        durations=[[1, 1], [1, 1]],
    )
    result = solve_cp_sat(inst, time_limit=10.0)
    assert result.status == "OPTIMAL"
    assert result.makespan == 2
    assert result.solver_gap == 0.0
    assert validate(inst, result.starts).valid


@pytest.mark.parametrize("scale", [(3, 3), (6, 6)])
def test_small_instances_optimal(scale):
    """小规模实例应快速求得最优解。"""
    n, m = scale
    inst = generate_batch(n, m, count=3, seed=5)[0]
    result = solve_cp_sat(inst, time_limit=30.0)
    assert result.status == "OPTIMAL", f"{inst.instance_id}: {result.status}"
    assert result.solver_gap == 0.0
    assert validate(inst, result.starts).valid


def test_solution_not_worse_than_heuristics():
    """CP-SAT 解（最优或近优）应不劣于全部启发式规则解。"""
    inst = generate_batch(6, 6, count=2, seed=5)[0]
    result = solve_cp_sat(inst, time_limit=30.0)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    best_heuristic = min(makespan for _, makespan in solve_all_rules(inst).values())
    assert result.makespan <= best_heuristic


def test_unassigned_starts_reported_none():
    """求解失败（如 horizon 上界错误导致不可行）时 starts 为 None。"""
    # 构造不可能的情况：本模型对任何合法实例都不会 INFEASIBLE；
    # 这里只验证 API 契约：status 非 OPTIMAL/FEASIBLE 时 starts=None
    inst = generate_batch(3, 3, count=1, seed=0)[0]
    result = solve_cp_sat(inst, time_limit=0.001, num_workers=1)  # 极短时限
    if result.status not in ("OPTIMAL", "FEASIBLE"):
        assert result.starts is None
        assert result.makespan is None

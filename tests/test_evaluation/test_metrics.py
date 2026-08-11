"""评估指标测试：gap 计算、可行性率、汇总统计。"""

from evaluation.metrics import evaluate_schedule, summarize
from problem.generator import generate_batch
from solver.heuristics import gt_schedule
from solver.ortools import solve_cp_sat


def test_feasible_schedule_gap():
    inst = generate_batch(6, 6, count=1, seed=0)[0]
    starts = gt_schedule(inst, rule="spt")
    eval_ = evaluate_schedule(inst, starts, reference_makespan=200)
    assert eval_.feasible
    assert eval_.makespan is not None
    # gap = (makespan - 200) / 200 * 100
    expected = (eval_.makespan - 200) / 200 * 100
    assert abs(eval_.gap_pct - expected) < 1e-9


def test_gap_zero_when_equal_to_reference():
    inst = generate_batch(6, 6, count=1, seed=0)[0]
    result = solve_cp_sat(inst, time_limit=10.0)
    assert result.status == "OPTIMAL"
    eval_ = evaluate_schedule(inst, result.starts, reference_makespan=result.makespan)
    assert eval_.feasible
    assert abs(eval_.gap_pct) < 1e-9


def test_infeasible_schedule():
    inst = generate_batch(3, 3, count=1, seed=0)[0]
    bad_starts = [[-1] * inst.m for _ in range(inst.n)]  # 全部未调度
    eval_ = evaluate_schedule(inst, bad_starts, reference_makespan=100)
    assert not eval_.feasible
    assert eval_.makespan is None
    assert eval_.gap_pct is None
    assert eval_.errors  # 有错误明细


def test_no_reference_gap_none():
    inst = generate_batch(3, 3, count=1, seed=0)[0]
    starts = gt_schedule(inst, rule="spt")
    eval_ = evaluate_schedule(inst, starts)  # 不给 reference
    assert eval_.feasible
    assert eval_.gap_pct is None


def test_summarize():
    insts = generate_batch(4, 4, count=3, seed=0)
    evals = []
    for inst in insts:
        starts = gt_schedule(inst, rule="spt")
        evals.append(evaluate_schedule(inst, starts, reference_makespan=500))
    # 混入一个不可行
    evals.append(evaluate_schedule(insts[0], [[-1, -1, -1, -1]] * 4, reference_makespan=500))

    stats = summarize(evals)
    assert stats["n"] == 4
    assert stats["feasible_rate"] == 0.75
    assert stats["avg_gap_pct"] is not None
    assert stats["best_gap_pct"] is not None
    assert stats["avg_makespan"] is not None
    assert stats["min_makespan"] <= stats["max_makespan"]


def test_summarize_empty():
    stats = summarize([])
    assert stats["n"] == 0
    assert stats["feasible_rate"] is None
    assert stats["avg_gap_pct"] is None

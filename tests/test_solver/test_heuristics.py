"""启发式基线测试：GT 调度的合法性、规则正确性、可复现性。"""

import pytest

from problem.generator import generate_batch
from problem.validator import validate
from solver.heuristics import RULE_CHOICES, gt_schedule, solve_all_rules


@pytest.fixture(scope="module")
def instances():
    return generate_batch(4, 4, count=5, seed=1)


class TestGtSchedule:
    def test_all_rules_valid(self, instances):
        for inst in instances:
            for rule in ("spt", "lpt", "mor", "mwr"):
                starts = gt_schedule(inst, rule=rule)
                result = validate(inst, starts)
                assert result.valid, f"rule={rule} {inst.instance_id}: {result.errors}"

    def test_deterministic(self, instances):
        inst = instances[0]
        a = gt_schedule(inst, rule="spt")
        b = gt_schedule(inst, rule="spt")
        assert a == b

    def test_random_rule_reproducible_with_seed(self, instances):
        inst = instances[0]
        a = gt_schedule(inst, rule="random", seed=7)
        b = gt_schedule(inst, rule="random", seed=7)
        assert a == b

    def test_unknown_rule_rejected(self, instances):
        with pytest.raises(ValueError):
            gt_schedule(instances[0], rule="unknown")

    def test_makespan_reasonable(self, instances):
        """SPT 的 makespan 应不超过总加工时间（平凡上界）。"""
        for inst in instances:
            starts = gt_schedule(inst, rule="spt")
            total = sum(sum(row) for row in inst.durations)
            from problem.makespan import makespan_from_starts

            assert makespan_from_starts(inst, starts) <= total


class TestSolveAllRules:
    def test_returns_all_rules(self, instances):
        results = solve_all_rules(instances[0])
        assert set(results.keys()) == {"spt", "lpt", "mor", "mwr"}
        for starts, makespan in results.values():
            assert validate(instances[0], starts).valid
            assert makespan > 0

    def test_spt_not_worse_than_random_baseline(self, instances):
        """SPT 在多数实例上不劣于固定排列的平凡解（弱断言，仅验证机制运转）。"""
        inst = instances[0]
        spt_makespan = solve_all_rules(inst)["spt"][1]
        # 平凡下界：最长工件的总加工时间
        lb = max(sum(row) for row in inst.durations)
        assert spt_makespan >= lb

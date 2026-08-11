"""FOARL 奖励函数测试。"""

import pytest

from problem.generator import generate_batch
from problem.validator import validate
from solver.heuristics import gt_schedule
from training.reward import (
    FEASIBILITY_REWARD,
    INFEASIBILITY_REWARD,
    combined_reward,
    feasibility_reward,
    optimality_reward,
)


@pytest.fixture(scope="module")
def instance():
    return generate_batch(4, 4, count=1, seed=0)[0]


@pytest.fixture(scope="module")
def good_starts(instance):
    return gt_schedule(instance, rule="spt")


class TestFeasibilityReward:
    def test_valid_schedule(self, instance, good_starts):
        assert feasibility_reward(validate(instance, good_starts)) == FEASIBILITY_REWARD

    def test_invalid_schedule(self, instance):
        bad = [[-1] * instance.m for _ in range(instance.n)]
        assert feasibility_reward(validate(instance, bad)) == INFEASIBILITY_REWARD


class TestOptimalityReward:
    def test_zero_gap(self):
        assert optimality_reward(100, 100) == 1.0

    def test_gap_100_percent(self):
        # makespan 2× 参考 → gap=1.0 → r_opt = 0.5
        assert optimality_reward(200, 100) == pytest.approx(0.5)

    def test_gap_10_percent(self):
        assert optimality_reward(110, 100) == pytest.approx(1 / 1.1)

    def test_bad_reference(self):
        with pytest.raises(ValueError):
            optimality_reward(100, 0)

    def test_better_than_reference_rejected(self):
        with pytest.raises(ValueError):
            optimality_reward(90, 100)


class TestCombinedReward:
    def test_infeasible_gets_zero(self, instance):
        bad = [[-1] * instance.m for _ in range(instance.n)]
        assert combined_reward(instance, validate(instance, bad), reference_makespan=100) == 0.0

    def test_feasible_with_reference(self, instance, good_starts):
        validation = validate(instance, good_starts)
        ref = validation.makespan  # 用自身作参考 → gap=0
        r = combined_reward(instance, validation, reference_makespan=ref, lambda_opt=1.0)
        assert r == pytest.approx(1.0 + 1.0)

    def test_lambda_scales_optimality(self, instance, good_starts):
        validation = validate(instance, good_starts)
        ref = int(validation.makespan * 0.8)  # 参考更优（gap > 0）
        r0 = combined_reward(instance, validation, reference_makespan=ref, lambda_opt=0.0)
        r2 = combined_reward(instance, validation, reference_makespan=ref, lambda_opt=2.0)
        # λ 越大，最优性奖励权重越高；可行项恒为 1
        assert r2 - r0 == pytest.approx(2.0 * optimality_reward(validation.makespan, ref))

    def test_no_reference_only_feasibility(self, instance, good_starts):
        validation = validate(instance, good_starts)
        assert combined_reward(instance, validation, reference_makespan=None) == FEASIBILITY_REWARD

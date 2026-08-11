"""FOARL 奖励函数（对齐 LLMCoSolver 论文，arXiv:2509.16865）。

联合奖励：r = r_feas + λ · r_opt
- r_feas 可行性奖励：输出格式合法（解析成功）且全部约束满足 → 1.0，否则 0.0
- r_opt  最优性奖励：与最优性 gap 成反比，r_opt = 1 / (1 + gap)，gap = (C - C_ref) / C_ref ≥ 0
- λ      最优性权重（configs/foarl/ 配置）

设计要点（论文经验）：
1. 可行性是硬门槛：任何约束违规（含解析失败）直接 r_feas=0，且 r_opt 也为 0（不可行解不奖励目标值）
2. 防止「贪婪违规」：SFT 单独会出现为了目标值牺牲约束的行为，联合奖励让违规解的期望奖励被可行性项压住
3. gap 用 CP-SAT 参考解（监督信号）计算，保证奖励有明确锚点
"""

from __future__ import annotations

from problem.instance import Instance
from problem.validator import ValidationResult, validate

# 奖励尺度
FEASIBILITY_REWARD = 1.0
INFEASIBILITY_REWARD = 0.0


def feasibility_reward(validation: ValidationResult) -> float:
    """可行性奖励：格式合法（已由解析层保证）+ 约束全满足 → 1.0，否则 0.0。"""
    return FEASIBILITY_REWARD if validation.valid else INFEASIBILITY_REWARD


def optimality_reward(makespan: int, reference_makespan: int) -> float:
    """最优性奖励：r_opt = 1 / (1 + gap)，gap = (makespan - ref) / ref ≥ 0。

    - gap = 0（达到参考解）→ r_opt = 1.0
    - gap 越大 → r_opt 越小（渐近趋于 0）
    - makespan 超过参考 100% → r_opt ≈ 0.5
    """
    if reference_makespan <= 0:
        raise ValueError(f"reference_makespan 必须为正: {reference_makespan}")
    gap = (makespan - reference_makespan) / reference_makespan
    if gap < 0:
        raise ValueError(f"makespan 不应优于 reference（{makespan} < {reference_makespan}），检查参考解来源")
    return 1.0 / (1.0 + gap)


def combined_reward(
    instance: Instance,
    validation: ValidationResult,
    reference_makespan: int | None,
    lambda_opt: float = 1.0,
) -> float:
    """联合奖励 r = r_feas + λ · r_opt。

    Args:
        instance: JSSP 实例（用于校验通过后取 makespan）
        validation: 排程验证结果（validate() 的返回值）
        reference_makespan: 参考 makespan（CP-SAT 监督解）；None 时最优性奖励记 0
        lambda_opt: 最优性权重 λ

    Returns:
        联合奖励值。不可行 → 0.0；可行但无参考 → r_feas（=1.0）
    """
    if not validation.valid or validation.makespan is None:
        return INFEASIBILITY_REWARD
    if reference_makespan is None:
        return FEASIBILITY_REWARD
    r_opt = optimality_reward(validation.makespan, reference_makespan)
    return FEASIBILITY_REWARD + lambda_opt * r_opt

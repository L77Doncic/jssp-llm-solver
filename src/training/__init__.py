"""training: SFT（sft.py 数据管线；训练入口 scripts/train_sft.py）与
FOARL（reward.py 奖励函数；GRPO 循环待阶段 4 实现）。
"""

from .reward import (
    FEASIBILITY_REWARD,
    INFEASIBILITY_REWARD,
    combined_reward,
    feasibility_reward,
    optimality_reward,
)

__all__ = [
    "FEASIBILITY_REWARD",
    "INFEASIBILITY_REWARD",
    "combined_reward",
    "feasibility_reward",
    "optimality_reward",
]

"""solver: 参考求解器 —— CP-SAT 封装（ortools.py，生成监督信号/评估基线）、
启发式基线（heuristics.py，SPT/MOR 等优先规则 + Giffler-Thompson 主动调度）。
"""

from .heuristics import RULE_CHOICES, gt_schedule, solve_all_rules
from .ortools import SolveResult, solve_cp_sat

__all__ = [
    "RULE_CHOICES",
    "gt_schedule",
    "solve_all_rules",
    "SolveResult",
    "solve_cp_sat",
]

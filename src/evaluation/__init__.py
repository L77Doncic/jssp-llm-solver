"""evaluation: 评估指标 —— gap 计算、可行性率、汇总统计（metrics.py）。
后续阶段补充：时间统计、泛化评估（跨规模/跨分布）、评估流水线脚本。
"""

from .metrics import ScheduleEval, evaluate_schedule, summarize

__all__ = ["ScheduleEval", "evaluate_schedule", "summarize"]

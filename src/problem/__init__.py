"""problem: JSSP 形式化 —— 实例数据结构（instance.py）、makespan 计算（makespan.py）、
可行性验证器（validator.py）、随机实例生成器（generator.py）。

约定：机器编号 0-based；instance_id 全局唯一贯穿全流程。详见 docs/jssp_definition.md。
"""

from .generator import generate_batch, generate_instance
from .instance import Instance, from_raw, validate_structure
from .makespan import compute_start_times, makespan_from_machine_order, makespan_from_starts
from .validator import ValidationResult, validate, validate_machine_order

__all__ = [
    "Instance",
    "from_raw",
    "validate_structure",
    "generate_batch",
    "generate_instance",
    "compute_start_times",
    "makespan_from_machine_order",
    "makespan_from_starts",
    "ValidationResult",
    "validate",
    "validate_machine_order",
]

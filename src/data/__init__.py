"""data: 数据流水线 —— 标准实例解析（parsers.py）、监督数据构建（build_supervised.py）、
train/val/test 划分（splits.py）。instance_id 贯穿全程，详见 CLAUDE.md。
"""

from .build_supervised import build_supervised
from .parsers import parse_jobshop_file
from .splits import load_splits, make_splits, save_splits

__all__ = [
    "build_supervised",
    "parse_jobshop_file",
    "load_splits",
    "make_splits",
    "save_splits",
]

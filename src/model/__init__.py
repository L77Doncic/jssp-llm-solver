"""model: 输入输出表示 —— TAI 文本模板、排程 JSON 编码与容错解析（format.py）。
后续阶段将加入本地模型推理封装（vLLM/transformers，见 CLAUDE.md）。
"""

from .format import INSTRUCTION, ParseResult, build_tai, encode_schedule, parse_schedule

__all__ = [
    "INSTRUCTION",
    "ParseResult",
    "build_tai",
    "encode_schedule",
    "parse_schedule",
]

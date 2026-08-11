"""阶段 2：输入输出表示 —— TAI 文本模板与排程 JSON 编解码。

输入（TAI，Text-Attributed Instance，对齐 LLMCoSolver 论文）：
    指令 + 实例文本描述（按工件列出 (机器,时间) 工序序列）+ 可选启发式特征提示。

输出：可解析、可验证的结构化排程 —— JSON 数组，逐工序一条记录：
    [{"job": 0, "op": 0, "machine": 3, "start": 0, "duration": 28}, ...]
解析器容错：容忍 markdown 代码块、前后说明文字、字段顺序任意。
解析结果直接喂给 problem.validator 做可行性校验。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from problem.instance import Instance

_JSON_BLOCK_RE = re.compile(r"\[[\s\S]*\]")


@dataclass
class ParseResult:
    ok: bool
    starts: list[list[int]] | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok


# ---- TAI 输入构建 ----

INSTRUCTION = (
    "You are a job shop scheduler. Given a Job Shop Scheduling Problem (JSSP) instance, "
    "output a feasible schedule that minimizes the makespan.\n"
    "Rules:\n"
    "1. Each job's operations must be processed in the given order.\n"
    "2. A machine can process at most one operation at a time.\n"
    "3. Operations cannot be preempted.\n"
    "Output ONLY a JSON array of operations: "
    '[{"job": j, "op": k, "machine": mu, "start": s, "duration": p}, ...], '
    "where every operation (j,k) appears exactly once.\n\n"
)


def build_tai(instance: Instance, heuristic_hint: str | None = None) -> str:
    """构建 TAI 输入文本。

    Args:
        instance: JSSP 实例
        heuristic_hint: 可选启发式特征提示（阶段 2 的对照实验项；如 'best-known makespan: 123'）

    Returns:
        完整 prompt 文本
    """
    parts = [INSTRUCTION]
    parts.append(
        f"Instance {instance.instance_id}: {instance.n} jobs x {instance.m} machines.\n"
        "Each job is a sequence of (machine, processing time) pairs.\n"
    )
    for j in range(instance.n):
        ops = " ".join(
            f"(m{instance.machines[j][k]},{instance.durations[j][k]})"
            for k in range(instance.m)
        )
        parts.append(f"Job {j}: {ops}")
    parts.append("")
    if heuristic_hint:
        parts.append(f"Hint: {heuristic_hint}")
        parts.append("")
    return "\n".join(parts)


# ---- 排程 JSON 编码（starts → 文本） ----

def encode_schedule(instance: Instance, starts: list[list[int]]) -> str:
    """把 starts 矩阵编码为 JSON 文本（按 job, op 排序，字段齐全）。"""
    records = []
    for j in range(instance.n):
        for k in range(instance.m):
            records.append({
                "job": j,
                "op": k,
                "machine": instance.machines[j][k],
                "start": starts[j][k],
                "duration": instance.durations[j][k],
            })
    return json.dumps(records, ensure_ascii=False)


# ---- 排程解析（文本 → starts） ----

def parse_schedule(instance: Instance, text: str) -> ParseResult:
    """把 LLM 输出文本解析为 starts 矩阵。

    容错策略：
    1. 提取文本中第一个 '[' 到最后一个 ']' 之间的片段（处理代码块/说明文字包裹）
    2. JSON 解析失败、字段缺失、与实例不匹配（机器/时间/缺工序）均返回错误信息
    3. **多余记录容忍**（2026-08-10 实测模型在 6x6 上会多输出若干 job 的工序：
       生成 90 条而实例只需 36 条，且前 36 条合法）——记录数超过 n×m 时，
       若「前 n×m 条」恰好完整覆盖全部工序（无重复无缺失）则取前缀，多余部分忽略；
       否则报错（真实质量缺陷）。
    4. 解析成功后由外部调用方用 validator 校验约束

    Returns:
        ParseResult；ok=True 时 starts 为 n×m 矩阵
    """
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return ParseResult(ok=False, error="未找到 JSON 数组片段")
    raw = match.group(0)

    try:
        records = json.loads(raw)
    except json.JSONDecodeError as e:
        return ParseResult(ok=False, error=f"JSON 解析失败: {e.msg}（位置 {e.pos}）")

    if not isinstance(records, list):
        return ParseResult(ok=False, error=f"期望 JSON 数组，得到 {type(records).__name__}")

    n, m = instance.n, instance.m
    n_ops = n * m

    # 先逐条做静态校验（字段、机器、时间），收集合法记录
    def parse_record(i, rec):
        if not isinstance(rec, dict):
            return None, f"第 {i} 条记录不是对象: {rec!r}"
        try:
            j = int(rec["job"])
            k = int(rec["op"])
            mu = int(rec["machine"])
            s = int(rec["start"])
            p = int(rec["duration"])
        except (KeyError, TypeError, ValueError) as e:
            return None, f"第 {i} 条记录字段非法（需要 job/op/machine/start/duration）: {e}"
        if not (0 <= j < n and 0 <= k < m):
            return None, f"工序 ({j},{k}) 越界"
        if mu != instance.machines[j][k]:
            return None, f"工序 ({j},{k}) 机器 {mu} 与实例不符（应为 {instance.machines[j][k]}）"
        if p != instance.durations[j][k]:
            return None, f"工序 ({j},{k}) 加工时间 {p} 与实例不符（应为 {instance.durations[j][k]}）"
        if s < 0:
            return None, f"工序 ({j},{k}) 开始时间非法: {s}"
        return (j, k, s), None

    parsed_items = []
    for i, rec in enumerate(records):
        item, err = parse_record(i, rec)
        if err:
            # 多余记录（> n*m）中出现坏记录可忽略；合法数量未达标则报错
            if len(parsed_items) >= n_ops:
                continue
            return ParseResult(ok=False, error=err)
        parsed_items.append(item)

    # 记录数超出：尝试取前 n×m 条（容忍模型多余输出）
    if len(parsed_items) > n_ops:
        prefix = parsed_items[:n_ops]
        keys = {(j, k) for j, k, _ in prefix}
        if len(keys) == n_ops:  # 前缀恰好完整覆盖
            parsed_items = prefix
        else:
            return ParseResult(
                ok=False,
                error=f"记录数 {len(parsed_items)} 超出 {n_ops} 且前缀未完整覆盖全部工序（覆盖 {len(keys)}/{n_ops}）",
            )

    starts = [[-1] * m for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for j, k, s in parsed_items:
        if (j, k) in seen:
            return ParseResult(ok=False, error=f"工序 ({j},{k}) 重复出现")
        seen.add((j, k))
        starts[j][k] = s

    missing = [(j, k) for j in range(n) for k in range(m) if (j, k) not in seen]
    if missing:
        return ParseResult(ok=False, error=f"缺少工序: {missing[:5]} ...（共 {len(missing)} 道）")

    return ParseResult(ok=True, starts=starts)

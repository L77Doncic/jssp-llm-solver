"""标准公开实例解析器：OR-Library（ft/la 系列）与 Taillard（ta 系列）。

两种格式的统一抽象（docs/jssp_definition.md §3.3）：
- 首行：`n m`
- 随后 n 行：每行 2m 个整数，按 (机器编号, 加工时间) 交替对
- 机器编号可能 0-based（ft06、ta）或 1-based（la 系列）→ 自动检测并归一化为 0-based

本解析器按「单实例文件」处理；多实例拼接文件（如 OR-Library 的 jobshop1.txt）
不在支持范围，如需支持请扩展本模块并补测试。
"""

from __future__ import annotations

from pathlib import Path

from problem.instance import Instance, from_raw


def _parse_rows(raw: str) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            rows.append([int(tok) for tok in stripped.split()])
        except ValueError as e:
            raise ValueError(f"无法解析为整数: {stripped!r}") from e
    if not rows:
        raise ValueError("文件为空")
    return rows


def parse_jobshop_file(path: str | Path) -> Instance:
    """解析单实例 JSSP 文件（OR-Library / Taillard 通用）。

    instance_id 取文件名（不含扩展名），如 'ft06' / 'la01' / 'ta01'。

    Raises:
        ValueError: 文件为空、维度不匹配、机器编号无法归一化等
    """
    path = Path(path)
    rows = _parse_rows(path.read_text(encoding="utf-8", errors="replace"))
    header = rows[0]
    if len(header) != 2:
        raise ValueError(f"首行应为 'n m'，得到 {header}")
    n, m = header
    if n <= 0 or m <= 0:
        raise ValueError(f"维度非法: n={n}, m={m}")
    if len(rows) - 1 != n:
        raise ValueError(f"数据行数 {len(rows) - 1} != n={n}")

    pairs: list[list[tuple[int, int]]] = []
    machine_nums: list[int] = []
    for row in rows[1:]:
        if len(row) != 2 * m:
            raise ValueError(f"行长度 {len(row)} != 2m={2 * m}: {row}")
        job_pairs = [(row[2 * k], row[2 * k + 1]) for k in range(m)]
        for mu, p in job_pairs:
            if p <= 0:
                raise ValueError(f"加工时间必须为正: {job_pairs}")
            machine_nums.append(mu)
        pairs.append(job_pairs)

    # 机器编号归一化：全部从 0 开始 → offset 0；全部从 1 开始 → offset 1；其他 → 报错
    min_mu, max_mu = min(machine_nums), max(machine_nums)
    if min_mu == 0:
        offset = 0
    elif min_mu == 1:
        offset = 1
    else:
        raise ValueError(f"机器编号无法归一化（min={min_mu}），应全部从 0 或 1 开始")
    if max_mu - offset >= m:
        raise ValueError(f"机器编号 {max_mu} 超出机器数 m={m}")

    return from_raw(instance_id=path.stem, n=n, m=m, pairs=pairs, machine_offset=offset)

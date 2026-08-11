"""makespan 计算。

支持两种排程表示：
1. starts：n×m 开始时间矩阵，starts[j][k] = 工序 (j,k) 的开始时间（未调度为 -1）
2. machine_order：每台机器上的工序列表，按列表顺序执行（最左调度）
"""

from __future__ import annotations

from collections import deque

from .instance import Instance


def makespan_from_starts(instance: Instance, starts: list[list[int]]) -> int:
    """给定各工序开始时间，返回 makespan（最后完工工序的完成时刻）。

    前提：starts 已通过 validator 校验（形状合法、非负、约束满足）。
    """
    n, m = instance.n, instance.m
    cmax = 0
    for j in range(n):
        for k in range(m):
            cmax = max(cmax, starts[j][k] + instance.durations[j][k])
    return cmax


def compute_start_times(instance: Instance, machine_order: list[list[tuple[int, int]]]) -> list[list[int]]:
    """给定每台机器上的工序序列，计算各工序的最早开始时间（最左调度）。

    做法：工序间偏序构成 DAG —— 边来自两类约束
      ① 工件顺序：工序 (j,k) → (j,k+1)
      ② 机器顺序：同一机器上排列中前一道工序 → 后一道工序
    按拓扑序取「所有前驱完工时间的最大值」即每个工序的最早开始时间，
    结果必然满足三条硬约束。若排列与工件顺序矛盾（偏序图存在环），
    说明该排列无可行调度，抛出 ValueError。

    Args:
        instance: JSSP 实例
        machine_order: 长度 m 的列表；每项是 (job, op) 元组序列

    Returns:
        starts: n×m 开始时间矩阵
    """
    n, m = instance.n, instance.m
    n_ops = n * m

    def idx(j: int, k: int) -> int:
        return j * m + k

    pred: list[list[int]] = [[] for _ in range(n_ops)]  # 前驱表，用于计算最早开始时间
    out: list[list[int]] = [[] for _ in range(n_ops)]   # 后继表，用于 Kahn 拓扑排序
    indeg = [0] * n_ops

    def add_edge(u: int, v: int) -> None:
        pred[v].append(u)
        out[u].append(v)
        indeg[v] += 1

    # ① 工件顺序边：O(j,k) 完成后 O(j,k+1) 才能开始
    for j in range(n):
        for k in range(m - 1):
            add_edge(idx(j, k), idx(j, k + 1))
    # ② 机器顺序边：同一机器排列中前一道工序完成后，后一道才能开工
    for seq in machine_order:
        for i in range(len(seq) - 1):
            add_edge(idx(*seq[i]), idx(*seq[i + 1]))

    # Kahn 拓扑排序：s(v) = max(0, max_{u∈pred(v)} s(u) + p(u))
    queue = deque(v for v in range(n_ops) if indeg[v] == 0)
    starts_flat = [0] * n_ops
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        j, k = divmod(u, m)
        s = 0
        for p in pred[u]:
            pj, pk = divmod(p, m)
            s = max(s, starts_flat[p] + instance.durations[pj][pk])
        starts_flat[u] = s
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if visited != n_ops:
        raise ValueError("machine_order 与工件顺序约束冲突（偏序图存在环），该排列无可行调度")

    starts = [[-1] * m for _ in range(n)]
    for j in range(n):
        for k in range(m):
            starts[j][k] = starts_flat[idx(j, k)]
    return starts


def makespan_from_machine_order(instance: Instance, machine_order: list[list[tuple[int, int]]]) -> int:
    """给定每台机器上的工序序列，返回最左调度的 makespan。"""
    return makespan_from_starts(instance, compute_start_times(instance, machine_order))

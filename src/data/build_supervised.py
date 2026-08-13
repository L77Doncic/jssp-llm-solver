"""监督数据构建流水线：生成实例 → CP-SAT 求解 → 保存 SFT 数据集 + 划分 + 统计。

数据流（instance_id 全程贯穿）：
    configs/data/pipeline.yaml
      → 按规模生成实例（数据盘 instances/{n}x{m}/）并逐实例 CP-SAT 求解
      → 监督数据集 JSONL（supervised/sft_dataset.jsonl，含 instance + solution）
      → train/val/test 划分（splits/）
      → 构建统计摘要（返回 dict 供脚本打印/存档）

求解失败（超时 UNKNOWN 等）的实例不写入监督集，但计入统计，绝不静默丢弃。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from problem.generator import generate_batch
from problem.instance import Instance
from problem.validator import validate
from solver.ortools import solve_cp_sat
from .splits import make_splits, save_splits


def save_instances(instances: list[Instance], instances_dir: str | Path) -> None:
    """保存实例本体（JSON），供复现与评估使用。"""
    instances_dir = Path(instances_dir)
    for inst in instances:
        path = instances_dir / f"{inst.n}x{inst.m}" / f"{inst.instance_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inst.to_json(), encoding="utf-8")


def solve_instances(
    instances: list[Instance],
    time_limit: float,
    num_workers: int,
) -> tuple[list[dict], dict]:
    """逐实例 CP-SAT 求解，返回 (记录列表, 统计)。

    记录结构：{"instance": dict, "solution": {starts, makespan, status, wall_time, solver_gap}}
    统计：{"total", "solved", "optimal", "skipped", "avg_wall_time", "by_scale": {...}}
    """
    records: list[dict] = []
    stats: dict = {"total": 0, "solved": 0, "optimal": 0, "skipped": 0, "avg_wall_time": 0.0}
    by_scale: dict[str, dict] = defaultdict(lambda: {"total": 0, "solved": 0, "optimal": 0})
    total_time = 0.0

    for inst in instances:
        key = f"{inst.n}x{inst.m}"
        stats["total"] += 1
        by_scale[key]["total"] += 1
        result = solve_cp_sat(inst, time_limit=time_limit, num_workers=num_workers)
        total_time += result.wall_time

        if result.starts is None:
            # UNKNOWN / INFEASIBLE：不写入监督集（INFEASIBLE 理论上不会出现）
            stats["skipped"] += 1
            print(f"  [skip] {inst.instance_id}: {result.status}")
            continue

        # 求解结果用验证器复核，防御性校验
        check = validate(inst, result.starts)
        if not check.valid:
            raise RuntimeError(f"CP-SAT 返回非法排程（{inst.instance_id}）: {check.errors}")

        stats["solved"] += 1
        by_scale[key]["solved"] += 1
        if result.status == "OPTIMAL":
            stats["optimal"] += 1
            by_scale[key]["optimal"] += 1

        records.append({
            "instance": inst.to_dict(),
            "solution": {
                "starts": result.starts,
                "makespan": result.makespan,
                "status": result.status,
                "solver_gap": result.solver_gap,
                "solver_wall_time": result.wall_time,
            },
        })

    stats["avg_wall_time"] = total_time / stats["total"] if stats["total"] else 0.0
    stats["by_scale"] = dict(by_scale)
    return records, stats


def write_records(records: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_supervised(config: dict, out_root: str | Path) -> dict:
    """执行完整监督数据流水线。

    Args:
        config: pipeline 配置（见 configs/data/pipeline.yaml）
        out_root: 数据根目录（本项目为 /root/autodl-tmp/jssp_data）

    Returns:
        构建统计摘要 dict（供脚本打印/存档）
    """
    out_root = Path(out_root)
    instances_dir = out_root / "instances"
    supervised_dir = out_root / "supervised"
    splits_dir = out_root / "splits"

    seed = config["seed"]
    p_min, p_max = config["p_range"]
    solver_cfg = config["solver"]
    split_cfg = config["splits"]

    all_ids: list[str] = []
    all_records: list[dict] = []
    scale_reports = []

    for scale in config["instance_scales"]:
        n, m, count = scale["n"], scale["m"], scale["count"]
        print(f"[scale {n}x{m}] 生成 {count} 实例 ...")
        instances = generate_batch(n, m, count, seed=seed, p_min=p_min, p_max=p_max)
        save_instances(instances, instances_dir)

        print(f"[scale {n}x{m}] CP-SAT 求解（time_limit={solver_cfg['time_limit']}s）...")
        records, stats = solve_instances(
            instances, time_limit=solver_cfg["time_limit"], num_workers=solver_cfg["num_workers"]
        )
        all_records.extend(records)
        all_ids.extend(inst.instance_id for inst in instances)
        scale_reports.append({"scale": f"{n}x{m}", **stats})
        print(f"  solved={stats['solved']}/{stats['total']} optimal={stats['optimal']} "
              f"avg_time={stats['avg_wall_time']:.3f}s")

    write_records(all_records, supervised_dir / "sft_dataset.jsonl")
    print(f"[dataset] 监督样本 {len(all_records)} 条 → supervised/sft_dataset.jsonl")

    splits = make_splits(all_ids, ratios=tuple(split_cfg["ratios"]), seed=split_cfg["seed"])
    save_splits(splits, splits_dir)
    print(f"[splits] train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    return {
        "seed": seed,
        "p_range": [p_min, p_max],
        "solver": solver_cfg,
        "splits": {name: len(ids) for name, ids in splits.items()},
        "total_instances": len(all_ids),
        "supervised_records": len(all_records),
        "scales": scale_reports,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

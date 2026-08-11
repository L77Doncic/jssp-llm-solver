#!/usr/bin/env python3
"""阶段 1 一键入口：生成实例 → CP-SAT 监督 → 划分 → 统计摘要。

用法（在服务器 /root/jssp 下执行）：
    python scripts/build_data.py                          # 默认验证档 configs/data/pipeline.yaml
    python scripts/build_data.py -c configs/data/pipeline_full.yaml   # 放量档
    python scripts/build_data.py --out /root/autodl-tmp/jssp_data     # 默认即数据盘

数据产出（全部按 instance_id 关联，落在数据盘）：
    <out>/instances/         实例本体 JSON（按规模分子目录）
    <out>/supervised/        监督数据集（JSONL：instance + solution）
    <out>/splits/            train/val/test 划分清单
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.build_supervised import build_supervised  # noqa: E402

DEFAULT_OUT = "/root/autodl-tmp/jssp_data"


def main() -> int:
    parser = argparse.ArgumentParser(description="JSSP 监督数据构建流水线")
    parser.add_argument("-c", "--config", default="configs/data/pipeline.yaml", help="流水线配置 YAML")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"数据输出根目录（默认 {DEFAULT_OUT}）")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] 配置不存在: {config_path}", file=sys.stderr)
        return 1
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    stats = build_supervised(config, out_root=args.out)

    # 统计摘要落盘（experiments 结构：config 冻结 + 结果存档）
    exp_dir = ROOT / "experiments" / "data_build"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (exp_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[summary] 已存档 → {exp_dir}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

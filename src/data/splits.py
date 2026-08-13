"""train/val/test 划分：按 instance_id 列表保存，保证跨阶段可复现。"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    instance_ids: list[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[str]]:
    """把实例 ID 列表按比例随机划分为 train/val/test。

    Args:
        instance_ids: 全部实例 ID
        ratios: (train, val, test) 比例，和为 1
        seed: 打乱种子，保证可复现

    Returns:
        {"train": [...], "val": [...], "test": [...]}；输出顺序稳定（按 ID 排序）
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios 必须为三个和为 1 的比例: {ratios}")
    if any(r < 0 for r in ratios):
        raise ValueError("ratios 不能为负")

    shuffled = list(instance_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def save_splits(splits: dict[str, list[str]], out_dir: str | Path) -> Path:
    """把划分结果写入 out_dir/<split>.json，返回 out_dir。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        ids = splits.get(name, [])
        (out_dir / f"{name}.json").write_text(
            json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out_dir


def load_splits(splits_dir: str | Path) -> dict[str, list[str]]:
    """从 out_dir/<split>.json 读回划分结果。"""
    splits_dir = Path(splits_dir)
    result = {}
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result

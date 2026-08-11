"""阶段 3：SFT —— Qwen2.5-7B-Instruct + LoRA 监督微调。

数据管线（instance_id 贯穿）：
    sft_dataset.jsonl（instance + solution）→ 按 splits 过滤 → TAI 模板构建输入、
    排程 JSON 编码输出 → tokenize（prompt 部分 label=-100，仅答案算损失）→ LoRA + Trainer。

模型与输出：LoRA adapter 保存到 experiments/<exp_name>/（config 冻结存档）。
"""

from __future__ import annotations

import json
from pathlib import Path

from model.format import build_tai, encode_schedule
from problem.instance import Instance

# ---- 数据集构建 ----


def load_sft_records(dataset_path: str | Path, split_ids: list[str] | None = None) -> list[dict]:
    """读取 sft_dataset.jsonl；split_ids 给定则只保留这些 instance_id。"""
    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if split_ids is not None and rec["instance"]["instance_id"] not in split_ids:
                continue
            records.append(rec)
    return records


def make_text_pair(rec: dict) -> tuple[str, str]:
    """把一条监督记录转成 (prompt, answer) 文本对。

    prompt = TAI 模板；answer = 排程 JSON。
    """
    instance = Instance.from_dict(rec["instance"])
    prompt = build_tai(instance)
    answer = encode_schedule(instance, rec["solution"]["starts"])
    return prompt, answer


def tokenize_pair(tokenizer, prompt: str, answer: str, max_length: int, mask_prompt: bool = True) -> dict:
    """把 (prompt, answer) 编码为模型输入；mask_prompt=True 时 prompt 部分 label=-100。"""
    full_text = prompt + answer + tokenizer.eos_token
    enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    labels = input_ids.clone()
    if mask_prompt:
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        # 若被截断，prompt 可能不完整——保守处理：截断场景下仍只 mask 到实际截断位置
        prompt_len = min(prompt_len, len(input_ids) - 1)
        labels[:prompt_len] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

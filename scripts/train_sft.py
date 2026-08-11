#!/usr/bin/env python3
"""阶段 3 入口：SFT 训练（Qwen2.5-7B-Instruct + LoRA）。

用法（服务器 /root/jssp 下，数据与模型就绪后）：
    python scripts/train_sft.py                          # 默认 configs/sft/sft.yaml
    python scripts/train_sft.py -c configs/sft/sft.yaml

产物：
    experiments/sft_qwen7b/    checkpoint + LoRA adapter + 训练日志
    （config 已由脚本冻结存档到该目录）
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import load_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="JSSP SFT 训练（LoRA）")
    parser.add_argument("-c", "--config", default="configs/sft/sft.yaml", help="SFT 配置 YAML")
    parser.add_argument("--scale-filter", default=None,
                        help="只训练指定规模，如 '6x6,10x10'（按 instance_id 中的规模段过滤）")
    parser.add_argument("--adapter", default=None,
                        help="续训起点：已有 LoRA adapter 目录（None = 从基座开始）")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # 依赖在函数内导入：确保执行时环境完整
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    from training.loss import chunked_causal_lm_loss, install_chunked_forward
    from training.sft import load_sft_records, make_text_pair, tokenize_pair

    # ---- 数据 ----
    splits = load_splits(config["data"]["splits_dir"])
    records = load_sft_records(config["data"]["dataset_path"], split_ids=splits.get("train"))
    val_records = load_sft_records(config["data"]["dataset_path"], split_ids=splits.get("val"))

    def scale_of(rec):
        # instance_id 形如 gen-6x6-42-00000 → 规模段 "6x6"
        return rec["instance"]["instance_id"].split("-")[1]

    if args.scale_filter:
        allowed = set(args.scale_filter.split(","))
        records = [r for r in records if scale_of(r) in allowed]
        val_records = [r for r in val_records if scale_of(r) in allowed]
        print(f"[data] scale-filter={args.scale_filter} → train={len(records)} val={len(val_records)}")
    else:
        print(f"[data] train={len(records)} val={len(val_records)}")

    # ---- 模型与分词器 ----
    base_model = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers 5.x：dtype 参数接收 torch dtype 对象（不接受字符串）
    import torch

    dtype_str = config["model"].get("dtype", "auto")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}.get(dtype_str, "auto")
    # 注意：transformers 5.14 默认即 sdpa（sdpa_attention_forward 函数级 dispatch），无需显式指定
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    # 训练禁用 KV cache：use_cache=True 会在前向为整条序列构建 KV cache（seq 17408 约 1 GiB），
    # 训练不做生成，纯属浪费显存（2026-08-10 p3 OOM 的成因之一）→ 显式关闭。
    model.config.use_cache = False
    print("[model] use_cache=False（训练不生成，省去全序列 KV cache 显存）")

    # ---- 损失：长序列 + 152K 词表下默认 ForCausalLMLoss 的 logits.float() 会 OOM
    # （seq 4096/batch 4 首步即炸；seq 16384 更甚）
    # ① chunked_causal_lm_loss：默认 loss_function 的替代（分块 CE + checkpoint 流式）；
    # ② install_chunked_forward：连全序列 logits 都不落盘（分块 lm_head）——
    #    2026-08-10 实测 20x20 seq≈17137 即使 use_cache=False + 流式 loss，backward 仍需
    #    logits.grad(~4.93 GiB) → 32GB 打爆；分块 lm_head 省下 ≈10 GiB，才真正可训。
    if config["training"].get("use_chunked_loss", True):
        import functools

        chunk_size = config["training"].get("chunk_size", 1024)
        model.loss_function = functools.partial(chunked_causal_lm_loss, chunk_size=chunk_size)
        print(f"[loss] 使用分块交叉熵（chunked_causal_lm_loss, chunk_size={chunk_size}）")
        install_chunked_forward(model, chunk_size=chunk_size)

    # ---- LoRA（--adapter 给定则续训已有 adapter）----
    lora_cfg = config["lora"]
    if args.adapter:
        from peft import PeftModel

        # 关键：PeftModel.from_pretrained 默认 is_trainable=False（推理模式，全参数冻结）
        # → 续训会 trainable=0，白跑。必须显式 is_trainable=True。
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        model.print_trainable_parameters()
        print(f"[lora] 续训 adapter: {args.adapter} (is_trainable=True)")
    else:
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ---- 数据长度预检（防呆：max_length 过短导致样本被截断的静默事故）----
    max_length = config["data"]["max_length"]
    mask_prompt = config["data"]["mask_prompt"]

    def precheck_lengths(recs, max_len, per_scale=500):
        """按规模抽样统计完整样本 token 长度；存在超限样本则中止训练。

        事故背景（2026-08-08/09）：max_length=1024 时 6x6 样本 1611 tokens 被静默截断，
        模型只学会输出 JSON 前半段，评估可行性率 0%。此预检保证同类错误在启动时暴露。
        """
        import collections

        stats = collections.defaultdict(lambda: [0, 0, 0])  # scale -> [sampled, over, max_len_seen]
        for rec in recs:
            key = scale_of(rec)
            s = stats[key]
            if s[0] >= per_scale:
                continue
            s[0] += 1
            prompt, answer = make_text_pair(rec)
            n = len(tokenizer(prompt + answer + tokenizer.eos_token)["input_ids"])
            s[2] = max(s[2], n)
            if n > max_len:
                s[1] += 1
        for key, (sampled, over, worst) in stats.items():
            flag = "❌" if over else "✅"
            print(f"[precheck] {key}: 抽样 {sampled} 条, 最大长度 {worst}, 上限 {max_len} {flag}")
        bad = {k: v for k, v in stats.items() if v[1] > 0}
        if bad:
            raise SystemExit(
                f"[precheck] 以下规模存在超限样本将被截断 → 已中止训练: "
                f"{ {k: f'{v[1]}/{v[0]} 条超限, 最大 {v[2]} > {max_length}' for k, v in bad.items()} }。"
                f"请调大 configs 中该规模的 max_length 后重试。"
            )

    precheck_lengths(records, max_length)

    def build_ds(recs):
        pairs = [make_text_pair(r) for r in recs]
        tokenized = [
            tokenize_pair(tokenizer, p, a, max_length=max_length, mask_prompt=mask_prompt)
            for p, a in pairs
        ]
        return Dataset.from_list(tokenized)

    train_ds = build_ds(records)
    val_ds = build_ds(val_records) if val_records else None

    # ---- 训练 ----
    tr_cfg = config["training"]
    out_dir = Path(tr_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    # transformers 5.x：warmup_ratio/logging_dir 已弃用 → warmup_steps；Trainer 不再接收 tokenizer
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tr_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        learning_rate=tr_cfg["learning_rate"],
        num_train_epochs=tr_cfg["num_epochs"],
        max_steps=tr_cfg.get("max_steps", -1),
        warmup_steps=tr_cfg.get("warmup_steps", 0),
        logging_steps=tr_cfg["logging_steps"],
        eval_steps=tr_cfg.get("eval_steps", 500),
        save_steps=tr_cfg["save_steps"],
        save_total_limit=tr_cfg.get("save_total_limit", 2),
        gradient_checkpointing=tr_cfg.get("gradient_checkpointing", True),
        bf16=True,
        report_to=[],
        seed=config["seed"],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))

    stats = {"train_samples": len(records), "val_samples": len(val_records)}
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[done] LoRA adapter → {out_dir / 'final'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""阶段 4 入口：FOARL 训练（GRPO + 可行性/最优性联合奖励）。

每轮（epochs 配置）：
  1. vLLM 加载当前 LoRA adapter → 每实例采样 n_samples 个 rollout
  2. 奖励：parse → validate → r = r_feas + λ·r_opt（参考 = 监督数据 CP-SAT 解）
  3. 组内归一化优势 A（GRPO）
  4. transformers 训练：ref（SFT adapter）与 θ 顺序计算逐 token logprob →
     loss = -mean(A·logπ_θ) + β·KL → backward → step → 保存新 adapter

用法（服务器 /root/jssp 下）：
    python scripts/train_foarl.py [-c configs/foarl/foarl.yaml]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="FOARL 训练（GRPO）")
    parser.add_argument("-c", "--config", default="configs/foarl/foarl.yaml")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="从第几个 epoch 开始（断点续跑；>1 时 adapter 用上一 epoch 产物）")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    import torch
    from training.foarl import group_advantages, grpo_loss, token_logprobs
    from training.reward import combined_reward
    from model.format import build_tai, parse_schedule
    from problem.instance import Instance
    from problem.validator import validate
    from training.sft import load_sft_records

    # ---- 数据（train split，按规模过滤）----
    from data.splits import load_splits

    splits = load_splits(config["data"]["splits_dir"])
    train_ids = set(splits.get("train", []))
    records = load_sft_records(config["data"]["dataset_path"], split_ids=train_ids)
    scales = set(config["data"]["scales"])

    def scale_of(rec):
        return rec["instance"]["instance_id"].split("-")[1]

    records = [r for r in records if scale_of(r) in scales][: config["data"]["max_instances"]]
    print(f"[data] FOARL 实例数: {len(records)}（scales={config['data']['scales']}）")
    ref_makespan = {r["instance"]["instance_id"]: r["solution"]["makespan"] for r in records}
    prompts = [build_tai(Instance.from_dict(r["instance"])) for r in records]

    # ---- vLLM 采样环境（CUDA_HOME 等，见 CLAUDE.md 踩坑 16）----
    import os

    cu13 = "/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13"
    os.environ.setdefault("CUDA_HOME", cu13)
    # LD_LIBRARY_PATH 必需：llm.sleep() 的 shutdown 路径 import cumem_allocator 需要 libnvrtc.so.13
    #（2026-08-11 epoch 2 崩溃根因：sleep 时 ImportError: libnvrtc.so.13）
    os.environ["LD_LIBRARY_PATH"] = f"{cu13}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    sft_adapter = config["model"]["sft_adapter"]
    current_adapter = sft_adapter  # 起点 = SFT adapter
    start_epoch = args.start_epoch
    if start_epoch > 1:
        # 断点续跑：adapter 用上一 epoch 产物
        prev = out_dir / f"epoch{start_epoch - 1}"
        if not prev.exists():
            print(f"[error] 续跑起点不存在: {prev}", file=sys.stderr)
            return 1
        current_adapter = str(prev)
        print(f"[resume] 从 epoch {start_epoch} 续跑，adapter = {current_adapter}")

    for epoch in range(start_epoch - 1, config["grpo"]["epochs"]):
        t_epoch = time.perf_counter()
        print(f"\n===== GRPO epoch {epoch + 1}/{config['grpo']['epochs']}（adapter: {current_adapter}）=====")

        # ---- 1. vLLM 采样 ----
        # 引擎生命周期：循环外创建一次，epoch 间 sleep/wake_up 切换 GPU
        #（vLLM 0.26 无 shutdown/close；del 会残留 EngineCore 子进程占显存，见 2026-08-11 事故）
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        rollout_path = out_dir / f"epoch{epoch + 1}_rollouts.jsonl"
        n_samples = config["sampling"]["n_samples"]

        # 采样复用：已落盘的 rollouts 直接加载（崩溃保护，2026-08-11 起）
        if rollout_path.exists():
            print(f"[采样] 复用已落盘 rollouts → {rollout_path}")
            with open(rollout_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            texts_by_key = {(r["iid"], r["sample"]): r["text"] for r in rows}
            texts = [
                [texts_by_key[(rec["instance"]["instance_id"], s)] for s in range(n_samples)]
                for rec in records
            ]
        else:
            if "llm" not in locals() or llm is None:
                def make_llm():
                    return LLM(
                        model=config["model"]["base_model"],
                        enable_lora=True,
                        max_lora_rank=32,
                        gpu_memory_utilization=0.85,
                        enforce_eager=True,
                    )

                try:
                    llm = make_llm()
                except Exception as e:
                    print(f"[采样] vLLM 启动失败（{type(e).__name__}），60s 后重试...")
                    time.sleep(60)
                    llm = make_llm()
            else:
                try:
                    llm.wake_up()
                    time.sleep(5)
                except Exception as e:
                    # wake_up 失败（如显存残留）：销毁引擎重建
                    print(f"[采样] llm.wake_up() 失败（{type(e).__name__}），销毁重建...")
                    try:
                        llm.sleep()
                    except Exception:
                        pass
                    del llm
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(15)
                    llm = make_llm()
            lora_req = LoRARequest("foarl", 1, current_adapter)
            max_nt = max(config["sampling"]["max_new_tokens"], 68 * max(Instance.from_dict(r["instance"]).n * Instance.from_dict(r["instance"]).m for r in records))
            params = SamplingParams(
                temperature=config["sampling"]["temperature"],
                top_p=config["sampling"]["top_p"],
                max_tokens=max_nt,
            )
            batch_prompts = [p for p in prompts for _ in range(n_samples)]  # 每实例 n_samples 份
            print(f"[采样] {len(batch_prompts)} 个请求（{len(records)} 实例 × {n_samples}）...")
            t0 = time.perf_counter()
            outputs = llm.generate(batch_prompts, params, lora_request=lora_req)
            print(f"[采样] 完成，耗时 {time.perf_counter() - t0:.1f}s")
            # 采样结果立即落盘（崩溃保护：训练阶段读盘，采样不重跑）
            with open(rollout_path, "w", encoding="utf-8") as f:
                for i, rec in enumerate(records):
                    iid = rec["instance"]["instance_id"]
                    for s in range(n_samples):
                        f.write(json.dumps({"iid": iid, "sample": s,
                                            "text": outputs[i * n_samples + s].outputs[0].text},
                                           ensure_ascii=False) + "\n")
            print(f"[采样] 结果已落盘 → {rollout_path}")
            # 构建 texts 供奖励/训练阶段使用（2026-08-11 bug：此前漏了这步导致 UnboundLocalError）
            texts = [
                [outputs[i * n_samples + s].outputs[0].text for s in range(n_samples)]
                for i in range(len(records))
            ]
            # sleep 释放 GPU 给训练阶段（引擎进程保留，下轮 wake_up 复用）；失败则重试一次
            try:
                llm.sleep()
            except Exception as e:
                print(f"[采样] llm.sleep() 失败（{type(e).__name__}），重试...")
                time.sleep(10)
                llm.sleep()
            time.sleep(5)
            torch.cuda.empty_cache()

        # ---- 2. 奖励 + 3. 优势 ----
        per_instance_rewards: dict[str, list[float]] = {}
        rollouts: dict[str, list[str]] = {}
        for i, rec in enumerate(records):
            iid = rec["instance"]["instance_id"]
            inst = Instance.from_dict(rec["instance"])
            rewards = []
            for s in range(n_samples):
                text = texts[i][s]
                parsed = parse_schedule(inst, text)
                if parsed.ok:
                    check = validate(inst, parsed.starts)
                    if check.valid:
                        rewards.append(combined_reward(inst, check, ref_makespan[iid], config["reward"]["lambda_opt"]))
                        continue
                rewards.append(0.0)  # 不可行：r_feas=0（combined_reward 对不可行为 0）
            per_instance_rewards[iid] = rewards
            rollouts[iid] = texts[i]

        feas_rate = sum(1 for rs in per_instance_rewards.values() if max(rs) > 1.0) / len(records)
        print(f"[奖励] 至少一个可行解的实例占比: {feas_rate * 100:.1f}%")
        advantages = {iid: group_advantages(rs) for iid, rs in per_instance_rewards.items()}

        # ---- 4. 训练阶段（ref → θ 顺序计算 logprob）----
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"], trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def logprob_for(adapter_path, trainable=False):
            model = AutoModelForCausalLM.from_pretrained(
                config["model"]["base_model"], dtype=torch.bfloat16, trust_remote_code=True
            )
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
            model = model.to("cuda")
            if not trainable:
                model.eval()
            logps = {}
            with torch.no_grad() if not trainable else torch.enable_grad():
                for rec in records:
                    iid = rec["instance"]["instance_id"]
                    texts = rollouts[iid]
                    prompt = prompts[records.index(rec)]
                    for s, text in enumerate(texts):
                        full = tokenizer(prompt + text, return_tensors="pt")["input_ids"].to("cuda")
                        # BOS 对齐：full 与 prompt 都用默认 add_special_tokens（都含 BOS）
                        plen = len(tokenizer(prompt)["input_ids"])
                        logps[(iid, s)] = token_logprobs(model, full, plen)
            return model, logps

        # ref logprob（SFT adapter，冻结）
        print("[训练] 计算 ref logprob（SFT adapter）...")
        t0 = time.perf_counter()
        ref_model, ref_logps = logprob_for(sft_adapter, trainable=False)
        del ref_model
        torch.cuda.empty_cache()
        print(f"  完成 {time.perf_counter() - t0:.1f}s")

        # θ logprob（当前 adapter，可训练）+ GRPO loss + step
        print("[训练] 计算 θ logprob 并更新 LoRA...")
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model"], dtype=torch.bfloat16, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, current_adapter, is_trainable=True)
        model = model.to("cuda")
        model.train()
        # 长 rollout（模型多输出 ~3000 tokens）+ 训练模式全激活会 OOM → checkpoint 重算
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["grpo"]["learning_rate"])

        total_loss = 0.0
        n_items = 0
        for rec in records:
            iid = rec["instance"]["instance_id"]
            prompt = prompts[records.index(rec)]
            for s, text in enumerate(rollouts[iid]):
                full = tokenizer(prompt + text, return_tensors="pt")["input_ids"].to("cuda")
                plen = len(tokenizer(prompt)["input_ids"])
                logp_theta = token_logprobs(model, full, plen)
                loss = grpo_loss(logp_theta, ref_logps[(iid, s)], advantages[iid][s], config["grpo"]["kl_coef"])
                total_loss += loss.item()
                n_items += 1
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grpo"]["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()

        # 保存新 adapter
        current_adapter = str(out_dir / f"epoch{epoch + 1}")
        model.save_pretrained(current_adapter)
        print(f"[训练] loss={total_loss / n_items:.4f} | 新 adapter → {current_adapter} | 耗时 {time.perf_counter() - t0:.1f}s")
        # 彻底清理：PeftModel 循环引用需 gc.collect() 才释放 base 权重（否则残留 ~13GB
        # 导致下一轮 vLLM 启动失败，2026-08-11 epoch 3 崩溃根因）
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(10)

        epoch_stats = {
            "epoch": epoch + 1,
            "feasible_rate": feas_rate,
            "avg_loss": total_loss / n_items,
            "elapsed_s": time.perf_counter() - t_epoch,
        }
        print(f"[epoch {epoch + 1}] 完成，耗时 {(time.perf_counter() - t_epoch) / 60:.1f} 分钟")
        stats_path = out_dir / "foarl_stats.jsonl"
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_stats, ensure_ascii=False) + "\n")

    print(f"\n[done] FOARL 完成，最终 adapter → {current_adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

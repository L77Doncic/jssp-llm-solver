#!/usr/bin/env python3
"""阶段 5 入口：评估模型输出的排程质量（可行性 / gap / 时间）。

流程：build_tai → 批量生成（transformers，left-padding，batch 并行）→
BoN 采样（每实例 n_samples 次）→ parse_schedule 解析 → validator 校验 →
取最优可行解 → 相对 reference 算 gap。
汇总输出：可行性率、平均/最优 gap、平均生成时间 → experiments/<exp>/results.json

用法（服务器，模型就绪后）：
    python scripts/evaluate_schedule.py [-c configs/eval/eval.yaml] [--limit N] [--batch-size 4]

说明：vLLM 因 flashinfer cu128 wheel 不支持 5090（SM 12x 需 CUDA 12.9+），
推理走 transformers 批量生成（2026-08-10 实测 batch4 ≈124 tokens/s，约 4x 单实例）。
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description="评估 LLM 排程输出")
    parser.add_argument("-c", "--config", default="configs/eval/eval.yaml")
    parser.add_argument("--limit", type=int, default=None, help="最多评估的实例数（小批量验证用）")
    parser.add_argument("--batch-size", type=int, default=4, help="推理 batch 大小（显存允许时调大提速）")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    from data.splits import load_splits
    from evaluation.metrics import ScheduleEval, summarize
    from model.format import build_tai, parse_schedule
    from problem.instance import Instance
    from problem.validator import validate
    from training.sft import load_sft_records

    # ---- 模型加载 ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = config["model"]["base_model"]
    adapter = config["model"].get("adapter")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # decoder-only 生成必须 left padding（右 padding 会污染生成结果）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype="auto", trust_remote_code=True, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[model] 基座 + LoRA adapter: {adapter}")
    else:
        print(f"[model] 零样本基座: {base_model}")
    model.eval()

    # ---- 数据 ----
    splits = load_splits(config["data"]["splits_dir"])
    split = config["data"]["split"]
    ids = set(splits.get(split, []))
    records = load_sft_records(config["data"]["dataset_path"], split_ids=ids)
    if args.limit:
        records = records[: args.limit]
    print(f"[data] {split} 实例数: {len(records)}")
    reference_of = {r["instance"]["instance_id"]: r["solution"]["makespan"] for r in records}

    # ---- 推理（批量 BoN）----
    n_samples = config["inference"]["n_samples"]
    gen_kwargs = dict(
        max_new_tokens=config["inference"].get("max_new_tokens", 4096),  # 下限，按规模放大
        temperature=config["inference"]["temperature"],
        top_p=config["inference"]["top_p"],
        do_sample=True,
    )
    batch_size = args.batch_size

    # 任务队列：(实例, 采样序号)；每个实例生成 n_samples 次
    tasks = [(rec, s) for rec in records for s in range(n_samples)]
    # 按实例聚合候选 makespan
    best_makespan: dict[str, int] = {}
    feasible_count: dict[str, int] = defaultdict(int)
    gen_times: list[float] = []

    t_all0 = time.perf_counter()
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i : i + batch_size]
        recs = [t[0] for t in chunk]
        instances = [Instance.from_dict(r["instance"]) for r in recs]

        # max_new_tokens 按该批最大规模放大（68×n×m 公式，防截断）
        max_nt = max(gen_kwargs["max_new_tokens"], 68 * max(inst.n * inst.m for inst in instances))
        prompts = [build_tai(inst) for inst in instances]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        # left-padding 下每条的真实 prompt 长度（非 pad token 数）
        gen_lens = [(enc["input_ids"][j] != tokenizer.pad_token_id).sum().item() for j in range(len(chunk))]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, **{**gen_kwargs, "max_new_tokens": max_nt})
        gen_times.append((time.perf_counter() - t0) / len(chunk))

        for j, (rec, inst) in enumerate(zip(recs, instances)):
            plen = gen_lens[j]
            text = tokenizer.decode(out[j][plen:], skip_special_tokens=True)
            parsed = parse_schedule(inst, text)
            iid = inst.instance_id
            if parsed.ok:
                check = validate(inst, parsed.starts)
                if check.valid:
                    feasible_count[iid] += 1
                    cur = check.makespan
                    if iid not in best_makespan or cur < best_makespan[iid]:
                        best_makespan[iid] = cur
        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(tasks))
            print(f"  [{done}/{len(tasks)}] 采样完成")

    # ---- 汇总 ----
    evals: list[ScheduleEval] = []
    for rec in records:
        iid = rec["instance"]["instance_id"]
        ref = reference_of.get(iid)
        if iid in best_makespan:
            best = best_makespan[iid]
            gap_pct = (best - ref) / ref * 100.0 if ref and ref > 0 else None
            evals.append(ScheduleEval(instance_id=iid, feasible=True, makespan=best, gap_pct=gap_pct))
        else:
            evals.append(ScheduleEval(instance_id=iid, feasible=False, makespan=None, gap_pct=None,
                                      errors=["无可行解（BoN 全部失败）"]))

    stats = summarize(evals)
    stats["avg_gen_time_per_sample_s"] = sum(gen_times) / len(gen_times) if gen_times else None
    stats["n_samples_bon"] = n_samples
    stats["total_wall_time_s"] = time.perf_counter() - t_all0
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"stats": stats, "per_instance": [
            {"instance_id": e.instance_id, "feasible": e.feasible,
             "makespan": e.makespan, "gap_pct": e.gap_pct}
            for e in evals
        ]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] → {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""失败模式分析：hint 模型 vs 纯 SFT 模型，生成失败分类（格式/内容/机器冲突/工序顺序）"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/jssp/src")

BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
OUT = "/root/jssp/experiments/failure_analysis/results.json"

os.environ["CUDA_HOME"] = "/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13"
os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from model.format import build_tai, parse_schedule
from problem.validator import validate
from problem.instance import Instance
from training.sft import load_sft_records
from data.splits import load_splits

splits = load_splits("/root/autodl-tmp/jssp_data/splits")
records = load_sft_records("/root/autodl-tmp/jssp_data/supervised/sft_dataset.jsonl",
                           split_ids=set(splits["test"]))
records = [r for r in records if r["instance"]["instance_id"].startswith("gen-6x6-")][:30]
print("[data] 30 个 6x6 test 实例 × 8 采样/模型")


def classify(inst, text):
    """失败类型分类：feasible / format / content / machine_conflict / precedence / unassigned"""
    parsed = parse_schedule(inst, text)
    if not parsed.ok:
        err = parsed.error or ""
        if "未找到" in err or "JSON 解析失败" in err:
            return "format_json"
        return "content_records"
    check = validate(inst, parsed.starts)
    if check.valid:
        return "feasible"
    if any("冲突" in e for e in check.errors):
        return "machine_conflict"
    if any("早于" in e for e in check.errors):
        return "precedence"
    if any("未调度" in e for e in check.errors):
        return "unassigned"
    return "constraint_other"


llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32,
          gpu_memory_utilization=0.85, enforce_eager=True)
params = SamplingParams(temperature=0.5, top_p=0.7, max_tokens=4096)

models = {
    "sft_plain": "/root/jssp/experiments/sft_qwen7b/final",
    "sft_hint": "/root/jssp/experiments/sft_qwen7b_hint/final",
}
results = {}
t0 = time.time()
for idx, (mname, adapter) in enumerate(models.items()):
    # 关键：LoRA int_id 必须全局唯一（vLLM 用 int_id 标识 LoRA，同 id 缓存冲突
    # 会导致后加载的 adapter 实际仍是前一个——2026-08-14 失败分析 bug 根因）
    lora_req = LoRARequest(mname, idx + 1, adapter)
    counter = Counter()
    per_instance = {}
    for rec in records:
        iid = rec["instance"]["instance_id"]
        inst = Instance.from_dict(rec["instance"])
        prompt = build_tai(inst)
        outs = llm.generate([prompt] * 8, params, lora_request=lora_req)
        types = [classify(inst, o.outputs[0].text) for o in outs]
        for t in types:
            counter[t] += 1
        per_instance[iid] = types
        print("[{}] {}/{} done".format(mname, len(per_instance), len(records)))
    total = sum(counter.values())
    results[mname] = {
        "distribution": {k: {"count": v, "pct": v / total * 100} for k, v in counter.most_common()},
        "per_instance_feasible": {iid: ts.count("feasible") for iid, ts in per_instance.items()},
    }
    print("[{}] {}".format(mname, dict(counter)))

results["total_time_s"] = time.time() - t0
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(json.dumps(results, indent=2))
print("[done] -> " + OUT)

#!/usr/bin/env python3
"""hint 模型的 hint 评估（论文口径公平对比）：hint 训练 + hint 推理，60 实例 6x6 BoN8"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/jssp/src")

BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
ADAPTER = "/root/jssp/experiments/sft_qwen7b_hint/final"
OUT = "/root/jssp/experiments/eval_hint_hint/results.json"

os.environ["CUDA_HOME"] = "/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13"
os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from model.format import build_tai, parse_schedule
from problem.validator import validate
from problem.instance import Instance
from problem.makespan import makespan_from_starts
from solver.heuristics import gt_schedule
from training.sft import load_sft_records
from data.splits import load_splits

splits = load_splits("/root/autodl-tmp/jssp_data/splits")
records = load_sft_records("/root/autodl-tmp/jssp_data/supervised/sft_dataset.jsonl",
                           split_ids=set(splits["test"]))
records = [r for r in records if r["instance"]["instance_id"].startswith("gen-6x6-")][:60]
ref_of = {r["instance"]["instance_id"]: r["solution"]["makespan"] for r in records}
print("[data] {} 个 6x6 test 实例（hint 模型 + hint 推理）".format(len(records)))

llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32,
          gpu_memory_utilization=0.85, enforce_eager=True)
lora_req = LoRARequest("jssp", 1, ADAPTER)
params = SamplingParams(temperature=0.5, top_p=0.7, max_tokens=4096)

feas = 0
gaps = []
best_gaps = []
t0 = time.time()
for idx, rec in enumerate(records):
    iid = rec["instance"]["instance_id"]
    inst = Instance.from_dict(rec["instance"])
    spt_mk = makespan_from_starts(inst, gt_schedule(inst, rule="spt"))
    prompt = build_tai(inst, heuristic_hint="A fast SPT heuristic yields makespan {}".format(spt_mk))
    outs = llm.generate([prompt] * 8, params, lora_request=lora_req)
    makespans = []
    for o in outs:
        parsed = parse_schedule(inst, o.outputs[0].text)
        if parsed.ok:
            check = validate(inst, parsed.starts)
            if check.valid:
                makespans.append(check.makespan)
    if makespans:
        feas += 1
        best = min(makespans)
        ref = ref_of[iid]
        gaps.append((best - ref) / ref * 100)
        best_gaps.append((min(makespans) - ref) / ref * 100)
    print("[{}/{}] {}: {} feasible".format(idx + 1, len(records), iid, len(makespans)))

results = {"n": len(records),
           "feasible_rate": feas / len(records),
           "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None,
           "best_gap_pct": min(best_gaps) if best_gaps else None,
           "elapsed_s": time.time() - t0}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(json.dumps(results, indent=2))
print("[done] -> " + OUT)

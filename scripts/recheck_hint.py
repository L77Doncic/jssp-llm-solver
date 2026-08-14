#!/usr/bin/env python3
"""复测：hint 模型在 eval_hint_sft 同批次（前 60 test 6x6）的可行率，消除批次差异"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/root/jssp/src")
BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
ADAPTER = "/root/jssp/experiments/sft_qwen7b_hint/final"
OUT = "/root/jssp/experiments/recheck_hint/results.json"
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
records = [r for r in records if r["instance"]["instance_id"].startswith("gen-6x6-")][:60]
ref_of = {r["instance"]["instance_id"]: r["solution"]["makespan"] for r in records}

llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32,
          gpu_memory_utilization=0.85, enforce_eager=True)
lora_req = LoRARequest("jssp", 1, ADAPTER)
params = SamplingParams(temperature=0.5, top_p=0.7, max_tokens=4096)

feas = 0
gaps = []
for rec in records:
    iid = rec["instance"]["instance_id"]
    inst = Instance.from_dict(rec["instance"])
    prompt = build_tai(inst)
    outs = llm.generate([prompt] * 8, params, lora_request=lora_req)
    best = None
    for o in outs:
        parsed = parse_schedule(inst, o.outputs[0].text)
        if parsed.ok:
            check = validate(inst, parsed.starts)
            if check.valid and (best is None or check.makespan < best):
                best = check.makespan
    if best is not None:
        feas += 1
        gaps.append((best - ref_of[iid]) / ref_of[iid] * 100)

results = {"n": 60, "feasible_rate": feas / 60,
           "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None,
           "same_batch_as_eval_hint_sft": True}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(json.dumps(results, indent=2))
print("[done] -> " + OUT)

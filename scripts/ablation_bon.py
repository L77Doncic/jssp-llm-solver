#!/usr/bin/env python3
"""消融：BoN 大小事后推导（60 实例，保存每样本结果，推导 N=1,2,4,8 可行率/gap）"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/jssp/src")

BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
ADAPTER = "/root/jssp/experiments/foarl_qwen7b/epoch2"
OUT = "/root/jssp/experiments/ablation_bon/results.json"

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
print("[data] {} 个 6x6 test 实例".format(len(records)))

llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32,
          gpu_memory_utilization=0.85, enforce_eager=True)
lora_req = LoRARequest("jssp", 1, ADAPTER)
params = SamplingParams(temperature=0.5, top_p=0.7, max_tokens=4096)

per_instance = {}
t0 = time.time()
for rec in records:
    iid = rec["instance"]["instance_id"]
    inst = Instance.from_dict(rec["instance"])
    prompt = build_tai(inst)
    outs = llm.generate([prompt] * 8, params, lora_request=lora_req)
    makespans = []
    for o in outs:
        parsed = parse_schedule(inst, o.outputs[0].text)
        if parsed.ok:
            check = validate(inst, parsed.starts)
            if check.valid:
                makespans.append(check.makespan)
    per_instance[iid] = makespans
    print("[{}/{}] {}: {} feasible samples".format(
        len(per_instance), len(records), iid, len(makespans)))


def bon_stats(n):
    feas = 0
    gaps = []
    for iid, ms in per_instance.items():
        best = None
        if ms:
            best = min(ms[:n])
        if best is not None:
            feas += 1
            ref = ref_of[iid]
            gaps.append((best - ref) / ref * 100)
    return {"n_bon": n, "feasible_rate": feas / len(per_instance),
            "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None}


results = {"per_instance": {k: {"makespans": v, "ref": ref_of[k]}
                           for k, v in per_instance.items()},
           "bon": [bon_stats(n) for n in [1, 2, 4, 8]],
           "total_time_s": time.time() - t0}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(json.dumps(results["bon"], indent=2))
print("[done] -> " + OUT)

#!/usr/bin/env python3
"""分布内多规模评估：10x10/15x15/20x20 各 60 test 实例（FOARL epoch2，BoN8）"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/jssp/src")

BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
ADAPTER = "/root/jssp/experiments/foarl_qwen7b/epoch2"
OUT = "/root/jssp/experiments/eval_multiscale/results.json"

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
selected = {}
for r in records:
    key = r["instance"]["instance_id"].split("-")[1]
    if key in ("10x10", "15x15", "20x20"):
        selected.setdefault(key, []).append(r)
for k in selected:
    selected[k] = selected[k][:60]
print("[data] " + " ".join("{}:{}".format(k, len(v)) for k, v in selected.items()))

llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32,
          gpu_memory_utilization=0.85, enforce_eager=True)
lora_req = LoRARequest("jssp", 1, ADAPTER)

results = {}
t0 = time.time()
for scale, recs in selected.items():
    feas = 0
    gaps = []
    for rec in recs:
        iid = rec["instance"]["instance_id"]
        inst = Instance.from_dict(rec["instance"])
        prompt = build_tai(inst)
        max_nt = max(4096, 68 * inst.n * inst.m)
        params = SamplingParams(temperature=0.5, top_p=0.7, max_tokens=max_nt)
        outs = llm.generate([prompt] * 8, params, lora_request=lora_req)
        best = None
        for o in outs:
            parsed = parse_schedule(inst, o.outputs[0].text)
            if parsed.ok:
                check = validate(inst, parsed.starts)
                if check.valid and (best is None or check.makespan < best):
                    best = check.makespan
        ref = rec["solution"]["makespan"]
        if best is not None:
            feas += 1
            gaps.append((best - ref) / ref * 100)
        print("[{}] {}/{} done".format(scale, feas, len(recs)))
    results[scale] = {"n": len(recs), "feasible_rate": feas / len(recs),
                      "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None}
    print("[{}] complete: feasible_rate {:.1f}% gap {}".format(
        scale, results[scale]["feasible_rate"] * 100,
        "{:.2f}%".format(results[scale]["avg_gap_pct"]) if results[scale]["avg_gap_pct"] else "N/A"))

results["total_time_s"] = time.time() - t0
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(json.dumps(results, indent=2))
print("[done] -> " + OUT)

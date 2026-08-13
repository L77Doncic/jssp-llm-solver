#!/usr/bin/env python3
"""公开基准泛化评估：ft/la/ta 123 实例，模型生成 vs best_known。vLLM 推理。"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/root/jssp/src")

BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
ADAPTER = "/root/jssp/experiments/foarl_qwen7b/epoch2"
RAW = "/root/autodl-tmp/jssp_data/raw"
OUT = "/root/jssp/experiments/eval_benchmark/results.json"

import os
os.environ["CUDA_HOME"] = "/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13"
os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from model.format import build_tai, parse_schedule
from problem.validator import validate
from data.parsers import parse_jobshop_file

manifest = json.loads(Path(RAW + "/benchmark_manifest.json").read_text())
items = [(k, v) for k, v in manifest.items() if not k.startswith("_") and v.get("best_known")]

llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32, gpu_memory_utilization=0.85, enforce_eager=True)
lora_req = LoRARequest("jssp", 1, ADAPTER)
params = SamplingParams(temperature=0.5, top_p=0.7, max_tokens=8192)

results = []
t0 = time.time()
for idx, (name, meta) in enumerate(items):
    inst = parse_jobshop_file(f"{RAW}/{name}.txt")
    prompt = build_tai(inst)
    outs = llm.generate([prompt] * 8, params, lora_request=lora_req)  # BoN8
    best = None
    for o in outs:
        parsed = parse_schedule(inst, o.outputs[0].text)
        if parsed.ok:
            check = validate(inst, parsed.starts)
            if check.valid and (best is None or check.makespan < best):
                best = check.makespan
    bk = meta["best_known"]
    gap = (best - bk) / bk * 100 if best else None
    results.append({"instance": name, "n": inst.n, "m": inst.m, "best_known": bk,
                    "model_makespan": best, "gap_pct": gap, "feasible": best is not None})
    print(f"[{idx+1}/{len(items)}] {name}: makespan={best} best_known={bk} gap={gap}")

feas = sum(1 for r in results if r["feasible"])
gaps = [r["gap_pct"] for r in results if r["gap_pct"] is not None]
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump({"stats": {"n": len(results), "feasible_rate": feas / len(results),
                     "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None,
                     "best_gap_pct": min(gaps) if gaps else None,
                     "total_time_s": time.time() - t0},
           "results": results}, open(OUT, "w"), indent=2)
print(f"[done] → {OUT}")

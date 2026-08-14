#!/usr/bin/env python3
"""TAI 特征消融：SPT 启发式提示 有/无 对比（60 个 6x6 test 实例，BoN8，FOARL epoch2）"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/root/jssp/src")

BASE = "/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct"
ADAPTER = "/root/jssp/experiments/foarl_qwen7b/epoch2"
OUT = "/root/jssp/experiments/ablation_tai/results.json"

os.environ["CUDA_HOME"] = "/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13"
os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from model.format import build_tai, parse_schedule
from problem.validator import validate
from problem.instance import Instance
from solver.heuristics import gt_schedule
from problem.makespan import makespan_from_starts
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

def eval_variant(with_hint):
    per_instance = {}
    t0 = time.time()
    for rec in records:
        iid = rec["instance"]["instance_id"]
        inst = Instance.from_dict(rec["instance"])
        if with_hint:
            spt_starts = gt_schedule(inst, rule="spt")
            spt_mk = makespan_from_starts(inst, spt_starts)
            prompt = build_tai(inst, heuristic_hint="A fast SPT heuristic yields makespan {}".format(spt_mk))
        else:
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
        print("[{}] {}/{}".format("hint" if with_hint else "plain", len(per_instance), len(records)))
    feas = sum(1 for ms in per_instance.values() if ms)
    gaps = [(min(ms) - ref_of[k]) / ref_of[k] * 100 for k, ms in per_instance.items() if ms]
    return {"variant": "with_spt_hint" if with_hint else "plain",
            "feasible_rate": feas / len(per_instance),
            "avg_gap_pct": sum(gaps) / len(gaps) if gaps else None,
            "best_gap_pct": min(gaps) if gaps else None,
            "elapsed_s": time.time() - t0}

results = {"variants": [eval_variant(False), eval_variant(True)]}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(json.dumps(results, indent=2))
print("[done] -> " + OUT)

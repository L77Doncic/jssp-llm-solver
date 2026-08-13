#!/usr/bin/env python3
"""基线对比：SPT/MOR 启发式 + CP-SAT（限时 30s）在公开基准 123 实例上的 gap。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/jssp/src")

from data.parsers import parse_jobshop_file
from solver.heuristics import solve_all_rules
from solver.ortools import solve_cp_sat

RAW = "/root/autodl-tmp/jssp_data/raw"
OUT = "/root/jssp/experiments/baseline_benchmark/results.json"
manifest = json.loads(Path(RAW + "/benchmark_manifest.json").read_text())
items = [(k, v) for k, v in manifest.items() if not k.startswith("_") and v.get("best_known")]

results = []
t0 = time.time()
for idx, (name, meta) in enumerate(items):
    inst = parse_jobshop_file(RAW + "/" + name + ".txt")
    bk = meta["best_known"]
    heur = solve_all_rules(inst)
    best_heur = min(m for _, m in heur.values())
    cp = solve_cp_sat(inst, time_limit=30.0, num_workers=8)
    cp_mk = cp.makespan
    cp_gap = (cp_mk - bk) / bk * 100 if cp_mk else None
    results.append({
        "instance": name, "n": inst.n, "m": inst.m, "best_known": bk,
        "spt": heur["spt"][1], "mor": heur["mor"][1],
        "best_heuristic": best_heur,
        "heuristic_gap_pct": (best_heur - bk) / bk * 100,
        "cp_sat_makespan": cp_mk,
        "cp_sat_gap_pct": cp_gap,
        "cp_sat_status": cp.status,
    })
    print("[{}/{}] {}: heur_gap={:.1f}% cp_gap={}".format(
        idx + 1, len(items), name,
        results[-1]["heuristic_gap_pct"],
        "{:.1f}%".format(cp_gap) if cp_gap is not None else "None"))

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
h_gaps = [r["heuristic_gap_pct"] for r in results]
c_gaps = [r["cp_sat_gap_pct"] for r in results if r["cp_sat_gap_pct"] is not None]
json.dump({"stats": {"n": len(results),
                     "avg_heuristic_gap_pct": sum(h_gaps) / len(h_gaps),
                     "avg_cp_sat_gap_pct": sum(c_gaps) / len(c_gaps) if c_gaps else None,
                     "total_time_s": time.time() - t0},
           "results": results}, open(OUT, "w"), indent=2)
print("[done] -> " + OUT)

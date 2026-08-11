#!/bin/bash
# 仅运行 SFT p3（20x20）—— p1a/p1b/p2 已完成（experiments/sft_qwen7b/final 为 p2 产物）。
# 成功才写 ALL_SFT_DONE 标记（与 run_sft_all.sh 的约定一致，供无人值守监督脚本判定）。
#
# 2026-08-10 修复背景：p3 首步 OOM。成因有二：
#   ① train_sft.py 训练前向 use_cache=True → 全序列 KV cache（seq~17137 约 1 GiB）纯浪费；
#   ② 即便分块 CE，backward 仍要全序列 logits.grad（~4.93 GiB）→ 32GB 打爆。
#   修复：install_chunked_forward 把训练前向替换为「分块 lm_head + loss」，全序列 logits 不落盘。
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 缓解显存碎片化
cd /root/jssp
P=/root/miniconda3/bin/python
LOG=experiments/sft_p3_run.log

if "$P" -u scripts/train_sft.py -c configs/sft/sft_p3.yaml --scale-filter 20x20 --adapter experiments/sft_qwen7b/final > "$LOG" 2>&1; then
    echo "[p3 done]" >> "$LOG"
    echo "ALL_SFT_DONE" >> "$LOG"
    echo "[ALL_SFT_DONE] 四阶段 SFT 训练全部完成" >> experiments/sft_all_run.log
    echo "[p3 done] ALL_SFT_DONE written"
else
    echo "[p3 FAILED]" >> "$LOG"
    echo "[p3 FAILED] see $LOG"
    exit 1
fi

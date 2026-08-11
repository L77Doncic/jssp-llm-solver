#!/bin/bash
# 分阶段 SFT 串联：p1a(6x6) → p1b(10x10) → p2(15x15) → p3(20x20)，同一 LoRA 顺序续训
# 任一阶段失败即中止（|| exit 1）；ALL_SFT_DONE 仅在 p3 成功后写入。
# 2026-08-09 配置修正：max_length 按实测样本长度（6x6 1611 / 10x10 4176 / 15x15 9518 / 20x20 17112）
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 缓解显存碎片化（配合分块 CE）
cd /root/jssp
P=/root/miniconda3/bin/python

run_phase() {
  local tag="$1"; shift
  local log="experiments/sft_${tag}_run.log"
  if "$P" -u "$@" > "$log" 2>&1; then
    echo "[$tag done]" >> "$log"
    return 0
  else
    echo "[$tag FAILED]" >> "$log"
    return 1
  fi
}

run_phase p1a scripts/train_sft.py -c configs/sft/sft_p1a.yaml --scale-filter 6x6 || exit 1
run_phase p1b scripts/train_sft.py -c configs/sft/sft_p1b.yaml --scale-filter 10x10 --adapter experiments/sft_qwen7b/final || exit 1
run_phase p2  scripts/train_sft.py -c configs/sft/sft_p2.yaml  --scale-filter 15x15 --adapter experiments/sft_qwen7b/final || exit 1
run_phase p3  scripts/train_sft.py -c configs/sft/sft_p3.yaml  --scale-filter 20x20 --adapter experiments/sft_qwen7b/final || exit 1

echo "ALL_SFT_DONE" >> experiments/sft_p3_run.log
echo "[ALL_SFT_DONE] 四阶段 SFT 训练全部完成" >> experiments/sft_all_run.log

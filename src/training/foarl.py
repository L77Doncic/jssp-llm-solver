"""FOARL 核心逻辑（对齐 LLMCoSolver 论文，arXiv:2509.16865）。

GRPO 循环（每轮）：
1. vLLM 加载当前 LoRA adapter → 对每个实例采样 n_samples 个 rollout（组）
2. 奖励：r = r_feas + λ·r_opt（reward.py），参考 makespan 取监督数据（CP-SAT 解）
3. 组内归一化优势：A_i = (r_i - mean(r)) / std(r)
4. 训练阶段（transformers）：
   - ref 模型（SFT adapter，固定）对 rollout 逐 token logprob → KL 参考
   - 当前策略对 rollout 逐 token logprob（带计算图）
   - loss = -mean(A_i · log π_θ) + β · KL(π_θ ‖ π_ref)，backward 更新 LoRA

显存策略：ref 与 θ 的 logprob 顺序计算（各 15GB，避免 32GB 卡双模型 OOM）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_advantages(rewards: list[float]) -> list[float]:
    """组内归一化优势（GRPO）：A_i = (r_i - mean) / (std + eps)。"""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    eps = 1e-6
    return [(r - mean) / (std + eps) for r in rewards]


def token_logprobs(model, input_ids: torch.Tensor, prompt_len: int, chunk_size: int = 512) -> torch.Tensor:
    """计算 rollout 部分的逐 token logprob（分块前向，省显存）。

    注意：本函数不切换模型模式（eval/train 由调用方管理）、不包 no_grad
    （ref 场景调用方包 torch.no_grad()；θ 场景需保留梯度用于 GRPO loss 反向）。

    Args:
        model: transformers 模型
        input_ids: [1, seq_len] 完整序列（prompt + rollout）
        prompt_len: prompt token 数
        chunk_size: 分块大小（长序列显存控制）

    Returns:
        logp: [rollout_len] rollout 部分每个 token 的 logprob
        （logp[t] = log p(input_ids[prompt_len+t] | 前缀)）
    """
    seq_len = input_ids.shape[1]
    rollout_len = seq_len - prompt_len
    logp_list: list[torch.Tensor] = []

    # 需要 logits 的位置 = rollout 部分的每个 token（其前缀为 prompt+前 t 个 rollout token）
    # 分块：对每个 chunk 的起点位置做 forward，取该位置的预测分布
    for start in range(prompt_len, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        # 输入 = [0, end)（预测 end 位置需要到 end 的输入）
        chunk_input = input_ids[:, :end]
        out = model(chunk_input)
        logits = out.logits  # [1, end-1, vocab]（shift 后 logits[t] 预测 token t+1）
        # 需要预测的位置：start..end-1（对应输入位置 start-1..end-2）
        if start == prompt_len:
            # 第一个 chunk：需要预测 [prompt_len, end-1] → 输入位置 [prompt_len-1, end-2]
            pred_logits = logits[0, prompt_len - 1 : end - 1, :]  # [end-prompt_len, vocab]
            target_ids = input_ids[0, prompt_len:end]             # [end-prompt_len]
        else:
            pred_logits = logits[0, start - 1 : end - 1, :]
            target_ids = input_ids[0, start:end]
        logp = F.log_softmax(pred_logits.float(), dim=-1)
        logp = logp.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # [n]
        logp_list.append(logp)
        del out, logits

    return torch.cat(logp_list)


def grpo_loss(
    logp_theta: torch.Tensor,
    logp_ref: torch.Tensor,
    advantage: float,
    kl_coef: float,
) -> torch.Tensor:
    """单个 rollout 的 GRPO 损失。

    loss = -A · mean(log π_θ) + β · mean(log π_ref - log π_θ)
    第二项即 KL(π_θ ‖ π_ref) 的蒙特卡洛估计（对数域差）。
    """
    policy_loss = -advantage * logp_theta.mean()
    kl = (logp_ref - logp_theta).mean()
    return policy_loss + kl_coef * kl

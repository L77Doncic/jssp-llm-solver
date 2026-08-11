"""阶段 3：SFT 损失 —— 分块交叉熵 + 分块 lm_head（规避长序列 logits OOM）。

两种相关机制（均保持与默认 `ForCausalLMLoss` 数值等价，同 shift、同 ignore_index、
同归一化）：

1. `chunked_causal_lm_loss`：默认 loss_function 的替代。transformers 5.x 的
   `ForCausalLMLoss` 会先 `logits.float()` 全量 fp32 拷贝（seq 4096/batch 4 即 OOM）。
   按 chunk 逐段计算 CE 并用 `torch.utils.checkpoint` 包住 fp32 计算，避免全部 chunk
   常驻图（否则总驻留仍为 S*vocab*4，seq 17137 时约 10.4 GiB）。

2. `chunked_lm_head_loss` + `install_chunked_forward`：**连全序列 logits 都不落盘**。
   训练前向替换为：对 hidden_states 按 chunk 与 lm_head 做矩阵乘 + CE，直接产出 loss，
   不构造 (S, vocab) 的 logits 张量及其梯度。seq 17137 时省下 ≈ 4.93（logits）+
   4.93（logits.grad）GiB，32GB 卡上可训 20x20。数学上逐 chunk 矩阵乘与整段完全一致
   （线性运算），loss 数值与机制 1 相同。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce(
    chunk_logits: torch.Tensor,
    chunk_labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
) -> torch.Tensor:
    """单个 chunk 的交叉熵（sum reduction）。fp32 计算，与默认损失数值对齐。"""
    chunk_logits = chunk_logits.reshape(-1, vocab_size).float()
    chunk_labels = chunk_labels.reshape(-1)
    return F.cross_entropy(
        chunk_logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
    )


def chunked_lm_head_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
    vocab_size: int,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """按序列 chunk 计算 `lm_head` logits + 交叉熵，**不落盘全序列 logits**。

    与默认 `ForCausalLMLoss` 语义一致：shift 右移 1、ignore_index、mean 归一化；
    与 `chunked_causal_lm_loss` 数值一致（逐 chunk 矩阵乘在浮点上等于整段矩阵乘）。
    峰值 ≈ chunk_size*vocab*4 B（chunk 1024 时 ≈ 0.62 GiB），不随序列长度累积。
    """
    # 复制默认 shift：labels = pad(labels, (0, 1), ignore); shift_labels = labels[..., 1:]
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()

    batch, seq_len, _ = hidden_states.shape
    total = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    n_valid = 0
    for i in range(0, seq_len, chunk_size):
        hs_chunk = hidden_states[:, i : i + chunk_size, :]
        chunk_logits = hs_chunk @ lm_head_weight.T  # bf16 (B, chunk, V)
        chunk_logits = chunk_logits.reshape(-1, vocab_size).float()  # fp32
        chunk_labels = shift_labels[:, i : i + chunk_size].reshape(-1)
        valid = chunk_labels != ignore_index
        cnt = valid.sum().item()
        if cnt == 0:
            continue
        n_valid += cnt
        total = total + F.cross_entropy(
            chunk_logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
        )
    return total / max(n_valid, 1)


def install_chunked_forward(model, chunk_size: int = 1024, ignore_index: int = -100) -> None:
    """把模型 forward 替换为「分块 lm_head + loss」训练路径（monkey-patch，非子类）。

    仅当 `labels` 存在且 `past_key_values` 为 None（训练/评估）时走分块路径；
    其余（推理、generate、speculative decoding）走原始 forward。
    规避全序列 (S, vocab) logits 与其梯度的驻留 → 32GB 卡上可训 seq≈17137（20x20）。
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    orig_forward = model.forward

    def _forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if labels is None or past_key_values is not None:
            return orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # 训练路径：hidden_states → 分块 logits+loss，不构造全序列 logits
        model_kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **model_kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        hs = hidden_states[:, slice_indices, :]

        loss = chunked_lm_head_loss(
            hs,
            labels,
            model.lm_head.weight,
            model.config.vocab_size,
            ignore_index=ignore_index,
            chunk_size=chunk_size,
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=hs.new_empty(hs.shape[0], 0, model.config.vocab_size),  # 训练不回传全 logits
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states if kwargs.get("output_hidden_states") else None,
            attentions=outputs.attentions if kwargs.get("output_attentions") else None,
        )

    model.forward = _forward
    print(
        f"[model] 训练前向替换为分块 lm_head + loss（chunk_size={chunk_size}），"
        "避免全序列 logits 驻留（20x20 seq≈17137 可训）"
    )


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    chunk_size: int = 1024,
    **kwargs,
) -> torch.Tensor:
    """等价于 `transformers.loss.loss_utils.ForCausalLMLoss`，但按序列分块 + checkpoint 流式。

    语义对齐默认实现：
      - shift：logits[i] 预测 labels[i+1]（labels 右移 1，末尾补 ignore_index）；
      - reduction：num_items_in_batch 给定 → sum/num_items_in_batch；
                   否则按有效（非 ignore_index）token 数取均值。
    """
    reduction = "sum" if num_items_in_batch is not None else "mean"

    # 复制默认 shift：labels = pad(labels, (0, 1), ignore); shift_labels = labels[..., 1:]
    shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()

    batch, seq_len, _ = logits.shape
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    n_valid = 0
    for i in range(0, seq_len, chunk_size):
        chunk_logits = logits[:, i : i + chunk_size, :]
        chunk_labels = shift_labels[:, i : i + chunk_size]
        valid = chunk_labels != ignore_index
        cnt = valid.sum().item()
        if cnt == 0:
            continue
        n_valid += cnt
        if chunk_logits.requires_grad:
            # checkpoint：fp32 拷贝逐 chunk 重算，避免全部 chunk 常驻图 → 长序列下累积 OOM
            ce = checkpoint(
                _chunk_ce,
                chunk_logits,
                chunk_labels,
                vocab_size,
                ignore_index,
                use_reentrant=False,
            )
        else:
            ce = _chunk_ce(chunk_logits, chunk_labels, vocab_size, ignore_index)
        total = total + ce

    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(logits.device)
        return total / num_items_in_batch
    return total / max(n_valid, 1)

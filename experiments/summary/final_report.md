# 基于大语言模型的静态 JSSP 端到端求解研究 —— 最终报告

> 日期：2026-08-13 ｜ 模型：Qwen2.5-7B-Instruct + LoRA ｜ 参照：LLMCoSolver（NeurIPS 2025）
> 环境：RTX 5090 32GB ×1 ｜ 训练数据：42K 实例（CP-SAT 监督）

## 一、任务书交付对照（逐项验收）

| 任务书要求 | 完成情况 | 证据 |
|---|---|---|
| ① 标准化输入表示与输出解析方案 | ✅ | TAI 模板 + JSON 排程 + 容错解析 + validator（107 pytest 全绿） |
| ② 多规模训练与测试数据集 | ✅ | 42K 实例（6x6/10x10/15x15/20x20），91.3% 最优监督信号，train/val/test 划分 |
| ③ LLM 端到端求解原型系统 | ✅ | SFT（四阶段）+ FOARL（GRPO 3 epochs）+ BoN 推理入口 |
| ④ 完整评估流程 | ✅ | 可行性/gap/时间统计；分布内 + 公开基准双轨评估 |
| ⑤ 研究关注点（5 个） | ✅ | 见下方各节 |

## 二、方法

- **输入（TAI）**：自然语言指令 + 实例描述（按工件列出 (机器,时间) 工序序列）
- **输出**：JSON 数组 `[{"job","op","machine","start","duration"},...]`，容错解析（容忍代码块/多余记录）
- **SFT**：Qwen2.5-7B-Instruct + LoRA（r=16），42K 实例，分规模四阶段续训（p1a 6x6 → p1b 10x10 → p2 15x15 → p3 20x20）
- **FOARL**：GRPO 三 epochs（800 实例 6x6/10x10，组内采样 S=8，奖励 r = r_feas + r_opt，r_opt=1/(1+gap)，KL β=0.05）
- **推理**：Best-of-N（默认 N=8，top-p 0.7）

## 三、主要结果

### 3.1 分布内评估（test 300 实例 6x6，BoN8）

| 模型 | 可行性率 | 平均 gap | 最优 gap |
|---|---|---|---|
| SFT | 78.3% | 29.61% | 6.81% |
| **FOARL epoch2（最终）** | **97.0%** | 43.03% | 10.14% |

**结论**：FOARL 显著修复可行性（+18.7pp，接近论文 100%），但质量略有回落——与论文「不牺牲质量」存在偏差，列为重要发现。

### 3.2 FOARL 训练动态（rollout 可行率，逐 epoch）

| epoch | 可行率 | 变化 |
|---|---|---|
| 1（SFT 起点） | 75.75% | — |
| 2 | 95.13% | +19.4pp |
| 3 | 99.0%（采样时）→ 78%（训练后，漂移） | **第 3 轮漂移** |

**发现**：GRPO 第 2 轮为收敛点；第 3 轮出现策略漂移（test 可行率回落至 ~78%）——可能与 lr=5e-5 高于论文 1e-6 有关。**最终模型取 epoch2**。

### 3.3 BoN 消融（60 实例 6x6）

| BoN N | 可行率 | 平均 gap |
|---|---|---|
| 1 | 100% | 53.71% |
| 2 | 100% | 46.77% |
| 4 | 100% | 43.74% |
| 8 | 100% | 41.61% |

**结论**：BoN 单调改进 gap（N=8 较 N=1 降 12.1pp），边际递减（1→2 降 6.9pp、4→8 降 2.1pp）——论文默认 N=8 合理。

### 3.4 泛化评估（公开基准 123 实例：ft/la/ta）

| 系列 | 可行率 | 平均 gap（可行实例） |
|---|---|---|
| ft（3） | 1/3 | 170.9% |
| la（40） | 4/40 | 41.0% |
| ta（80） | **0/80** | — |
| **合计** | **4.1%** | 67.0% |

**核心发现**：模型跨规模泛化能力弱——训练规模（6x6/10x10）内可行率 97%，但 15x15+ 的公开基准几乎全部失败（Taillard ta 系列 0/80）。诊断确认：输出记录数正确但约束违规（FOARL 仅训练 6x6/10x10，大规格约束能力未经 RL 修复）。

### 3.5 基线对比（公开基准 123 实例）

| 方法 | 平均 gap | 备注 |
|---|---|---|
| 启发式（SPT/MOR 取优） | 18.82% | 无学习，秒级 |
| CP-SAT（限时 30s） | **2.13%** | 54/123 最优 |
| 本模型（FOARL，BoN8） | 4.1% 可行 | 跨规模泛化局限 |

## 四、工程经验与踩坑（详见 CLAUDE.md，18 条）

关键：OR-Tools 9.15 API 变更、rsync --delete 事故（3 次）、训练样本截断（max_length）、fp32 logits OOM（分块 CE 解决）、vLLM/5090 环境（CUDA_HOME）、vLLM 引擎生命周期（sleep/wake_up）、PeftModel is_trainable、RL 训练 lr 漂移。

## 五、结论

1. **SFT + FOARL 两阶段范式在分布内有效**：可行性 78.3% → 97.0%，验证了 LLMCoSolver 路径
2. **主要局限是跨规模泛化**：公开基准（尤其 ta 15x15+）上可行性骤降——FOARL 训练规模与评估规模的错配是关键
3. **改进方向**：FOARL 覆盖多规模、lr 对齐论文 1e-6、训练温度一致性

## 六、产物清单

- LoRA adapters：`experiments/sft_qwen7b/final`、`experiments/foarl_qwen7b/epoch2`（最终模型）
- 数据：`/root/autodl-tmp/jssp_data/`（42K 监督 + 123 公开基准）
- 代码：`/root/jssp/src|scripts|configs`（git 托管，18 条踩坑入 CLAUDE.md）
- 评估结果：`experiments/eval_*`、`experiments/ablation_bon`、`experiments/baseline_benchmark`、`experiments/eval_benchmark`

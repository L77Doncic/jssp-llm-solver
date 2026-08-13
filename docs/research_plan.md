# 研究计划与实现路径

> 对齐任务书「主要研究内容」的 5 个部分；本文档是 `CLAUDE.md`「实现路径」的展开版。进度更新时同步更新 CLAUDE.md「当前阶段」。

## 阶段总览

| 阶段 | 内容 | 对应研究内容 | 交付物 | 状态 |
|------|------|------------|--------|------|
| 0 | 结构搭建 + 问题定义 | ① | 目录、CLAUDE.md、docs | ✅ 完成 |
| 1 | 参考求解器与数据构建 | ③ | OR-Tools CP-SAT 封装；实例生成器；SFT 数据集 + splits | ✅ 完成（42K 实例） |
| 2 | 输入输出表示 | ② | TAI 文本模板；JSON 排程格式；解析器 + 验证器 + 测试 | ✅ 完成 |
| 3 | SFT | ④ | 四阶段分规模 LoRA 训练完成 | ✅ 完成 |
| 4 | FOARL | ④ | GRPO 3 epochs（可行率 75.8%→99.0%） | ✅ 完成 |
| 5 | 评估与泛化 | ⑤ | 分布内+公开基准+基线+消融 全套评估 | ✅ 完成 |
| 6 | 消融与产出 | — | 输入特征/输出结构/BoN/规模消融；论文或报告 | 视目标 |

## 环境（2026-08-06 定稿）

- **运行环境**：RTX 5090（32GB）×1，`ssh -p 26022 root@connect.westb.seetacloud.com`；Python 3.12.3 + PyTorch 2.8.0+cu128
- **LLM 本地部署，不使用任何 API**；代码目录 `/root/jssp`，数据目录 `/root/autodl-tmp/jssp_data`（50GB）
- **主模型：Qwen2.5-7B-Instruct（LoRA）** —— 论文同款配置，32GB 显存训练舒适（16-18GB）；升级备选 Qwen2.5-14B（QLoRA 7-8GB），对照备选 Llama-3.1-8B
- 依赖：ortools / transformers / peft / datasets / accelerate / bitsandbytes / vllm（推理与 rollout）

## 关键技术决策点（D1–D6）

- **D1 基座模型（已定）**：Qwen2.5-7B-Instruct + LoRA，本地部署（transformers + peft 训练，vLLM 推理）。论文已验证 7B 有效；32GB 显存 LoRA 训练余量充足（实测 16-18GB，QLoRA 仅 7-8GB）。升级路线 14B QLoRA。
- **D2 监督求解器**：OR-Tools **CP-SAT**（中规模近优）；小实例（≤10×10）通常直接最优，支撑严格 gap 评估。参考信号质量直接决定 SFT 上限。
- **D3 输入表示（TAI）**：实例文本（标准工序序列描述）+ 调度启发式特征（候选：机器负载、SPT 初始解、Giffler-Thompson 部分解）。**留作阶段 2 的对照实验**。
- **D4 输出格式**：JSON 数组逐工序 `{job, op, machine, start, duration}`；解析失败=不可行=低奖励。验证成本低，BoN 友好。
- **D5 奖励设计**：$r = r_{feas} + \lambda r_{opt}$；$r_{opt}$ 用 gap（对 CP-SAT 解）的负相关函数；$\lambda$ 进 `configs/foarl/` 配置。
- **D6 泛化协议**：训练规模 S，测试跨规模（±）与跨分布（随机 vs 结构化 vs 公开基准 ta/la/ft）；对比基线：SPT/MOR 等规则、CP-SAT（限时）、通用 LLM 零样本。

## 数据构建方案（阶段 1 细化）

1. `src/problem/generator.py`：随机实例生成器 —— 控制 (n, m)、加工时间分布（如 U[1,99]）、机器顺序随机（保证可行性）；产出统一 JSON + instance_id；
2. `src/data/`：raw 解析（OR-Library/Taillard 两种惯例，见 `jssp_definition.md` §3.3）；
3. `src/solver/ortools.py`：CP-SAT 封装，输入实例 → 排程（JSON），可配时间上限；产出最优/近优解；
4. 规模规划：6×6 / 10×10 / 15×15 / 20×20（训练）；ta01–ta80、la01–la40（公开测试）；SFT 初始 50K，验证流水线后再放量；
5. `data/splits/`：按 instance_id 出 train/val/test 清单，种子可复现。

## 训练方案（阶段 3–4）

- SFT：transformers + peft LoRA；文本模板：`指令 + 实例 TAI + 排程 JSON`；损失仅计算答案部分（模板区 mask）；
- FOARL：GRPO 自实现（参照 LLMCoSolver 论文描述）；采样 N 组 → 可行性/最优性奖励 → 组内归一化优势 + KL 惩罚更新 LoRA；
- 推理：BoN（默认 N=8，top-p 0.7），选最优可行解；
- 显存预估：7B + LoRA：训练 ≥24GB；推理 vLLM 或 4bit 量化可降。

## 评估协议（阶段 5）

| 维度 | 指标 |
|------|------|
| 可行性 | 可行解比例（含格式合法率） |
| 优化质量 | 平均/最优 gap%（对 CP-SAT 或已知最优），makespan 分布 |
| 求解时间 | 端到端生成时间（模型）vs 参考求解器（限时） |
| 泛化 | 跨规模、跨分布、公开基准（ta/la/ft）上的可行率与 gap |

对照实验（论文研究关注点 5）：TAI 特征有无、输出结构（JSON vs 自由文本）、BoN 大小、SFT 数据量、模型规模。

## 风险与应对

| 风险 | 应对 |
|------|------|
| JSSP 输出格式难学（长序列 JSON 出错率高） | 结构化格式 + 格式奖励 + 小实例起步；必要时模板化输出 |
| 监督信号质量不足（CP-SAT 限时解差） | 小实例取最优；公开基准已知最优做上限验证 |
| 7B 训练资源不足 | 先小模型跑通全流程，LoRA + 梯度检查点降显存 |
| 泛化差（跨规模掉点） | 多规模混合训练、课程式规模递进（可作 FOARL 扩展） |

## 里程碑

- M1：数据流水线跑通（生成→CP-SAT→JSON 数据集）＋ 验证器测试通过
- M2：SFT 模型可在 6×6/10×10 输出 >90% 可行解
- M3：FOARL 后可行率≈100%，gap 收敛
- M4：公开基准（ft/la/ta）评估 + 泛化实验完成

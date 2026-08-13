# 基于大语言模型的静态 JSSP 端到端求解研究

> 项目类型：科研探索（源自笔试题）
> 运行环境：RTX 5090 服务器（本地部署，无 API）
> 状态：**✅ 全部完成（2026-08-13 独立验收通过，可交付）**
> 最终报告：[experiments/summary/final_report.md](experiments/summary/final_report.md)

## 项目目标

构建一个面向**静态作业车间调度问题（JSSP）**的大语言模型端到端求解器：
以生产实例信息为输入，直接输出满足**工序顺序约束**与**机器独占约束**的可行排程，
在 **makespan** 等目标上取得较优性能，并具备跨规模泛化能力。

## 核心结果

| 指标 | 结果 |
|---|---|
| 训练数据 | 42K 实例（6x6/10x10/15x15/20x20），91.3% 最优监督信号 |
| SFT（Qwen2.5-7B + LoRA） | 四阶段分规模训练，全部完成 |
| FOARL（GRPO ×3 epochs） | rollout 可行率 75.75% → 95.13% → 99.0% |
| **分布内可行率（6x6）** | **SFT 78.3% → FOARL 97.0%** |
| 跨规模泛化 | 10x10 35.0% → 15x15 0%（模型主要局限） |
| 公开基准（ft/la/ta 123 实例） | 可行率 4.1% vs 启发式 gap 18.8% vs CP-SAT gap 2.1% |
| BoN 消融（N=1→8） | gap 53.7% → 41.6%（单调改进） |

## 技术方案

- **主模型**：Qwen2.5-7B-Instruct（本地部署，LoRA r=16）
- **训练范式**：SFT（四阶段分规模续训）→ FOARL（GRPO + 可行性/最优性联合奖励，KL β=0.05）—— 参照 LLMCoSolver（NeurIPS 2025）
- **监督信号**：OR-Tools CP-SAT 求解器（42K 实例）
- **推理**：vLLM 0.26 + Best-of-N 采样（N=8）

## 目录结构

```
jssp/                          # 代码目录（/root/jssp）
├── CLAUDE.md          # 项目维护手册（含 18 条踩坑经验）
├── README.md          # 本文件
├── docs/              # 问题定义、研究计划、论文解读、参考文献
├── src/               # 核心代码：problem / data / model / solver / training / evaluation / utils
├── configs/           # 实验配置：data / sft / foarl / eval
├── scripts/           # 运行入口（数据构建/SFT/FOARL/评估/消融/基线）
├── experiments/       # 实验产物：最终报告 + 各实验结果（summary/ 与结果文件入库）
└── tests/             # 单元测试（107 全绿）

数据：/root/autodl-tmp/jssp_data/（42K 监督数据 + 123 公开基准，数据盘不入库）
```

## 关键文档

- [最终报告](experiments/summary/final_report.md) — 任务书交付对照 + 全部实验数据
- [JSSP 问题定义](docs/jssp_definition.md)
- [研究计划](docs/research_plan.md)
- [LLMCoSolver 论文解读](docs/llmcosolver_notes.md)
- [参考文献](docs/references.md)

## 维护约定

所有开发操作遵循 `CLAUDE.md` 中的结构约定与流程规范；结构或经验有变化时同步更新 `CLAUDE.md`。

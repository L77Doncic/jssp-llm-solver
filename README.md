# 基于大语言模型的静态 JSSP 端到端求解研究

> 项目类型：科研探索（源自笔试题）
> 运行环境：RTX 5090 服务器（本地部署，无 API）
> 状态：**阶段 1 — 数据构建**（代码重写完成，流水线验证中）

## 项目目标

构建一个面向**静态作业车间调度问题（JSSP）**的大语言模型端到端求解器：
以生产实例信息为输入，直接输出满足**工序顺序约束**与**机器独占约束**的可行排程，
在 **makespan** 等目标上取得较优性能，并具备跨规模泛化能力。

## 技术方案

- **主模型**：Qwen2.5-7B-Instruct（本地部署，LoRA 微调；对照备选 Llama-3.1-8B）
- **训练范式**：SFT（监督微调）→ FOARL（可行性与最优性感知的强化学习，GRPO）—— 参照 LLMCoSolver（NeurIPS 2025）
- **监督信号**：OR-Tools CP-SAT 求解器
- **推理**：vLLM + Best-of-N 采样

## 目录结构

```
jssp/                          # 代码目录（/root/jssp）
├── CLAUDE.md          # 项目维护手册（AI 协作约定，每次开工先读）
├── docs/              # 问题定义、研究计划、论文解读、参考资料
├── src/               # 核心代码：problem / data / model / solver / training / evaluation / utils
├── configs/           # 实验配置：data / sft / foarl
├── scripts/           # 运行入口（一键流水线）
├── experiments/       # 实验产物（一实验一目录）
├── tests/             # 单元测试
└── notebooks/         # 探索性分析

数据：/root/autodl-tmp/jssp_data/{raw,instances,supervised,rl,splits}
```

## 关键文档

- [JSSP 问题定义](docs/jssp_definition.md) — 形式化定义、约束、复杂度、标准实例
- [研究计划与实现路径](docs/research_plan.md) — 分阶段路线图
- [LLMCoSolver 论文解读](docs/llmcosolver_notes.md) — 参考方法详解
- [参考文献与资源](docs/references.md)

## 维护约定

所有开发操作遵循 `CLAUDE.md` 中的结构约定与流程规范；结构或经验有变化时同步更新 `CLAUDE.md`。

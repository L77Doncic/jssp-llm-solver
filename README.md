# 基于大语言模型的静态作业车间调度（JSSP）端到端求解系统

> 以生产实例信息为输入、直接输出可行排程的 LLM 端到端求解器：Qwen2.5-7B + LoRA，
> 采用「SFT + FOARL（GRPO）」两阶段训练范式，单卡 RTX 5090 全本地完成
> 数据构建 → 训练 → 推理 → 评估全流程。

[![状态: 全部完成](https://img.shields.io/badge/状态-全部完成-brightgreen)](#状态)
[![规模: 6x6 - 20x20](https://img.shields.io/badge/规模-6x6_~_20x20-blue)](#核心结果)
[![基座: Qwen2.5-7B](https://img.shields.io/badge/基座-Qwen2.5--7B--Instruct-blue)](#技术方案)
[![方法: SFT + FOARL](https://img.shields.io/badge/方法-SFT_%2B_FOARL(GRPO)-orange)](#技术方案)
[![测试: 107 passed](https://img.shields.io/badge/测试-107_passed-green)](#测试与维护)
[![Release: v1](https://img.shields.io/badge/Release-v1-8A2BE2)](https://github.com/L77Doncic/jssp-llm-solver/releases/tag/v1)

## 状态

- **✅ 全部交付完成**（2026-08-13 独立验收通过）：任务书要求的 12 项交付全部达成，无降级
- **🔄 改进迭代中**（2026-08-14 起）：FOARL 快速验证、lr 对齐论文、输出结构化约束等实验排入路线图（详见 [CLAUDE.md 改进路线图](CLAUDE.md)）
- 服务器已关闭，本仓库为唯一事实源：代码 + 报告 + 模型权重 + 数据集均在此（GitHub + Release v1）

## 核心结果

### 分布内与跨规模（BoN=8）

| 规模 | 可行率 | 平均 gap | 说明 |
|---|---:|---:|---|
| **6×6（分布内）** | **97.0%** | 43.0% | SFT 78.3% → FOARL 97.0%，两阶段范式有效性验证 |
| 10×10（未见规模） | 35.0% | 50.8% | 跨规模泛化衰减 |
| 15×15 / 20×20 | ~0% | — | 主要局限（归因分析见报告 §6） |

### 消融实验

| 实验 | 结果 |
|---|---|
| Best-of-N（N=1→8，6×6） | gap 53.7% → **41.6%**（单调改进，边际递减，论文默认 N=8 得到验证） |
| **TAI 启发式特征（训练注入，负结果）** | 训练阶段注入 SPT 提示**破坏可行性**：78.3% → **8.3%**；机制 = 输出格式纪律崩溃（JSON 格式错误 43.3% vs 纯 SFT 8.8%） |
| TAI 对可行实例的质量增益 | gap 29.6% → **10.7%**（破坏可行性，但提升质量——训练/推理注入完整三组对照见报告 §5.6） |

### 公开基准泛化（ft/la/ta 共 123 实例，从未见过）

| 方法 | 可行率 | gap |
|---|---:|---:|
| 本系统（LLM + BoN=8） | 4.1% | — |
| CP-SAT（限时参考） | 100% | **2.1%** |
| 启发式基线（SPT 等） | 100% | 18.8% |

> 结论：分布内可行排程生成验证了 LLM 端到端求解 JSSP 的可行性；跨规模泛化与约束保持是当前主要瓶颈，报告 §6 给出了完整归因（错误指数累积 + FOARL 规模错配）。

## 技术方案

| 环节 | 方案 | 说明 |
|---|---|---|
| 输入表示 | TAI（Text-Attributed Instance） | 自然语言问题描述 + 启发式特征，JSON 排程输出（容错解析 + 约束校验） |
| 监督数据 | 42K 实例（6×6 / 10×10 / 15×15 / 20×20） | OR-Tools CP-SAT 求解，91.3% 达到最优 |
| 基座模型 | Qwen2.5-7B-Instruct + LoRA（r=16） | 本地部署，32GB 单卡可训练 |
| **SFT** | 四阶段分规模续训 | 每阶段独立 max_length 防截断；分块 CE + 分块 lm_head 突破显存瓶颈 |
| **FOARL** | GRPO × 3 epochs | 可行性 + 最优性联合奖励；rollout 可行率 75.8% → 99.0%；epoch3 策略漂移为科学发现（报告 §4.5） |
| 推理 | vLLM 0.26 + Best-of-N（N=8） | Blackwell/5090 环境适配完成 |
| 评估 | 可行性 / gap / 时间 / 泛化 / 基线 / 消融 | 5 组独立评估，结果全部落盘 |

## 快速恢复与资产（GitHub Release v1）

服务器关闭后，新环境一条命令恢复全部代码与资产（详细步骤见 [CLAUDE.md §0 新环境上手指南](CLAUDE.md)）：

```bash
git clone https://github.com/L77Doncic/jssp-llm-solver.git /root/jssp && cd /root/jssp
```

| 资产 | 内容 | 获取方式 |
|---|---|---|
| `weights.tgz` | SFT / FOARL LoRA 权重（573MB） | Release v1 下载 |
| `weights_hint.tgz` | TAI 训练注入对照权重 | Release v1 下载 |
| `data.tgz` | 42K 监督数据 + 划分 + 123 公开基准（13MB） | Release v1 下载 |
| `instances.tgz` | 42K 实例文件（4.9MB，可选） | Release v1 下载 |
| 基座 Qwen2.5-7B-Instruct（15GB） | — | hf-mirror 重下载（CLAUDE.md §0.1） |

## 报告与文档

| 文档 | 位置 |
|---|---|
| **技术报告（Markdown）** | [experiments/summary/technical_report.md](experiments/summary/technical_report.md) |
| **技术报告（PDF，10 页美化版）** | [experiments/summary/technical_report.pdf](experiments/summary/technical_report.pdf)（封面 / 三线表 / 思源字体，pandoc + LaTeX 构建） |
| 最终交付报告（任务书对照） | [experiments/summary/final_report.md](experiments/summary/final_report.md) |
| JSSP 问题定义 | [docs/jssp_definition.md](docs/jssp_definition.md) |
| 研究计划 | [docs/research_plan.md](docs/research_plan.md) |
| LLMCoSolver 论文解读 | [docs/llmcosolver_notes.md](docs/llmcosolver_notes.md) |
| 参考文献 | [docs/references.md](docs/references.md) |
| **项目维护手册（宪法，含 18 条踩坑经验）** | [CLAUDE.md](CLAUDE.md) |

## 目录结构

```
jssp/
├── CLAUDE.md          # 项目维护手册：新环境恢复 / 改进路线图 / 踩坑经验
├── README.md          # 本文件
├── docs/              # 问题定义、研究计划、论文解读、参考文献
├── src/               # 核心代码：problem / data / model / solver / training / evaluation / utils
├── configs/           # 实验配置（YAML）：data / sft / foarl
├── scripts/           # 运行入口：数据构建 / SFT / FOARL / 评估 / 消融 / 基线
├── experiments/       # 一实验一目录：config 冻结 + 日志 + 结果 + summary；final_report / technical_report
└── tests/             # 单元测试（107 全绿）
```

数据与权重不入库，落在数据盘 `/root/autodl-tmp/jssp_data/`（42K 监督数据 + 123 公开基准）。

## 测试与维护

```bash
python -m pytest tests/ -q    # 期望 107 passed
```

- **代码分层**：逻辑进 `src/<模块>/`，入口进 `scripts/`，超参一律进 `configs/` YAML
- **一实验一目录**：config 冻结存档 + 结果结构化落盘，随机种子显式配置
- **文档同步**：约定与经验变更必须同步更新 `CLAUDE.md`

## 论文参照与致谢

本项目的「SFT + FOARL」两阶段训练范式直接借鉴 LLMCoSolver：

> Jiang et al., *LLMCoSolver: Code-based Self-Correction and Contextual Reinforcement Learning for Combinatorial Optimization*,
> NeurIPS 2025（[arXiv:2509.16865](https://arxiv.org/abs/2509.16865) · [代码](https://github.com/Summer142857/LLMCoSolver)）

基线对照：OR-Tools CP-SAT（[Google OR-Tools](https://developers.google.com/optimization)）与经典启发式（SPT/MOR 等）；公开基准来自 [JSPLIB](https://github.com/tamy0612/JSPLIB)（ft/la/ta 共 123 实例）。

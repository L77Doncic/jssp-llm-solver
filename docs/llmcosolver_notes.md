# LLMCoSolver 论文解读（核心参考方法）

> Jiang X, Wu Y, Li M, Cao Z, Zhang Y. **Large Language Models as End-to-end Combinatorial Optimization Solvers**[C]. NeurIPS 2025.
> arXiv: [2509.16865](https://arxiv.org/abs/2509.16865) ｜ 代码: [github.com/Summer142857/LLMCoSolver](https://github.com/Summer142857/LLMCoSolver)

## 1. 核心思想

让 LLM 直接充当组合优化（CO）**端到端求解器**：输入问题实例的文本描述，直接输出解（文本），**无需**代码生成、无需调用外部求解器、无需针对每个问题改架构。解决的问题模型被统一为「条件文本生成」：`P(解 | 实例描述)`。

## 2. 两阶段训练策略（本项目的直接参照）

### 阶段一：监督微调 SFT
- 把 CO 当作 **next-token prediction** 任务；
- 监督信号 = 专业求解器的解（TSP 用 LKH3，其他问题用 Gurobi/OR-Tools 等）；
- **Text-Attributed Instance (TAI)**：在纯文本描述基础上附加廉价启发式特征（如 TSP 的 top-k 最近邻），引导探索；
- **LoRA** 参数高效微调，基座 **Qwen2.5-7B**；
- 规模：每问题 ~**500K** 条实例（SFT）。

### 阶段二：FOARL（Feasibility-and-Optimality-Aware RL）
- **动机（关键经验）**：SFT 后的模型会「贪婪违规」——为了优化目标值而违反约束（如 CVRP 超容量、OP 超距离）。生成"看起来合理但不可行"的解。
- **联合奖励** $r = r_{feas} + \lambda \cdot r_{opt}$：
  - $r_{feas}$：可行性奖励 —— 输出格式合法 + 全部约束满足；
  - $r_{opt}$：最优性奖励 —— 与最优性 gap 成反比（gap 越小奖励越高）。
- **GRPO**（Group Relative Policy Optimization）更新：组内归一化优势 + KL 约束，**无需 critic 网络**；
- 规模：每问题 ≤ **3,200** 条实例（FOARL）。

### 推理：Best-of-N (BoN)
生成 N 个候选解（默认 **N=8**，top-p=0.7），选**最优的可行解**。N 是质量-延迟旋钮：N ∈ {1,2,4,8,16,32,64} 平滑权衡。

## 3. 覆盖问题与结果

覆盖 7 个 NP-hard CO 问题：**TSP、OP、CVRP、MIS、MVC、PFSP、JSSP**（单一框架，无问题定制）。

- **可行性**：SFT+RL+BoN 后在 **6/7 问题达到 100% 可行**；
- **最优性**：平均 gap **1.03%–8.20%**（7B 模型）；
- 超越 GPT-4o、DeepSeek-R1、GPT-o1 等通用 LLM 及常见领域启发式；
- 例：CVRP 平均 gap 4.53%，显著优于同时间预算下 Gurobi 的 35.09%（Gurobi 在紧凑 MIP 类问题如 MIS/MVC 上仍更强）；
- 消融：FOARL 的增益主要来自**恢复不可行实例的可行性**且不牺牲解质量；去掉 TAI 启发式特征会显著掉点。

## 4. 对 JSSP 的具体启示（本项目落点）

LLMCoSolver 已验证该范式在 JSSP 上成立，但 JSSP 在其论文中属于「覆盖验证」，并非深耕对象。本项目的深化空间：

1. **JSSP 专属输入编码**：TAI 中注入调度领域特征（如机器负载、关键路径、SPT/MOR 规则解）——启发式特征对 TAI 增益已被论文证实，JSSP 上值得专门设计；
2. **输出格式**：JSSP 排程的自然语言表示（如逐工序 JSON `{job, op, machine, start, duration}`）需**可解析 + 可验证**；格式错误是可行性奖励的惩罚项；
3. **监督信号质量**：JSSP 用 CP-SAT 生成近优解（小实例可得最优解，用于严格 gap 评估）；
4. **泛化**：跨规模（如 6×6 训练 → 30×20 测试）与跨分布（随机 vs 结构化）是论文遗留的开放问题，正是本项目研究关注点「泛化能力」所在；
5. **BoN 策略**：JSSP 可行性验证成本低（O(nm log nm)），BoN 采样天然适合。

## 5. 论文中 JSSP 相关的可复用结论

- JSSP 实例文本化的建议形式：按工件列出工序序列（机器+时间），与标准数据格式一致，LLM 易理解；
- 可行排程的表示建议用结构化格式（如 JSON 数组），比自由文本更利于解析与奖励计算；
- 训练数据规模参考：SFT 每问题 500K、FOARL 3.2K —— 本项目可从小规模（如 50K SFT）起步验证流程。

## 6. 相关延伸工作（视野）

- **Starjob**（arXiv 2025）：首个大规模 JSSP 监督数据集（120K–130K 实例），微调 Llama-3.1-8B（4bit+LoRA），超越 PDR 与神经方法 L2D：DMU 上平均提升 11.28%，Taillard 上 3.29%；
- **IEEE ICBAIE 2025**：SPT 规则构建 20K/120K 数据微调 Llama3.1-8B / Qwen2.5-7B，可行率最高 68.8% —— 佐证「零样本 LLM 直接生成 JSSP 排程几乎不可行，必须微调」；
- **JSSP-4W + 结构化推理（Hu & Zhao, J. Manufacturing Systems 2026）**：六步结构化推理 + 约束感知损失，可行率 97%，gap 5.14%–19.27%；
- **DScheLLM**（arXiv 2026）：双系统 LLM 动态调度（本项目为静态，不直接相关）。

这些工作共同确认：**微调 + 可行性感知机制**是 LLM 端到端求解 JSSP 的关键配方。

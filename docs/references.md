# 参考文献与资源

## 核心文献（项目直接依据）

1. **Jiang X, Wu Y, Li M, Cao Z, Zhang Y.** Large Language Models as End-to-end Combinatorial Optimization Solvers[C]. NeurIPS 2025.
   - arXiv: https://arxiv.org/abs/2509.16865（2509.16865）
   - OpenReview/Proceedings: https://proceedings.neurips.cc/paper_files/paper/2025/hash/f0c6c93e1dbff84daf6082fef7e1a094-Abstract-Conference.html
   - 开源代码（本项目的直接参考实现）: https://github.com/Summer142857/LLMCoSolver

## JSSP 经典文献

2. **Garey M R, Johnson D S, Sethi R.** The Complexity of Flowshop and Jobshop Scheduling[J]. Mathematics of Operations Research, 1976. — JSSP 强 NP-hard 证明
3. **Fisher H, Thompson G L.** Probabilistic learning combinations of local job-shop scheduling rules[J]. Industrial Scheduling, 1963. — ft06/ft10/ft20 实例出处（ft10 最优 930）
4. **Lawrence S.** Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques[R]. 1984. — la01–la40 实例出处
5. **Taillard E.** Benchmarks for basic scheduling problems[J]. European Journal of Operational Research, 1993. — ta01–ta80 实例出处

## LLM / 学习求解 JSSP 相关（2025–2026 进展）

6. **Starjob: Dataset for LLM-Driven Job Shop Scheduling**（Abgaryan & Cazenave, arXiv 2025 / ICLR 投稿）— 首个大规模 JSSP 监督数据集（120K–130K 实例），微调 Llama-3.1-8B 超越 PDR 与 L2D
   - https://www.semanticscholar.org/paper/055a073ec1399aaba42dcf1b880e4cc57ccaf923
   - OpenReview: https://openreview.net/forum?id=t0fU6t3Skw
7. **Leveraging Large Language Models for End-to-End Job Shop Scheduling Scheme Generation**（IEEE ICBAIE 2025）— SPT 规则构建 20K/120K 数据微调 Llama3.1-8B / Qwen2.5-7B，可行率最高 68.8%，证明零样本不可行
   - https://ieeexplore.ieee.org/document/11326598
8. **Hu & Zhao.** 结构化推理 + 约束感知损失的 JSSP-4W 双增强框架（Journal of Manufacturing Systems, 2026）— 可行率 97%，gap 5.14%–19.27%
   - https://www.sciencedirect.com/author/57198597221/weidong-zhao
9. **DScheLLM: Enabling Dynamic Scheduling through a Fine-Tuned Dual-System LLM**（arXiv 2026）— 动态调度，与本项目（静态）关系较弱，留作扩展参考
   - https://www.semanticscholar.org/paper/b629e5b66c506a37dc2beda06ac69a6ad362af6c

## 标准数据集与工具

- **OR-Library JSSP 数据集**（ft/la/orb 系列）: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html
- **Taillard 基准**: http://jobshop.jjv.nl/ 或 Taillard 个人主页（ta01–ta80 与已知最优）
- **OR-Tools CP-SAT**（监督信号生成与对比求解器）: https://developers.google.com/optimization
- **Giffler-Thompson 算法**：优先分配规则生成主动调度的经典算法（基线/特征用）
- **Qwen2.5 系列模型**: https://huggingface.co/Qwen（7B 为默认基座，1.5B/3B 为快速验证选项）

## 备注

- 引用 1（LLMCoSolver）是本项目范式源头；引用 6–8 是 2025–2026 年把 LLM 端到端范式落地到 JSSP 的直接相关工作，构成「本项目与既有工作的差异化定位」依据（详见 `llmcosolver_notes.md` §4）。
- 资料随调研持续补充，新增条目请标注日期。

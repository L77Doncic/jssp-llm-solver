# CLAUDE.md — 项目维护手册（服务器版）

本文件是本项目的「宪法」：每次开始工作前先读本文件；开发过程中若结构约定或经验发生变化，**必须同步更新本文件**。

> **⚠️ 项目已迁移/或即将迁移到新服务器（2026-08-14）**：若本文件出现在新环境，先读「§0 新环境上手指南」——它包含环境重建、资产迁移与待办实验的完整说明。

## 项目概述

**目标**：构建一个面向**静态作业车间调度问题（JSSP）**的大语言模型端到端求解器 —— 模型以生产实例信息为输入，直接输出满足约束的可行排程，在 makespan 上取得较优性能并具备泛化能力。

**状态**：✅ **全部交付完成**（2026-08-13 独立验收通过）。当前处于「改进迭代」阶段，待办见「§改进路线图」。

**来源**：笔试任务书（`Vibe Coding 笔试.docx` 为原始需求文档，勿删）。
**核心参考**：LLMCoSolver（Jiang et al., NeurIPS 2025, arXiv:2509.16865；代码: github.com/Summer142857/LLMCoSolver）—— 本项目直接借鉴其 **SFT + FOARL 两阶段训练**范式。

---

## §0 新环境上手指南（迁移必读）

新服务器拿到项目后，按此顺序 30 分钟内可恢复完整工作环境。

### 0.1 资产清单（什么必须迁移，什么可重建）

| 资产 | 位置（旧服务器） | 大小 | 迁移策略 |
|---|---|---|---|
| **代码+文档+配置** | git 仓库（GitHub: L77Doncic/jssp-llm-solver） | ~400KB | ✅ `git clone` 即可 |
| 最终报告/PDF | experiments/summary/ | ~250KB | ✅ 在 GitHub |
| **LoRA 模型权重** | experiments/{sft_qwen7b/final, foarl_qwen7b/epoch1-3} | 573MB（打包） | ✅ **GitHub Release v1 下载**（见下） |
| 42K 监督数据 + 划分 + 123 基准 | /root/autodl-tmp/jssp_data/ | 13MB（打包） | ✅ **GitHub Release v1 下载** |
| 42K 实例文件 | /root/autodl-tmp/jssp_data/instances/ | 4.9MB（打包） | ✅ Release v1（可选，可重建） |
| 模型基座 Qwen2.5-7B | /root/autodl-tmp/jssp_data/models/ | 15GB | ✅ 可重下载（见 0.4） |
| 实验产物（results.json 等） | experiments/*/results.json | 小 | ✅ 已在 GitHub（gitignore 白名单） |

**资产下载（GitHub Release v1，任何新环境一条命令恢复）**：
```bash
# Release 页面: https://github.com/L77Doncic/jssp-llm-solver/releases/tag/v1
RELEASE=https://github.com/L77Doncic/jssp-llm-solver/releases/download/v1

git clone https://github.com/L77Doncic/jssp-llm-solver.git /root/jssp
cd /root/jssp

# 1) 权重（LoRA adapters）→ experiments/
mkdir -p experiments && curl -L -o /tmp/weights.tgz $RELEASE/weights.tgz
tar xzf /tmp/weights.tgz -C experiments/   # 得到 sft_qwen7b/final、foarl_qwen7b/epoch{1,2,3}

# 2) 数据 → 数据盘
mkdir -p /root/autodl-tmp/jssp_data && curl -L -o /tmp/data.tgz $RELEASE/data.tgz
tar xzf /tmp/data.tgz -C /root/autodl-tmp/jssp_data/   # supervised/ + splits/ + raw/

# 3) 实例（可选）→ 数据盘
curl -L -o /tmp/instances.tgz $RELEASE/instances.tgz
tar xzf /tmp/instances.tgz -C /root/autodl-tmp/jssp_data/

# 4) 模型基座（15GB，国内走 hf-mirror）
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct')"

# 5) 验证
python -m pytest tests/ -q   # 期望 107 passed
```

### 0.2 环境搭建（conda + 依赖）

```bash
# 1. conda 环境（Python 3.12）
conda create -n jssp python=3.12 -y && conda activate jssp

# 2. 核心依赖（torch 装 GPU 版，见第 3 步）
pip install ortools pyyaml pytest numpy \
            transformers peft datasets accelerate bitsandbytes

# 3. PyTorch + CUDA（按新机 CUDA 驱动版本选；5090/Blackwell 需 CUDA >= 12.9）
#    若驱动 13.x: pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
#    版本纪律：torch 与 torchaudio 必须同版本配对（踩坑 6）
pip install "torch==2.11.0+cu130" "torchaudio==2.11.0+cu130" --index-url https://download.pytorch.org/whl/cu130

# 4. vLLM（可选，评估/FOARL 加速；需 flashinfer）
pip install vllm flashinfer-python ninja

# 5. 验证
python -c "import torch, transformers, peft, ortools; print(torch.__version__, torch.cuda.is_available())"
```

### 0.3 vLLM 环境变量（Blackwell/5090 必配，踩坑 16）

```bash
export PATH=/root/miniconda3/bin:$PATH
export CUDA_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
# 使用 vllm 前必 export；建议写 ~/.bashrc
```

### 0.4 数据与模型重建（若未迁移）

```bash
# 模型基座（hf-mirror 国内镜像，必须禁用 xet，踩坑 9）
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/root/autodl-tmp/jssp_data/models/Qwen2.5-7B-Instruct')"

# 42K 监督数据重建（CP-SAT 求解，约 47 小时，建议后台跑）
python scripts/build_data.py -c configs/data/pipeline_full.yaml --out /root/autodl-tmp/jssp_data

# 123 公开基准（ft/la/ta）
# 来源：GitHub tamy0612/JSPLIB 的 instances/ 目录；已知最优: thomasWeise/jsspInstancesAndResults
# 产物 benchmark_manifest.json 格式见 data/raw/（可从旧服务器迁移此 1MB 目录最省事）
```

### 0.5 模型权重重建（仅当 Release 下载不可用时，需 ~30 小时）

| 权重 | 重建命令 | 耗时 |
|---|---|---|
| SFT（sft_qwen7b/final） | `bash scripts/run_sft_all.sh` | ~29h（四阶段） |
| FOARL（foarl_qwen7b/epoch2） | `python scripts/train_foarl.py -c configs/foarl/foarl.yaml --start-epoch 2`（起点=SFT） | ~10h |

**优先从 GitHub Release v1 下载**（§0.1 命令），重建是兜底方案。

### 0.6 测试验证

```bash
cd /root/jssp && python -m pytest tests/ -q   # 期望 107 passed
```

---

## 改进路线图（当前阶段：交付完成后的迭代）

> 按优先级排序。每个实验的结果记入 experiments/<exp>/summary.md 并更新本文件。

### 1. FOARL 快速验证实验（15×15 可行率修复验证，优先级最高）
- **动机**：报告 §6.2 归因②（FOARL 规模错配导致 15×15 可行率 0%）；论文（Table 2 large 档 16.25% gap/100% 可行）证明 FOARL 覆盖规模即可修复
- **设计**：100 个 15×15 实例 × 1 epoch FOARL（数据：从 supervised 数据集抽 15x15，参考 = CP-SAT 解）
  - 配置参考：configs/foarl/foarl.yaml，改 `scales: [15x15]`、`max_instances: 100`、epochs=1，输出目录 experiments/foarl_15x15_quick
- **预期**：15×15 可行率 0% → 显著回升（若成立则验证归因②；不成立则归因需修正）
- **耗时**：4-6h（15×15 rollout ~9500 tokens/采样）
- **评估**：scripts/evaluate_schedule.py（adapter 指向新权重，60 实例 15x15）

### 2. TAI 训练注入对照实验（✅ 完成 2026-08-14）
- 训练完成：experiments/sft_qwen7b_hint（6×6 + SPT 提示，与 p1a 同配置，5h）
- 评估完成（60 实例 × BoN8）：**重要负结果**——SPT 提示训练注入破坏可行性（78.3% → 8.3%，训练阶段分布偏移），但对可行实例质量有增益（gap 29.61% → 10.70%）；完整三组对照（训练注入×推理注入）见报告 §5.6
- hint 权重已上传 GitHub Release v1（weights_hint.tgz）
- 若需重跑：`python scripts/train_sft.py -c configs/sft/sft_p1a_hint.yaml --scale-filter 6x6`（~9h）

### 3. lr 对齐论文（1e-6）重训 FOARL
- 动机：epoch3 漂移归因于 lr=5e-5 过高（论文 1e-6）
- 设计：configs/foarl/foarl.yaml 改 learning_rate: 1e-6，重跑 3 epochs，对比漂移是否消失

### 4. 多规模 FOARL（若时间充足）
- FOARL 数据覆盖 6×6~20×20（论文口径），预期修复全部规模的约束违规
- 成本高（40-60h），先看实验 1 结果再决定

### 5. 输出结构化约束（受限解码）
- 机制性修复错误指数累积（报告 §6.2 归因①）：用语法约束采样保证 JSON 结构正确
- 参考：outlines / guidance / vLLM guided decoding

---

## 论文 JSSP 结果对照（改进目标参照）

| 规模档（100 实例/档） | 论文 gap | 论文可行率 | 我们（同口径待测） |
|---|---|---|---|
| small 5×5–10×10 | 2.86% | 100% | 6×6 97% 可行 / gap 43%（BoN8） |
| medium 10×10–15×15 | 9.56% | 100% | 10×10 35% 可行 |
| large 15×15–20×20 | 16.25% | 100% | 15×15/20×20 0% 可行 |

论文要点：SFT 数据 500K/问题、规模 U[5,20] 随机（**训练覆盖大规格**）、FOARL ≤3200 实例/问题、context 20000 tokens、监督 CP-SAT 300s 时限。Taillard 泛化（N=8）：15×15 14.22% / 20×15 21.16% / 20×20 23.30%。

## 运行环境（本地部署，无任何 API）（原服务器参考；新环境按 §0.2 重建）

| 项 | 值 |
|---|---|
| GPU 服务器 | RTX 5090（32GB）×1，`ssh -p 26022 root@connect.westb.seetacloud.com` |
| 系统环境 | Python 3.12.3（miniconda3 base），PyTorch 2.8.0+cu128，CUDA 13.0 驱动 |
| 关键依赖 | ortools 9.15、transformers 5.14、peft 0.20、datasets、accelerate、bitsandbytes、vllm（安装中） |
| 版本纪律 | **torch 2.8.0+cu128 是锚点**：配套 torchaudio 必须同为 2.8.0（pip 曾误装 2.11 导致 libcudart.so.13 缺失）；任何 pip 操作后验证 `torch.__version__` 不被升级 |
| 代码目录 | `/root/jssp`（本文件所在目录；git 仓库，服务器为唯一事实源） |
| 数据目录 | `/root/autodl-tmp/jssp_data`（50GB 数据盘；根分区仅 30GB，**大文件一律放数据盘**） |
| 模型下载 | HuggingFace 国内镜像：环境变量 `HF_ENDPOINT=https://hf-mirror.com` |
| SSH 凭据 | **不接受公钥**，必须 `sshpass -e ssh -p 26022 -o StrictHostKeyChecking=no root@connect.westb.seetacloud.com`（密码在 `SSHPASS` 环境变量，由 `/root/.jssp_ssh_env` 注入，**严禁写入任何项目文件**）；seetacloud 网关间歇性断开连接，命令需加 `-o ConnectTimeout=25 -o ServerAliveInterval=10` 并重试 |

**模型选型决策（2026-08-06）**：主模型 **Qwen2.5-7B-Instruct**（LLMCoSolver 论文同款配置，JSSP 上已验证 100% 可行率、gap 1.03%–8.20%；32GB 卡上 LoRA 训练舒适）；升级备选 Qwen2.5-14B（QLoRA）；对照备选 Llama-3.1-8B（Starjob 验证模型）。

## 当前阶段（2026-08-14 更新：交付完成，进入改进迭代；历史过程记录保留如下）

- **阶段 0/1/2** ✅ 完成（结构、42K 数据集、TAI+JSON 表示，107 测试全绿）
- **阶段 3：SFT —— 发现并修复训练配置错误（进行中）**
  - ❌ **首轮训练无效（2026-08-08）**：`sft.yaml` 的 `max_length: 1024` 过短，6x6 完整样本 1611 tokens、10x10 ~3500、15x15/20x20 更长 → **全部训练样本被截断** → 模型只学会输出 JSON 前半段（生成 2317 字符后停，无 `]`），评估可行性率 0.0%
  - ✅ 诊断方法：单实例生成 → 检查 token 长度 vs max_length → 确认 100% 截断
  - 🔄 **修复方案：分规模续训同一 LoRA**（`train_sft.py` 已支持 `--scale-filter` 与 `--adapter`）
    - 验证档：`sft_verify.yaml`（6x6 全量 + max_length 2048 + 60 步，约 15 分钟）→ 生成验证完整 JSON + 约束 ✅
    - ⚠️ **2026-08-09 发现 OOM**：p1（seq 4096, batch 4）首步即 `torch.OutOfMemoryError`——transformers 5.x `ForCausalLMLoss` 执行 `logits.float()`，在 (B,S,152K 词表) 全量张量上做 fp32 拷贝，内存随 batch×seq 线性涨；p2/p3（seq 8192/16384）即便 batch 1 也会被 fp32 logits 打爆
    - ✅ **OOM 修复（2026-08-09，已重启训练）**：① `src/training/loss.py` 新增分块 CE（`training.use_chunked_loss: true`）——fp32 峰值从 B*S*vocab 降为 chunk_size*vocab（默认 2048 ≈1.24GiB），与默认 `ForCausalLMLoss` 数值等价（测试 diff ~1e-6）；② 批次下调（有效 batch 不变）；③ `run_sft_all.sh` 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    - ✅ **续训 adapter 修复**：`PeftModel.from_pretrained(..., is_trainable=True)`（默认 `is_trainable=False` 全参数冻结 → trainable=0 白跑）
    - ⚠️ **样本长度实测（2026-08-09，防呆核心数据）**：完整样本（prompt+答案+eos）token 数按规模实测：**6x6 最大 1611 / 10x10 最大 4176（P99 4174）/ 15x15 最大 9518（P99 9511）/ 20x20 最大 17112（P99 17109）**。此前估算严重偏低（10x10 估 3500 实 4176），导致初版配置（4096/8192/16384）除 6x6 外**全部截断**，训练中途发现并停止
    - ✅ **防呆预检（train_sft.py 内置）**：训练启动前按规模抽样 500 条统计完整样本长度，超限即中止并报错（不再静默截断）——2026-08-08 max_length=1024 静默截断事故的工程防线
    - ⚠️ **显存实测边界（2026-08-10 修正）**：sdpa 已默认生效（transformers 5.14 函数级 dispatch）；此前「seq 18432 单样本 OOM（差 1.16GB）→ max_length 取 17408」是**只测前向**的结论。**完整训练（fwd+bwd）在 seq≈17137 时即便 use_cache=False + 流式 CE 仍 OOM**——backward 需全序列 logits.grad（~4.93 GiB）叠加在前向 22.8 GiB 上。真正解法是**分块 lm_head（连 logits 都不落盘）**：fwd+bwd 峰值 26.9 GiB / 余 2.9 GiB（chunk 512），20x20 全量样本可训
    - ✅ **p1a/p1b/p2 完成（2026-08-10 确认）**：p1a（990 步）→ p1b（~990 步）→ p2（990 步，train_loss 0.0585，`sft_qwen7b/final` = p2 产物）；p3 首步 OOM（2026-08-10）→ 修复后重启
    - 🔄 **四阶段训练（p1a/p1b/p2 已完成，p3 运行中）**：`run_sft_all.sh` 全链 + 断点续训 `run_sft_p3.sh`（p3 独立跑，成功才写 `ALL_SFT_DONE`）
      - p1a：`sft_p1a.yaml`（6x6，max_length 4096，batch 2×16，2 epochs，~9h）✅
      - p1b：`sft_p1b.yaml`（10x10，max_length 5120，batch 1×32，2 epochs，~9h）✅
      - p2：`sft_p2.yaml`（15x15，max_length 10240，batch 1×8，2 epochs，~8h）✅（990 步全完成）
      - p3：`sft_p3.yaml`（20x20，max_length 17408，batch 1×2，1 epoch，~3h）🔄 802 步，13.3s/步
      - 全部完成标记：`experiments/sft_p3_run.log` 含 `ALL_SFT_DONE`
  - 训练完成后 LoRA 在 `experiments/sft_qwen7b/final`
- **阶段 4：FOARL（2026-08-11，进行中）**
  - ✅ 实现完成：`scripts/train_foarl.py` + `src/training/foarl.py`（GRPO：vLLM 采样 → 奖励 r_feas+λ·r_opt → 组内优势 → ref/θ 顺序 logprob → LoRA 更新 + KL）
  - ✅ 冒烟测试通过（4 实例 × 2 采样 × 1 epoch，3.5 分钟全链路）
  - 🔄 正式训练运行中：`foarl.yaml`（800 实例 6x6/10x10 × 8 采样 × 3 epochs，日志 `experiments/foarl_run.log`）
  - 已知坑：训练阶段 OOM → `model.gradient_checkpointing_enable()`（rollout 长文本 ~3000 tokens）
- **阶段 5 备料**：评估脚本已备（批量 transformers 推理版，left-padding，batch 4 ≈124 tok/s）；vLLM 7B+LoRA 已验证可用（CUDA_HOME 方案）
- 后续阶段（自动推进中）：评估 → 泛化 → 基线 → 消融 → 报告

### 交付清单（对照任务书，全部必须完成，严禁降级 —— 2026-08-09 用户明确要求）

**✅ 已完成**
1. 标准化输入表示与输出解析方案（TAI + JSON + 容错解析 + validator）
2. 多规模训练与测试数据集（42K，4 规模，91.3% 最优 + splits）

**🔄 进行中**
3. SFT 四阶段训练（p1a→p1b→p2→p3，`run_sft_all.sh`，完成标记 `ALL_SFT_DONE`）

**⬜ 待执行（按序，全部必做）**
4. vLLM 加载验证（训练完成后第一步）：vLLM 0.26 + Qwen2.5-7B + LoRA；不兼容则 FOARL 用 transformers generate（慢但可用）
5. 推理求解封装：模型 + BoN 采样的统一求解入口（原型系统可演示部分）
6. 完整评估：可行性率 / gap / 求解时间（BoN 脚本已备并修复 max_new_tokens 截断隐患）
7. **FOARL（研究内容④，两阶段范式必须完整）**：GRPO 循环（奖励函数 reward.py 已就绪）+ rollout + 约束违规修复验证
8. 公开基准数据：ft06/10/20、la01-40、ta01-80 下载（parsers.py 已就绪）→ 泛化评估输入
9. 泛化实验：跨规模（训练规模 vs 未见规模）、跨分布、公开基准上的可行率与 gap
10. 基线对比：启发式（SPT/LPT/MOR/MWR）、CP-SAT（限时）、通用 LLM 零样本
11. 消融实验：TAI 特征有无、输出结构、BoN 大小（N∈{1,2,4,8,16}）、模型规模
12. 最终交付物：总结报告/演示（对照任务书五、预期成果逐项验收）

执行纪律：每步完成后在 `experiments/<exp>/summary.md` 记录结果；所有实验随机种子显式配置；无人值守监督脚本按此清单驱动。

## 踩坑与经验（2026-08-06 记录）

1. **OR-Tools 9.15 API 变更**：`var.Value()` 已移除，必须用 `solver.Value(var)`；`solver.ObjectiveValue()` / `solver.BestObjectiveBound()` 不受影响。升级 ortools 前检查 API 兼容。
2. **rsync 同步纪律**：单文件 rsync 会把文件直接放到目标根目录（`rsync a.py dest/` → `dest/a.py`，丢失相对路径）——**一律整目录同步**（`rsync src/ dest/src/`），且改动后先删服务器 `__pycache__` 再跑测试。
3. **导入约定**：`src/` 各包以顶层包方式挂载（conftest/scripts 注入 sys.path），跨包导入用绝对路径（`from problem...` / `from solver...`），包内用相对导入（`from .`）；**禁止跨包相对导入**（`from ..`）。
4. **Giffler-Thompson 最左调度**：机器排列 → 开始时间必须用 DAG 拓扑排序（工件顺序边 + 机器顺序边），朴素贪心（单调 job_ready）在排列与工件顺序矛盾时会出错；排列含环时报错而非静默出错。
5. **数据质量**：6x6/10x10 的 CP-SAT 基本秒出最优（监督信号完美）；15x15/20x20 需 15-30s 且常为近优——大规模实例的监督信号 gap 需在 SFT 前统计上报（见 `experiments/data_build_full/build_stats.json`）。
6. **CUDA 运行时匹配**：本机 torch 2.8.0+cu128 自带 libcudart.so.12；pip 若把 torchaudio 升级到 2.11（cu13）会因缺 libcudart.so.13 直接崩——装任何依赖后必须验证 torch/torchaudio 版本配对。
7. **Python 输出缓冲**：nohup 重定向日志时 stdout 是块缓冲，长任务日志可能长时间为空——调试时用 `python -u` 或 `PYTHONUNBUFFERED=1`。
8. **rsync 纪律（血泪教训，已犯三次）**：**多源 rsync + `--delete` 会把目标根目录中不在源列表的一切删光**（2026-08-06/07 三次误删服务器项目布局：①多源+--delete 展开根目录并删文件；②单文件多源放错位置；③`--exclude 'data'` 误伤 `src/data/`）。**服务器同步铁律**：①每次同步前先在脑子里走一遍「源路径/目标路径/排除规则」三个字段；②**永远单源**：`rsync -av --delete /root/JSSP/ dest:/root/jssp/`（源=整个项目根时 --delete 才安全），或**纯增量**（`rsync -av src/ dest:/root/jssp/src/`，不加 --delete）；③绝不用「多个文件/目录 + --delete」的组合；④排除规则用前缀锚定（`--exclude '/data'` 只排除根级 data，不误伤 `src/data/`）。
9. **huggingface_hub 1.x**：`huggingface-cli` 已弃用；国内镜像下载必须 `export HF_ENDPOINT=https://hf-mirror.com` 且 **`HF_HUB_DISABLE_XET=1`**（否则走 xet 协议 401 失败）。用 Python API：`snapshot_download(repo_id=..., local_dir=...)`。
10. **transformers 5.14 API 变更**（写训练/推理代码前先对照）：`from_pretrained` 的 `torch_dtype` 改名 `dtype`（且只接受 torch dtype 对象，不接受字符串）；`TrainingArguments.warmup_ratio` → `warmup_steps`、`logging_dir` 弃用；**`Trainer.__init__` 不再接受 `tokenizer` 参数**（data_collator 自带）；`huggingface-cli` 弃用。已在 `train_sft.py` 中适配。
11. **长序列 SFT 的 fp32 logits OOM（2026-08-09）**：transformers 5.x `ForCausalLMLoss` 先 `logits.float()` 再算 CE，fp32 张量 = B×S×vocab(152K)。seq 4096/batch 4 首步即炸（需 9.28GiB 仅剩 7.01GiB）；seq 16384 即便 batch 1 也要 ~19.9GiB。**解法**：`src/training/loss.py` 分块 CE——把序列切成 chunk 逐段 `sum` 计算再按有效 token 归一，fp32 峰值恒为 chunk_size×vocab（≈1.24GiB），数值与默认等价；配置 `training.use_chunked_loss: true`（train_sft.py 里 `model.loss_function` 直接替换）。
12. **`PeftModel.from_pretrained` 默认 `is_trainable=False`（推理模式，全参冻结）**：`--adapter` 续训时若漏传 `is_trainable=True`，`print_trainable_parameters()` 显示 0，训练白跑。2026-08-09 已在 train_sft.py 修复并显式注释。
13. **bash `set -e` + `A && B` 陷阱**：`set -e` 下 `cmd && echo ok` 中 cmd 失败不会退出（cmd 在 && 列表中非最后命令，豁免）——原 `run_sft_all.sh` 因此阶段失败后仍继续跑后续阶段，且末尾无条件 `echo ALL_SFT_DONE` 造成「未完成却标记完成」。**改法**：封装 `run_phase() { if "$P" "$@"; then ...; else ...; return 1; fi }` + `run_phase ... || exit 1`，完成标记仅在 p3 成功后写入。
14. **`pgrep -f`/`pkill -f` 自匹配**：SSH 远程执行 `pgrep -f 'train_sft.py'` 时，远程 `bash -c` 包装进程命令行本身含该字符串 → 恒匹配（误判训练在跑/误杀 SSH 会话）。**解法**：`pgrep -f '[t]rain_sft.py'`（字符类技巧）。supervisor 的 `sft_running_remote` 已按此实现。
15. **SSH 网关抖动 → supervisor 误报「训练已停止」（2026-08-09）**：seetacloud 网关（116.172.66.188:26022）间歇性断开连接（`Connection closed`/`Permission denied`）。`unattended_supervisor.sh` 的 `remote()` 探测失败时输出为空 → `grep -q YES` 失败 → `sft_running_remote` 返回 NO → 误触发 diagnose 会话（实际训练健康）。**解法**：SSH 命令加 `-o ConnectTimeout=25 -o ServerAliveInterval=10` 并失败重试；探测逻辑把「无输出」视为 unknown（当 running 处理）而非 stopped。
16. **vLLM 在 5090 的启用方法（2026-08-10，subagent 排查确认）**：vLLM 0.26 + flashinfer 0.6.16.post3 本身支持 SM 12x，报错根因是 flashinfer 的 CUDA 版本判定走 `get_cuda_path()`（无 `CUDA_HOME` 时回退 `/usr/local/cuda` → 软链 12.8 → 判定 12.8<12.9 → "SM 12.x requires CUDA >= 12.9"）。**修复（使用 vllm 前必 export，建议写 ~/.bashrc）**：
   ```bash
   export PATH=/root/miniconda3/bin:$PATH
   export CUDA_HOME=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13
   export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
   export VLLM_ATTENTION_BACKEND=FLASH_ATTN
   export VLLM_USE_FLASHINFER_SAMPLER=0   # 跳过 flashinfer JIT 采样（nvcc 13.3 与 headers 13.0 错配）
   ```
   已用 Qwen2.5-1.5B 实测（引擎启动 29s、生成/serve 正常）；**7B + LoRA 实测通过（2026-08-10）**：`LLM(model=..., enable_lora=True, max_lora_rank=32)` + `LoRARequest("jssp", 1, "/root/jssp/experiments/sft_qwen7b/final")` 加载生成正常。吞吐预期 7B 独占 30-60 tok/s。
17. **模型输出纪律问题与容错解析（2026-08-10）**：四阶段 SFT 后模型在 **6x6 上会多输出工序**（生成 90 条记录而实例只需 36 条，job 0-14 均匀分布；10x10/15x15/20x20 输出数量恰好正确）。诊断确认**前 n×m 条记录完全合法可行**（makespan 665 vs 参考 502）→ 属「停止纪律」问题而非内容错误。**解法**：`parse_schedule` 容错——记录数超过 n×m 时若前缀恰好完整覆盖全部工序则取前缀（多余忽略），否则报错。评估温度 0.5、greedy 均受影响；此问题是 FOARL 的修复对象之一。
16. **长序列 SFT 的 backward logits.grad OOM + `model.loss_function` 赋值陷阱（2026-08-10）**：20x20 样本实测**全部** 16989–17137 tokens（p50 17071，非长尾分布），截断不可行。即便 ①`use_cache=False`（省 ~1 GiB 全序列 KV cache，训练不生成纯浪费）②分块 CE 用 `torch.utils.checkpoint` 流式（不驻留全部分块 fp32），**backward 仍需全序列 logits.grad（bf16 seq×vocab×2 ≈ 4.93 GiB）叠加在前向 22.8 GiB 上 → 32GB 打爆**。此前「seq 18432 OOM」只测前向，漏了 backward。**真正解法**：`install_chunked_forward`（`src/training/loss.py`）把训练 forward 替换为「对 hidden_states 按 chunk 与 lm_head 矩阵乘 + 直接算 CE」，**全序列 logits 及其梯度都不落盘**（省 ≈10 GiB），数值与默认 loss 完全等价（测试 diff=0.0）；实测 seq 17408 fwd+bwd 峰值 26.9 GiB / 余 2.9 GiB。**另一个陷阱**：`model.loss_function = X` 必须在 **PeftModel 包裹之前**设在原始模型上——设在 PeftModel 上不会下传到内部 base model（其 forward 里 `self.loss_function` 仍取默认）。相关配置：`training.use_chunked_loss` + `training.chunk_size`（p3 取 512）。

## 目录结构与职责约定

```
jssp/                          # /root/jssp，代码与文档
├── CLAUDE.md            # 本文件：项目维护手册，随项目演进持续更新
├── README.md            # 项目门面：概述 + 导航
├── .gitignore           # 数据/权重/实验产物不入库
├── docs/                # 一切文档：问题定义、研究计划、论文解读、文献
├── src/                 # 核心代码（Python 包，各包有 __init__.py 说明职责）
│   ├── problem/         #   JSSP 形式化：实例数据结构、约束、makespan 计算、实例生成器
│   ├── data/            #   数据流水线：解析 raw → 生成实例 → 求解器监督 → 划分
│   ├── model/           #   LLM 侧：TAI 文本编码模板、排程输出格式定义与解析、本地模型推理封装
│   ├── solver/          #   参考求解器封装：OR-Tools CP-SAT、启发式基线（SPT 等）
│   ├── training/        #   SFT（LoRA）与 FOARL（GRPO）训练脚本
│   ├── evaluation/      #   可行性验证、gap 计算、时间统计、泛化评估
│   └── utils/           #   通用工具（种子、日志、IO）
├── configs/             # 实验配置（YAML），按阶段分子目录
│   ├── data/            #   实例生成与监督数据参数
│   ├── sft/             #   微调参数（模型、LoRA、数据、优化器）
│   └── foarl/           #   RL 参数（奖励权重 λ、GRPO、KL、BoN）
├── scripts/             # 一键运行入口（串联各阶段流水线）
├── experiments/         # 实验产物：一实验一目录（config 冻结存档 + logs + checkpoints + results + summary.md）
├── tests/               # pytest 单元测试（与 src 同构）
└── notebooks/           # 探索性分析（EDA、可视化）

数据实际存放：/root/autodl-tmp/jssp_data/{raw,instances,supervised,rl,splits}
```

## 开发规范（必须遵守）

1. **代码分层**：新代码只写入 `src/<对应模块>/`，运行入口写 `scripts/`，参数一律进 `configs/` YAML（**不硬编码超参**）。
2. **实例 ID 贯穿**：实例生成时分配 `instance_id`，数据、监督、划分、实验记录全程用同一 ID 关联。
3. **实验可复现**：每个实验在 `experiments/<exp_name>/` 冻结一份 config 副本；所有随机种子显式配置。
4. **一实验一目录**：不把多组实验的结果混在一个目录；结果文件用 JSON/CSV 结构化保存。
5. **测试先行**：核心逻辑（约束校验、makespan 计算、输出解析）必须配 pytest 测试，放 `tests/`，与 `src/` 同构。
6. **数据不入库**：数据与权重由脚本重建，落在数据盘 `/root/autodl-tmp/jssp_data`；`.gitignore` 已配置，不要手动破坏。
7. **文档同步**：文档改动写 `docs/`；**项目约定/经验/进度变化必须同步更新本 CLAUDE.md**，并保持与 README 一致。
8. **远程开发**：以服务器为唯一事实源；本机工作台修改后必须 rsync 到服务器并以服务器验证结果为准。

## 领域知识速览（详细见 docs/）

### JSSP 定义要点
- n 个工件 × m 台机器；每工件 m 道工序，工序 (j,k) 在指定机器 μ[j][k] 上加工 p[j][k] 时间
- **三条硬约束**：①工序顺序（前序工序完成后才能开始下一道）②机器独占（一台机器同一时刻只加工一道工序）③不可抢占
- 目标：最小化 makespan Cmax = max 完工时间；决策变量 = 各工序开始时间（或各机器上工序排列）
- 复杂度：强 NP-hard；经典公开实例：ft06/10/20（Fisher-Thompson）、la01-40（Lawrence）、ta01-80（Taillard）
- 标准数据集格式：OR-Library 与 Taillard 均为「首行 n m，随后 n 行每行 m 个(机器,时间)对」；机器编号 ft/ta 从 0 起、la 从 1 起

### LLMCoSolver 方法要点（本项目参照）
- **两阶段训练**：SFT（~500K 实例/问题，Qwen2.5-7B + LoRA）→ FOARL（≤3200 实例）
- **TAI 输入**：Text-Attributed Instance = 自然语言问题描述 + 廉价启发式特征（如 top-k 近邻）
- **FOARL 奖励** = 可行性奖励（格式合法 + 全部约束满足）+ 最优性奖励（与 gap 成反比），GRPO 更新（无需 critic）
- **推理**：Best-of-N 采样（N=8 默认），选最优可行解 —— 可行性与质量可权衡
- **成效**：7 个 CO 问题（含 JSSP）100% 可行性（6/7），平均 gap 1.03%–8.20%，超越 GPT-4o 等通用 LLM
- **关键教训**：SFT 单独会出现「贪婪违规」（为了目标值牺牲约束），FOARL 专门修复此问题

## 实现路径（路线图）

1. **问题形式化**（已完成，docs/jssp_definition.md）：数据模型、约束、目标、验证器
2. **参考求解器**：OR-Tools CP-SAT 封装，中小规模求近似最优解作监督信号；SPT 等启发式作基线
3. **数据构建**（当前）：随机生成多规模实例（如 6x6 ~ 30x20）→ 求解器出解 → 组成 SFT 数据集 + 划分
4. **输入输出表示**：TAI 文本模板（JSON 排程输出，可解析可验证）
5. **SFT**：LoRA 微调 Qwen2.5-7B-Instruct（本地，transformers + peft；推理/rollout 用 vLLM）
6. **FOARL**：GRPO + 可行性/最优性联合奖励，修复约束违规（本地采样）
7. **评估与泛化**：可行性率、gap、时间、跨规模/跨分布泛化，对照基线（启发式、OR-Tools、通用 LLM）
8. **消融与论文化**（视目标而定）

## 环境操作备忘

- conda 不自动激活：命令前 `source /root/miniconda3/etc/profile.d/conda.sh`，或用绝对路径 `/root/miniconda3/bin/python`、`/root/miniconda3/bin/pip`
- pip 镜像：阿里云（`/etc/pip.conf` 已配置）；HuggingFace 下载模型走 `hf-mirror.com`（`export HF_ENDPOINT=https://hf-mirror.com`）
- 后台长任务：`nohup ... > log 2>&1 &`，日志统一放 `experiments/<exp>/logs/`
- GPU 检查：`nvidia-smi`；单卡 5090 32GB

## 维护规则（本文件自身）

- 每次会话开始：读本文件 + `docs/research_plan.md`，对照「当前阶段」继续推进
- 阶段推进、结构变更、关键经验、踩坑记录：**当次会话内同步更新本文件**
- 本文件变更后如影响 README（目录树等），一并同步

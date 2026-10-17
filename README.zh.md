<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[![Unit-tests](https://github.com/pytorch/rl/actions/workflows/test-linux.yml/badge.svg)](https://github.com/pytorch/rl/actions/workflows/test-linux.yml)
[![Nightly](https://github.com/pytorch/rl/actions/workflows/nightly_orchestrator.yml/badge.svg)](https://pytorch.github.io/rl/nightly-status/)
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://pytorch.org/rl/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)](https://pytorch.github.io/rl/dev/bench/)
[![CI Timing](https://img.shields.io/badge/CI%20Timing-blue.svg)](https://pytorch.github.io/rl/ci-timing/)
[![codecov](https://codecov.io/gh/pytorch/rl/branch/main/graph/badge.svg?token=HcpK1ILV6r)](https://codecov.io/gh/pytorch/rl)
[![Flaky Tests](https://img.shields.io/endpoint?url=https://pytorch.github.io/rl/flaky/badge.json)](https://pytorch.github.io/rl/flaky/)
[![X / Twitter Follow](https://img.shields.io/twitter/follow/torchrl1?style=social)](https://twitter.com/torchrl1)
[![Python version](https://img.shields.io/pypi/pyversions/torchrl.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<a href="https://pypi.org/project/torchrl"><img src="https://img.shields.io/pypi/v/torchrl" alt="pypi version"></a>
<a href="https://pypi.org/project/torchrl-nightly"><img src="https://img.shields.io/pypi/v/torchrl-nightly?label=nightly" alt="pypi nightly version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/torchrl?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)](https://pepy.tech/project/torchrl)
[![Downloads](https://static.pepy.tech/personalized-badge/torchrl-nightly?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads%20(nightly))](https://pepy.tech/project/torchrl-nightly)
[![Discord Shield](https://dcbadge.vercel.app/api/server/cZs26Qq3Dd)](https://discord.gg/cZs26Qq3Dd)

# TorchRL

<p align="center">
  <img src="docs/source/_static/img/icon.png" width="200" alt="TorchRL logo">
</p>

TorchRL 是专为强化学习（Reinforcement Learning）、序贯决策、机器人学与动力学仿真打造的 **PyTorch 原生**工具包。它并非单一算法的孤立实现，也不是狭隘的基准测试集，而是一整套高度可组合的模块化组件，用于构建与 PyTorch 编程范式紧密契合的现代强化学习系统。近期演进特别强化了循环神经网络 RL（Recurrent RL）、基于 MuJoCo 的连续控制、多智能体协同训练（MARL）、经验回放池与数据收集器基础设施，以及可复用的损失函数与价值估计组件。

本库围绕三大核心设计理念构建：

1. **统一数据语义**：数据在穿透整个训练循环的过程中，应全程具备显式命名字段、嵌套结构、批次维度（Batch Dimensions）与设备属性（Device）。
2. **完全解耦的模块化抽象**：环境（Environments）、策略（Policies）、经验回放池（Replay Buffers）、目标优化函数（Objectives）与数据收集器（Collectors）均作为独立模块运作，替换任意组件均无需重写其他技术栈。
3. **跨规模无缝伸缩**：科研与工程代码可在不改变基础数据模型的前提下，平滑拓展至向量化（Vectorized）、多进程（Multiprocess）、分布式（Distributed）、`torch.compile` 编译加速、循环时序、多智能体、基于模型（Model-based）或离线（Offline）工作流。

上述通用数据模型正是 [TensorDict](https://github.com/pytorch/tensordict/) —— 一种具备 PyTorch 原生张量算子支持、设备间高效传输、共享内存优化、内存映射（Memmap）、惰性视图（Lazy Views）以及 `nn.Module` 包装能力的类字典张量容器。

[快速入门](https://pytorch.org/rl/stable/index.html#getting-started) |
[API 参考文档](https://pytorch.org/rl/stable/reference/index.html) |
[进阶教程](https://pytorch.org/rl/stable#tutorials) |
[知识库](https://pytorch.org/rl/stable/reference/knowledge_base.html) |
[示例项目](examples/) |
[SOTA 算法实现库](sota-implementations/)

## 近期核心亮点 (Recent highlights)

TorchRL 0.13 及近期演化周期带来了若干重磅提升：

- **极速循环 RL 路径**：内置 `scan` 算子以及基于 Triton 的 GRU/LSTM 隐状态重置优化；
- **原生 MuJoCo 仿真支持**：新增定制化 MuJoCo 环境、卫星控制示例与宏控制（Macro-control）分层策略；
- **全方位多智能体支持**：原生集成 MAPPO、IPPO、`MultiAgentGAE`、价值归一化工具（Value-normalization）与 Mixer 网络配置；
- **更强劲的数据收集与回放**：引入异步优先级写入、有序存储访问、紧凑观测表示、事后经验重放（HER），并提供基于 CUDA 的优先级回放池预编译 Wheel 包；
- **全新环境变换与价值估计**：引入 `ActionScaling`、`FlattenAction`、`NextObservationDelta`、紧凑时移估计器与分块前向传递（Chunked Forwards）。

## 核心工作心智模型 (A quick mental model)

TorchRL 将强化学习交互抽象为一个在若干高度复用组件之间流动传递的 `TensorDict`：

```text
TensorDict
  -> 策略网络 (Policy Module) 写入动作 (actions) 与对数概率 (log-probs)
  -> 环境 (Environment) 读取动作并写入下一步观测 (next observations)、奖励 (rewards) 与终止标志 (done flags)
  -> 收集器 (Collector) 聚合来自单个或多个工作进程的轨迹批次
  -> 经验回放池 (Replay Buffer) 存储、采样、优先级排序并变换数据
  -> 损失模块 (Loss Module) 读取命名字段并计算可微损失
  -> 优化器 (Optimizer) 更新常规 PyTorch 模型参数
```

同一个 `TensorDict` 容器可以贯穿携带状态观测、图像像素、动作、奖励、掩码（Masks）、循环隐状态、智能体分组、采样索引、优先级或自定义任务指标。这从根本上消除了冗余胶水代码与黑盒假设。

## 极速演示 (Quick demo)

在本地执行一段交互采集（Rollout），只需在 PyTorch 神经网络模块与环境之间传递 `TensorDict`：

```python
import torch
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.envs import PendulumEnv, StepCounter, TransformedEnv

# 创建带有常规变换堆栈的 PyTorch 原生环境
env = TransformedEnv(PendulumEnv(), StepCounter(max_steps=200))

# 策略本质上是包装了显式 TensorDict 键名约定的标准 nn.Module
policy = TensorDictModule(
    nn.Sequential(
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.Linear(64, 1),
        nn.Tanh(),
    ),
    in_keys=["observation"],
    out_keys=["action"],
)

rollout = env.rollout(max_steps=32, policy=policy)
assert rollout.batch_size == torch.Size([32])
assert rollout["next", "reward"].shape[:1] == torch.Size([32])
```

该模式完全通用：批处理环境、多智能体任务、收集器、经验回放池、循环时序模块、数据变换与损失计算均采用完全一致的键名与 `TensorDict` 接口。

## 现代 TorchRL 架构深度解析

### 1. 以 TensorDict 为第一公民的数据管线

传统强化学习代码极易积累特殊特化逻辑：一个环境返回元组，另一个返回字典，循环隐状态单独成数组，掩码散落在各处，损失函数对张量布局暗含假设。TorchRL 借助 `TensorDict` 将所有假设显式化。

`TensorDict` 在完整保留结构命名字段的同时，支持标准 PyTorch 算子操作：

```python
# 批量操作保留结构，作用于每一个兼容的值
batch = torch.stack(list_of_tensordicts, dim=0)
batch = batch.reshape(-1)
batch = batch.to("cuda")
mini_batch = batch[:128]

# 嵌套键（Nested Keys）让多智能体、循环状态和下一时刻状态清晰明确
reward = batch["next", "reward"]
agent_obs = batch["agents", "observation"]
hidden = batch["recurrent_state", "h"]
```

### 2. 环境与变换 (Environments and transforms)

TorchRL 包含原生环境、主流仿真库适配器，以及用于并行加速的向量化容器。环境 API 显式暴露观测、动作、奖励与终止状态的规范（Specs），在启动耗时数小时的大规模训练前即可预先校验形状、设备、数据类型与数值边界。

环境生态支持：
- PyTorch 原生环境（如 `PendulumEnv` 及定制 MuJoCo 任务）；
- 流行环境库封装：Gymnasium、Gym、DM Control、Brax、Jumanji、PettingZoo、VMAS、OpenSpiel、Safety-Gymnasium、Isaac Lab 等；
- 并行向量化容器：`SerialEnv`、`ParallelEnv` 与批处理包装器；
- 丰富的环境变换（Transforms）：观测归一化、图像转换、奖励缩放、动作掩码、动作缩放、自动重置、多帧堆叠（Frame Stacking）、状态重构等。

```python
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, TransformedEnv
from torchrl.envs.libs.gym import GymEnv

base_env = GymEnv("HalfCheetah-v4", device="cuda:0")
env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
    ),
)
```

### 3. 数据收集器与执行模型 (Collectors and execution models)

数据收集器（Collectors）是连接策略与环境的桥梁。收集器管理执行循环、组织轨迹批次、调度设备，并支持在环境持续推进采样的同时动态同步更新策略权重。

TorchRL 提供单进程、异步、多进程以及跨机器的分布式数据收集器：

```python
from torchrl.collectors import Collector

collector = Collector(
    create_env_fn=env,
    policy=policy,
    frames_per_batch=1024,
    total_frames=1_000_000,
)

for data in collector:
    # data 为保留时间序列、环境维度与键值结构的 TensorDict
    train_step(data)
```

### 4. 经验回放池与离线数据集 (Replay buffers and offline data)

TorchRL 的经验回放池高度模块化：底层存储（Storage）、采样器（Sampler）、写入器（Writer）、合并函数（Collate）、变换、预取、优先级更新与设备调度完全解耦。

```python
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer

buffer = TensorDictPrioritizedReplayBuffer(
    storage=LazyMemmapStorage(1_000_000),
    alpha=0.7,
    beta=0.5,
    batch_size=256,
    prefetch=2,
)

buffer.extend(collector_batch)
sample = buffer.sample()
```

### 5. 模块、分布与策略 (Modules, distributions, and policies)

TorchRL 模块是具备显式输入输出键绑定的 PyTorch 模块，提供智能体 Actor、Critic、Actor-Critic 复合算子、循环模块、探索模块、世界模型（World Models）、Decision Transformers、机器人模仿学习模型等。

构建随机策略 Actor 示例：

```python
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal

params = TensorDictModule(
    nn.Sequential(
        nn.LazyLinear(256),
        nn.Tanh(),
        nn.Linear(256, 2),
        NormalParamExtractor(),
    ),
    in_keys=["observation"],
    out_keys=["loc", "scale"],
)

actor = ProbabilisticActor(
    params,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=TanhNormal,
    distribution_kwargs={"low": -1.0, "high": 1.0},
    return_log_prob=True,
)
```

### 6. 目标损失、回报估计与训练器 (Objectives, returns, and trainers)

TorchRL 损失函数从 `TensorDict` 读取命名字段计算可微损失，覆盖经典与前沿算法：
- 策略梯度与 Actor-Critic：PPO, SAC, TD3, REDQ, CrossQ, IMPALA
- Q 学习与离线 RL：DQN, IQL, CQL, Decision Transformer
- 模仿学习与基于模型：Dreamer/DreamerV3, GAIL, 行为克隆 (BC), ACT
- 价值估计器：GAE, TD(lambda), V-trace, 向量化优势计算

```python
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

loss = ClipPPOLoss(actor_network=actor, critic_network=critic)
advantage = GAE(value_network=critic, gamma=0.99, lmbda=0.95)

data = advantage(data)
losses = loss(data)
loss_value = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
```

### 7. 多智能体、基于模型与大模型后训练

- **多智能体 (MARL)**：原生嵌套表示智能体状态（`("agents", "observation")`），深度支持 MAPPO、IPPO、`MultiAgentGAE`、`PopArtValueNorm` 与集中式 Critic。
- **前沿前沿大模型后训练 (LLM Post-Training)**：提供对话容器、Hugging Face / vLLM / SGLang 交互接口、GRPO 与 SFT 优化目标。详见 [LLM 专题文档](https://pytorch.org/rl/stable/reference/llms.html) 与 [GRPO 实现](sota-implementations/grpo)。

---

## 快速导航 (Where to start)

| 您的探索目标 | 推荐起点 |
| :--- | :--- |
| 掌握基础环境交互与 TensorDict 数据流 | [快速入门指南](https://pytorch.org/rl/stable/index.html#getting-started) 与上述 Demo 范例 |
| 训练经典连续控制智能体 | [PPO](sota-implementations/ppo/)、[SAC](sota-implementations/sac/) 或 [TD3](sota-implementations/td3/) 算法实现 |
| 定制环境数据预处理与变换 | [环境变换组件 (Environment transforms)](https://pytorch.org/rl/stable/reference/envs_transforms.html) |
| 大规模扩展数据采集吞吐 | [收集器架构 (Collectors)](https://pytorch.org/rl/stable/reference/collectors.html) 与 [分布式收集器](examples/distributed/collectors/) |
| 存储海量或优先级回放数据 | [经验回放池 (Replay buffers)](https://pytorch.org/rl/stable/reference/data_replaybuffers.html) |
| 研发循环神经网络（RNN/GRU/LSTM）策略 | [循环模块](https://pytorch.org/rl/stable/reference/modules_rnn.html) 与 [状态生命周期文档](https://pytorch.org/rl/stable/reference/recurrent_state_lifecycle.html) |
| 研发多智能体协同系统 | [多智能体目标函数](https://pytorch.org/rl/stable/reference/objectives_multiagent.html) 与 [MARL 示例](examples/multiagent/) |
| 探索 MuJoCo 宏分层动作策略 | [宏控制原语 (Macro primitives)](https://pytorch.org/rl/stable/reference/macro_primitives.html) 与 MuJoCo 教程 |
| 探索大语言模型强化学习后训练 (RLHF/RLAIF) | [LLM 专题参考](https://pytorch.org/rl/stable/reference/llms.html) 与 [GRPO 实现](sota-implementations/grpo/) |

---

## 安装指南 (Installation)

TorchRL 0.13 要求 **Python 3.10+**、**PyTorch 2.1+** 以及 **TensorDict 0.13.x**。

### 1. 安装最新稳定版本

```bash
pip install torchrl
```

对于绝大多数用户（包括使用 CPU 优先级回放或无需优先级回放的工作流），官方 PyPI 预编译包即可满足全部需求。从 TorchRL 0.13 开始，针对需要使用 CUDA 加速优先级回放池的用户，还发布了 Linux CUDA 轮子（将 `cu128` 替换为与您的 PyTorch 环境匹配的 CUDA 版本）：

```bash
pip install "torchrl==0.14.0+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
```

### 2. 安装常用可选依赖项

```bash
pip install "torchrl[utils]"              # Hydra、日志与开发辅助工具
pip install "torchrl[gym_continuous]"     # Gymnasium 连续控制环境套件
pip install "torchrl[atari]"              # Atari 游戏支持
pip install "torchrl[offline-data]"       # 离线强化学习数据集与数据工具
pip install "torchrl[marl]"               # 多智能体环境库 (VMAS, PettingZoo 等)
pip install "torchrl[llm-vllm]"           # Linux 上搭配 vLLM 后端的 LLM API
pip install "torchrl[llm-sglang]"         # Linux 上搭配 SGLang 后端的 LLM API
```

### 3. 安装每日构建版本 (Nightly)

```bash
pip install --pre tensordict-nightly torchrl-nightly
```

### 4. 本地源码可编辑安装

```bash
git clone https://github.com/pytorch/tensordict
git clone https://github.com/pytorch/rl
uv pip install --no-deps -e tensordict
uv pip install --no-deps -e rl
```

---

## 文档与学习资源

- [官方稳定版文档](https://pytorch.org/rl/stable/)
- [API 完整参考手册](https://pytorch.org/rl/stable/reference/index.html)
- [进阶实战教程](https://pytorch.org/rl/stable#tutorials)
- [核心知识库 (Knowledge Base)](https://pytorch.org/rl/stable/reference/knowledge_base.html)
- [基准测试性能大盘](https://pytorch.github.io/rl/dev/bench/)
- [CI 每日构建状态](https://pytorch.github.io/rl/nightly-status/)

入门论文与演讲：
- [TorchRL 学术论文](https://arxiv.org/abs/2306.00577)
- [TalkRL 播客专访：Vincent Moens 深度畅谈 TorchRL](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
- [PyTorch Day 2022：TorchRL 官方介绍演讲](https://youtu.be/cIKMhZoykEE)
- [PyTorch 2.0 官方答疑：TorchRL 专场](https://www.youtube.com/live/myEfUoYrbts?feature=share)

---

## 论文引用 (Citation)

如果您在学术研究中使用了 TorchRL，请引用官方论文：

```bibtex
@misc{bou2023torchrl,
      title={TorchRL: A data-driven decision-making library for PyTorch},
      author={Albert Bou and Matteo Bettini and Sebastian Dittert and Vikash Kumar and Shagun Sodhani and Xiaomeng Yang and Gianni De Fabritiis and Vincent Moens},
      year={2023},
      eprint={2306.00577},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 问题反馈与社区贡献

- **问题反馈**：若发现 Bug，请在仓库中提交 Issue。关于 PyTorch 强化学习的泛化讨论，欢迎访问 [PyTorch 官方 RL 论坛](https://discuss.pytorch.org/c/reinforcement-learning/6)。
- **参与贡献**：详见 [CONTRIBUTING.md](CONTRIBUTING.md) 贡献指引，以及官方设立的[需求招募清单](https://github.com/pytorch/rl/issues/509)。在对应 Issue 下回复 `/assign` 即可认领任务。
- **本地代码规范检查**：
  ```bash
  pre-commit install
  ```

## 项目状态与开源协议

TorchRL 目前作为 PyTorch Beta 特性发布。我们致力于在重大改动前提供跨多个版本的废弃警告（Deprecation Warnings）。

TorchRL 遵循宽松的 **MIT 开源许可证**。详见 [LICENSE](LICENSE)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月27日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！

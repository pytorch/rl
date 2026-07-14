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

TorchRL is a PyTorch-native toolkit for reinforcement learning, decision making,
robotics, and simulation. It is not a single algorithm implementation or a
narrow benchmark suite: it is a collection of composable pieces for building RL
systems while keeping the code close to the PyTorch programming model. Recent
work has made this especially strong for recurrent RL, MuJoCo-based control,
multi-agent training, replay-buffer and collector infrastructure, and reusable
loss/value-estimation components.

The library is built around three ideas:

1. Data should have names, structure, batch dimensions, and devices all the way
   through the training loop.
2. Environments, policies, replay buffers, objectives, and collectors should be
   independent modules that can be swapped without rewriting the rest of the
   stack.
3. Research code should scale from a local prototype to vectorized,
   multiprocess, distributed, compiled, recurrent, multi-agent, model-based, or
   offline workflows without changing the data model.

That common data model is [TensorDict](https://github.com/pytorch/tensordict/),
a dictionary-like tensor container with PyTorch operations, device transfers,
shared-memory support, memmaps, lazy views, and `nn.Module` wrappers.

[Getting started](https://pytorch.org/rl/stable/index.html#getting-started) |
[API reference](https://pytorch.org/rl/stable/reference/index.html) |
[Tutorials](https://pytorch.org/rl/stable#tutorials) |
[Knowledge base](https://pytorch.org/rl/stable/reference/knowledge_base.html) |
[Examples](examples/) |
[SOTA implementations](sota-implementations/)

## Recent highlights

TorchRL 0.13 and the preceding development cycle bring several user-visible
improvements that are worth surfacing up front:

- faster recurrent RL paths, including scan and Triton GRU/LSTM reset handling;
- custom MuJoCo environments, satellite examples, and macro-control policies;
- stronger multi-agent coverage through MAPPO, IPPO, `MultiAgentGAE`,
  value-normalization utilities, and mixer configs;
- better collector and replay-buffer ergonomics, including async prioritized
  writes, ordered storage access, compact observations, HER, and optional CUDA
  wheels for CUDA-based prioritized replay-buffer kernels;
- new transforms and value-estimator improvements such as `ActionScaling`,
  `FlattenAction`, `NextObservationDelta`, compact shifted estimators, and
  chunked forwards.

## A quick mental model

TorchRL represents an RL interaction as a TensorDict that moves through a small
number of reusable components:

```text
TensorDict
  -> policy module writes actions and log-probs
  -> environment reads actions and writes next observations, rewards, done flags
  -> collector batches trajectories from one or many workers
  -> replay buffer stores, samples, prioritizes, and transforms data
  -> loss module reads named keys and writes differentiable losses
  -> optimizer updates ordinary PyTorch parameters
```

The same object can carry observations, pixels, actions, rewards, masks,
recurrent states, agent groups, sampled indices, priorities, or custom task
fields. The result is less glue code and fewer hidden assumptions about
what each algorithm or environment returns.

## Quick demo

A local rollout is just a TensorDict passed between a PyTorch module and an
environment:

```python
import torch
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.envs import PendulumEnv, StepCounter, TransformedEnv

# A PyTorch-native environment with an ordinary transform stack.
env = TransformedEnv(PendulumEnv(), StepCounter(max_steps=200))

# Policies are regular nn.Modules wrapped with explicit TensorDict keys.
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

Nothing in this pattern is specific to Pendulum. The same keys-and-TensorDict
interface is used by batched environments, multi-agent tasks, collectors,
replay buffers, recurrent modules, transforms, and losses.

## What TorchRL is today

### TensorDict-first pipelines

RL code tends to accumulate special cases: tuples from one environment, dicts
from another, separate arrays for recurrent states, masks next to data rather
than inside it, and losses that silently assume a particular batch layout.
TorchRL uses TensorDict to make those assumptions explicit.

TensorDict supports common tensor operations while preserving named fields:

```python
# These operations preserve the structure and operate on every compatible value.
batch = torch.stack(list_of_tensordicts, dim=0)
batch = batch.reshape(-1)
batch = batch.to("cuda")
mini_batch = batch[:128]

# Nested keys make multi-agent, recurrent, and next-state data explicit.
reward = batch["next", "reward"]
agent_obs = batch["agents", "observation"]
hidden = batch["recurrent_state", "h"]
```

This is the reason TorchRL components compose: a collector can emit a TensorDict,
a replay buffer can store it without losing structure, a transform can add or
remove keys, and a loss can read exactly the keys it needs.

### Environments and transforms

TorchRL includes native environments, wrappers for popular environment libraries,
and vectorized containers for running many environments at once. The environment
API exposes specs for observations, actions, rewards, and done flags, so policies
and transforms can check shapes, devices, dtypes, and bounds before a training
job runs for hours.

Environment support includes:

- PyTorch-native environments such as `PendulumEnv` and custom MuJoCo tasks.
- Wrappers for Gymnasium, Gym, DM Control, Brax, Jumanji, PettingZoo, VMAS,
  OpenSpiel, Safety-Gymnasium, Isaac Lab, and other optional libraries.
- `SerialEnv`, `ParallelEnv`, and batched wrappers for local vectorization and
  multiprocessing.
- Environment transforms for observation normalization, image conversion,
  reward transforms, action masking, action scaling, auto-reset, frame stacking,
  state reconstruction, and more.

Transforms are first-class TorchRL modules. They can run on-device, participate
in specs, and be inserted, removed, or composed without wrapping the whole
environment in opaque adapter layers.

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

### Collectors and execution models

Collectors are the bridge between policies and environments. A collector owns
the execution loop, batches trajectories, handles devices, and can update policy
weights while environments keep running.

TorchRL includes single-process, async, multiprocess, and distributed
collectors. This lets the same policy and loss code be used across small smoke
tests, GPU-heavy simulation, CPU environment farms, or asynchronous evaluation
setups.

```python
from torchrl.collectors import Collector

collector = Collector(
    create_env_fn=env,
    policy=policy,
    frames_per_batch=1024,
    total_frames=1_000_000,
)

for data in collector:
    # data is a TensorDict with time, environment, and key structure preserved.
    train_step(data)
```

For larger jobs, the collector family adds async execution, multiple worker
processes, weight updaters, evaluator loops, profiling hooks, and fake-data
helpers for testing downstream code without stepping an expensive environment.

### Replay buffers and offline data

TorchRL replay buffers are modular: storage, sampler, writer, collate function,
transforms, prefetching, priority updates, and device movement are separate
pieces. That makes it possible to use the same interface for simple in-memory
replay, memmap-backed storage, prioritized replay, CUDA-aware sampling, offline
datasets, HER, or custom storage layouts.

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

Replay buffers understand TensorDict structure, so they can store trajectories,
nested agent data, recurrent states, HER relabeling metadata, or offline
datasets without flattening everything into parallel Python containers.

### Modules, distributions, and policies

TorchRL modules are ordinary PyTorch modules with explicit input and output
keys. The library provides actors, critics, actor-critic operators, recurrent
modules, distribution wrappers, exploration modules, world models, decision
transformers, robot-learning models, and helper utilities for inferring specs
from environments.

A stochastic actor can be assembled from familiar PyTorch layers:

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

The explicit key contract makes it clear what data a module consumes and
produces, and it allows losses, collectors, and transforms to be reconfigured
without editing the model itself.

### Objectives, returns, and trainers

TorchRL objectives are loss modules that read TensorDict keys, compute losses,
and expose configurable key mappings. They cover policy-gradient methods,
actor-critic algorithms, Q-learning, offline RL, imitation learning, model-based
RL, and multi-agent RL.

Examples include PPO, SAC, DQN, TD3, REDQ, IQL, CQL, Decision Transformer,
Dreamer, CrossQ, GAIL, behavior cloning, ACT, MAPPO, IPPO, and QMIX/VDN.
Value-estimator utilities provide GAE, TD(lambda), V-trace, lambda returns,
multi-agent advantages, and vectorized return computation.

```python
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

loss = ClipPPOLoss(actor_network=actor, critic_network=critic)
advantage = GAE(value_network=critic, gamma=0.99, lmbda=0.95)

data = advantage(data)
losses = loss(data)
loss_value = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
```

For higher-level workflows, TorchRL also provides trainer utilities and Hydra
configuration dataclasses that assemble environments, networks, collectors,
losses, optimizers, loggers, hooks, and schedules into reproducible recipes.

### Multi-agent, model-based, and imitation learning

Multi-agent data is represented as TensorDict structure rather than a separate
parallel convention. Agent observations, actions, rewards, masks, and shared
state can live under nested keys such as `("agents", "observation")`, while
losses and modules declare which keys they use.

TorchRL supports multi-agent environments and algorithms through VMAS,
PettingZoo, Melting Pot, SMACv2, OpenSpiel, multi-agent trainers, and dedicated
objectives. The 0.13 line adds MAPPO, IPPO, `MultiAgentGAE`, `ValueNorm`,
`PopArtValueNorm`, `RunningValueNorm`, and cross-agent critic utilities.

The same component style also covers model-based and imitation-learning work:
Dreamer/DreamerV3 objectives and RSSM modules, Decision Transformer components,
behavior cloning losses, and ACT-style action chunking all share the same
TensorDict and key-dispatch conventions as the online RL algorithms.

### Additional specialized workflows

TorchRL also includes support for specialized workflows, including LLM
post-training experiments. The LLM stack provides conversation containers,
Hugging Face/vLLM/SGLang integration points, GRPO and SFT objectives, async
collectors, weight-update helpers, and tool-use transforms. Entry points include
the [LLM reference](https://pytorch.org/rl/stable/reference/llms.html) and the
[GRPO implementation](sota-implementations/grpo).

### Performance and PyTorch integration

TorchRL is designed to stay close to PyTorch execution. Components are
TensorDict-aware, vectorized where possible, and increasingly friendly to
`torch.compile`, CUDA, shared memory, memmaps, and distributed execution.

Performance-sensitive areas include:

- vectorized return and advantage computation;
- recurrent GRU/LSTM reset handling with scan and Triton backends;
- compact sequence layouts for recurrent value estimation;
- async collectors and policy weight synchronization;
- prioritized replay and CUDA-aware replay-buffer paths;
- memmap-backed data movement for large offline or distributed jobs.

## What is new in TorchRL 0.13

TorchRL 0.13 is a broad release. The most impactful changes are in recurrent
RL performance, MuJoCo-native workflows, multi-agent training, model-based and
imitation-learning components, replay/collector throughput, and compatibility
with old or optional dependency stacks.

### Recurrent RL

- Triton and scan recurrent backends for GRU/LSTM reset handling.
- Recurrent integration tests and a recurrent state lifecycle guide.
- Compact and shifted value-estimator improvements, chunked forwards, and a
  dynamic value-estimator registry across loss modules.
- Recurrent matmul precision controls exposed through public module utilities.

### MuJoCo, robotics, and macro control

- Custom MuJoCo environments with selectable physics backends.
- New `MujocoEnv` task base plus locomotion tasks, `SatelliteEnv`, and
  `CubeBowlEnv`.
- Satellite MuJoCo SAC examples.
- Macro-control primitives and tutorials for low-frequency semantic actions
  expanded into multi-step low-level control sequences.

### Multi-agent, imitation, and model-based RL

- MAPPO and IPPO losses.
- `MultiAgentGAE` and value-normalization utilities.
- DreamerV3 losses and RSSM V3 modules.
- `BCLoss`, `ACTLoss`, and `ACTModel` for behavior cloning and action chunking.
- QMIX/VDN trainer configuration support and improved multi-agent trainer
  ergonomics.

### Data, transforms, and compatibility

- HER support through `HERReplayBuffer` and `HindsightStrategy`.
- Action and observation transforms such as `ActionScaling`, `FlattenAction`,
  `ExpandAs`, `NextObservationDelta`, `NextStateReconstructor`, and
  `TerminateTransform`.
- Async prioritized replay-buffer writes, ordered read/write APIs, optional
  trajectory IDs, compact observations, and safer collector weight syncs.
- Compatibility fixes across Gym/Atari, PettingZoo, Robohive, optional
  dependency, setup, documentation, vLLM, and SGLang workflows.

## Where to start

| If you want to... | Start with... |
| --- | --- |
| Learn the basic environment and TensorDict loop | [Getting started](https://pytorch.org/rl/stable/index.html#getting-started) and the quick demo above |
| Train a classic continuous-control agent | [PPO](sota-implementations/ppo/), [SAC](sota-implementations/sac/), or [TD3](sota-implementations/td3/) implementations |
| Build custom environment preprocessing | [Environment transforms](https://pytorch.org/rl/stable/reference/envs_transforms.html) |
| Scale data collection | [Collectors](https://pytorch.org/rl/stable/reference/collectors.html) and [distributed collectors](examples/distributed/collectors/) |
| Store large or prioritized data | [Replay buffers](https://pytorch.org/rl/stable/reference/data_replaybuffers.html) |
| Work with recurrent policies | [Recurrent modules](https://pytorch.org/rl/stable/reference/modules_rnn.html) and [state lifecycle docs](https://pytorch.org/rl/stable/reference/recurrent_state_lifecycle.html) |
| Train multi-agent systems | [Multi-agent objectives](https://pytorch.org/rl/stable/reference/objectives_multiagent.html) and [multi-agent examples](examples/multiagent/) |
| Explore MuJoCo macro policies | [Macro primitives](https://pytorch.org/rl/stable/reference/macro_primitives.html) and MuJoCo tutorials |
| Try language-model post-training experiments | [LLM reference](https://pytorch.org/rl/stable/reference/llms.html) and [GRPO](sota-implementations/grpo/) |

## Installation

TorchRL 0.13 targets Python 3.10+, PyTorch 2.1+, and TensorDict 0.13.x.

Install the stable release:

```bash
pip install torchrl
```

This standard PyPI wheel is the right default for most users, including CPU
prioritized replay buffers and workloads that do not use prioritized replay.
Starting with TorchRL 0.13, Linux CUDA wheels are also published for users who
want the CUDA-based prioritized replay-buffer implementations. Install the
CUDA wheel from the PyTorch wheel index that matches your PyTorch CUDA runtime
(replace `cu128` with the CUDA build you use):

```bash
pip install "torchrl==0.13.0+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
```

The CUDA wheel is optional: if you do not need CUDA prioritized replay buffers,
or if your prioritized replay buffers run on CPU, keep using `pip install
torchrl`.

Install common optional dependencies:

```bash
pip install "torchrl[utils]"              # Hydra, logging, and development utilities
pip install "torchrl[gym_continuous]"     # Gymnasium continuous-control environments
pip install "torchrl[atari]"              # Atari support
pip install "torchrl[offline-data]"       # Offline datasets and data helpers
pip install "torchrl[marl]"               # Multi-agent environment libraries
pip install "torchrl[llm-vllm]"           # LLM API with vLLM backend on Linux
pip install "torchrl[llm-sglang]"         # LLM API with SGLang backend on Linux
```

Some optional libraries are platform- or Python-version-specific. If you are
building a reproducible environment, install PyTorch first from the appropriate
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install TorchRL and the optional extras you need.

Install the nightly builds when working against nightly PyTorch:

```bash
pip install --pre tensordict-nightly torchrl-nightly
```

For local development, keep the TorchRL and TensorDict checkouts on compatible
branches and avoid re-resolving an already selected PyTorch build:

```bash
git clone https://github.com/pytorch/tensordict
git clone https://github.com/pytorch/rl
uv pip install --no-deps -e tensordict
uv pip install --no-deps -e rl
```

The C++ extension paths used by prioritized replay buffers require a compatible
PyTorch version. If you see undefined-symbol errors, consult the
[versioning issues guide](knowledge_base/VERSIONING_ISSUES.md).

## Documentation and learning resources

- [Stable documentation](https://pytorch.org/rl/stable/)
- [API reference](https://pytorch.org/rl/stable/reference/index.html)
- [Tutorials](https://pytorch.org/rl/stable#tutorials)
- [Knowledge base](https://pytorch.org/rl/stable/reference/knowledge_base.html)
- [Benchmarks](https://pytorch.github.io/rl/dev/bench/)
- [Nightly CI status](https://pytorch.github.io/rl/nightly-status/)

Introductory material:

- [TorchRL paper](https://arxiv.org/abs/2306.00577)
- [TalkRL podcast episode](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
- [TorchRL intro at PyTorch Day 2022](https://youtu.be/cIKMhZoykEE)
- [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Examples, tutorials, and implementations

TorchRL ships examples for small features and complete training recipes:

- [SOTA implementations](sota-implementations/) for PPO, SAC, DQN, TD3, REDQ,
  Decision Transformer, Dreamer, CrossQ, GAIL, IMPALA, multi-agent algorithms,
  GRPO, and more.
- [Examples](examples/) for distributed collectors, replay buffers, RLHF,
  MuJoCo satellite control, and other focused workflows.
- [Tutorials](https://pytorch.org/rl/stable#tutorials) for environment design,
  transforms, collectors, losses, recurrent state handling, MuJoCo macros, and
  end-to-end training.

The implementations are meant to be readable starting points, not black-box
benchmarks. They show how TorchRL components fit together and can be copied into
research code when a full trainer abstraction is not the right fit.

## Ecosystem and publications

TorchRL is domain-agnostic and is used across robotics, control, simulation,
drug discovery, multi-agent RL, combinatorial optimization, and research
infrastructure. Selected projects and papers include:

- [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement
  learning of generative chemical agents for drug discovery.
- [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking
  multi-agent reinforcement learning.
- [BricksRL](https://arxiv.org/abs/2406.17490): A platform for democratizing
  robotics and reinforcement learning research and education with LEGO.
- [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An
  efficient and flexible platform for reinforcement learning in drone control.
- [RL4CO](https://arxiv.org/abs/2306.17100): Reinforcement learning for
  combinatorial optimization.
- [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf):
  A unified framework for robot learning.

## Citation

If you use TorchRL, please cite:

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

## Asking questions

If you find a bug, please open an issue in this repository. For broader RL in
PyTorch questions, use the [PyTorch reinforcement learning forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full
contribution guide and the [call for contributions](https://github.com/pytorch/rl/issues/509)
for open areas where help is especially useful.

For local development, install pre-commit hooks with:

```bash
pre-commit install
```

## Status and license

TorchRL is released as a PyTorch beta feature. Breaking changes can happen, but
TorchRL aims to introduce them with deprecation warnings over multiple release
cycles.

TorchRL is licensed under the MIT License. See [LICENSE](LICENSE) for details.

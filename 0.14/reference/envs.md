# torchrl.envs package

TorchRL offers a comprehensive API to handle environments of different backends, making it easy to swap
environments in an experiment with minimal effort. The library provides wrappers for popular RL frameworks
including Gym, DMControl, Brax, Jumanji, and many others.

The [`EnvBase`](generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) class serves as the foundation, providing a unified interface that uses
[`tensordict.TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for data organization. This design allows the framework to be generic and
handle an arbitrary number of inputs and outputs, as well as nested or batched data structures.

## Key Features

- **Unified API**: Consistent interface across different environment backends
- **Vectorization**: Built-in support for parallel and batched environments
- **Transforms**: Powerful transform system for preprocessing observations and actions
- **Multi-agent**: Native support for multi-agent RL with no additional infrastructure
- **Flexible backends**: Easy integration with Gym, DMControl, Brax, and custom environments

## Quick Example

```
from torchrl.envs import GymEnv, ParallelEnv, TransformedEnv
from torchrl.envs.transforms import RewardSum, StepCounter

# Create a single environment
env = GymEnv("Pendulum-v1")

# Add transforms
env = TransformedEnv(env, RewardSum())

# Create parallel environments
def make_env():
 return TransformedEnv(
 GymEnv("Pendulum-v1"),
 StepCounter(max_steps=200)
 )

parallel_env = ParallelEnv(4, make_env)

# Run a rollout
rollout = parallel_env.rollout(100)
```

## Documentation Sections

- [Environment API](envs_api.html)

- [Env specs: locks and batch size](envs_api.html#env-specs-locks-and-batch-size)
- [Auto-wrapping recurrent transforms via the `policy=` argument](envs_api.html#auto-wrapping-recurrent-transforms-via-the-policy-argument)
- [Compiling envs via the `compile=` constructor argument](envs_api.html#compiling-envs-via-the-compile-constructor-argument)
- [Env methods](envs_api.html#env-methods)
- [Base classes](envs_api.html#base-classes)
- [Custom native TorchRL environments](envs_api.html#custom-native-torchrl-environments)
- [MuJoCo custom environments](envs_api.html#mujoco-custom-environments)
- [Domain-specific](envs_api.html#domain-specific)
- [Helpers](envs_api.html#helpers)
- [Vectorized and Parallel Environments](envs_vectorized.html)

- [Vectorized environment classes](envs_vectorized.html#vectorized-environment-classes)
- [Partial steps and partial resets](envs_vectorized.html#partial-steps-and-partial-resets)
- [Async environments](envs_vectorized.html#async-environments)
- [Transforms](envs_transforms.html)

- [Forward and inverse transforms](envs_transforms.html#forward-and-inverse-transforms)
- [Designing your own Transform](envs_transforms.html#designing-your-own-transform)
- [Available Transforms](envs_transforms.html#available-transforms)
- [Functional transforms](envs_transforms.html#functional-transforms)
- [Environments with masked actions](envs_transforms.html#environments-with-masked-actions)
- [Macro-control primitives](macro_primitives.html)

- [The central design choice](macro_primitives.html#the-central-design-choice)
- [What does "reach" mean?](macro_primitives.html#what-does-reach-mean)
- [Choosing a macro transform](macro_primitives.html#choosing-a-macro-transform)
- [Example 1: humanoid actuator-control macros](macro_primitives.html#example-1-humanoid-actuator-control-macros)
- [Example 2: satellite attitude slews](macro_primitives.html#example-2-satellite-attitude-slews)
- [Example 3: cube-to-bowl robot primitives](macro_primitives.html#example-3-cube-to-bowl-robot-primitives)
- [Custom Cartesian solvers and partial pose constraints](macro_primitives.html#custom-cartesian-solvers-and-partial-pose-constraints)
- [Designing target-driven macros for a new environment](macro_primitives.html#designing-target-driven-macros-for-a-new-environment)
- [Comparison](macro_primitives.html#comparison)
- [When to specialize](macro_primitives.html#when-to-specialize)
- [Multi-agent Environments](envs_multiagent.html)

- [MarlGroupMapType](generated/torchrl.envs.MarlGroupMapType.html)
- [check_marl_grouping](generated/torchrl.envs.check_marl_grouping.html)
- [Library Wrappers](envs_libraries.html)

- [Available wrappers](envs_libraries.html#available-wrappers)
- [Auto-resetting Environments](envs_libraries.html#auto-resetting-environments)
- [Dynamic Specs](envs_libraries.html#dynamic-specs)
- [Recorders](envs_recorders.html)

- [Recording videos](envs_recorders.html#recording-videos)
- [IsaacLab Integration](isaaclab.html)

- [IsaacLabWrapper](isaaclab.html#isaaclabwrapper)
- [Collector](isaaclab.html#collector)
- [Replay Buffer](isaaclab.html#replay-buffer)
- [TorchRL-Specific Gotchas](isaaclab.html#torchrl-specific-gotchas)
# CraftGroundWrapper

torchrl.envs.CraftGroundWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/craftground.html#CraftGroundWrapper)

CraftGround (Minecraft) environment wrapper.

GitHub: [yhs0602/CraftGround](https://github.com/yhs0602/CraftGround)

Documentation: [https://yhs0602.github.io/CraftGround/](https://yhs0602.github.io/CraftGround/)

Paper: Yun et al., "CraftGround: A Flexible Reinforcement Learning
Environment Based on the Latest Minecraft" (2025).

CraftGround runs a lightweight, headless-capable Minecraft client
instrumented through a Fabric mod, and exposes it as a gymnasium
environment. Observations are ego-centric RGB frames; actions follow
either the MineDojo-style multi-discrete layout (v1) or a
MineRL-human-like dictionary layout (v2).

The wrapped environment is a *sandbox*: the underlying `step` always
returns a zero reward and never terminates. Rewards and termination
conditions are meant to be composed on top with TorchRL transforms
(see the example below), or by sending Minecraft commands through
`env.add_command(...)` and reading the resulting state.

Note

**Minecraft ownership and licensing.** TorchRL does not
distribute Minecraft. On the first [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset), CraftGround's Gradle
project downloads the Minecraft client from Mojang's servers onto the
local machine and runs it in offline mode. Users are expected to own a
valid Minecraft: Java Edition license; offline mode bypasses
authentication, not ownership. Usage of Minecraft is governed by the
Minecraft EULA ([https://www.minecraft.net/eula](https://www.minecraft.net/eula)), including its
restrictions on commercial exploitation. Never redistribute the
downloaded game files (e.g. in public Docker images or CI caches).
CraftGround is a separate, optional dependency that TorchRL does not
vendor or redistribute. Its upstream repository currently ships a
GPL-3.0 license file while its package metadata reports MIT; consult the
upstream licensing information before distribution. See
`knowledge_base/MINECRAFT.md` in the TorchRL repository for details.

Note

The environment is spawned lazily: constructing the wrapper only
binds an IPC channel. The Minecraft client (a Java subprocess built and
launched through Gradle) starts on the first [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset), which can
take several minutes on the very first run while Gradle downloads
Minecraft and compiles the mod. Requires a JDK (OpenJDK 21) and, on
headless machines, a virtual display (e.g. `Xvfb`); see
`knowledge_base/MINECRAFT.md`.

Note

Only the `RAW` (default) and `PNG` screen encoding modes are
supported. `RAW` frames have shape `(H, W, 3)`, `PNG` frames
`(3, W, H)`, both `torch.uint8`. With a binocular configuration
(`eye_distance > 0`) a second `pixels_2` entry is added.

Parameters:

**env** (*craftground.environment.environment.CraftGroundEnvironment*) - the
CraftGround environment instance to wrap.

Keyword Arguments:

- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent
([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)), otherwise a one-hot encoding
will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)). Defaults to `False`.
Only used with the v1 (MineDojo-style) action space.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the
data is to be cast. Defaults to `torch.device("cpu")`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - only `torch.Size([])` is
supported, as CraftGround environments are not vectorized.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.

Variables:

**available_envs** - an empty list; CraftGround environments are described
by a `craftground.InitialEnvironmentConfig` rather than selected
from a registry of task ids.

Examples

```
>>> import craftground 
>>> from torchrl.envs import TransformedEnv, StepCounter
>>> from torchrl.envs.libs.craftground import CraftGroundWrapper
>>> base = craftground.make(port=8023) 
>>> env = CraftGroundWrapper(base) 
>>> # the sandbox emits no reward: compose one with a transform,
>>> # e.g. a step-count budget via StepCounter
>>> env = TransformedEnv(env, StepCounter(max_steps=100)) 
>>> td = env.rollout(3) 
>>> assert td["next", "pixels"].dtype is torch.uint8
```
# CraftGroundEnv

torchrl.envs.CraftGroundEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/craftground.html#CraftGroundEnv)

CraftGround (Minecraft) environment built from a configuration.

See [`CraftGroundWrapper`](torchrl.envs.CraftGroundWrapper.html#torchrl.envs.CraftGroundWrapper) for behavior details and licensing notes.
The constructor builds the environment through `craftground.make(...)`.

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

Keyword Arguments:

- **initial_env_config** (*craftground.InitialEnvironmentConfig**,**optional*) - the world/observation configuration (image size, game mode, world
type, seed, initial commands, ...). Defaults to a fresh
`InitialEnvironmentConfig()`.
- **mc_version** (*str**,**optional*) - the Minecraft version to run. Only
`"1.21"` is currently functional upstream. Defaults to
`"1.21"`.
- **port** (*int**,**optional*) - the IPC port used to communicate with the
Minecraft process. A free port is picked automatically if the
given one is busy. Defaults to `8000`.
- **action_space_version** (*craftground.ActionSpaceVersion**,**optional*) - the
action layout, either `V1_MINEDOJO` (multi-discrete) or
`V2_MINERL_HUMAN` (dict of booleans plus a continuous camera).
Defaults to `V1_MINEDOJO`.
- **env_path** (*str**,**optional*) - path to a custom CraftGround Gradle project.
Defaults to the project shipped with the installed
`craftground-runtime-mc121` package.
- **use_shared_memory** (*bool**,**optional*) - if `True`, uses the shared-memory
IPC backend instead of TCP sockets. Defaults to `False`.
- **verbose** (*bool**,**optional*) - enables CraftGround's verbose logging.
Defaults to `False`.
- **craftground_kwargs** (*dict**,**optional*) - additional keyword arguments
forwarded verbatim to `craftground.make`.
- ****kwargs** - additional keyword arguments passed to
[`CraftGroundWrapper`](torchrl.envs.CraftGroundWrapper.html#torchrl.envs.CraftGroundWrapper) (e.g. `device`).

Examples

```
>>> from craftground.initial_environment_config import ( 
... InitialEnvironmentConfig, WorldType)
>>> from torchrl.envs.libs.craftground import CraftGroundEnv
>>> env = CraftGroundEnv( 
... initial_env_config=InitialEnvironmentConfig(
... image_width=114, image_height=64,
... world_type=WorldType.SUPERFLAT,
... ),
... port=8023,
... )
>>> td = env.reset() 
>>> td["pixels"].shape 
torch.Size([64, 114, 3])
```
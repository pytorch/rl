# IsaacLabWrapper

torchrl.envs.IsaacLabWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/isaac_lab.html#IsaacLabWrapper)

A wrapper for IsaacLab environments.

Parameters:

- **env** (*isaaclab.envs.ManagerBasedRLEnv**or**equivalent*) - the environment
instance to wrap. `ManagerBasedEnv`, `ManagerBasedRLEnv`,
`DirectRLEnv` and `DirectMARLEnv` are all supported.
- **categorical_action_encoding** (*bool**,**optional*) - if `True`, categorical
specs will be converted to the TorchRL equivalent ([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)),
otherwise a one-hot encoding will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)).
Defaults to `False`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `True`.
- **native_autoreset** (*bool**,**optional*) - if `True`, keeps Isaac Lab's native
auto-reset observations in the collector hot path and avoids the
synthetic reset call in `step_and_maybe_reset()`.
The terminal `"next"` observation remains unavailable and is
marked with `NaN`; the native reset observation is cloned into
the next root observation.
Defaults to `False`.
- **from_tiled_camera** (*bool**,**optional*) - if `True`, reads pixels from an
Isaac Lab tiled camera sensor and writes them under `pixels_key`.
This is the recommended headless rendering path. Defaults to
`False`.
- **tiled_camera_name** (*str**,**optional*) - Name of the sensor in
`env.scene.sensors`. Defaults to `"tiled_camera"`.
- **tiled_camera_data_type** (*str**,**optional*) - Camera data type to read.
Defaults to `"rgb"`.
- **pixels_key** (*NestedKey**,**optional*) - TensorDict key where pixels are
written. Defaults to `"pixels"`.
- **pixels_dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype used for the output pixels.
If `torch.uint8` is requested and the camera returns floating
point data, values are scaled from `[0, 1]` to `[0, 255]`.
Defaults to `torch.uint8`.
- **pixels_channels** (*int**,**optional*) - Number of channels to keep from the
camera output. Defaults to `3`.

For other arguments, see the [`torchrl.envs.GymWrapper`](torchrl.envs.GymWrapper.html#torchrl.envs.GymWrapper) documentation.

Refer to [the Isaac Lab doc for installation instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

## Per-index reset

Isaac Lab's underlying envs let the caller reset an arbitrary subset of
sub-environments without disturbing the others. This wrapper plumbs that
capability through the standard torchrl `"_reset"` mask: when the
tensordict passed to [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) carries a partial `"_reset"` boolean
mask (i.e. neither all `True` nor all `False`), only the masked
sub-envs are reset and the others keep their state and `episode_length_buf`.
The transform stack (`RewardSum`, `InitTracker`, recurrent primers,
`VecNormV2`, ...) fires on the reset rows only, exactly like a normal
reset.

The per-index reset path is gated on `native_autoreset=True`: with the
default `native_autoreset=False` it would conflict with the
[`VecGymEnvTransform`](torchrl.envs.transforms.VecGymEnvTransform.html#torchrl.envs.transforms.VecGymEnvTransform)-based obs-swap path
that `step_and_maybe_reset()` triggers on every
"done" row (this would double-reset the affected envs). Set
`native_autoreset=True` if you want partial-reset semantics.

## State-based reset

For deterministic branching from a snapshot, snapshot the scene with
`get_state()` and restore it with `env.reset(td, set_state=True,
scene_state=snapshot)` (or the `reset_to_state()` convenience, which
routes through the same path). Manager-based envs only; both work in
conjunction with the partial `"_reset"` mask.

Note

The snapshot is passed as the `scene_state` *keyword argument*
rather than inside the tensordict. For a stateless env the reset state is
torch-native (tensordict / `torch.Tensor` entries declared in
`state_spec`), so it lives naturally in the tensordict. Isaac Lab's
scene state is an opaque, non-torch-native object: carrying it in the
tensordict would mean wrapping it as a `NonTensor` `state_spec` entry
and threading it through the transform and step-MDP machinery, whereas a
kwarg is simpler and keeps simulator state out of the data path.

Example

```
>>> # This code block ensures that the Isaac app is started in headless mode
>>> from scripts_isaaclab.app import AppLauncher
>>> import argparse
```

```
>>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
>>> AppLauncher.add_app_launcher_args(parser)
>>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
>>> app_launcher = AppLauncher(args_cli)
```

```
>>> # Imports and env
>>> import gymnasium as gym
>>> import isaaclab_tasks # noqa: F401
>>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
>>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
```

```
>>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
>>> env = IsaacLabWrapper(env)
```
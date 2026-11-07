# GenesisWrapper

torchrl.envs.GenesisWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/genesis.html#GenesisWrapper)

TorchRL wrapper around a Genesis physics scene.

Genesis is a torch-native physics engine for general-purpose robotics
and embodied AI. This wrapper keeps tensors on-device end-to-end: no
numpy round-trips, no gym-style shims.

Customization is done by subclassing and overriding the hooks below;
`GenesisWrapper(scene)` works out of the box on any built scene with
sensible defaults, so it's still useful for one-off experiments.

Subclass hooks (all optional):

- `_make_obs()` - read observations off `self._scene` and return a
`dict[str, Tensor]` (or a single `Tensor`). Default: for each entity
with DoFs, emit `{entity.name}_qpos` and `{entity.name}_qvel` via
`get_dofs_position()` / `get_dofs_velocity()`.
- `_apply_action()` - push the agent's action into the scene before
`scene.step()` is called. Default: feed `action[..., :n_dofs]` of
the first actuated entity as a position target via
`control_dofs_position()`.
- `_compute_reward()` - compute the per-step reward. Called once per
physics substep when `frame_skip > 1` and the results are summed.
Default: `0`.
- `_compute_done()` - compute the truncation flag. Default:
`self._current_step >= self._max_steps`.

Parameters:

- **scene** (*gs.Scene*) - a pre-built Genesis scene.
- **max_steps** (*int**,**optional*) - truncation horizon consulted by the default
`_compute_done()`. Defaults to `1000`.
- **frame_skip** (*int**,**optional*) - physics steps per env step. Defaults to `1`.
- **from_pixels** (*bool**,**optional*) - if `True`, render a `pixels` entry
into the observation using the first camera on the scene.
Cameras must be added via `scene.add_camera()` **before**
`scene.build()` (Genesis cannot add cameras post-build);
a missing camera raises `ValueError`. Defaults to `False`.
- **pixels_key** (*str**,**optional*) - key under which the rendered frame is
stored in the returned tensordict. Defaults to `"pixels"`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - torch device for returned tensors.
When `None` (the default), inferred from `gs.device` -- i.e.
the device Genesis itself is running on. This avoids a silent
per-step copy when the sim is on GPU but the wrapper was
defaulting to CPU.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - batch size for the env. If omitted,
inferred from `scene.n_envs` (`()` for non-batched scenes,
`(n_envs,)` for batched scenes). If provided, validated against
`scene.n_envs`; a mismatch raises `ValueError`.
- **allow_done_after_reset** (*bool**,**optional*) - passed through to
[`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase). Defaults to `False`.

Examples

Out-of-the-box use on an arbitrary scene:

```
>>> import genesis as gs
>>> from torchrl.envs import GenesisWrapper
>>> gs.init(backend=gs.cpu)
>>> scene = gs.Scene(show_viewer=False)
>>> scene.add_entity(gs.morphs.Plane())
>>> scene.add_entity(
... gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
... )
>>> scene.build()
>>> env = GenesisWrapper(scene)
>>> td = env.rollout(10)
```

Subclassing for a custom task:

```
>>> class ReachEnv(GenesisWrapper):
... def __init__(self, scene, target, **kwargs):
... self._franka = scene.entities[1] # set before super().__init__
... self._target = target
... super().__init__(scene, **kwargs)
... def _make_obs(self):
... return {"qpos": self._franka.get_dofs_position()}
... def _apply_action(self, action):
... self._franka.control_dofs_position(action)
... def _compute_reward(self, action):
... ee = self._franka.get_link("hand").get_pos()
... return -(ee - self._target).norm()
```
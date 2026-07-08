# RenderConfig

*class*torchrl.render.RenderConfig(*ckpt: str | ~pathlib.Path, policy: str | ~collections.abc.Callable[[...], ~typing.Any], env: str | ~collections.abc.Callable[[...], ~typing.Any], num_trajs: int = 1, format: ~typing.Literal['ipynb', 'mp4', 'gif', 'frames', 'npz', 'jsonl'] = 'mp4', out: str | ~pathlib.Path | None = None, max_steps: int | None = None, fps: float = 30.0, camera: list[str] = <factory>, camera_layout: ~typing.Literal['single', 'grid', 'horizontal', 'vertical', 'separate'] = 'single', deterministic: bool = True, exploration_mode: ~typing.Literal['deterministic', 'mode', 'mean', 'random'] | None = None, seed: int | None = 0, device: ~torch.device | str = 'cpu', policy_device: ~torch.device | str | None = None, env_device: ~torch.device | str | None = None, render_backend: ~typing.Literal['auto', 'pixels', 'env', 'null'] = 'auto', notebook_render_backend: ~typing.Literal['auto', 'static', 'mujoco_wasm', 'mujoco-wasm'] = 'auto', notebook_rollout_mode: ~typing.Literal['saved', 'live', 'both'] = 'saved', env_backend: ~typing.Literal['auto', 'torchrl', 'gym', 'gymnasium', 'mujoco', 'dm_control', 'isaaclab'] = 'auto', env_kwargs: dict[str, ~typing.Any] = <factory>, policy_kwargs: dict[str, ~typing.Any] = <factory>, checkpoint_key: str | None = None, state_dict_key: str | None = None, strict_load: bool = True, auto_load_policy: bool = True, policy_eval: bool = True, obs_key: ~tensordict._nestedkey.NestedKey = 'observation', action_key: ~tensordict._nestedkey.NestedKey = 'action', done_key: ~tensordict._nestedkey.NestedKey = 'done', reward_key: ~tensordict._nestedkey.NestedKey = 'reward', pixel_key: ~tensordict._nestedkey.NestedKey = 'pixels', from_pixels: bool = False, pixels_only: bool = False, render_mode: str | None = None, save_rollout: bool = False, save_tensordicts: bool = False, save_frames: bool = False, frame_dir: str | ~pathlib.Path | None = None, artifact_dir: str | ~pathlib.Path | None = None, metadata: str | ~pathlib.Path | None = None, overwrite: bool = False, video_codec: str | None = None, mujoco_model_path: str | ~pathlib.Path | None = None, mujoco_asset_paths: list[str | ~pathlib.Path] = <factory>, mujoco_qpos_key: ~tensordict._nestedkey.NestedKey | None = None, notebook_viewer_port: int = 5178, dry_run: bool = False, validate_only: bool = False*)[[source]](../../_modules/torchrl/render/config.html#RenderConfig)

Configuration for rendering policy rollouts.

Parameters:

- **ckpt** - Local checkpoint path passed to the policy factory.
- **policy** - Policy factory or `"module:object"` import specification.
- **env** - Environment factory or `"module:object"` import specification.
- **num_trajs** - Number of trajectories to render.
- **format** - Artifact format to write.
- **notebook_rollout_mode** - For `format="ipynb"`, whether rollouts are
collected before notebook creation (`"saved"`), inside notebook
cells (`"live"`), or both (`"both"`).

Examples

```
>>> from torchrl.render import RenderConfig
>>> cfg = RenderConfig(
... ckpt="policy.pt",
... policy="project.policy:make_policy",
... env="project.env:make_env",
... max_steps=10,
... )
>>> cfg.num_trajs
1
```

to_dict() → dict[str, Any][[source]](../../_modules/torchrl/render/config.html#RenderConfig.to_dict)

Returns a JSON-serializable dictionary representation.

to_json(***kwargs: Any*) → str[[source]](../../_modules/torchrl/render/config.html#RenderConfig.to_json)

Returns this configuration as formatted JSON.
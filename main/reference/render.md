# Rendering applications

`torchrl.render` provides reusable utilities behind the `rlrender` and
`torchrl-render` commands. The command-line entry point imports trusted user
policy and environment factories, loads a local checkpoint, collects one or more
rollouts, captures RGB frames from TensorDict pixels or `env.render()`, and
writes a reproducible artifact.

Notebook artifacts can also include an optional MuJoCo WASM sidecar viewer. In
that mode, the notebook imports helper functions from `torchrl.render` to
start a local Vite viewer, load an MJCF scene in browser-side MuJoCo, and stream
qpos trajectories into the live iframe. When the environment exposes native
MuJoCo state or wraps a Gymnasium MuJoCo environment,
[`MujocoStateReader`](generated/torchrl.render.backends.MujocoStateReader.html#torchrl.render.backends.MujocoStateReader) records qpos directly from
the simulator rather than deriving it from policy observations. By default,
`rlrender` collects and saves trajectories before writing the notebook. Use
`--notebook-rollout-mode live` to write a notebook that constructs the policy
and environment inside the kernel and generates trajectories when its cells
are run. Use `--notebook-rollout-mode both` to save collected rollouts and
also keep an in-notebook collection cell. The generated notebook should stay
thin: reusable display, playback, rollout, and acknowledgement helpers live in
TorchRL rather than being copied into each notebook.

The MuJoCo WASM viewer requires Node.js and either `npm` or `pnpm`. The
viewer installs the generated Vite project's pinned JavaScript dependencies
when `node_modules` is absent, which requires network access. The generated
`node_modules` directory is reused when present.

Factories can be addressed as `module.submodule:callable` or as a local file
path such as `/path/to/render_factories.py:make_env`. The base TorchRL package
does not install video or image encoders for this feature. Use the optional
rendering dependencies when writing MP4, GIF, PNG, or YAML-backed configs:

```
uv run --extra rendering rlrender --help
```

## Core API

| [`render_policy`](generated/torchrl.render.render_policy.html#torchrl.render.render_policy)(config) | Renders a policy according to `config` and writes the requested artifact. |
| --- | --- |
| [`make_render_env`](generated/torchrl.render.make_render_env.html#torchrl.render.make_render_env)(config[, checkpoint]) | Builds and prepares an environment for rendering. |
| [`load_render_policy`](generated/torchrl.render.load_render_policy.html#torchrl.render.load_render_policy)(config[, env, ...]) | Builds and prepares a policy for rendering. |
| [`collect_render_rollouts`](generated/torchrl.render.collect_render_rollouts.html#torchrl.render.collect_render_rollouts)(env, policy, config) | Collects sequential render rollouts. |
| [`write_render_artifact`](generated/torchrl.render.write_render_artifact.html#torchrl.render.write_render_artifact)(result, config) | Writes the configured render artifact and sidecar metadata. |
| [`import_from_string`](generated/torchrl.render.import_from_string.html#torchrl.render.import_from_string)(spec) | Imports an object from a `"module:attribute"` string. |
| [`call_with_supported_kwargs`](generated/torchrl.render.call_with_supported_kwargs.html#torchrl.render.call_with_supported_kwargs)(factory, ...) | Calls a user factory with a spec object or supported keyword arguments. |
| [`load_checkpoint`](generated/torchrl.render.load_checkpoint.html#torchrl.render.load_checkpoint)(path[, map_location, ...]) | Loads a local PyTorch checkpoint. |
| [`save_render_checkpoint`](generated/torchrl.render.save_render_checkpoint.html#torchrl.render.save_render_checkpoint)(path, model, *[, ...]) | Writes a checkpoint in the layout expected by rlrender factories. |
| [`checkpoint_hash`](generated/torchrl.render.checkpoint_hash.html#torchrl.render.checkpoint_hash)(path) | Compute a SHA256 digest over all checkpoint bytes. |
| [`infer_state_dict`](generated/torchrl.render.infer_state_dict.html#torchrl.render.infer_state_dict)(payload[, key]) | Infers a model state dict from common checkpoint payload layouts. |
| [`parse_nested_key`](generated/torchrl.render.parse_nested_key.html#torchrl.render.parse_nested_key)(value) | Parses dotted strings into TensorDict nested keys. |
| [`key_to_string`](generated/torchrl.render.key_to_string.html#torchrl.render.key_to_string)(key) | Formats a TensorDict nested key for config and metadata output. |
| [`normalize_policy`](generated/torchrl.render.normalize_policy.html#torchrl.render.normalize_policy)(policy, config) | Normalizes a policy into a TensorDict-compatible callable. |
| [`add_step_counter`](generated/torchrl.render.add_step_counter.html#torchrl.render.add_step_counter)(env, max_steps) | Adds a [`StepCounter`](generated/torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) when supported. |
| [`seed_env`](generated/torchrl.render.seed_env.html#torchrl.render.seed_env)(env, seed) | Seeds an environment if it exposes a known seed method. |
| [`normalize_env`](generated/torchrl.render.normalize_env.html#torchrl.render.normalize_env)(env, config) | Normalizes external environments into TorchRL wrappers when feasible. |
| [`write_mujoco_wasm_viewer`](generated/torchrl.render.write_mujoco_wasm_viewer.html#torchrl.render.write_mujoco_wasm_viewer)(output_dir, ...[, ...]) | Writes a local Vite viewer for MuJoCo WASM notebook rendering. |
| [`display_mujoco_wasm_viewer`](generated/torchrl.render.display_mujoco_wasm_viewer.html#torchrl.render.display_mujoco_wasm_viewer)(viewer_dir, *[, ...]) | Starts and displays a generated MuJoCo WASM viewer in a notebook. |
| [`send_mujoco_wasm_qpos`](generated/torchrl.render.send_mujoco_wasm_qpos.html#torchrl.render.send_mujoco_wasm_qpos)(qpos, *[, port, ...]) | Sends one qpos vector to a live MuJoCo WASM notebook viewer. |
| [`play_mujoco_wasm_trajectory`](generated/torchrl.render.play_mujoco_wasm_trajectory.html#torchrl.render.play_mujoco_wasm_trajectory)(qpos, *[, dt, ...]) | Plays a qpos trajectory in a live MuJoCo WASM notebook viewer. |
| [`extract_qpos_trajectory`](generated/torchrl.render.extract_qpos_trajectory.html#torchrl.render.extract_qpos_trajectory)(rollout[, qpos_key]) | Extracts a qpos trajectory from a rollout TensorDict. |

## Configuration and results

| [`RenderConfig`](generated/torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)(ckpt, policy, ~typing.Any], ...) | Configuration for rendering policy rollouts. |
| --- | --- |
| [`RenderEnvSpec`](generated/torchrl.render.RenderEnvSpec.html#torchrl.render.RenderEnvSpec)(device, seed, max_steps, ...) | Context object passed to environment factories. |
| [`RenderPolicySpec`](generated/torchrl.render.RenderPolicySpec.html#torchrl.render.RenderPolicySpec)(ckpt_path, checkpoint, ...) | Context object passed to policy factories. |
| [`RenderResult`](generated/torchrl.render.RenderResult.html#torchrl.render.RenderResult)(artifact_path, trajectories, ...) | Result returned by [`torchrl.render.render_policy()`](generated/torchrl.render.render_policy.html#torchrl.render.render_policy). |
| [`FrameBundle`](generated/torchrl.render.FrameBundle.html#torchrl.render.FrameBundle)(frames, ~typing.Any], step, ...) | One rendered step containing one or more named camera frames. |
| [`TensorDictPolicyAdapter`](generated/torchrl.render.TensorDictPolicyAdapter.html#torchrl.render.TensorDictPolicyAdapter)(policy, obs_key, ...) | Adapts plain tensor policies to a TensorDict policy callable. |

## Backends

| [`RenderBackend`](generated/torchrl.render.backends.RenderBackend.html#torchrl.render.backends.RenderBackend)(*args, **kwargs) | Protocol implemented by rlrender frame-capture backends. |
| --- | --- |
| [`MujocoStateReader`](generated/torchrl.render.backends.MujocoStateReader.html#torchrl.render.backends.MujocoStateReader)() | Reads simulator state from TorchRL-native and Gym MuJoCo environments. |
| [`TensorDictPixelsBackend`](generated/torchrl.render.backends.TensorDictPixelsBackend.html#torchrl.render.backends.TensorDictPixelsBackend)() | Captures frames from TensorDict pixel entries. |
| [`EnvRenderBackend`](generated/torchrl.render.backends.EnvRenderBackend.html#torchrl.render.backends.EnvRenderBackend)() | Captures frames by calling `env.render()`. |
| [`NullRenderBackend`](generated/torchrl.render.backends.NullRenderBackend.html#torchrl.render.backends.NullRenderBackend)() | Fallback backend used when no RGB renderer is available. |

## Lower-level helpers

| [`build_parser`](generated/torchrl.render.cli.build_parser.html#torchrl.render.cli.build_parser)() | Builds the rlrender command-line parser. |
| --- | --- |
| [`config_from_args`](generated/torchrl.render.cli.config_from_args.html#torchrl.render.cli.config_from_args)(args) | Constructs a [`RenderConfig`](generated/torchrl.render.RenderConfig.html#torchrl.render.RenderConfig) from parsed CLI args. |
| [`main`](generated/torchrl.render.cli.main.html#torchrl.render.cli.main)([argv]) | Entry point for `rlrender` and `torchrl-render`. |

| [`build_notebook`](generated/torchrl.render.notebook.build_notebook.html#torchrl.render.notebook.build_notebook)(result, config) | Builds a minimal reproducible render report notebook. |
| --- | --- |
| [`write_render_notebook`](generated/torchrl.render.notebook.write_render_notebook.html#torchrl.render.notebook.write_render_notebook)(result, config, path) | Writes a render report notebook as plain ipynb JSON. |

| [`normalize_frame`](generated/torchrl.render.video.normalize_frame.html#torchrl.render.video.normalize_frame)(frame) | Converts a tensor-like image into an `H x W x 3` uint8 array. |
| --- | --- |
| [`normalize_frame_output`](generated/torchrl.render.video.normalize_frame_output.html#torchrl.render.video.normalize_frame_output)(output) | Normalizes renderer output into named RGB frames. |
| [`compose_frame_grid`](generated/torchrl.render.video.compose_frame_grid.html#torchrl.render.video.compose_frame_grid)(frames[, layout]) | Composes multiple frames into one RGB image. |
| [`encode_video`](generated/torchrl.render.video.encode_video.html#torchrl.render.video.encode_video)(frames, path, fps, *[, video_codec]) | Encodes RGB frames as an MP4 using TorchRL's torchcodec writer. |
| [`encode_gif`](generated/torchrl.render.video.encode_gif.html#torchrl.render.video.encode_gif)(frames, path, fps) | Encodes RGB frames as an animated GIF using Pillow. |
| [`write_png`](generated/torchrl.render.video.write_png.html#torchrl.render.video.write_png)(frame, path) | Writes one RGB frame as a PNG file using Pillow. |
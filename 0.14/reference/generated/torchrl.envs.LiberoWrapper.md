# LiberoWrapper

torchrl.envs.LiberoWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/libero.html#LiberoWrapper)

LIBERO environment wrapper.

GitHub: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

Paper: [https://arxiv.org/abs/2306.03310](https://arxiv.org/abs/2306.03310) (LIBERO: Benchmarking Knowledge
Transfer for Lifelong Robot Learning, Liu et al., 2023)

LIBERO is a benchmark of language-conditioned tabletop manipulation
tasks (robosuite/MuJoCo based) widely used to evaluate and fine-tune
Vision-Language-Action policies (e.g. OpenVLA, SimpleVLA-RL). This
wrapper exposes a LIBERO environment with the canonical VLA TensorDict
schema (see [the VLA reference page](../vla.html#ref-vla)): a `uint8`
`[C, H, W]` camera `image` (and optionally a `wrist_image`) plus a
proprioceptive `state` under `observation`, the
`language_instruction` string at the root, and a boolean `success`
entry. Episodes terminate on success and truncate after
`max_episode_steps`; the reward is the binary success indicator.

Initial-state control is first-class: LIBERO ships a fixed set of
initial simulator states per task (50 for the standard evaluation
protocol), and the rollout-grouping mechanism GRPO-style group
advantages require (n rollouts per initial state) is built in via
`group_repeats` (see [`MCAdvantage`](torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage)).

Note

Rendering is offscreen (no display needed). On headless Linux set
`MUJOCO_GL=egl` (or `osmesa`); on macOS the default (`cgl`)
works. One environment instance hosts one MuJoCo simulation: use
[`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) or an
[`AsyncEnvPool`](torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool) with one instance per process
for parallel collection. With EGL, [`LiberoEnv`](torchrl.envs.LiberoEnv.html#torchrl.envs.LiberoEnv)
accepts a `render_gpu_device_id` argument to pin robosuite rendering
to an EGL-visible GPU device.

Note

Importing `libero` for the first time normally prompts on stdin
for a dataset path. This wrapper pre-seeds the default LIBERO config
(honoring `LIBERO_CONFIG_PATH`) so that headless workers never
block on the prompt.

Parameters:

**env** (`libero.libero.envs.OffScreenRenderEnv`) - the environment to
wrap.

Keyword Arguments:

- **instruction** (*str**,**optional*) - the language instruction. Defaults to
the instruction parsed from the task's BDDL file.
- **init_states** (*np.ndarray**,**optional*) - a `[N, D]` array of initial
simulator states to reset to (as returned by the LIBERO
benchmark for each task). Required for `init_state_mode` /
`group_repeats`. Defaults to `None` (native random resets).
- **init_state_mode** (*str**,**optional*) - how the initial state is selected
at reset when `init_states` is provided. One of
`"random"` (sample uniformly), `"cycle"` (walk through the
init states in order - the fixed-50-trials evaluation protocol)
or `"fixed"` (always `init_state_id`). Defaults to
`"random"`.
- **init_state_id** (*int**,**optional*) - the initial state used with
`init_state_mode="fixed"`. Defaults to `0`.
- **group_repeats** (*int**,**optional*) - grouped-rollout mode: the same
initial state is replayed for `group_repeats` consecutive
episodes (one GRPO group) before the next one is selected, and
an integer `group_id` observation entry identifies the group.
Defaults to `None`.
- **group_id_offset** (*int**,**optional*) - added to the group ids. Group ids
must be unique across concurrently collecting workers for group
advantages to be computed correctly: give each worker a disjoint
offset (e.g. `worker_idx * 10**6`). Defaults to `0`.
- **group_id_mode** (*str**,**optional*) - how grouped-rollout ids are stamped.
`"episode"` uses the local group counter
(`episode_count // group_repeats`). `"init_state"` uses the
selected initial-state id and requires `init_state_mode` to be
`"cycle"` or `"fixed"`. The latter is useful when several
parallel workers share the same task and should form GRPO groups
by initial state even if their episode lengths differ. Defaults to
`"episode"`.
- **camera** (*str**,**optional*) - the camera rendered under
`("observation", "image")`. Defaults to `"agentview"`.
- **wrist_camera** (*str**,**optional*) - if provided (e.g.
`"robot0_eye_in_hand"`), this camera is exposed under
`("observation", "wrist_image")`. Defaults to `None`.
- **from_pixels** (*bool**,**optional*) - if `True`, expose the rendered
`camera` frame as a root `pixels` entry too (HWC `uint8`),
in addition to the CHW `("observation", "image")` the policy
consumes. This is the torchrl pixels-rendering convention and
feeds [`VideoRecorder`](torchrl.record.VideoRecorder.html#torchrl.record.VideoRecorder) (default
`in_keys=["pixels"]`) with no extra render cost. Defaults to
`False`.
- **flip_image** (*bool**,**optional*) - rotate the rendered images by 180
degrees. LIBERO renders upside down relative to the standard
human-view orientation used by the OpenVLA / SimpleVLA-RL
pipelines. Defaults to `True`.
- **proprio_keys** (*tuple**of**str**,**optional*) - the low-dimensional observation
entries concatenated (in order) into `("observation", "state")`.
Defaults to `("robot0_eef_pos", "robot0_eef_quat",
"robot0_gripper_qpos")`.
- **max_episode_steps** (*int**,**optional*) - truncation horizon, in env steps.
Defaults to `512` (the SimpleVLA-RL setting).
- **settle_steps** (*int**,**optional*) - no-op steps (zero delta, gripper open)
executed after setting the initial state so the simulation
settles, as in the OpenVLA evaluation protocol. Defaults to
`10`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device of the output tensordicts.

Variables:

**available_envs** (*list**of**str*) - the available task-suite names.

Examples

```
>>> import os
>>> from libero.libero import benchmark, get_libero_path
>>> from libero.libero.envs import OffScreenRenderEnv
>>> from torchrl.envs import LiberoWrapper
>>> suite = benchmark.get_benchmark_dict()["libero_spatial"]()
>>> task = suite.get_task(0)
>>> env = OffScreenRenderEnv(
... bddl_file_name=os.path.join(
... get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
... ),
... camera_heights=256,
... camera_widths=256,
... )
>>> env = LiberoWrapper(env, instruction=task.language)
>>> td = env.reset()
>>> td["observation", "image"].shape
torch.Size([3, 256, 256])
```
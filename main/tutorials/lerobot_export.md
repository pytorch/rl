Note

Go to the end
to download the full example code.

# Exporting Rollout data to LeRobot

**Author**: [TorchRL contributors](https://github.com/pytorch/rl)

An RL policy learns from the data it collects while interacting with an
environment. During online training, a replay buffer stores those transitions
so the policy can sample them again. We may also want to keep that experience
after collection. It can help us inspect a policy's behavior, share recordings,
or use the data for offline RL or imitation learning.

TorchRL replay buffers provide an export API for moving stored experience into
a dataset. A dataset adapter maps TensorDict fields to the destination format
and writes the data. This separates the way we collect and store experience
from the details of each dataset format.

TorchRL provides built-in adapters for several common dataset formats,
including LeRobot. LeRobot provides a common format for robotics datasets. It
is designed to store large collections from real-world robot runs efficiently,
and its integration with the Hugging Face Hub makes those datasets easy to
share and use with LeRobot tools.

In this tutorial, we'll collect interactions with a MicroDuck simulation
environment into a replay buffer, as in online RL. We'll export the buffer's
contents to LeRobot, then load a batch back into TorchRL as a TensorDict. An
upcoming tutorial will show how a LeRobot dataset can seed a replay buffer, so
an RL run can start learning from already-collected data.

## What you will learn

- Collect robot observations, actions and camera frames with a TorchRL collector.
- See how episode and step counters keep trajectories in order in a replay buffer.
- Describe the data to export and save it as a LeRobot dataset.
- Inspect the recording and load it back into TorchRL for further use.

Install TorchRL with the `lerobot` extra, plus `mujoco` and `ipython` to run
this tutorial. The `lerobot` extra requires Python 3.12 or newer.

```
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torchrl
from huggingface_hub import HfApi
from IPython.display import Video
from torch.utils.data import DataLoader
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, tensordict_collate
from torchrl.data.datasets import LeRobotDatasetAdapter
from torchrl.envs import Compose, MicroDuckEnv, StepCounter, TrajCounter, TransformedEnv

# Reuse the example's TensorDict policy. This path also works when Sphinx copies
# this tutorial into docs/source/reference/generated/tutorials.
EXAMPLE_DIR = Path(torchrl.__file__).resolve().parents[1] / "examples" / "microduck"
sys.path.insert(0, str(EXAMPLE_DIR))
from heuristic_gait import MicroDuckGaitActor # noqa: E402
```

```
/__w/_temp/lerobot-gallery-XfmHnm/repo/torchrl/_utils.py:132: DeprecationWarning: TorchRL support for torch<2.13 (found torch 2.11.0+cpu) is deprecated and will be removed in v0.17. torch 2.11 and 2.12 remain supported until then for stacks that pin them, such as Isaac Lab 3.0. Upgrade to torch>=2.13 to silence this warning.
 warnings.warn(
```

## A complete example

Let's start by running the complete path once. This short example collects a
MicroDuck episode into a replay buffer, exports it as a LeRobot dataset, and
displays the recorded video. We will come back to each part below and see how
the collector, replay buffer, and dataset adapter fit together.

```
TUTORIAL_FAST = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
QUICK_STEPS = 100 if TUTORIAL_FAST else 250
QUICK_WIDTH, QUICK_HEIGHT = 320, 240
quick_policy = MicroDuckGaitActor()
quick_base_env = MicroDuckEnv(
 download=True,
 backend="mujoco",
 tasks=MicroDuckEnv.tracking_task(0.03, **quick_policy.config.task_kwargs()),
 action_scale=0.35,
 from_pixels=True,
 render_width=QUICK_WIDTH,
 render_height=QUICK_HEIGHT,
 camera_id=-1,
 seed=0,
)
quick_env = TransformedEnv(
 quick_base_env,
 Compose(
 StepCounter(max_steps=QUICK_STEPS, step_count_key="frame"),
 TrajCounter(out_key="episode"),
 ),
)
quick_collector = Collector(
 quick_env,
 quick_policy,
 frames_per_batch=QUICK_STEPS,
 total_frames=QUICK_STEPS,
)
try:
 quick_batch = next(iter(quick_collector))
finally:
 quick_collector.shutdown()

quick_replay = ReplayBuffer(storage=LazyTensorStorage(QUICK_STEPS), batch_size=8)
quick_replay.extend(
 quick_batch.select("pixels", "observation", "action", "episode", "frame").reshape(
 -1
 )
)
quick_features = {
 "observation.state": {
 "dtype": "float32",
 "shape": (quick_batch["observation"].shape[-1],),
 },
 "action": {
 "dtype": "float32",
 "shape": (MicroDuckEnv.NUM_JOINTS,),
 "names": list(MicroDuckEnv.JOINT_NAMES),
 },
 "observation.images.front": {
 "dtype": "video",
 "shape": (3, QUICK_HEIGHT, QUICK_WIDTH),
 },
}
quick_adapter = LeRobotDatasetAdapter(
 repo_id="local/microduck-quick-start",
 root=Path(tempfile.mkdtemp(prefix="torchrl-microduck-quick-")) / "dataset",
 fps=round(1 / quick_base_env.dt),
 features=quick_features,
 key_map={
 "observation.state": "observation",
 "observation.images.front": "pixels",
 },
 task="Walk forward slowly while keeping balance.",
 robot_type="microduck",
 batch_size=32,
)
quick_dataset = quick_replay.export(
 quick_adapter, trajectory_key="episode", step_key="frame"
)
quick_video_paths = sorted(quick_dataset.root.rglob("*.mp4"))
assert quick_video_paths
Video(filename=str(quick_video_paths[0]), embed=True, width=QUICK_WIDTH)
```

```
2026-11-06 04:04:20,401 [torchrl][INFO] Downloading the MicroDuck assets from https://github.com/pollen-robotics/microduck_rl/archive/d424a0c899f6b33cbd3daeb279913134349c0b63.zip to /root/.cache/torchrl/microduck/microduck_rl-d424a0c899f6b33cbd3daeb279913134349c0b63 [END]
/__w/_temp/lerobot-gallery-XfmHnm/repo/torchrl/collectors/_single.py:255: FutureWarning: The env passed to Collector is missing transforms required by the policy (InitTracker). From torchrl v0.15 the collector will append them automatically. To enable that behavior now (and silence this warning), pass `auto_register_policy_transforms=True`. To opt out permanently, pass `auto_register_policy_transforms=False`.
 return super().__call__(*args, **kwargs)
2026-11-06 04:04:23,625 [torchrl][INFO] Initialized LazyTensorStorage with torch.Size([250]) shape [END]

Map: 0%| | 0/250 [00:00<?, ? examples/s]
Map: 100%|██████████| 250/250 [00:00<00:00, 2503.18 examples/s]
```

 Your browser does not support the video tag.
 

Now let's look at the same process one piece at a time. We will collect a
longer recording below so we can inspect how episode IDs, replay entries, and
the LeRobot schema are built.

## Setting up the collection

A normal run collects 500 frames at 640 by 480, which makes a 10-second video
at 50 fps. With no falls, this gives two 250-step episodes. A fall can split
the same frame budget across more episodes. In docs CI,
`TORCHRL_TUTORIALS_FAST=1` collects 300 frames across three 100-step
episodes. Each run writes a new dataset because the LeRobot writer expects a
new destination. To keep the result at a known location, set
`TORCHRL_LEROBOT_OUTPUT` to a new directory before running the tutorial.

```
EPISODE_STEPS = 100 if TUTORIAL_FAST else 250
NUM_EPISODES = 3 if TUTORIAL_FAST else 2
TOTAL_FRAMES = NUM_EPISODES * EPISODE_STEPS
WIDTH, HEIGHT = (320, 240) if TUTORIAL_FAST else (640, 480)
REPO_ID = os.environ.get("TORCHRL_LEROBOT_REPO_ID", "local/microduck-walk")
PRIVATE_DATASET = True
OUTPUT_ROOT = (
 Path(os.environ["TORCHRL_LEROBOT_OUTPUT"]).expanduser().absolute()
 if "TORCHRL_LEROBOT_OUTPUT" in os.environ
 else Path(tempfile.mkdtemp(prefix="torchrl-microduck-lerobot-"))
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
```

## Creating the environment and policy

We will use [`MicroDuckEnv`](../reference/generated/torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv), a simulated robot environment.
Alongside its numerical observation, we ask the environment for camera images
with `from_pixels=True`. These images are stored under the `"pixels"` key.

The walking controller comes from the MicroDuck example. It is a TensorDict
module, so a collector can use it as a policy in the same way it would use a
learned policy.

```
policy = MicroDuckGaitActor()
base_env = MicroDuckEnv(
 download=True,
 backend="mujoco",
 tasks=MicroDuckEnv.tracking_task(0.03, **policy.config.task_kwargs()),
 action_scale=0.35,
 from_pixels=True,
 render_width=WIDTH,
 render_height=HEIGHT,
 camera_id=-1,
 seed=0,
)
```

## Keeping track of episodes

LeRobot stores data as episodes, with frames in time order. A collector,
however, returns batches of a fixed size: a batch can contain the end of one
episode and the beginning of another. If we collect from several environments,
their transitions may also be interleaved. This example uses one environment,
but the same question arises: how can we recover the episodes after putting
those transitions in a replay buffer?

We will add two entries to each transition with environment transforms:
`TrajCounter` gives an `"episode"` identifier to each
episode, and `StepCounter` records the step since reset
under `"frame"`. Its `max_steps` argument also ends an episode at the
chosen limit by setting `("next", "truncated")`. An episode may end earlier
if the robot falls.

Later, we will pass these two keys to `ReplayBuffer.export` to group the
transitions by episode and order the frames within each group.

```
env = TransformedEnv(
 base_env,
 Compose(
 StepCounter(max_steps=EPISODE_STEPS, step_count_key="frame"),
 TrajCounter(out_key="episode"),
 ),
)
# LeRobot uses a frame rate to give the recorded data timestamps. We derive it
# from the simulation's control period so video and numerical data stay aligned.
fps = round(1 / base_env.dt)
assert abs(fps * base_env.dt - 1) < 1e-6
```

## Collecting data in a replay buffer

A [`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) runs the policy in the environment
and yields batches of transitions. In an off-policy training loop, we might
add each batch to a replay buffer, sample from the buffer and use those samples
to update the policy. Here we will use the same collection-and-storage pattern
and export the collected experience once the loop finishes.

A collector batch has one dimension for environments and one for time. Let's
inspect the first batch before putting it in replay. This run has one
environment, so its batch contains one 500-step sequence.

This example holds the whole recording in memory. For a larger collection,
`LazyMemmapStorage` can store replay data on disk instead.

```
collector = Collector(
 env,
 policy,
 frames_per_batch=EPISODE_STEPS,
 total_frames=TOTAL_FRAMES,
)
collector_iterator = iter(collector)
first_batch = next(collector_iterator)
```

```
/__w/_temp/lerobot-gallery-XfmHnm/repo/torchrl/collectors/_single.py:255: FutureWarning: The env passed to Collector is missing transforms required by the policy (InitTracker). From torchrl v0.15 the collector will append them automatically. To enable that behavior now (and silence this warning), pass `auto_register_policy_transforms=True`. To opt out permanently, pass `auto_register_policy_transforms=False`.
 return super().__call__(*args, **kwargs)
```

The first batch has an environment dimension followed by time. The image
shape also shows that the simulator returns each frame in height, width,
channel order.

```
(
 first_batch.batch_size,
 first_batch.keys(include_nested=True, leaves_only=True),
 first_batch["pixels"].shape,
)
```

```
(torch.Size([1, 250]), _TensorDictKeysView(['task_id', 'command', 'frame', 'action', 'observation', 'fallen', 'pixels', 'episode', 'done', 'terminated', 'truncated', ('next', 'observation'), ('next', 'command'), ('next', 'task_id'), ('next', 'fallen'), ('next', 'pixels'), ('next', 'frame'), ('next', 'episode'), ('next', 'reward'), ('next', 'done'), ('next', 'terminated'), ('next', 'truncated'), ('collector', 'traj_ids')],
 include_nested=True,
 leaves_only=True), torch.Size([1, 250, 480, 640, 3]))
```

The replay buffer will store one row per transition. We select the fields we
need for export, dropping the other fields, including the next camera frame,
and flatten the environment and time dimensions into one transition dimension.

```
transition_keys = (
 "pixels",
 "observation",
 "action",
 "episode",
 "frame",
 ("next", "reward"),
 ("next", "done"),
 ("next", "terminated"),
 ("next", "truncated"),
)
transitions = first_batch.select(*transition_keys).reshape(-1)
(
 transitions.batch_size,
 transitions.keys(include_nested=True, leaves_only=True),
 transitions["pixels"].shape,
)
```

```
(torch.Size([250]), _TensorDictKeysView(['pixels', 'observation', 'action', 'episode', 'frame', ('next', 'reward'), ('next', 'done'), ('next', 'terminated'), ('next', 'truncated')],
 include_nested=True,
 leaves_only=True), torch.Size([250, 480, 640, 3]))
```

Add the prepared first batch, then prepare each later batch in the same way.
This is the collection-and-storage pattern used in off-policy training. Here
we export the complete buffer instead of sampling it to update a policy.

```
replay = ReplayBuffer(storage=LazyTensorStorage(TOTAL_FRAMES), batch_size=8)
replay.extend(transitions)
try:
 for batch in collector_iterator:
 replay.extend(batch.select(*transition_keys).reshape(-1))
finally:
 collector.shutdown()

frames = replay[:]
```

```
2026-11-06 04:04:29,067 [torchrl][INFO] Initialized LazyTensorStorage with torch.Size([500]) shape [END]
```

During off-policy training, we would sample batches like this one to update a
policy. We won't train a policy here, but we can still inspect a sample before
exporting the complete contents of the buffer.

```
training_batch = replay.sample()
training_batch.batch_size
```

```
torch.Size([8])
```

## Looking at the episode counters

Let's look at the transition between the first two episodes. Each row below
contains an episode ID followed by its step number. We find the first change
in ID and show the transitions around it: the step number starts again after
the environment resets.

```
episode_and_frame = torch.stack(
 (frames["episode"].reshape(-1), frames["frame"].reshape(-1)), dim=-1
)
next_episode = (episode_and_frame[:, 0] != episode_and_frame[0, 0]).nonzero()[0].item()
episode_and_frame[max(0, next_episode - 2) : next_episode + 2]
```

```
tensor([[ 0, 248],
 [ 0, 249],
 [ 1, 0],
 [ 1, 1]])
```

We can also check how much data we collected. At the default settings, the
buffer has room for all 500 frames. A fall can create an extra episode, but
the collector stops once it has filled that frame budget.

```
recording_summary = {
 "frames": len(replay),
 "episodes": frames["episode"].unique().numel(),
 "fps": fps,
 "mean_reward": frames["next", "reward"].mean().item(),
}
recording_summary
```

```
{'frames': 500, 'episodes': 2, 'fps': 50, 'mean_reward': 0.13022974133491516}
```

## Describing the dataset

The replay buffer now has one row per transition. Let's check the shapes of
the numerical fields before describing them in the dataset schema.

```
frames.batch_size, frames["observation"].shape, frames["action"].shape
```

```
(torch.Size([500]), torch.Size([500, 56]), torch.Size([500, 14]))
```

LeRobot's feature schema gives each saved field a name, shape and dtype. We
map the policy observation to `"observation.state"` and describe the action
and transition signals as numeric features.

```
features = {
 "observation.state": {
 "dtype": "float32",
 "shape": (frames["observation"].shape[-1],),
 },
 "action": {
 "dtype": "float32",
 "shape": (MicroDuckEnv.NUM_JOINTS,),
 "names": list(MicroDuckEnv.JOINT_NAMES),
 },
 "next.reward": {"dtype": "float32", "shape": (1,)},
 "next.done": {"dtype": "bool", "shape": (1,)},
 "next.terminated": {"dtype": "bool", "shape": (1,)},
 "next.truncated": {"dtype": "bool", "shape": (1,)},
}
```

We use `key_map` to connect a LeRobot feature name to the TensorDict key
that holds its data. The camera frames are stored in TorchRL as `"pixels"`.
Their final three dimensions are height, width, channels, while the LeRobot
camera feature is declared in channels, height, width order. The adapter
converts the layout when it writes the video.

```
frames["pixels"].shape[-3:]
```

```
torch.Size([480, 640, 3])
```

```
features["observation.images.front"] = {
 "dtype": "video",
 "shape": (3, HEIGHT, WIDTH),
}
key_map = {
 "observation.state": "observation",
 "observation.images.front": "pixels",
}
(features["observation.images.front"], key_map)
```

```
({'dtype': 'video', 'shape': (3, 480, 640)}, {'observation.state': 'observation', 'observation.images.front': 'pixels'})
```

The action field stores the controller's normalized action vector, not a
measured joint position. MicroDuck scales it by `action_scale` and adds it
to its standing actuator targets. The observation combines robot state with
the tracking command and gait clock. We save the task and controller settings
so these numbers can be interpreted later.

```
recording_configuration = {
 "environment": "MicroDuckEnv",
 "backend": "mujoco",
 "task": "tracking_task",
 "forward_command_m_s": 0.03,
 "action_scale": base_env.action_scale,
 "gait_controller": asdict(policy.config),
 "fps": fps,
 "seed": 0,
}
recording_configuration
```

```
{'environment': 'MicroDuckEnv', 'backend': 'mujoco', 'task': 'tracking_task', 'forward_command_m_s': 0.03, 'action_scale': 0.35, 'gait_controller': {'frequency_hz': 1.8913, 'hip_amplitude': 0.999, 'knee_amplitude': 0.9097, 'ankle_amplitude': 0.0317, 'lateral_amplitude': 0.9584, 'lateral_phase_offset': -0.1624, 'phase_offset': -1.5237, 'pitch_kp': -9.9495, 'pitch_kd': -0.6119, 'ankle_pitch_kp': 10.9934, 'ankle_pitch_kd': 0.6033, 'ramp_duration_s': 0.4}, 'fps': 50, 'seed': 0}
```

```
adapter = LeRobotDatasetAdapter(
 repo_id=REPO_ID,
 root=OUTPUT_ROOT / "dataset",
 fps=fps,
 features=features,
 key_map=key_map,
 task="Walk forward slowly while keeping balance.",
 robot_type="microduck",
 batch_size=32,
)
```

## Exporting the episodes

The `episode` values identify trajectories, and `frame` gives each
transition's position within its episode. We can inspect the episode lengths
before exporting.

```
frames["episode"].unique(return_counts=True)
```

```
(tensor([0, 1]), tensor([250, 250]))
```

`trajectory_key` tells the exporter which transitions belong together, and
`step_key` tells it their order within each episode. The exporter groups by
episode ID and sorts by step, so replay entries do not need to be arranged by
episode. IDs must be unique if multiple workers contribute data. The
`TrajCounter` documentation shows how to share a counter
across worker environments. Each retained episode must have consecutive
steps. A partially retained episode is exported as a fragment; the missing
transitions are not recreated.

```
native = replay.export(adapter, trajectory_key="episode", step_key="frame")
# Keep the task and controller settings with the exported data.
(native.root / "recording_config.json").write_text(
 json.dumps(recording_configuration, indent=2) + "\n"
)
torchrl.torchrl_logger.info("Saved the LeRobot recording to %s", native.root)
loaded = type(native)(repo_id=REPO_ID, root=native.root, video_backend="pyav")
assert loaded.num_frames == len(replay)
```

```
Map: 0%| | 0/250 [00:00<?, ? examples/s]
Map: 76%|███████▋ | 191/250 [00:00<00:00, 1881.55 examples/s]
Map: 100%|██████████| 250/250 [00:00<00:00, 1814.31 examples/s]

Map: 0%| | 0/250 [00:00<?, ? examples/s]
Map: 76%|███████▋ | 191/250 [00:00<00:00, 1892.49 examples/s]
Map: 100%|██████████| 250/250 [00:00<00:00, 1827.58 examples/s]
2026-11-06 04:04:45,141 [torchrl][INFO] Saved the LeRobot recording to /tmp/torchrl-microduck-lerobot-mnot3hgs/dataset [END]
```

## Loading the data back into TorchRL

To use the recordings in a TorchRL pipeline, we can load them back as
TensorDict entries. `from_external` wraps the LeRobot dataset as a PyTorch
dataset. We can use a [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to fetch a batch,
just as we would with another offline dataset. LeRobot decodes the camera
frames as float32 tensors in channels, height, width order, with values between
zero and one.

```
imported = LeRobotDatasetAdapter().from_external(loaded)
batch = next(iter(DataLoader(imported, batch_size=8, collate_fn=tensordict_collate)))
torch.testing.assert_close(batch["action"], frames["action"][:8])
assert batch["observation", "image", "front"].shape == (8, 3, HEIGHT, WIDTH)
batch
```

**TensorDict** batch_size=[8]

- action: f32[14]
- episode: i64[]
- frame: i64[]
- index: i64[]
- language_instruction: NonTensorStack
- next: 4 entries

- done: bool[1]
- reward: f32[1]
- terminated: bool[1]
- truncated: bool[1]
- observation: 2 entries

- image: 1 entry

- front: f32[3,480,640]
- state: f32[56]
- task_index: i64[]
- timestamp: f32[]

Notice that this example saved the current observation and selected
transition signals, but not `("next", "observation")`. An offline RL loss
that needs the next observation will need that field as well, or a way to
construct it from adjacent frames within an episode. Choose the fields to
retain according to the algorithm that will consume the dataset.

## Inspecting the recording

Let's also look at the saved camera video. This is the encoded file from the
exported dataset, so it lets us inspect what was actually written. A video
file can contain more than one episode; LeRobot's episode metadata identifies
where each episode begins and ends.

```
video_paths = sorted(native.root.rglob("*.mp4"))
assert video_paths
Video(filename=str(video_paths[0]), embed=True, width=WIDTH)
```

 Your browser does not support the video tag.
 

## Sharing the dataset

Keeping the dataset locally is enough for this tutorial. If we want someone
else to inspect or reuse it, we can optionally upload it to the Hugging Face
Hub. The LeRobot dataset's `push_to_hub` method uploads the data, metadata
and videos together.

To try this, sign in with `hf auth login` (or set `HF_TOKEN`), choose a
new repository name in your namespace, and run the tutorial with:

```
TORCHRL_LEROBOT_REPO_ID=your-name/microduck-walk \
TORCHRL_LEROBOT_PUSH=1 \
 python tutorials/sphinx-tutorials/lerobot_export.py
```

The example creates a private repository and refuses to overwrite an existing
one. Set `PRIVATE_DATASET = False` above if you want it to be public.

```
if os.environ.get("TORCHRL_LEROBOT_PUSH", "0") == "1":
 if REPO_ID.startswith("local/"):
 raise ValueError(
 "Set TORCHRL_LEROBOT_REPO_ID to your own Hub namespace before uploading."
 )
 HfApi().create_repo(
 repo_id=REPO_ID,
 repo_type="dataset",
 private=PRIVATE_DATASET,
 exist_ok=False,
 )
 native.push_to_hub(
 private=PRIVATE_DATASET, push_videos=True, tags=["torchrl", "microduck"]
 )
 hub_url = f"https://huggingface.co/datasets/{REPO_ID}"
 visualizer_url = (
 "https://huggingface.co/spaces/lerobot/visualize_dataset"
 f"?path=/{REPO_ID}/episode_0"
 )
 torchrl.torchrl_logger.info("Dataset: %s", hub_url)
 torchrl.torchrl_logger.info("Visualizer: %s", visualizer_url)
```

After uploading, open the visualizer URL printed above and sign in to view a
private dataset. It shows camera playback alongside the state and action data
for each episode.

## Conclusion

We started with a policy interacting with an environment, collected its
transitions into replay, and used episode and step counters to export an
ordered LeRobot dataset. We then loaded a batch back into TorchRL and inspected
the recorded video.

We can now review or share the recording, or sample from it in TorchRL with
[`LeRobotExperienceReplay`](../reference/generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay). These datasets can
also be a starting point for offline or imitation learning. The same pattern
works with a learned policy or another environment, provided the saved fields
and episode identifiers match the intended use. When using recordings as
demonstrations, inspect their behavior and suitability for the task first.

## Further reading

- gs_storage for more on collectors and replay buffers.
- [`LeRobotDatasetAdapter`](../reference/generated/torchrl.data.datasets.LeRobotDatasetAdapter.html#torchrl.data.datasets.LeRobotDatasetAdapter) for supported features
and export requirements.
- [`LeRobotExperienceReplay`](../reference/generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay) for trajectory
sampling from LeRobot datasets in TorchRL.
- [LeRobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
and [visualization tools](https://huggingface.co/docs/lerobot/using_dataset_tools).

**Total running time of the script:** (0 minutes 28.191 seconds)

[`Download Jupyter notebook: lerobot_export.ipynb`](../_downloads/61d38028156aeff4fa28fed78de648c2/lerobot_export.ipynb)

[`Download Python source code: lerobot_export.py`](../_downloads/04c83a4a078c3b8f38a4d14315268f60/lerobot_export.py)

[`Download zipped: lerobot_export.zip`](../_downloads/8f9ec502016ad319c76dda843569efc5/lerobot_export.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)
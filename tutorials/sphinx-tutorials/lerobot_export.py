"""
Exporting Rollout data to LeRobot
=================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _lerobot_export_tuto:

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

What you will learn
-------------------

- Collect robot observations, actions and camera frames with a TorchRL collector.
- See how episode and step counters keep trajectories in order in a replay buffer.
- Describe the data to export and save it as a LeRobot dataset.
- Inspect the recording and load it back into TorchRL for further use.

Install TorchRL with the ``lerobot`` extra, plus ``mujoco`` and ``ipython`` to run
this tutorial. The ``lerobot`` extra requires Python 3.12 or newer.
"""

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
from heuristic_gait import MicroDuckGaitActor  # noqa: E402

# %%
# Setting up the collection
# -------------------------
#
# We will collect 1,500 frames, with at most 500 steps per episode. Each run
# writes a new dataset because the LeRobot writer expects a new destination.
# To keep the result at a known location, set ``TORCHRL_LEROBOT_OUTPUT`` to a
# new directory before running the tutorial.

EPISODE_STEPS = 500
TOTAL_FRAMES = 1500
WIDTH, HEIGHT = 320, 240
REPO_ID = os.environ.get("TORCHRL_LEROBOT_REPO_ID", "local/microduck-walk")
PRIVATE_DATASET = True
OUTPUT_ROOT = (
    Path(os.environ["TORCHRL_LEROBOT_OUTPUT"]).expanduser().absolute()
    if "TORCHRL_LEROBOT_OUTPUT" in os.environ
    else Path(tempfile.mkdtemp(prefix="torchrl-microduck-lerobot-"))
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# %%
# Creating the environment and policy
# -----------------------------------
#
# We will use :class:`~torchrl.envs.MicroDuckEnv`, a simulated robot environment.
# Alongside its numerical observation, we ask the environment for camera images
# with ``from_pixels=True``. These images are stored under the ``"pixels"`` key.
#
# The walking controller comes from the MicroDuck example. It is a TensorDict
# module, so a collector can use it as a policy in the same way it would use a
# learned policy.

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

# %%
# Keeping track of episodes
# -------------------------
#
# LeRobot stores data as episodes, with frames in time order. A collector,
# however, returns batches of a fixed size: a batch can contain the end of one
# episode and the beginning of another. If we collect from several environments,
# their transitions may also be interleaved. This example uses one environment,
# but the same question arises: how can we recover the episodes after putting
# those transitions in a replay buffer?
#
# We will add two entries to each transition with environment transforms:
# :class:`~torchrl.envs.TrajCounter` gives an ``"episode"`` identifier to each
# episode, and :class:`~torchrl.envs.StepCounter` records the step since reset
# under ``"frame"``. Its ``max_steps`` argument also ends an episode at the
# chosen limit by setting ``("next", "truncated")``. An episode may end earlier
# if the robot falls.
#
# Later, we will pass these two keys to ``ReplayBuffer.export`` to group the
# transitions by episode and order the frames within each group.

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

# %%
# Collecting data in a replay buffer
# ----------------------------------
#
# A :class:`~torchrl.collectors.Collector` runs the policy in the environment
# and yields batches of transitions. In an off-policy training loop, we might
# add each batch to a replay buffer, sample from the buffer and use those samples
# to update the policy. Here we will use the same collection-and-storage pattern
# and export the collected experience once the loop finishes.
#
# A collector batch has one dimension for environments and one for time. Let's
# inspect the first batch before putting it in replay. This run has one
# environment, so its batch contains one 500-step sequence.
#
# This example holds the whole recording in memory. For a larger collection,
# :class:`~torchrl.data.LazyMemmapStorage` can store replay data on disk instead.

collector = Collector(
    env,
    policy,
    frames_per_batch=EPISODE_STEPS,
    total_frames=TOTAL_FRAMES,
)
collector_iterator = iter(collector)
first_batch = next(collector_iterator)

# %%
# The first batch has an environment dimension followed by time. The image
# shape also shows that the simulator returns each frame in height, width,
# channel order.

(
    first_batch.batch_size,
    first_batch.keys(include_nested=True, leaves_only=True),
    first_batch["pixels"].shape,
)

# %%
# The replay buffer will store one row per transition. We select the fields we
# need for export, dropping the other fields, including the next camera frame,
# and flatten the environment and time dimensions into one transition dimension.

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

# %%
# Add the prepared first batch, then prepare each later batch in the same way.
# This is the collection-and-storage pattern used in off-policy training. Here
# we export the complete buffer instead of sampling it to update a policy.

replay = ReplayBuffer(storage=LazyTensorStorage(TOTAL_FRAMES), batch_size=8)
replay.extend(transitions)
try:
    for batch in collector_iterator:
        replay.extend(batch.select(*transition_keys).reshape(-1))
finally:
    collector.shutdown()

frames = replay[:]

# %%
# During off-policy training, we would sample batches like this one to update a
# policy. We won't train a policy here, but we can still inspect a sample before
# exporting the complete contents of the buffer.

training_batch = replay.sample()
training_batch.batch_size

# %%
# Looking at the episode counters
# -------------------------------
#
# Let's look at the transition between the first two episodes. Each row below
# contains an episode ID followed by its step number. We find the first change
# in ID and show the transitions around it: the step number starts again after
# the environment resets.

episode_and_frame = torch.stack(
    (frames["episode"].reshape(-1), frames["frame"].reshape(-1)), dim=-1
)
next_episode = (episode_and_frame[:, 0] != episode_and_frame[0, 0]).nonzero()[0].item()
episode_and_frame[max(0, next_episode - 2) : next_episode + 2]

# %%
# We can also check how much data we collected. At the default settings, the
# buffer has room for all 1,500 frames. A fall can create an extra episode, as
# the collector resets the environment and continues until it has enough data.

recording_summary = {
    "frames": len(replay),
    "episodes": frames["episode"].unique().numel(),
    "fps": fps,
    "mean_reward": frames["next", "reward"].mean().item(),
}
recording_summary

# %%
# Describing the dataset
# ----------------------
#
# The replay buffer now has one row per transition. Let's check the shapes of
# the numerical fields before describing them in the dataset schema.

frames.batch_size, frames["observation"].shape, frames["action"].shape

# %%
# LeRobot's feature schema gives each saved field a name, shape and dtype. We
# map the policy observation to ``"observation.state"`` and describe the action
# and transition signals as numeric features.

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

# %%
# We use ``key_map`` to connect a LeRobot feature name to the TensorDict key
# that holds its data. The camera frames are stored in TorchRL as ``"pixels"``.
# Their final three dimensions are height, width, channels, while the LeRobot
# camera feature is declared in channels, height, width order. The adapter
# converts the layout when it writes the video.

frames["pixels"].shape[-3:]

# %%

features["observation.images.front"] = {
    "dtype": "video",
    "shape": (3, HEIGHT, WIDTH),
}
key_map = {
    "observation.state": "observation",
    "observation.images.front": "pixels",
}
(features["observation.images.front"], key_map)

# %%
# The action field stores the controller's normalized action vector, not a
# measured joint position. MicroDuck scales it by ``action_scale`` and adds it
# to its standing actuator targets. The observation combines robot state with
# the tracking command and gait clock. We save the task and controller settings
# so these numbers can be interpreted later.

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

# %%

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

# %%
# Exporting the episodes
# ----------------------
#
# The ``episode`` values identify trajectories, and ``frame`` gives each
# transition's position within its episode. We can inspect the episode lengths
# before exporting.

frames["episode"].unique(return_counts=True)

# %%
# ``trajectory_key`` tells the exporter which transitions belong together, and
# ``step_key`` tells it their order within each episode. The exporter groups by
# episode ID and sorts by step, so replay entries do not need to be arranged by
# episode. IDs must be unique if multiple workers contribute data. The
# :class:`~torchrl.envs.TrajCounter` documentation shows how to share a counter
# across worker environments. Each retained episode must have consecutive
# steps. A partially retained episode is exported as a fragment; the missing
# transitions are not recreated.

native = replay.export(adapter, trajectory_key="episode", step_key="frame")
# Keep the task and controller settings with the exported data.
(native.root / "recording_config.json").write_text(
    json.dumps(recording_configuration, indent=2) + "\n"
)
torchrl.torchrl_logger.info("Saved the LeRobot recording to %s", native.root)
loaded = type(native)(repo_id=REPO_ID, root=native.root, video_backend="pyav")
assert loaded.num_frames == len(replay)

# %%
# Loading the data back into TorchRL
# ----------------------------------
#
# To use the recordings in a TorchRL pipeline, we can load them back as
# TensorDict entries. ``from_external`` wraps the LeRobot dataset as a PyTorch
# dataset. We can use a :class:`~torch.utils.data.DataLoader` to fetch a batch,
# just as we would with another offline dataset. LeRobot decodes the camera
# frames as float32 tensors in channels, height, width order, with values between
# zero and one.
imported = LeRobotDatasetAdapter().from_external(loaded)
batch = next(iter(DataLoader(imported, batch_size=8, collate_fn=tensordict_collate)))
torch.testing.assert_close(batch["action"], frames["action"][:8])
assert batch["observation", "image", "front"].shape == (8, 3, HEIGHT, WIDTH)
batch

# %%
# Notice that this example saved the current observation and selected
# transition signals, but not ``("next", "observation")``. An offline RL loss
# that needs the next observation will need that field as well, or a way to
# construct it from adjacent frames within an episode. Choose the fields to
# retain according to the algorithm that will consume the dataset.
#
# Inspecting the recording
# ------------------------
#
# Let's also look at the saved camera video. This is the encoded file from the
# exported dataset, so it lets us inspect what was actually written. A video
# file can contain more than one episode; LeRobot's episode metadata identifies
# where each episode begins and ends.

video_paths = sorted(native.root.rglob("*.mp4"))
assert video_paths
Video(filename=str(video_paths[0]), embed=True, width=WIDTH)

# %%
# Sharing the dataset
# -------------------
#
# Keeping the dataset locally is enough for this tutorial. If we want someone
# else to inspect or reuse it, we can optionally upload it to the Hugging Face
# Hub. The LeRobot dataset's ``push_to_hub`` method uploads the data, metadata
# and videos together.
#
# To try this, sign in with ``hf auth login`` (or set ``HF_TOKEN``), choose a
# new repository name in your namespace, and run the tutorial with::
#
#     TORCHRL_LEROBOT_REPO_ID=your-name/microduck-walk \
#     TORCHRL_LEROBOT_PUSH=1 \
#         python tutorials/sphinx-tutorials/lerobot_export.py
#
# The example creates a private repository and refuses to overwrite an existing
# one. Set ``PRIVATE_DATASET = False`` above if you want it to be public.

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

# %%
# After uploading, open the visualizer URL printed above and sign in to view a
# private dataset. It shows camera playback alongside the state and action data
# for each episode.
#
# Conclusion
# ----------
#
# We started with a policy interacting with an environment, collected its
# transitions into replay, and used episode and step counters to export an
# ordered LeRobot dataset. We then loaded a batch back into TorchRL and inspected
# the recorded video.
#
# We can now review or share the recording, or sample from it in TorchRL with
# :class:`~torchrl.data.datasets.LeRobotExperienceReplay`. These datasets can
# also be a starting point for offline or imitation learning. The same pattern
# works with a learned policy or another environment, provided the saved fields
# and episode identifiers match the intended use. When using recordings as
# demonstrations, inspect their behavior and suitability for the task first.
#
# Further reading
# ---------------
#
# - :ref:`gs_storage` for more on collectors and replay buffers.
# - :class:`~torchrl.data.datasets.LeRobotDatasetAdapter` for supported features
#   and export requirements.
# - :class:`~torchrl.data.datasets.LeRobotExperienceReplay` for trajectory
#   sampling from LeRobot datasets in TorchRL.
# - `LeRobot dataset v3 <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>`_
#   and `visualization tools <https://huggingface.co/docs/lerobot/using_dataset_tools>`_.

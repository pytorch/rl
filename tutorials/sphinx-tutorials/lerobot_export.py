"""
Exporting Rollout data to LeRobot
=================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _lerobot_export_tuto:

An RL policy learns from interactions with an environment. At each step, the
policy receives an observation and chooses an action; the environment then
returns a new observation, a reward and a signal indicating whether the episode
has ended. During training, a replay buffer can hold these transitions so that
an algorithm can sample from them and update the policy.

Sometimes we want to keep a record of those interactions. A recording can help
us inspect a robot's behavior, share data with a teammate, or reuse trajectories
later for offline or imitation learning. For that, we need to save observations
and actions together with their episode order and camera frames in a format
that other tools can read.

`LeRobot <https://github.com/huggingface/lerobot>`_ provides a dataset format
for this kind of robot data, along with tools for inspecting and sharing it.
TorchRL collects transitions in TensorDicts; exporting them to LeRobot gives us
an episode-based dataset with camera video that we can share and later read
back into TorchRL. We will use a fixed walking controller in the MicroDuck
simulator to produce the data, and a replay buffer to store it before export.

What you will learn
-------------------

- Collect robot observations, actions and camera frames with a TorchRL collector.
- See how episode and step counters keep trajectories in order in a replay buffer.
- Describe the data to export and save it as a LeRobot dataset.
- Inspect the recording and load it back into TorchRL for further use.

This tutorial requires Python 3.12 or newer, MuJoCo, IPython, and TorchRL
installed with the ``lerobot`` extra. The MicroDuck assets download on first use.
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
# We will collect 1,500 frames, with at most 500 steps per episode. The shorter
# settings are used when building the documentation. Each run writes a new
# dataset: the LeRobot writer expects a destination that does not exist yet.
# To keep the result at a known location, set ``TORCHRL_LEROBOT_OUTPUT`` to a
# new directory before running the tutorial.

FAST = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
EPISODE_STEPS = 120 if FAST else 500
TOTAL_FRAMES = 240 if FAST else 1500
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
# learned policy. Its gait is tuned for the default action scale of 0.35.

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
# Each MicroDuck batch has an environment dimension and a time dimension. We
# flatten them so that each replay entry is one transition. We keep the current
# observation, action and camera frame, the two counters, and the reward and
# episode-end signals. Keeping only the fields we need also avoids storing a
# second copy of each image under ``"next"``.
#
# This example holds the whole recording in memory. For a larger collection,
# :class:`~torchrl.data.LazyMemmapStorage` can store replay data on disk instead.

replay = ReplayBuffer(storage=LazyTensorStorage(TOTAL_FRAMES), batch_size=8)
collector = Collector(
    env,
    policy,
    frames_per_batch=EPISODE_STEPS,
    total_frames=TOTAL_FRAMES,
)
try:
    for batch in collector:
        replay.extend(
            batch.select(
                "pixels",
                "observation",
                "action",
                "episode",
                "frame",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                ("next", "truncated"),
            ).reshape(-1)
        )
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
# Before exporting, we need to tell LeRobot what data each entry contains. Its
# feature schema describes the names, shapes and dtypes of the data to save.
# We will save the policy observation as ``"observation.state"``, the camera
# as ``"observation.images.front"``, and the action, reward and end signals.
#
# The names in our replay buffer do not all match the LeRobot names. For
# example, the camera is called ``"pixels"`` in TorchRL. ``key_map`` connects
# the LeRobot feature names to the corresponding TensorDict keys. The simulator
# returns images in height, width, channels order, while our schema declares
# channels, height, width. The adapter converts the images when it writes them.
#
# The saved action is the policy's normalized action vector, not a measured
# joint position. The environment scales it by ``action_scale`` and adds it to
# the standing actuator targets. The observation includes the robot's
# state, its command and gait-clock features. Knowing what these values mean
# matters when someone later uses the dataset, so we will also save information
# about the task and controller alongside it.

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (frames["observation"].shape[-1],),
    },
    "observation.images.front": {"dtype": "video", "shape": (3, HEIGHT, WIDTH)},
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
adapter = LeRobotDatasetAdapter(
    repo_id=REPO_ID,
    root=OUTPUT_ROOT / "dataset",
    fps=fps,
    features=features,
    key_map={"observation.state": "observation", "observation.images.front": "pixels"},
    task="Walk forward slowly while keeping balance.",
    robot_type="microduck",
    batch_size=32,
)

# %%
# Exporting the episodes
# ----------------------
#
# We can now export the buffer. ``trajectory_key`` tells the exporter which
# transitions belong together, and ``step_key`` tells it their order within an
# episode. The exporter groups by ID and sorts by step before passing the data
# to LeRobot; the entries do not need to be arranged by episode in replay.
#
# This example uses one environment. When several workers contribute data, make
# sure episode IDs are unique across all of them. :class:`~torchrl.envs.TrajCounter`
# can be shared between workers, as shown in its documentation. Each retained
# episode must also have consecutive steps. If the buffer retains only part of
# an episode, that fragment is exported; missing transitions are not recreated.
#
# Once export finishes, the returned LeRobot dataset is saved locally, including
# its metadata and encoded camera video. We also write the collection settings so
# that someone using the data can interpret the observations and actions.

native = replay.export(adapter, trajectory_key="episode", step_key="frame")
# Keep the task and controller settings with the exported data.
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

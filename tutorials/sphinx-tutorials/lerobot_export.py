"""
From a MicroDuck rollout to a LeRobot dataset
=============================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _lerobot_export_tuto:

Let's record a little robot dataset. We'll walk MicroDuck in MuJoCo, record the
data in a replay buffer, and export it as a native LeRobot dataset. At the end
you'll have MP4 videos you can watch locally, and an optional upload step to
share the same episodes on the Hugging Face Hub.

Note: This is a small data collection example, with no policy training. The
controller is the closed-form gait from TorchRL's MicroDuck example.

What you will learn
-------------------

- Record pixels, actions, rewards, and episode boundaries with TorchRL building
  blocks.
- Export replay entries in episode order, with compressed LeRobot camera videos.
- Reopen the result and load a batch of decoded frames into TensorDict.
- Upload the complete dataset and open it in the LeRobot Dataset Visualizer.

Run this from a TorchRL checkout with Python 3.12 or newer::

    uv run --python 3.12 --no-sources --extra lerobot --with ipython \
        python tutorials/sphinx-tutorials/lerobot_export.py

The ``lerobot`` extra installs the native dataset writer and video dependencies;
MuJoCo is in the checkout's default development dependencies. ``--no-sources``
selects released packages, since LeRobot's supported torch versions may lag the
TensorDict Git main branch. The pinned MicroDuck assets download on first use.
On a headless Linux machine, set ``MUJOCO_GL=egl`` before starting Python if your
MuJoCo installation uses EGL. macOS uses its native offscreen renderer.

By default we collect 120 frames with a 60-step limit per episode. Early falls
trigger a reset and can add episodes. ``TORCHRL_TUTORIALS_FAST=1`` reduces this
to 24 frames for a quick smoke run. Nothing is uploaded by default.
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
from torchrl.data import LazyTensorStorage, ReplayBuffer, tensordict_collate
from torchrl.data.datasets import LeRobotDatasetAdapter
from torchrl.envs import Compose, MicroDuckEnv, StepCounter, TrajCounter, TransformedEnv

# Reuse the example's TensorDict policy. This path also works when Sphinx copies
# this tutorial into docs/source/reference/generated/tutorials.
EXAMPLE_DIR = Path(torchrl.__file__).resolve().parents[1] / "examples" / "microduck"
sys.path.insert(0, str(EXAMPLE_DIR))
from heuristic_gait import MicroDuckGaitActor  # noqa: E402

# %%
# Make the recording small and repeatable
# ---------------------------------------
#
# A fresh temporary directory makes reruns easy: the adapter creates a new
# dataset and refuses to overwrite an existing one. Set
# ``TORCHRL_LEROBOT_OUTPUT=/path/to/a/new/run`` to keep a predictable location.
# Pick a new run directory each time.

FAST = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"
EPISODE_STEPS = 12 if FAST else 60
TOTAL_FRAMES = 2 * EPISODE_STEPS
WIDTH, HEIGHT = 240, 160
REPO_ID = os.environ.get("TORCHRL_LEROBOT_REPO_ID", "local/microduck-walk")
PRIVATE_DATASET = True
OUTPUT_ROOT = (
    Path(os.environ["TORCHRL_LEROBOT_OUTPUT"]).expanduser().absolute()
    if "TORCHRL_LEROBOT_OUTPUT" in os.environ
    else Path(tempfile.mkdtemp(prefix="torchrl-microduck-lerobot-"))
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# %%
# Collect with a policy and two transforms
# ----------------------------------------
#
# ``from_pixels=True`` adds uint8 HWC camera frames to the observation. The gait
# is already a TensorDict module: ``env.rollout`` calls it and records its action
# next to the observation that produced it.
#
# ``StepCounter`` limits each episode and marks the last transition as truncated.
# ``TrajCounter`` gives resets distinct episode IDs. These are stored observations,
# so the exporter can order frames even if the replay buffer wraps around later.
# A physical fall can end an episode earlier; ``break_when_any_done=False`` lets
# rollout reset and keep collecting until the frame budget is reached.

policy = MicroDuckGaitActor()
base_env = MicroDuckEnv(
    download=True,
    backend="mujoco",
    tasks=MicroDuckEnv.tracking_task(0.03, **policy.config.task_kwargs()),
    action_scale=1.0,
    from_pixels=True,
    render_width=WIDTH,
    render_height=HEIGHT,
    camera_id=-1,
    seed=0,
)
env = TransformedEnv(
    base_env,
    Compose(
        StepCounter(max_steps=EPISODE_STEPS, step_count_key="frame"),
        TrajCounter(out_key="episode"),
    ),
)
# Match timestamps to the simulation's actual control period, rather than its
# display metadata. LeRobot takes an integer frame rate.
try:
    fps = round(1 / base_env.dt)
    assert abs(fps * base_env.dt - 1) < 1e-6
    with torch.no_grad():
        rollout = env.rollout(TOTAL_FRAMES, policy=policy, break_when_any_done=False)
finally:
    env.close()

# %%
# Keep the frames in replay
# -------------------------
#
# MicroDuck returns a batch shaped ``[1, time]``. Flatten the environment and
# time dimensions so each replay entry is one transition. We retain current
# pixels, observations, actions, counters, and transition signals. Selecting
# these fields avoids storing a second copy of the camera under ``"next"``.

frames = rollout.select(
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
replay = ReplayBuffer(storage=LazyTensorStorage(TOTAL_FRAMES), batch_size=8)
replay.extend(frames)

# Sphinx and notebooks display expression values without extra logging code.
recording_summary = {
    "frames": len(replay),
    "episodes": frames["episode"].unique().numel(),
    "fps": fps,
    "mean_reward": frames["next", "reward"].mean().item(),
}
recording_summary

# %%
# Describe the LeRobot features
# -----------------------------
#
# The feature schema says which columns to persist, their dtypes and shapes.
# LeRobot names cameras ``observation.images.<name>``. Here the schema uses CHW
# order, while the source pixels are HWC. ``key_map`` connects those column names
# to the keys already in replay. Add another camera by adding a feature and its
# mapping.
#
# Here ``observation.state`` is MicroDuck's policy observation, which includes
# proprioception, commands and gait-clock features. The action is a normalized
# offset around the robot's standing actuator targets; it is not a measured
# joint angle. Save the task, controller and action scaling alongside a dataset
# if you plan to replay actions on another simulator or robot.

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
# Export, then reopen the saved dataset
# -------------------------------------
#
# ``ReplayBuffer.export`` protects replay data while the adapter reads it.
# Supplying both counter keys groups frames by episode and sorts by step number.
# The adapter rejects gaps and duplicates within an episode. A partially retained
# episode is saved as a fragment starting at LeRobot frame zero; it does not gain
# a fabricated terminal transition.
#
# Export returns only after the parquet files, episode metadata, statistics and
# MP4 videos are finalized. H.264 keeps the videos broadly playable in browsers.
# Encoding uses temporary PNGs to preserve every input frame; leave room for one
# episode of uncompressed camera images. Choose a smaller ``batch_size`` if host
# staging memory is tight. The native writer also buffers numeric episode data.

native = replay.export(adapter, trajectory_key="episode", step_key="frame")
# This small provenance file is uploaded with the native dataset.
recording_configuration = {
    "environment": "MicroDuckEnv",
    "backend": "mujoco",
    "task": "tracking_task",
    "forward_command_m_s": 0.03,
    "action_scale": 1.0,
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

# ``from_external`` supplies a lazy Dataset view. LeRobot decodes cameras to
# float32 CHW tensors in [0, 1], ready for a standard TensorDict DataLoader.
# Imported numeric values follow LeRobot's formatting.
imported = LeRobotDatasetAdapter().from_external(loaded)
batch = next(iter(DataLoader(imported, batch_size=8, collate_fn=tensordict_collate)))
torch.testing.assert_close(batch["action"], frames["action"][:8])
assert batch["observation", "image", "front"].shape == (8, 3, HEIGHT, WIDTH)
batch

# %%
# Watch the exported MP4
# ----------------------
#
# This preview uses the saved video, so you're checking the compressed export.
# LeRobot v3 can pack several episodes into the same MP4; the visualizer uses
# episode metadata to play their correct time ranges. The dataset directory also
# contains ``meta/info.json`` and the parquet tables for state, actions and tasks.

video_paths = sorted(native.root.rglob("*.mp4"))
assert video_paths
Video(filename=str(video_paths[0]), embed=True, width=WIDTH)

# %%
# Share it on the Hub
# -------------------
#
# Sign in once with ``hf auth login`` (or set ``HF_TOKEN``), then rerun with your
# own repository ID and opt into upload::
#
#     TORCHRL_LEROBOT_REPO_ID=your-name/microduck-walk \
#     TORCHRL_LEROBOT_PUSH=1 \
#     uv run --python 3.12 --no-sources --extra lerobot --with ipython \
#         python tutorials/sphinx-tutorials/lerobot_export.py
#
# Use a new repository name: the example creates it as private and refuses
# existing repositories. The native ``push_to_hub`` uploads the parquet data, metadata, dataset card and
# videos together. Set ``PRIVATE_DATASET = False`` above when you want to
# publish it publicly. Keep ``push_videos=True`` so viewers
# can play the camera recordings.

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
(hub_url, visualizer_url, native.root)

# %%
# Once uploaded, open ``visualizer_url`` in your browser and sign in to view the
# private dataset. Camera playback and state/action charts follow the same
# timestamps.
#
# To use a shared dataset for offline TorchRL training, see
# :class:`~torchrl.data.datasets.LeRobotExperienceReplay`, which supports trajectory
# sampling and lazy video references. You can also load the native dataset on
# another machine and adapt it with ``LeRobotDatasetAdapter().from_external``.
#
# Conclusion
# ----------
#
# We collected a short robot recording with a TensorDict policy and env
# transforms, stored transitions in replay, and exported a native LeRobot v3
# dataset with synchronized camera videos. The same export step works with
# collector data when you supply globally unique episode IDs and consecutive
# step counters. Check the videos and episode boundaries before building a larger
# demonstration set.
#
# Further reading
# ---------------
#
# - :ref:`microduck_tuto` for the environment, task presets and gait controller.
# - :class:`~torchrl.data.datasets.LeRobotDatasetAdapter` for supported types,
#   concurrency rules and incomplete exports.
# - `LeRobot dataset v3 <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>`_.
# - `LeRobot visualization tools <https://huggingface.co/docs/lerobot/using_dataset_tools>`_.

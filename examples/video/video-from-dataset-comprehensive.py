# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This example is a more comprehensive version of video-from-dataset.py
"""
import argparse

from torchrl.data import SliceSampler
from torchrl.data.datasets import (
    AtariDQNExperienceReplay,
    GenDGRLExperienceReplay,
    MinariExperienceReplay,
    OpenXExperienceReplay,
    RobosetExperienceReplay,
    VD4RLExperienceReplay,
)
from torchrl.envs import Compose
from torchrl.record import CSVLogger, VideoRecorder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    default="GenDGRL",
    choices=["OpenX", "Atari", "VD4RL", "GenDGRL", "Roboset", "Minari"],
)
parser.add_argument(
    "--dataset",
    default="ninja-1M_E",
    choices=[
        # OpenX
        "cmu_stretch",
        "stanford_robocook_converted_externally_to_rlds",
        # Atari
        "Pong/5",
        # VD4RL
        "main/walker_walk/expert/64px",
        # GenDGRL
        "bigfish-1M_E",
        "ninja-1M_E",
        # Roboset
        "DAPG(human)/door_v2d-v1",
        # Minari
        "kitchen-complete-v1",
    ],
)
args = parser.parse_args()

logger = CSVLogger(args.source, video_format="mp4", video_fps=4)
if args.source == "OpenX":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("observation", "image")])
    dataset = OpenXExperienceReplay(
        args.dataset,
        download=False,
        streaming=True,
        num_slices=num_slices,
        batch_size=128,
        strict_length=False,
        pad=0,
        transform=t,
    )
elif args.source == "Atari":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("pixels")])
    # We need to unsqueeze the B&W image
    dataset = AtariDQNExperienceReplay(
        args.dataset,
        batch_size=128,
        num_slices=num_slices,
        transform=Compose(
            lambda data: data.set(
                "pixels",
                data.get("observation").unsqueeze(-3).repeat_interleave(3, dim=-3),
            ),
            t,
        ),
    )
elif args.source == "VD4RL":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("pixels")])
    dataset = VD4RLExperienceReplay(
        args.dataset, batch_size=128, num_slices=num_slices, transform=t
    )
elif args.source == "GenDGRL":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("observation")])
    dataset = GenDGRLExperienceReplay(
        args.dataset,
        batch_size=128,
        sampler=SliceSampler(num_slices=num_slices, end_key=("next", "done")),
        transform=t,
    )
elif args.source == "Roboset":
    num_slices = 1
    t = VideoRecorder(
        logger,
        tag=args.dataset,
        in_keys=[("info", "visual_dict", "rgb:view_1:224x224:2d")],
    )
    dataset = RobosetExperienceReplay(
        args.dataset,
        batch_size=128,
        sampler=SliceSampler(num_slices=num_slices, end_key=("next", "done")),
        transform=t,
    )
elif args.source == "Minari":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("pixels")])
    dataset = MinariExperienceReplay(
        args.dataset,
        batch_size=128,
        sampler=SliceSampler(num_slices=num_slices, end_key=("next", "done")),
        transform=t,
    )
for i, _ in enumerate(dataset):  # data does not have a consistent shape
    t.dump()
    if i == 4:
        break

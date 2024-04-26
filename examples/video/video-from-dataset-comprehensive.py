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
from torchrl.envs import Compose, UnsqueezeTransform
from torchrl.record import CSVLogger, VideoRecorder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    default="Minari",
    choices=["OpenX", "Atari", "VD4RL", "GenDGRL", "Roboset", "Minari"],
)
parser.add_argument(
    "--dataset",
    default="kitchen-complete-v1",
    choices=[
        "cmu_stretch",
        "Pong/5",
        "main/walker_walk/random/64px",
        "bigfish-1M_E",
        "DAPG(human)/door_v2d-v1",
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
        transform=t,
    )
    for i, data in enumerate(dataset):  # data does not have a consistent shape
        t.dump()
        if i == 4:
            break
elif args.source == "Atari":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("observation")])
    # We need to unsqueeze the B&W image
    dataset = AtariDQNExperienceReplay(
        args.dataset,
        batch_size=128,
        num_slices=num_slices,
        transform=Compose(
            UnsqueezeTransform(in_keys=["observation"], unsqueeze_dim=-3), t
        ),
    )
    for i, data in enumerate(dataset):  # data does not have a consistent shape
        t.dump()
        if i == 4:
            break
elif args.source == "VD4RL":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("pixels")])
    dataset = VD4RLExperienceReplay(
        args.dataset, batch_size=128, num_slices=num_slices, transform=t
    )
    for i, data in enumerate(dataset):  # data does not have a consistent shape
        t.dump()
        if i == 4:
            break
elif args.source == "GenDGRL":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("observation")])
    dataset = GenDGRLExperienceReplay(
        args.dataset,
        batch_size=128,
        sampler=SliceSampler(num_slices=num_slices, end_key=("next", "done")),
        transform=t,
    )
    for i, data in enumerate(dataset):  # data does not have a consistent shape
        t.dump()
        if i == 4:
            break
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
    for i, data in enumerate(dataset):  # data does not have a consistent shape
        t.dump()
        if i == 4:
            break
elif args.source == "Minari":
    num_slices = 1
    t = VideoRecorder(logger, tag=args.dataset, in_keys=[("pixels")])
    dataset = MinariExperienceReplay(
        args.dataset,
        batch_size=128,
        sampler=SliceSampler(num_slices=num_slices, end_key=("next", "done")),
        transform=t,
    )
    for i, data in enumerate(dataset):  # data does not have a consistent shape
        t.dump()
        if i == 4:
            break

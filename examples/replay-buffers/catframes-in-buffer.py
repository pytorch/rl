# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import (
    CatFrames,
    Compose,
    DMControlEnv,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
)

# Number of frames to stack together
frame_stack = 4
# Dimension along which the stack should occur
stack_dim = -4
# Max size of the buffer
max_size = 100_000
# Batch size of the replay buffer
training_batch_size = 32

seed = 123


def main():
    catframes = CatFrames(
        N=frame_stack,
        dim=stack_dim,
        in_keys=["pixels_trsf"],
        out_keys=["pixels_trsf"],
    )
    env = TransformedEnv(
        DMControlEnv(
            env_name="cartpole",
            task_name="balance",
            device="cpu",
            from_pixels=True,
            pixels_only=True,
        ),
        Compose(
            ToTensorImage(
                from_int=True,
                dtype=torch.float32,
                in_keys=["pixels"],
                out_keys=["pixels_trsf"],
                shape_tolerant=True,
            ),
            UnsqueezeTransform(
                dim=stack_dim, in_keys=["pixels_trsf"], out_keys=["pixels_trsf"]
            ),
            catframes,
            StepCounter(),
        ),
    )
    env.set_seed(seed)

    transform, sampler = catframes.make_rb_transform_and_sampler(
        batch_size=training_batch_size,
        traj_key=("collector", "traj_ids"),
        strict_length=True,
    )

    rb_transforms = Compose(
        ToTensorImage(
            from_int=True,
            dtype=torch.float32,
            in_keys=["pixels", ("next", "pixels")],
            out_keys=["pixels_trsf", ("next", "pixels_trsf")],
            shape_tolerant=True,
        ),  # C W' H' -> C W' H' (unchanged due to shape_tolerant)
        UnsqueezeTransform(
            dim=stack_dim,
            in_keys=["pixels_trsf", ("next", "pixels_trsf")],
            out_keys=["pixels_trsf", ("next", "pixels_trsf")],
        ),  # 1 C W' H'
        transform,
    )

    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=max_size, device="cpu"),
        sampler=sampler,
        batch_size=training_batch_size,
        transform=rb_transforms,
    )

    data = env.rollout(1000, break_when_any_done=False)
    rb.extend(data)

    training_batch = rb.sample()
    print(training_batch)


if __name__ == "__main__":
    main()

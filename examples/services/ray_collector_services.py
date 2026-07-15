# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Collect with Ray-owned replay and inference services.

The driver only constructs TorchRL objects. Replay storage, policy inference,
and environment collection run in dedicated Ray actors; TorchRL creates and
distributes the restricted transport clients internally.

Run from the repository root with ``python examples/services/ray_collector_services.py``.
"""

from __future__ import annotations

from functools import partial

from tensordict.nn import TensorDictModule
from torch import nn

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.distributed import RayCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules.inference_server import InferenceServer


def make_env() -> GymEnv:
    """Create one CPU environment inside a collector actor."""
    return GymEnv("Pendulum-v1", device="cpu")


def make_policy() -> TensorDictModule:
    """Create the policy inside the inference actor."""
    return TensorDictModule(
        nn.Sequential(nn.Linear(3, 1), nn.Tanh()),
        in_keys=["observation"],
        out_keys=["action"],
    )


def main() -> None:
    ray_init_config = {
        "num_cpus": 4,
        "include_dashboard": False,
        "log_to_driver": False,
    }
    replay = TensorDictReplayBuffer(
        storage=partial(LazyTensorStorage, 1_000),
        batch_size=4,
        service_backend="ray",
        service_backend_options={
            "ray_init_config": ray_init_config,
            "remote_config": {"num_cpus": 1},
        },
        transport="auto",
    )
    inference = InferenceServer(
        policy_factory=make_policy,
        service_backend="ray",
        service_backend_options={
            "remote_config": {"num_cpus": 1},
        },
        transport="auto",
    )
    collector = None
    try:
        collector = RayCollector(
            create_env_fn=[make_env],
            policy=inference,
            replay_buffer=replay,
            collector_class="single",
            frames_per_batch=8,
            total_frames=16,
            remote_configs={"num_cpus": 1, "num_gpus": 0},
            sync=True,
        )
        for _ in collector:
            pass

        sample = replay.sample()
        torchrl_logger.info(
            "Collected %s transitions; sampled batch shape is %s.",
            len(replay),
            tuple(sample.shape),
        )
    finally:
        if collector is not None:
            collector.shutdown()
        inference.shutdown()
        replay.shutdown()


if __name__ == "__main__":
    main()

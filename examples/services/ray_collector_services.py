# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Collect with scoped Ray ownership and distributed tensor transports.

The driver only constructs TorchRL objects. Replay storage, policy inference,
and environment collection run in dedicated Ray actors. TensorDict payloads
use Gloo while TorchRL creates and distributes the restricted clients.

Run from the repository root with ``python examples/services/ray_collector_services.py``.
"""

from __future__ import annotations

from functools import partial

from tensordict.nn import TensorDictModule
from torch import nn

from torchrl import service_backend, transport_backend
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
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
    replay = inference = collector = None
    try:
        with service_backend("ray"), transport_backend("distributed"):
            replay = TensorDictReplayBuffer(
                storage=partial(LazyTensorStorage, 1_000),
                batch_size=4,
                service_backend_options={
                    "ray_init_config": ray_init_config,
                    "remote_config": {"num_cpus": 1},
                },
                transport_options={"backend": "gloo"},
            )
            inference = InferenceServer(
                policy_factory=make_policy,
                service_backend_options={
                    "remote_config": {"num_cpus": 1},
                },
                transport_options={"backend": "gloo"},
            )
            collector = Collector(
                create_env_fn=make_env,
                num_collectors=1,
                policy=inference,
                replay_buffer=replay,
                backend_options={
                    "collector_class": "single",
                    "remote_configs": {"num_cpus": 1, "num_gpus": 0},
                },
                frames_per_batch=8,
                total_frames=16,
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
        if inference is not None:
            inference.shutdown()
        if replay is not None:
            replay.shutdown()


if __name__ == "__main__":
    main()

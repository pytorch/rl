# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def test_imports():
    from torchrl.data import (
        PrioritizedReplayBuffer,
        ReplayBuffer,
        TensorSpec,
    )  # noqa: F401
    from torchrl.envs import Transform, TransformedEnv  # noqa: F401
    from torchrl.envs.gym_like import GymLikeEnv  # noqa: F401
    from torchrl.modules import SafeModule  # noqa: F401
    from torchrl.objectives.common import LossModule  # noqa: F401

    PrioritizedReplayBuffer(alpha=1.1, beta=1.1)

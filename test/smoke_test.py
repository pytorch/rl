# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def test_imports():
    from torchrl.data import (
        ReplayBuffer,
        PrioritizedReplayBuffer,
        TensorDict,
        TensorSpec,
    )
    from torchrl.envs import TransformedEnv, Transform
    from torchrl.envs.gym_like import GymLikeEnv
    from torchrl.modules import TensorDictModule
    from torchrl.objectives.costs.common import LossModule

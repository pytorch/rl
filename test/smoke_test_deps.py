# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tempfile

import pytest

from torchrl.envs.libs.gym import gym_backend


def test_dm_control():
    import dm_control  # noqa: F401
    import dm_env  # noqa: F401
    from dm_control import suite  # noqa: F401
    from dm_control.suite.wrappers import pixels  # noqa: F401
    from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv  # noqa

    assert _has_dmc
    env = DMControlEnv("cheetah", "run")
    env.reset()


@pytest.mark.skip(reason="Not implemented yet")
def test_dm_control_pixels():
    from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv  # noqa

    env = DMControlEnv("cheetah", "run", from_pixels=True)
    env.reset()


def test_gym():
    try:
        import gymnasium as gym
    except ImportError as err:
        ERROR = err
        try:
            import gym  # noqa: F401
        except ImportError as err:
            raise ImportError(
                f"gym and gymnasium load failed. Gym got error {err}."
            ) from ERROR

    from torchrl.envs.libs.gym import _has_gym, GymEnv  # noqa

    assert _has_gym
    from _utils_internal import PONG_VERSIONED

    env = GymEnv(PONG_VERSIONED())
    env.reset()


def test_tb():
    from torch.utils.tensorboard import SummaryWriter

    _has_tb = True

    assert _has_tb
    test_rounds = 100
    while test_rounds > 0:
        try:
            with tempfile.TemporaryDirectory() as directory:
                writer = SummaryWriter(log_dir=directory)
                writer.add_scalar("a", 1, 1)
            break
        except OSError:
            # OS error could be raised randomly
            # depending on the test machine
            test_rounds -= 1


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)

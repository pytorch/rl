import argparse
import tempfile

import pytest
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv
from torchrl.envs.libs.gym import _has_gym, GymEnv

if _has_gym:
    import gym
    from packaging import version

    gym_version = version.parse(gym.__version__)
    PONG_VERSIONED = (
        "ALE/Pong-v5" if gym_version > version.parse("0.20.0") else "Pong-v4"
    )
else:
    # placeholders
    PONG_VERSIONED = "ALE/Pong-v5"

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tb = True
except ImportError:
    _has_tb = False


def test_dm_control():
    import dm_control  # noqa: F401
    import dm_env  # noqa: F401
    from dm_control import suite  # noqa: F401
    from dm_control.suite.wrappers import pixels  # noqa: F401

    assert _has_dmc
    env = DMControlEnv("cheetah", "run")
    env.reset()


@pytest.mark.skip(reason="Not implemented yet")
def test_dm_control_pixels():
    env = DMControlEnv("cheetah", "run", from_pixels=True)
    env.reset()


def test_gym():
    import gym  # noqa: F401

    assert _has_gym
    env = GymEnv(PONG_VERSIONED)
    env.reset()


def test_tb():
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

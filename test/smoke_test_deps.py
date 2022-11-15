import argparse
import tempfile

import pytest
from torchrl._utils import implement_for
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv
from torchrl.envs.libs.gym import _has_gym, GymEnv

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
    env = GymEnv(_pong_versioned())
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


@implement_for("gym", None, "0.21")
def _pong_versioned():  # noqa: F811
    return "Pong-v4"


@implement_for("gym", "0.21", None)
def _pong_versioned():  # noqa: F811
    return "ALE/Pong-v5"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)

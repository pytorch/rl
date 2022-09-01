import tempfile

import pytest
import torch
from packaging import version

_has_tb = False
if version.parse(torch.__version__) > version.parse("1.11.0"):
    # 1.10 and before were using distutils, which migrated to packaging
    from torch.utils.tensorboard import SummaryWriter
    _has_tb = True
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv
from torchrl.envs.libs.gym import _has_gym, GymEnv


def test_dm_control():
    assert _has_dmc
    env = DMControlEnv("cheetah", "run")
    env.reset()


# Disabling this for now
# def test_dm_control_pixels():
#     env = DMControlEnv("cheetah", "run", from_pixels=True)
#     env.reset()


def test_gym():
    assert _has_gym
    env = GymEnv("ALE/Pong-v5")
    env.reset()

@pytest.mark.skipif(not _has_tb, reason="tensorboard could not be loaded")
def test_tb():
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

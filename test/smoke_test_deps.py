import tempfile

from torch.utils.tensorboard import SummaryWriter
from torchrl.envs import DMControlEnv, GymEnv
from torchrl.envs.libs.dm_control import _has_dmc
from torchrl.envs.libs.gym import _has_gym


def test_dm_control():
    import dm_control
    import dm_env
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

    assert _has_dmc
    env = DMControlEnv("cheetah", "run")
    env.reset()


# Disabling this for now
# def test_dm_control_pixels():
#     env = DMControlEnv("cheetah", "run", from_pixels=True)
#     env.reset()


def test_gym():
    import gym

    assert _has_gym
    env = GymEnv("ALE/Pong-v5")
    env.reset()


def test_tb():
    with tempfile.TemporaryDirectory() as directory:
        writer = SummaryWriter(log_dir=directory)
        writer.add_scalar("a", 1, 1)

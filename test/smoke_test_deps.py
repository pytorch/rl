import tempfile

from torch.utils.tensorboard import SummaryWriter
from torchrl.envs import DMControlEnv, GymEnv


def test_dm_control():
    env = DMControlEnv("cheetah", "run")
    env.reset()


# Disabling this for now
# def test_dm_control_pixels():
#     env = DMControlEnv("cheetah", "run", from_pixels=True)
#     env.reset()


def test_gym():
    env = GymEnv("ALE/Pong-v5")
    env.reset()


def test_gym_pixels():
    env = GymEnv("ALE/Pong-v5", from_pixels=True)
    env.reset()


def test_tb():
    with tempfile.TemporaryDirectory() as directory:
        writer = SummaryWriter(log_dir=directory)
        writer.add_scalar("a", 1, 1)

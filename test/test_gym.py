import os
from collections import defaultdict

import pytest
import torch

from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.envs import GymEnv
from torchrl.envs.libs.gym import _has_gym, _get_envs as _get_envs_gym

import yaml

try:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "configs", "atari.yaml"), "r") as file:
        atari_confs = yaml.load(file, Loader=yaml.FullLoader)
    _atari_found = True
except:
    _atari_found = False
    atari_confs = defaultdict(lambda: "")


@pytest.mark.skipif(not _atari_found, reason="no _atari_found found")
@pytest.mark.skipif(not _has_gym, reason="no gym library found")
@pytest.mark.parametrize("env_name", atari_confs["atari_envs"])
@pytest.mark.parametrize("env_suffix", atari_confs["version"])
@pytest.mark.parametrize("frame_skip", [1, 2, 3, 4])
def test_atari(env_name, env_suffix, frame_skip):
    env = GymEnv("-".join([env_name, env_suffix]), frame_skip=frame_skip)
    env.rollout(n_steps=50)


# TODO: check gym envs in a smart, efficient way
# @pytest.mark.skipif(not _has_gym, reason="no gym library found")
# @pytest.mark.parametrize("env_name", _get_envs_gym())
# @pytest.mark.parametrize("from_pixels", [False, True])
# def test_gym(env_name, from_pixels):
#     print(f"testing {env_name} with from_pixels={from_pixels}")
#     torch.manual_seed(0)
#     env = GymEnv(env_name, frame_skip=4, from_pixels=from_pixels)
#     env.set_seed(0)
#     td1 = env.rollout(n_steps=10, auto_reset=True)
#     tdb = env.rollout(n_steps=10, auto_reset=True)
#     if not tdb.get("done").sum():
#         tdc = env.rollout(n_steps=10, auto_reset=False)
#     torch.manual_seed(0)
#     env = GymEnv(env_name, frame_skip=4, from_pixels=from_pixels)
#     env.set_seed(0)
#     td2 = env.rollout(n_steps=10, auto_reset=True)
#     assert_allclose_td(td1, td2)


if __name__ == "__main__":
    pytest.main([__file__, "--capture", "no"])

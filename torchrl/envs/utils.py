from typing import Any

import pkg_resources
from torch.autograd.grad_mode import _DecoratorContextManager

from torchrl.data.tensordict.tensordict import _TensorDict

AVAILABLE_LIBRARIES = {pkg.key for pkg in pkg_resources.working_set}


def step_tensor_dict(tensor_dict: _TensorDict, next_tensor_dict: _TensorDict = None) -> _TensorDict:
    keys = [key for key in tensor_dict.keys() if key.rfind("next_") == 0]
    select_tensor_dict = tensor_dict.select(*keys).clone()
    for key in keys:
        select_tensor_dict.rename_key(key, key[5:], safe=True)
    if next_tensor_dict is not None:
        return next_tensor_dict.update(select_tensor_dict)
    else:
        return select_tensor_dict


def get_available_libraries():
    return SUPPORTED_LIBRARIES


def _check_gym():
    return "gym" in AVAILABLE_LIBRARIES


def _check_gym_atari():
    if not _check_gym():
        return False
    return "atari-py" in AVAILABLE_LIBRARIES


def _check_mario():
    return "gym-super-mario-bros" in AVAILABLE_LIBRARIES


def _check_dmcontrol():
    return "dm-control" in AVAILABLE_LIBRARIES


def _check_dmlab():
    return "deepmind-lab" in AVAILABLE_LIBRARIES


SUPPORTED_LIBRARIES = {
    "gym": _check_gym(),  # OpenAI
    "gym[atari]": _check_gym_atari(),  #
    "vizdoom": None,  # 1.2k, https://github.com/mwydmuch/ViZDoom
    "ml-agents": None,  # 11.5k, unity, https://github.com/Unity-Technologies/ml-agents
    "pysc2": None,  # 7.3k, DM, https://github.com/deepmind/pysc2
    "deepmind_lab": _check_dmlab(),
    # 6.5k DM, https://github.com/deepmind/lab, https://github.com/deepmind/lab/tree/master/python/pip_package
    "serpent.ai": None,  # 6k, https://github.com/SerpentAI/SerpentAI
    "gfootball": None,  # 2.8k G, https://github.com/google-research/football
    "dm_control": _check_dmcontrol(),  # 2.3k DM, https://github.com/deepmind/dm_control
    "habitat": None,  # 1.2k FB, https://github.com/facebookresearch/habitat-sim
    "meta-world": None,  # 500, https://github.com/rlworkgroup/metaworld
    "minerl": None,  # 300, https://github.com/minerllabs/minerl
    "multi-agent-emergence-environments": None,
    # 1.2k, OpenAI, https://github.com/openai/multi-agent-emergence-environments
    "openspiel": None,  # 2.8k, DM, https://github.com/deepmind/open_spiel
    "procgen": None,  # 500, OpenAI, https://github.com/openai/procgen
    "pybullet": None,  # 641, https://github.com/benelot/pybullet-gym
    "realworld_rl_suite": None,  # 250, G, https://github.com/google-research/realworldrl_suite
    "rlcard": None,  # 1.4k, https://github.com/datamllab/rlcard
    "screeps": None,  # 2.3k https://github.com/screeps/screeps
    "gym-super-mario-bros": _check_mario(),
}

EXPLORATION_MODE = False


class set_exploration_mode(_DecoratorContextManager):
    def __init__(self, mode: bool = True):
        super().__init__()
        self.mode = mode

    def __enter__(self) -> None:
        global EXPLORATION_MODE
        self.prev = EXPLORATION_MODE
        EXPLORATION_MODE = self.mode

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global EXPLORATION_MODE
        EXPLORATION_MODE = self.prev


def exploration_mode() -> bool:
    return EXPLORATION_MODE

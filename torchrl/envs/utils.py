# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import pkg_resources
from torch.autograd.grad_mode import _DecoratorContextManager

from torchrl.data.tensordict.tensordict import TensorDictBase

AVAILABLE_LIBRARIES = {pkg.key for pkg in pkg_resources.working_set}


class _classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def step_mdp(
    tensordict: TensorDictBase,
    next_tensordict: TensorDictBase = None,
    keep_other: bool = True,
    exclude_reward: bool = True,
    exclude_done: bool = True,
    exclude_action: bool = True,
) -> TensorDictBase:
    """Creates a new tensordict that reflects a step in time of the input tensordict.

    Given a tensordict retrieved after a step, returns another tensordict with all the :obj:`'next_'` prefixes are removed,
    i.e. all the :obj:`'next_some_other_string'` keys will be renamed onto :obj:`'some_other_string'` keys.


    Args:
        tensordict (TensorDictBase): tensordict with keys to be renamed
        next_tensordict (TensorDictBase, optional): destination tensordict
        keep_other (bool, optional): if True, all keys that do not start with :obj:`'next_'` will be kept.
            Default is True.
        exclude_reward (bool, optional): if True, the :obj:`"reward"` key will be discarded
            from the resulting tensordict.
            Default is True.
        exclude_done (bool, optional): if True, the :obj:`"done"` key will be discarded
            from the resulting tensordict.
            Default is True.
        exclude_action (bool, optional): if True, the :obj:`"action"` key will be discarded
            from the resulting tensordict.
            Default is True.

    Returns:
         A new tensordict (or next_tensordict) with the "next_*" keys renamed without the "next_" prefix.

    Examples:
    This funtion allows for this kind of loop to be used:
        >>> td_out = []
        >>> env = make_env()
        >>> policy = make_policy()
        >>> td = env.reset()
        >>> for i in range(max_steps):
        >>>     td = env.step(td)
        >>>     next_td = step_mdp(td)
        >>>     assert next_td is not td # make sure that keys are not overwritten
        >>>     td_out.append(td)
        >>>     td = next_td
        >>> td_out = torch.stack(td_out, 0)
        >>> print(td_out) # should contain keys 'observation', 'next_observation', 'action', 'reward', 'done' or similar

    """
    other_keys = []
    prohibited = set()
    if exclude_done:
        prohibited.add("done")
    else:
        other_keys.append("done")
    if exclude_reward:
        prohibited.add("reward")
    else:
        other_keys.append("reward")
    if exclude_action:
        prohibited.add("action")
    else:
        other_keys.append("action")
    keys = [key for key in tensordict.keys() if key.startswith("next_")]
    if len(keys) == 0:
        raise RuntimeError(
            "There was no key starting with 'next_' in the provided TensorDict: ",
            tensordict,
        )
    new_keys = [key[5:] for key in keys]
    prohibited = prohibited.union(keys).union(new_keys)
    if keep_other:
        other_keys = [key for key in tensordict.keys() if key not in prohibited]
    select_tensordict = tensordict.select(*other_keys, *keys)
    for new_key, key in zip(new_keys, keys):
        select_tensordict.rename_key(key, new_key, safe=True)
    if next_tensordict is not None:
        return next_tensordict.update(select_tensordict)
    else:
        return select_tensordict


def get_available_libraries():
    """Returns all the supported libraries."""
    return SUPPORTED_LIBRARIES


def _check_gym():
    """Returns True if the gym library is installed."""
    return "gym" in AVAILABLE_LIBRARIES


def _check_gym_atari():
    """Returns True if the gym library is installed and atari envs can be found."""
    if not _check_gym():
        return False
    return "atari-py" in AVAILABLE_LIBRARIES


def _check_mario():
    """Returns True if the "gym-super-mario-bros" library is installed."""
    return "gym-super-mario-bros" in AVAILABLE_LIBRARIES


def _check_dmcontrol():
    """Returns True if the "dm-control" library is installed."""
    return "dm-control" in AVAILABLE_LIBRARIES


def _check_dmlab():
    """Returns True if the "deepmind-lab" library is installed."""
    return "deepmind-lab" in AVAILABLE_LIBRARIES


SUPPORTED_LIBRARIES = {
    "gym": _check_gym(),  # OpenAI
    "gym[atari]": _check_gym_atari(),  #
    "dm_control": _check_dmcontrol(),
    "habitat": None,
    "gym-super-mario-bros": _check_mario(),
    # "vizdoom": None,  # gym based, https://github.com/mwydmuch/ViZDoom
    # "openspiel": None,  # DM, https://github.com/deepmind/open_spiel
    # "pysc2": None,  # DM, https://github.com/deepmind/pysc2
    # "deepmind_lab": _check_dmlab(),
    # DM, https://github.com/deepmind/lab, https://github.com/deepmind/lab/tree/master/python/pip_package
    # "serpent.ai": None,  # https://github.com/SerpentAI/SerpentAI
    # "gfootball": None,  # 2.8k G, https://github.com/google-research/football
    # DM, https://github.com/deepmind/dm_control
    # FB, https://github.com/facebookresearch/habitat-sim
    # "meta-world": None,  # https://github.com/rlworkgroup/metaworld
    # "minerl": None,  # https://github.com/minerllabs/minerl
    # "multi-agent-emergence-environments": None,
    # OpenAI, https://github.com/openai/multi-agent-emergence-environments
    # "procgen": None,  # OpenAI, https://github.com/openai/procgen
    # "pybullet": None,  # https://github.com/benelot/pybullet-gym
    # "realworld_rl_suite": None,
    # G, https://github.com/google-research/realworldrl_suite
    # "rlcard": None,  # https://github.com/datamllab/rlcard
    # "screeps": None,  # https://github.com/screeps/screeps
    # "ml-agents": None,
}

EXPLORATION_MODE = None


class set_exploration_mode(_DecoratorContextManager):
    """Sets the exploration mode of all ProbabilisticTDModules to the desired mode.

    Args:
        mode (str): mode to use when the policy is being called.

    Examples:
        >>> policy = Actor(action_spec, module=network, default_interaction_mode="mode")
        >>> env.rollout(policy=policy, max_steps=100)  # rollout with the "mode" interaction mode
        >>> with set_exploration_mode("random"):
        >>>     env.rollout(policy=policy, max_steps=100)  # rollout with the "random" interaction mode

    """

    def __init__(self, mode: str = "mode"):
        super().__init__()
        self.mode = mode

    def __enter__(self) -> None:
        global EXPLORATION_MODE
        self.prev = EXPLORATION_MODE
        EXPLORATION_MODE = self.mode

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global EXPLORATION_MODE
        EXPLORATION_MODE = self.prev


def exploration_mode() -> Union[str, None]:
    """Returns the exploration mode currently set."""
    return EXPLORATION_MODE

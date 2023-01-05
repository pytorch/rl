# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pkg_resources
import torch
from tensordict.nn.probabilistic import (  # noqa
    interaction_mode as exploration_mode,
    set_interaction_mode as set_exploration_mode,
)
from tensordict.tensordict import TensorDictBase

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
    _run_check: bool = True,
) -> TensorDictBase:
    """Creates a new tensordict that reflects a step in time of the input tensordict.

    Given a tensordict retrieved after a step, returns the :obj:`"next"` indexed-tensordict.

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
         A new tensordict (or next_tensordict) containing the tensors of the t+1 step.

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

    prohibited.add("next")
    if keep_other:
        other_keys = [key for key in tensordict.keys() if key not in prohibited]
    select_tensordict = tensordict.select(*other_keys)
    select_tensordict = select_tensordict.update(tensordict.get("next"))

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


def check_env_specs(env):
    """Tests an environment specs against the results of short rollout.

    This test function should be used as a sanity check for an env wrapped with
    torchrl's EnvBase subclasses: any discrepency between the expected data and
    the data collected should raise an assertion error.

    A broken environment spec will likely make it impossible to use parallel
    environments.

    """
    fake_tensordict = env.fake_tensordict().flatten_keys(".")
    real_tensordict = env.rollout(3).flatten_keys(".")

    keys1 = set(fake_tensordict.keys())
    keys2 = set(real_tensordict.keys())
    assert keys1 == keys2
    fake_tensordict = fake_tensordict.unsqueeze(real_tensordict.batch_dims - 1)
    fake_tensordict = fake_tensordict.expand(*real_tensordict.shape)
    fake_tensordict = fake_tensordict.to_tensordict()
    assert (
        fake_tensordict.apply(lambda x: torch.zeros_like(x))
        == real_tensordict.apply(lambda x: torch.zeros_like(x))
    ).all()
    for key in keys2:
        assert fake_tensordict[key].shape == real_tensordict[key].shape

    # test dtypes
    real_tensordict = env.rollout(3)  # keep empty structures, for example dict()
    for key, value in real_tensordict.items():
        _check_dtype(key, value, env.observation_spec, env.input_spec)


def _check_dtype(key, value, obs_spec, input_spec):
    if key in {"reward", "done"}:
        return
    elif key == "next":
        for _key, _value in value.items():
            _check_dtype(_key, _value, obs_spec, input_spec)
        return
    elif key in input_spec.keys(yield_nesting_keys=True):
        assert input_spec[key].is_in(value), (input_spec[key], value)
        return
    elif key in obs_spec.keys(yield_nesting_keys=True):
        assert obs_spec[key].is_in(value), (input_spec[key], value)
        return
    else:
        raise KeyError(key)

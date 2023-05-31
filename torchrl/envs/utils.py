# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pkg_resources
import torch
from tensordict.nn.probabilistic import (  # noqa
    # Note: the `set_interaction_mode` and their associated arg `default_interaction_mode` are being deprecated!
    #       Please use the `set_/interaction_type` ones above with the InteractionType enum instead.
    #       See more details: https://github.com/pytorch/rl/issues/1016
    interaction_mode as exploration_mode,
    interaction_type as exploration_type,
    InteractionType as ExplorationType,
    set_interaction_mode as set_exploration_mode,
    set_interaction_type as set_exploration_type,
)
from tensordict.tensordict import TensorDictBase

__all__ = [
    "exploration_mode",
    "exploration_type",
    "set_exploration_mode",
    "set_exploration_type",
    "ExplorationType",
    "check_env_specs",
    "step_mdp",
    "make_composite_from_td",
]
AVAILABLE_LIBRARIES = {pkg.key for pkg in pkg_resources.working_set}


def _convert_exploration_type(*, exploration_mode, exploration_type):
    if exploration_mode is not None:
        return ExplorationType.from_str(exploration_mode)
    return exploration_type


class _classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def step_mdp(
    tensordict: TensorDictBase,
    next_tensordict: TensorDictBase = None,
    keep_other: bool = True,
    exclude_reward: bool = True,
    exclude_done: bool = False,
    exclude_action: bool = True,
) -> TensorDictBase:
    """Creates a new tensordict that reflects a step in time of the input tensordict.

    Given a tensordict retrieved after a step, returns the :obj:`"next"` indexed-tensordict.
    THe arguments allow for a precise control over what should be kept and what
    should be copied from the ``"next"`` entry. The default behaviour is:
    move the observation entries, reward and done states to the root, exclude
    the current action and keep all extra keys (non-action, non-done, non-reward).

    Args:
        tensordict (TensorDictBase): tensordict with keys to be renamed
        next_tensordict (TensorDictBase, optional): destination tensordict
        keep_other (bool, optional): if ``True``, all keys that do not start with :obj:`'next_'` will be kept.
            Default is ``True``.
        exclude_reward (bool, optional): if ``True``, the :obj:`"reward"` key will be discarded
            from the resulting tensordict. If ``False``, it will be copied (and replaced)
            from the ``"next"`` entry (if present).
            Default is ``True``.
        exclude_done (bool, optional): if ``True``, the :obj:`"done"` key will be discarded
            from the resulting tensordict. If ``False``, it will be copied (and replaced)
            from the ``"next"`` entry (if present).
            Default is ``False``.
        exclude_action (bool, optional): if ``True``, the :obj:`"action"` key will
            be discarded from the resulting tensordict. If ``False``, it will
            be kept in the root tensordict (since it should not be present in
            the ``"next"`` entry).
            Default is ``True``.

    Returns:
         A new tensordict (or next_tensordict) containing the tensors of the t+1 step.

    Examples:
    This funtion allows for this kind of loop to be used:
        >>> from tensordict import TensorDict
        >>> td = TensorDict({
        ...     "done": torch.zeros((), dtype=torch.bool),
        ...     "reward": torch.zeros(()),
        ...     "extra": torch.zeros(()),
        ...     "next": TensorDict({
        ...         "done": torch.zeros((), dtype=torch.bool),
        ...         "reward": torch.zeros(()),
        ...         "obs": torch.zeros(()),
        ...     }, []),
        ...     "obs": torch.zeros(()),
        ...     "action": torch.zeros(()),
        ... }, [])
        >>> print(step_mdp(td))
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_done=True))  # "done" is dropped
        TensorDict(
            fields={
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_reward=True))  # "reward" is dropped
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_action=False))  # "action" persists at the root
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, keep_other=False))  # "extra" is missing
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """
    other_keys = []
    prohibited = set()
    if exclude_action:
        prohibited.add("action")
    else:
        other_keys.append("action")
    if exclude_done:
        prohibited.add("done")
    else:
        other_keys.append("done")
    if exclude_reward:
        prohibited.add("reward")
    else:
        other_keys.append("reward")

    prohibited.add("next")
    if keep_other:
        # TODO: make this work with nested keys
        other_keys = [key for key in tensordict.keys() if key not in prohibited]
    select_tensordict = tensordict.select(*other_keys, strict=False)
    excluded = []
    if exclude_reward:
        excluded.append("reward")
    if exclude_done:
        excluded.append("done")
    next_td = tensordict.get("next")
    if len(excluded):
        next_td = next_td.exclude(*excluded)
    select_tensordict = select_tensordict.update(next_td)

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


def check_env_specs(env, return_contiguous=True, check_dtype=True, seed=0):
    """Tests an environment specs against the results of short rollout.

    This test function should be used as a sanity check for an env wrapped with
    torchrl's EnvBase subclasses: any discrepency between the expected data and
    the data collected should raise an assertion error.

    A broken environment spec will likely make it impossible to use parallel
    environments.

    Args:
        env (EnvBase): the env for which the specs have to be checked against data.
        return_contiguous (bool, optional): if ``True``, the random rollout will be called with
            return_contiguous=True. This will fail in some cases (e.g. heterogeneous shapes
            of inputs/outputs). Defaults to True.
        check_dtype (bool, optional): if False, dtype checks will be skipped.
            Defaults to True.
        seed (int, optional): for reproducibility, a seed is set.

    Caution: this function resets the env seed. It should be used "offline" to
    check that an env is adequately constructed, but it may affect the seeding
    of an experiment and as such should be kept out of training scripts.

    """
    torch.manual_seed(seed)
    env.set_seed(seed)

    fake_tensordict = env.fake_tensordict().flatten_keys(".")
    real_tensordict = env.rollout(3, return_contiguous=return_contiguous)
    # # remove private keys
    # real_tensordict = real_tensordict.exclude(
    #     *[
    #         key
    #         for key in real_tensordict.keys(True)
    #         if (isinstance(key, str) and key.startswith("_"))
    #         or (
    #             isinstance(key, tuple) and any(subkey.startswith("_") for subkey in key)
    #         )
    #     ]
    # )
    real_tensordict = real_tensordict.flatten_keys(".")

    keys1 = set(fake_tensordict.keys(True))
    keys2 = set(real_tensordict.keys(True))
    if keys1 != keys2:
        raise AssertionError(
            "The keys of the fake tensordict and the one collected during rollout do not match:"
            f"Got fake-real: {keys1-keys2} and real-fake: {keys2-keys1}"
        )
    fake_tensordict = fake_tensordict.unsqueeze(real_tensordict.batch_dims - 1)
    fake_tensordict = fake_tensordict.expand(*real_tensordict.shape)
    fake_tensordict = fake_tensordict.to_tensordict()
    if (
        fake_tensordict.apply(lambda x: torch.zeros_like(x))
        != real_tensordict.apply(lambda x: torch.zeros_like(x))
    ).all():
        raise AssertionError(
            "zeroing the two tensordicts did not make them identical. "
            f"Check for discrepancies:\nFake=\n{fake_tensordict}\nReal=\n{real_tensordict}"
        )
    for key in keys2:
        if fake_tensordict[key].shape != real_tensordict[key].shape:
            raise AssertionError(
                f"The shapes of the real and fake tensordict don't match for key {key}. "
                f"Got fake={fake_tensordict[key].shape} and real={real_tensordict[key].shape}."
            )
        if check_dtype and (fake_tensordict[key].dtype != real_tensordict[key].dtype):
            raise AssertionError(
                f"The dtypes of the real and fake tensordict don't match for key {key}. "
                f"Got fake={fake_tensordict[key].dtype} and real={real_tensordict[key].dtype}."
            )

    # test dtypes
    real_tensordict = env.rollout(3)  # keep empty structures, for example dict()
    for key, value in real_tensordict[..., -1].items():
        _check_isin(key, value, env.observation_spec, env.input_spec)

    print("check_env_specs succeeded!")


def _check_isin(key, value, obs_spec, input_spec):
    if key in {"reward", "done"}:
        return
    elif key == "next":
        for _key, _value in value.items():
            _check_isin(_key, _value, obs_spec, input_spec)
        return
    elif key in input_spec["_action_spec"].keys(True):
        if not input_spec["_action_spec"][key].is_in(value):
            raise AssertionError(
                f"action_spec.is_in failed for key {key}. "
                f"Got action_spec={input_spec['_action_spec'][key]} and real={value}."
            )
        return

    elif key in input_spec.keys(True):
        if not input_spec[key].is_in(value):
            raise AssertionError(
                f"input_spec.is_in failed for key {key}. "
                f"Got input_spec={input_spec[key]} and real={value}."
            )
        return
    elif key in obs_spec.keys(True):
        if not obs_spec[key].is_in(value):
            raise AssertionError(
                f"obs_spec.is_in failed for key {key}. "
                f"Got obs_spec={obs_spec[key]} and real={value}."
            )
        return
    else:
        raise KeyError(
            f"key {key} was not found in input spec with keys {input_spec.keys(True)} or obs spec with keys {obs_spec.keys(True)}"
        )


def _selective_unsqueeze(tensor: torch.Tensor, batch_size: torch.Size, dim: int = -1):
    shape_len = len(tensor.shape)

    if shape_len < len(batch_size):
        raise RuntimeError(
            f"Tensor has less dims than batch_size. shape:{tensor.shape}, batch_size: {batch_size}"
        )
    if tensor.shape[: len(batch_size)] != batch_size:
        raise RuntimeError(
            f"Tensor does not have given batch_size. shape:{tensor.shape}, batch_size: {batch_size}"
        )

    if shape_len == len(batch_size):
        return tensor.unsqueeze(dim=dim)
    return tensor


class classproperty:
    """A class-property object.

    Usage: Allows for iterators coded as properties.
    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def _sort_keys(element):
    if isinstance(element, tuple):
        return "_-|-_".join(element)
    return element


def make_composite_from_td(data):
    """Creates a CompositeSpec instance from a tensordict, assuming all values are unbounded.

    Args:
        data (tensordict.TensorDict): a tensordict to be mapped onto a CompositeSpec.

    Examples:
        >>> from tensordict import TensorDict
        >>> data = TensorDict({
        ...     "obs": torch.randn(3),
        ...     "action": torch.zeros(2, dtype=torch.int),
        ...     "next": {"obs": torch.randn(3), "reward": torch.randn(1)}
        ... }, [])
        >>> spec = make_composite_from_td(data)
        >>> print(spec)
        CompositeSpec(
            obs: UnboundedContinuousTensorSpec(
                 shape=torch.Size([3]), space=None, device=cpu, dtype=torch.float32, domain=continuous),
            action: UnboundedContinuousTensorSpec(
                 shape=torch.Size([2]), space=None, device=cpu, dtype=torch.int32, domain=continuous),
            next: CompositeSpec(
                obs: UnboundedContinuousTensorSpec(
                     shape=torch.Size([3]), space=None, device=cpu, dtype=torch.float32, domain=continuous),
                reward: UnboundedContinuousTensorSpec(
                     shape=torch.Size([1]), space=ContinuousBox(minimum=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True), maximum=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)), device=cpu, dtype=torch.float32, domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
        >>> assert (spec.zero() == data.zero_()).all()
    """
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

    # custom funtion to convert a tensordict in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in data.items()
        },
        shape=data.shape,
    )
    return composite

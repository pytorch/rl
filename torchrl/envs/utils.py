# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import torch

from tensordict import is_tensor_collection, unravel_key
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
from tensordict.tensordict import LazyStackedTensorDict, NestedKey, TensorDictBase

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


from torchrl.data import CompositeSpec


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
    reward_key: NestedKey = "reward",
    done_key: NestedKey = "done",
    action_key: NestedKey = "action",
) -> TensorDictBase:
    """Creates a new tensordict that reflects a step in time of the input tensordict.

    Given a tensordict retrieved after a step, returns the :obj:`"next"` indexed-tensordict.
    The arguments allow for a precise control over what should be kept and what
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
        reward_key (key, optional): the key where the reward is written. Defaults
            to "reward".
        done_key (key, optional): the key where the done is written. Defaults
            to "done".
        action_key (key, optional): the key where the action is written. Defaults
            to "action".

    Returns:
         A new tensordict (or next_tensordict) containing the tensors of the t+1 step.

    Examples:
    This funtion allows for this kind of loop to be used:
        >>> from tensordict import TensorDict
        >>> import torch
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
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_done=True))  # "done" is dropped
        TensorDict(
            fields={
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_reward=False))  # "reward" is kept
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_action=False))  # "action" persists at the root
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, keep_other=False))  # "extra" is missing
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """
    if isinstance(tensordict, LazyStackedTensorDict):
        if next_tensordict is not None:
            next_tensordicts = next_tensordict.unbind(tensordict.stack_dim)
        else:
            next_tensordicts = [None] * len(tensordict.tensordicts)
        out = torch.stack(
            [
                step_mdp(
                    td,
                    next_tensordict=ntd,
                    keep_other=keep_other,
                    exclude_reward=exclude_reward,
                    exclude_done=exclude_done,
                    exclude_action=exclude_action,
                    reward_key=reward_key,
                    done_key=done_key,
                    action_key=action_key,
                )
                for td, ntd in zip(tensordict.tensordicts, next_tensordicts)
            ],
            tensordict.stack_dim,
        )
        if next_tensordict is not None:
            next_tensordict.update(out)
            return next_tensordict
        return out

    action_key = unravel_key(action_key)
    done_key = unravel_key(done_key)
    reward_key = unravel_key(reward_key)

    excluded = set()
    if exclude_reward:
        excluded = {reward_key}
    if exclude_done:
        excluded = excluded.union({done_key})
    if exclude_action:
        excluded = excluded.union({action_key})
    next_td = tensordict.get("next")
    out = next_td.empty()

    total_key = ()
    if keep_other:
        for key in tensordict.keys():
            if key != "next":
                _set(tensordict, out, key, total_key, excluded)
    elif not exclude_action:
        _set_single_key(tensordict, out, action_key)
    for key in next_td.keys():
        _set(next_td, out, key, total_key, excluded)
    if next_tensordict is not None:
        return next_tensordict.update(out)
    else:
        return out


def _set_single_key(source, dest, key, clone=False):
    # key should be already unraveled
    if isinstance(key, str):
        key = (key,)
    for k in key:
        val = source.get(k)
        if is_tensor_collection(val):
            new_val = dest.get(k, None)
            if new_val is None:
                new_val = val.empty()
                # dest.set(k, new_val)
                dest._set_str(k, new_val, inplace=False, validated=True)
            source = val
            dest = new_val
        else:
            if clone:
                val = val.clone()
            # dest.set(k, val)
            dest._set_str(k, val, inplace=False, validated=True)


def _set(source, dest, key, total_key, excluded):
    total_key = total_key + (key,)
    non_empty = False
    if unravel_key(total_key) not in excluded:
        val = source.get(key)
        if is_tensor_collection(val):
            new_val = dest.get(key, None)
            if new_val is None:
                new_val = val.empty()
            non_empty_local = False
            for subkey in val.keys():
                non_empty_local = (
                    _set(val, new_val, subkey, total_key, excluded) or non_empty_local
                )
            if non_empty_local:
                # dest.set(key, new_val)
                dest._set_str(key, new_val, inplace=False, validated=True)
            non_empty = non_empty_local
        else:
            non_empty = True
            # dest.set(key, val)
            dest._set_str(key, val, inplace=False, validated=True)
    return non_empty


def get_available_libraries():
    """Returns all the supported libraries."""
    return SUPPORTED_LIBRARIES


def _check_gym():
    """Returns True if the gym library is installed."""
    return importlib.util.find_spec("gym") is not None


def _check_gym_atari():
    """Returns True if the gym library is installed and atari envs can be found."""
    if not _check_gym():
        return False
    return importlib.util.find_spec("atari-py") is not None


def _check_mario():
    """Returns True if the "gym-super-mario-bros" library is installed."""
    return importlib.util.find_spec("gym-super-mario-bros") is not None


def _check_dmcontrol():
    """Returns True if the "dm-control" library is installed."""
    return importlib.util.find_spec("dm_control") is not None


def _check_dmlab():
    """Returns True if the "deepmind-lab" library is installed."""
    return importlib.util.find_spec("deepmind_lab") is not None


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


def _per_level_env_check(data0, data1, check_dtype):
    """Checks shape and dtype of two tensordicts, accounting for lazy stacks."""
    if isinstance(data0, LazyStackedTensorDict) and isinstance(
        data1, LazyStackedTensorDict
    ):
        if data0.stack_dim != data1.stack_dim:
            raise AssertionError(f"Stack dimension mismatch: {data0} vs {data1}.")
        for _data0, _data1 in zip(data0.tensordicts, data1.tensordicts):
            _per_level_env_check(_data0, _data1, check_dtype=check_dtype)
        return
    else:
        keys0 = set(data0.keys())
        keys1 = set(data1.keys())
        if keys0 != keys1:
            raise AssertionError(f"Keys mismatch: {keys0} vs {keys1}")
        for key in keys0:
            _data0 = data0[key]
            _data1 = data1[key]
            if _data0.shape != _data1.shape:
                raise AssertionError(
                    f"The shapes of the real and fake tensordict don't match for key {key}. "
                    f"Got fake={_data0.shape} and real={_data0.shape}."
                )
            if isinstance(_data0, TensorDictBase):
                _per_level_env_check(_data0, _data1, check_dtype=check_dtype)
            else:
                if check_dtype and (_data0.dtype != _data1.dtype):
                    raise AssertionError(
                        f"The dtypes of the real and fake tensordict don't match for key {key}. "
                        f"Got fake={_data0.dtype} and real={_data1.dtype}."
                    )


def check_env_specs(env, return_contiguous=True, check_dtype=True, seed=0):
    """Tests an environment specs against the results of short rollout.

    This test function should be used as a sanity check for an env wrapped with
    torchrl's EnvBase subclasses: any discrepancy between the expected data and
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

    fake_tensordict = env.fake_tensordict()
    real_tensordict = env.rollout(3, return_contiguous=return_contiguous)

    if return_contiguous:
        fake_tensordict = fake_tensordict.unsqueeze(real_tensordict.batch_dims - 1)
        fake_tensordict = fake_tensordict.expand(*real_tensordict.shape)
    else:
        fake_tensordict = torch.stack([fake_tensordict.clone() for _ in range(3)], -1)
    if (
        fake_tensordict.apply(lambda x: torch.zeros_like(x))
        != real_tensordict.apply(lambda x: torch.zeros_like(x))
    ).any():
        raise AssertionError(
            "zeroing the two tensordicts did not make them identical. "
            f"Check for discrepancies:\nFake=\n{fake_tensordict}\nReal=\n{real_tensordict}"
        )

    # Checks shapes and eventually dtypes of keys at all nesting levels
    _per_level_env_check(fake_tensordict, real_tensordict, check_dtype=check_dtype)

    # Check specs
    last_td = real_tensordict[..., -1]
    _action_spec = env.input_spec["_action_spec"]
    _state_spec = env.input_spec["_state_spec"]
    _obs_spec = env.output_spec["_observation_spec"]
    _reward_spec = env.output_spec["_reward_spec"]
    _done_spec = env.output_spec["_done_spec"]
    for name, spec in (
        ("action", _action_spec),
        ("state", _state_spec),
        ("done", _done_spec),
        ("obs", _obs_spec),
    ):
        if spec is None:
            spec = CompositeSpec(shape=env.batch_size, device=env.device)
        td = last_td.select(*spec.keys(True, True), strict=True)
        if not spec.is_in(td):
            raise AssertionError(
                f"spec check failed at root for spec {name}={spec} and data {td}."
            )
    for name, spec in (
        ("reward", _reward_spec),
        ("done", _done_spec),
        ("obs", _obs_spec),
    ):
        if spec is None:
            spec = CompositeSpec(shape=env.batch_size, device=env.device)
        td = last_td.get("next").select(*spec.keys(True, True), strict=True)
        if not spec.is_in(td):
            raise AssertionError(
                f"spec check failed at root for spec {name}={spec} and data {td}."
            )

    print("check_env_specs succeeded!")


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
        element = unravel_key(element)
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

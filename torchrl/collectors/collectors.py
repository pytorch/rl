# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import _pickle
import abc
import inspect
import os
import queue
import sys
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

from multiprocessing import connection, queues
from multiprocessing.managers import SyncManager

from textwrap import indent
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import multiprocessing as mp
from torch.utils.data import IterableDataset

from torchrl._utils import (
    _check_for_faulty_process,
    _ProcessNoWarn,
    accept_remote_rref_udf_invocation,
    prod,
    RL_WARNINGS,
    VERBOSE,
)
from torchrl.collectors.utils import split_trajectories
from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import StepCounter, TransformedEnv
from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    _convert_exploration_type,
    ExplorationType,
    set_exploration_type,
)

_TIMEOUT = 1.0
_MIN_TIMEOUT = 1e-3  # should be several orders of magnitude inferior wrt time spent collecting a trajectory
_MAX_IDLE_COUNT = int(os.environ.get("MAX_IDLE_COUNT", 1000))

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

_is_osx = sys.platform.startswith("darwin")


class RandomPolicy:
    """A random policy for data collectors.

    This is a wrapper around the action_spec.rand method.

    Args:
        action_spec: TensorSpec object describing the action specs

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.data.tensor_specs import BoundedTensorSpec
        >>> action_spec = BoundedTensorSpec(-torch.ones(3), torch.ones(3))
        >>> actor = RandomPolicy(action_spec=action_spec)
        >>> td = actor(TensorDict({}, batch_size=[])) # selects a random action in the cube [-1; 1]
    """

    def __init__(self, action_spec: TensorSpec, action_key: NestedKey = "action"):
        self.action_spec = action_spec
        self.action_key = action_key

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        if isinstance(self.action_spec, CompositeSpec):
            return td.update(self.action_spec.rand())
        else:
            return td.set(self.action_key, self.action_spec.rand())


class _Interruptor:
    """A class for managing the collection state of a process.

    This class provides methods to start and stop collection, and to check
    whether collection has been stopped. The collection state is protected
    by a lock to ensure thread-safety.
    """

    # interrupter vs interruptor: google trends seems to indicate that "or" is more
    # widely used than "er" even if my IDE complains about that...
    def __init__(self):
        self._collect = True
        self._lock = mp.Lock()

    def start_collection(self):
        with self._lock:
            self._collect = True

    def stop_collection(self):
        with self._lock:
            self._collect = False

    def collection_stopped(self):
        with self._lock:
            return self._collect is False


class _InterruptorManager(SyncManager):
    """A custom SyncManager for managing the collection state of a process.

    This class extends the SyncManager class and allows to share an Interruptor object
    between processes.
    """

    pass


_InterruptorManager.register("_Interruptor", _Interruptor)


def recursive_map_to_cpu(dictionary: OrderedDict) -> OrderedDict:
    """Maps the tensors to CPU through a nested dictionary."""
    return OrderedDict(
        **{
            k: recursive_map_to_cpu(item)
            if isinstance(item, OrderedDict)
            else item.cpu()
            if isinstance(item, torch.Tensor)
            else item
            for k, item in dictionary.items()
        }
    )


def _policy_is_tensordict_compatible(policy: nn.Module):
    sig = inspect.signature(policy.forward)

    if isinstance(policy, TensorDictModuleBase):
        return True
    if (
        len(sig.parameters) == 1
        and hasattr(policy, "in_keys")
        and hasattr(policy, "out_keys")
    ):
        warnings.warn(
            "Passing a policy that is not a TensorDictModuleBase subclass but has in_keys and out_keys "
            "will soon be deprecated. We'd like to motivate our users to inherit from this class (which "
            "has very few restrictions) to make the experience smoother.",
            category=DeprecationWarning,
        )
        # if the policy is a TensorDictModule or takes a single argument and defines
        # in_keys and out_keys then we assume it can already deal with TensorDict input
        # to forward and we return True
        return True
    elif not hasattr(policy, "in_keys") and not hasattr(policy, "out_keys"):
        # if it's not a TensorDictModule, and in_keys and out_keys are not defined then
        # we assume no TensorDict compatibility and will try to wrap it.
        return False

    # if in_keys or out_keys were defined but policy is not a TensorDictModule or
    # accepts multiple arguments then it's likely the user is trying to do something
    # that will have undetermined behaviour, we raise an error
    raise TypeError(
        "Received a policy that defines in_keys or out_keys and also expects multiple "
        "arguments to policy.forward. If the policy is compatible with TensorDict, it "
        "should take a single argument of type TensorDict to policy.forward and define "
        "both in_keys and out_keys. Alternatively, policy.forward can accept "
        "arbitrarily many tensor inputs and leave in_keys and out_keys undefined and "
        "TorchRL will attempt to automatically wrap the policy with a TensorDictModule."
    )


class DataCollectorBase(IterableDataset, metaclass=abc.ABCMeta):
    """Base class for data collectors."""

    _iterator = None

    def _get_policy_and_device(
        self,
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ] = None,
        device: Optional[DEVICE_TYPING] = None,
        observation_spec: TensorSpec = None,
    ) -> Tuple[TensorDictModule, torch.device, Union[None, Callable[[], dict]]]:
        """Util method to get a policy and its device given the collector __init__ inputs.

        From a policy and a device, assigns the self.device attribute to
        the desired device and maps the policy onto it or (if the device is
        ommitted) assigns the self.device attribute to the policy device.

        Args:
            create_env_fn (Callable or list of callables): an env creator
                function (or a list of creators)
            create_env_kwargs (dictionary): kwargs for the env creator
            policy (TensorDictModule, optional): a policy to be used
            device (int, str or torch.device, optional): device where to place
                the policy
            observation_spec (TensorSpec, optional): spec of the observations

        """
        if policy is None:
            if not hasattr(self, "env") or self.env is None:
                raise ValueError(
                    "env must be provided to _get_policy_and_device if policy is None"
                )
            policy = RandomPolicy(self.env.input_spec["full_action_spec"])
        elif isinstance(policy, nn.Module):
            # TODO: revisit these checks when we have determined whether arbitrary
            # callables should be supported as policies.
            if not _policy_is_tensordict_compatible(policy):
                # policy is a nn.Module that doesn't operate on tensordicts directly
                # so we attempt to auto-wrap policy with TensorDictModule
                if observation_spec is None:
                    raise ValueError(
                        "Unable to read observation_spec from the environment. This is "
                        "required to check compatibility of the environment and policy "
                        "since the policy is a nn.Module that operates on tensors "
                        "rather than a TensorDictModule or a nn.Module that accepts a "
                        "TensorDict as input and defines in_keys and out_keys."
                    )

                try:
                    # signature modified by make_functional
                    sig = policy.forward.__signature__
                except AttributeError:
                    sig = inspect.signature(policy.forward)
                required_params = {
                    str(k)
                    for k, p in sig.parameters.items()
                    if p.default is inspect._empty
                }
                next_observation = {
                    key: value for key, value in observation_spec.rand().items()
                }
                # we check if all the mandatory params are there
                if not required_params.difference(set(next_observation)):
                    in_keys = [str(k) for k in sig.parameters if k in next_observation]
                    if not hasattr(self, "env") or self.env is None:
                        out_keys = ["action"]
                    else:
                        out_keys = self.env.action_keys
                    output = policy(**next_observation)

                    if isinstance(output, tuple):
                        out_keys.extend(f"output{i+1}" for i in range(len(output) - 1))

                    policy = TensorDictModule(
                        policy, in_keys=in_keys, out_keys=out_keys
                    )
                else:
                    raise TypeError(
                        f"""Arguments to policy.forward are incompatible with entries in
env.observation_spec (got incongruent signatures: fun signature is {set(sig.parameters)} vs specs {set(next_observation)}).
If you want TorchRL to automatically wrap your policy with a TensorDictModule
then the arguments to policy.forward must correspond one-to-one with entries
in env.observation_spec that are prefixed with 'next_'. For more complex
behaviour and more control you can consider writing your own TensorDictModule.
"""
                    )

        try:
            policy_device = next(policy.parameters()).device
        except Exception:
            policy_device = (
                torch.device(device) if device is not None else torch.device("cpu")
            )

        device = torch.device(device) if device is not None else policy_device
        get_weights_fn = None
        if policy_device != device:
            param_and_buf = dict(policy.named_parameters())
            param_and_buf.update(dict(policy.named_buffers()))

            def get_weights_fn(param_and_buf=param_and_buf):
                return TensorDict(param_and_buf, []).apply(lambda x: x.data)

            policy_cast = deepcopy(policy).requires_grad_(False).to(device)
            # here things may break bc policy.to("cuda") gives us weights on cuda:0 (same
            # but different)
            try:
                device = next(policy_cast.parameters()).device
            except StopIteration:  # noqa
                pass
        else:
            policy_cast = policy
        return policy_cast, device, get_weights_fn

    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        """Updates the policy weights if the policy of the data collector and the trained policy live on different devices.

        Args:
            policy_weights (TensorDictBase, optional): if provided, a TensorDict containing
                the weights of the policy to be used for the udpdate.

        """
        if policy_weights is not None:
            self.policy_weights.apply(lambda x: x.data).update_(policy_weights)
        elif self.get_weights_fn is not None:
            self.policy_weights.apply(lambda x: x.data).update_(self.get_weights_fn())

    def __iter__(self) -> Iterator[TensorDictBase]:
        return self.iterator()

    def next(self):
        try:
            if self._iterator is None:
                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            out.clear_device_()
            return out
        except StopIteration:
            return None

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError

    @abc.abstractmethod
    def iterator(self) -> Iterator[TensorDictBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string


@accept_remote_rref_udf_invocation
class SyncDataCollector(DataCollectorBase):
    """Generic data collector for RL problems. Requires an environment constructor and a policy.

    Args:
        create_env_fn (Callable): a callable that returns an instance of
            :class:`~torchrl.envs.EnvBase` class.
        policy (Callable): Policy to be executed in the environment.
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
            If ``None`` is provided, the policy used will be a
            :class:`~torchrl.collectors.RandomPolicy` instance with the environment
            ``action_spec``.
        frames_per_batch (int): A keyword-only argument representing the total
            number of elements in a batch.
        total_frames (int): A keyword-only argument representing the total
            number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
             Endless collectors can be created by passing ``total_frames=-1``.
        device (int, str or torch.device, optional): The device on which the
            policy will be placed.
            If it differs from the input policy device, the
            :meth:`~.update_policy_weights_` method should be queried
            at appropriate times during the training loop to accommodate for
            the lag between parameter configuration at various times.
            Defaults to ``None`` (i.e. policy is kept on its original device).
        storing_device (int, str or torch.device, optional): The device on which
            the output :class:`tensordict.TensorDict` will be stored. For long
            trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``"cpu"``.
        create_env_kwargs (dict, optional): Dictionary of kwargs for
            ``create_env_fn``.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span over multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e. no maximum number of steps).
        init_random_frames (int, optional): Number of frames for which the
            policy is ignored before it is called. This feature is mainly
            intended to be used in offline/model-based settings, where a
            batch of random trajectories can be used to initialize training.
            If provided, it will be rounded up to the closest multiple of frames_per_batch.
            Defaults to ``None`` (i.e. no random frames).
        reset_at_each_iter (bool, optional): Whether environments should be reset
            at the beginning of a batch collection.
            Defaults to ``False``.
        postproc (Callable, optional): A post-processing transform, such as
            a :class:`~torchrl.envs.Transform` or a :class:`~torchrl.data.postprocs.MultiStep`
            instance.
            Defaults to ``None``.
        split_trajs (bool, optional): Boolean indicating whether the resulting
            TensorDict should be split according to the trajectories.
            See :func:`~torchrl.collectors.utils.split_trajectories` for more
            information.
            Defaults to ``False``.
        exploration_type (ExplorationType, optional): interaction mode to be used when
            collecting data. Must be one of ``ExplorationType.RANDOM``, ``ExplorationType.MODE`` or
            ``ExplorationType.MEAN``.
            Defaults to ``ExplorationType.RANDOM``
        return_same_td (bool, optional): if ``True``, the same TensorDict
            will be returned at each iteration, with its values
            updated. This feature should be used cautiously: if the same
            tensordict is added to a replay buffer for instance,
            the whole content of the buffer will be identical.
            Default is False.
        interruptor (_Interruptor, optional):
            An _Interruptor object that can be used from outside the class to control rollout collection.
            The _Interruptor class has methods ´start_collection´ and ´stop_collection´, which allow to implement
            strategies such as preeptively stopping rollout collection.
            Default is ``False``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> collector = SyncDataCollector(
        ...     create_env_fn=env_maker,
        ...     policy=policy,
        ...     total_frames=2000,
        ...     max_frames_per_traj=50,
        ...     frames_per_batch=200,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=False,
        ...     device="cpu",
        ...     storing_device="cpu",
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)
        >>> del collector

    The collector delivers batches of data that are marked with a ``"time"``
    dimension.

    Examples:
        >>> assert data.names[-1] == "time"

    """

    def __init__(
        self,
        create_env_fn: Union[
            EnvBase, "EnvCreator", Sequence[Callable[[], EnvBase]]  # noqa: F821
        ],  # noqa: F821
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ],
        *,
        frames_per_batch: int,
        total_frames: int,
        device: DEVICE_TYPING = None,
        storing_device: DEVICE_TYPING = None,
        create_env_kwargs: dict | None = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        split_trajs: bool | None = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        exploration_mode=None,
        return_same_td: bool = False,
        reset_when_done: bool = True,
        interruptor=None,
    ):
        from torchrl.envs.batched_envs import _BatchedEnv

        self.closed = True

        exploration_type = _convert_exploration_type(
            exploration_mode=exploration_mode, exploration_type=exploration_type
        )
        if create_env_kwargs is None:
            create_env_kwargs = {}
        if not isinstance(create_env_fn, EnvBase):
            env = create_env_fn(**create_env_kwargs)
        else:
            env = create_env_fn
            if create_env_kwargs:
                if not isinstance(env, _BatchedEnv):
                    raise RuntimeError(
                        "kwargs were passed to SyncDataCollector but they can't be set "
                        f"on environment of type {type(create_env_fn)}."
                    )
                env.update_kwargs(create_env_kwargs)

        if storing_device is None:
            if device is not None:
                storing_device = device
            elif policy is not None:
                try:
                    policy_device = next(policy.parameters()).device
                except (AttributeError, StopIteration):
                    policy_device = torch.device("cpu")
                storing_device = policy_device
            else:
                storing_device = torch.device("cpu")

        self.storing_device = torch.device(storing_device)
        self.env: EnvBase = env
        self.closed = False
        if not reset_when_done:
            raise ValueError("reset_when_done is deprectated.")
        self.reset_when_done = reset_when_done
        self.n_env = self.env.batch_size.numel()

        (self.policy, self.device, self.get_weights_fn,) = self._get_policy_and_device(
            policy=policy,
            device=device,
            observation_spec=self.env.observation_spec,
        )

        if isinstance(self.policy, nn.Module):
            self.policy_weights = TensorDict(dict(self.policy.named_parameters()), [])
            self.policy_weights.update(
                TensorDict(dict(self.policy.named_buffers()), [])
            )
        else:
            self.policy_weights = TensorDict({}, [])

        self.env: EnvBase = self.env.to(self.device)
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        if self.max_frames_per_traj is not None and self.max_frames_per_traj > 0:
            # let's check that there is no StepCounter yet
            for key in self.env.output_spec.keys(True, True):
                if isinstance(key, str):
                    key = (key,)
                if "step_count" in key:
                    raise ValueError(
                        "A 'step_count' key is already present in the environment "
                        "and the 'max_frames_per_traj' argument may conflict with "
                        "a 'StepCounter' that has already been set. "
                        "Possible solutions: Set max_frames_per_traj to 0 or "
                        "remove the StepCounter limit from the environment transforms."
                    )
            env = self.env = TransformedEnv(
                self.env, StepCounter(max_steps=self.max_frames_per_traj)
            )

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        if (
            init_random_frames is not None
            and init_random_frames % frames_per_batch != 0
            and RL_WARNINGS
        ):
            warnings.warn(
                f"init_random_frames ({init_random_frames}) is not exactly a multiple of frames_per_batch ({frames_per_batch}), "
                f" this results in more init_random_frames than requested"
                f" ({-(-init_random_frames // frames_per_batch) * frames_per_batch})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )

        self.postproc = postproc
        if self.postproc is not None and hasattr(self.postproc, "to"):
            self.postproc.to(self.storing_device)
        if frames_per_batch % self.n_env != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch ({frames_per_batch}) is not exactly divisible by the number of batched environments ({self.n_env}), "
                f" this results in more frames_per_batch per iteration that requested"
                f" ({-(-frames_per_batch // self.n_env) * self.n_env})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.frames_per_batch = -(-frames_per_batch // self.n_env)
        self.exploration_type = (
            exploration_type if exploration_type else DEFAULT_EXPLORATION_TYPE
        )
        self.return_same_td = return_same_td

        self._tensordict = env.reset()
        traj_ids = torch.arange(self.n_env, device=env.device).view(self.env.batch_size)
        self._tensordict.set(
            ("collector", "traj_ids"),
            traj_ids,
        )

        with torch.no_grad():
            self._tensordict_out = self.env.fake_tensordict()
        # If the policy has a valid spec, we use it
        if (
            hasattr(self.policy, "spec")
            and self.policy.spec is not None
            and all(v is not None for v in self.policy.spec.values(True, True))
        ):
            if any(
                key not in self._tensordict_out.keys(isinstance(key, tuple))
                for key in self.policy.spec.keys(True, True)
            ):
                # if policy spec is non-empty, all the values are not None and the keys
                # match the out_keys we assume the user has given all relevant information
                # the policy could have more keys than the env:
                policy_spec = self.policy.spec
                if policy_spec.ndim < self._tensordict_out.ndim:
                    policy_spec = policy_spec.expand(self._tensordict_out.shape)
                for key, spec in policy_spec.items(True, True):
                    if key in self._tensordict_out.keys(isinstance(key, tuple)):
                        continue
                    self._tensordict_out.set(key, spec.zero())

        else:
            # otherwise, we perform a small number of steps with the policy to
            # determine the relevant keys with which to pre-populate _tensordict_out.
            # This is the safest thing to do if the spec has None fields or if there is
            # no spec at all.
            # See #505 for additional context.
            self._tensordict_out.update(self._tensordict)
            with torch.no_grad():
                self._tensordict_out = self.policy(self._tensordict_out.to(self.device))

        self._tensordict_out = (
            self._tensordict_out.unsqueeze(-1)
            .expand(*env.batch_size, self.frames_per_batch)
            .clone()
            .zero_()
        )
        # in addition to outputs of the policy, we add traj_ids to
        # _tensordict_out which will be collected during rollout
        self._tensordict_out = self._tensordict_out.to(self.storing_device)
        self._tensordict_out.set(
            ("collector", "traj_ids"),
            torch.zeros(
                *self._tensordict_out.batch_size,
                dtype=torch.int64,
                device=self.storing_device,
            ),
        )
        self._tensordict_out.refine_names(..., "time")

        if split_trajs is None:
            split_trajs = False
        self.split_trajs = split_trajs
        self._exclude_private_keys = True
        self.interruptor = interruptor
        self._frames = 0
        self._iter = -1

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Sets the seeds of the environments stored in the DataCollector.

        Args:
            seed (int): integer representing the seed to be used for the environment.
            static_seed(bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            Output seed. This is useful when more than one environment is contained in the DataCollector, as the
            seed will be incremented for each of these. The resulting seed is the seed of the last environment.

        Examples:
            >>> from torchrl.envs import ParallelEnv
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> env_fn = lambda: GymEnv("Pendulum-v1")
            >>> env_fn_parallel = ParallelEnv(6, env_fn)
            >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
            >>> collector = SyncDataCollector(env_fn_parallel, policy, total_frames=300, frames_per_batch=100)
            >>> out_seed = collector.set_seed(1)  # out_seed = 6

        """
        return self.env.set_seed(seed, static_seed=static_seed)

    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterates through the DataCollector.

        Yields: TensorDictBase objects containing (chunks of) trajectories

        """
        if self.storing_device.type == "cuda":
            stream = torch.cuda.Stream(self.storing_device, priority=-1)
            event = stream.record_event()
        else:
            event = None
            stream = None
        with torch.cuda.stream(stream):
            total_frames = self.total_frames

            while self._frames < self.total_frames:
                self._iter += 1
                tensordict_out = self.rollout()
                self._frames += tensordict_out.numel()
                if self._frames >= total_frames:
                    self.env.close()

                if self.split_trajs:
                    tensordict_out = split_trajectories(
                        tensordict_out, prefix="collector"
                    )
                if self.postproc is not None:
                    tensordict_out = self.postproc(tensordict_out)
                if self._exclude_private_keys:

                    def is_private(key):
                        if isinstance(key, str) and key.startswith("_"):
                            return True
                        if isinstance(key, tuple) and any(
                            _key.startswith("_") for _key in key
                        ):
                            return True
                        return False

                    excluded_keys = [
                        key for key in tensordict_out.keys(True) if is_private(key)
                    ]
                    tensordict_out = tensordict_out.exclude(
                        *excluded_keys, inplace=True
                    )
                if self.return_same_td:
                    # This is used with multiprocessed collectors to use the buffers
                    # stored in the tensordict.
                    if event is not None:
                        event.record()
                        event.synchronize()
                    yield tensordict_out
                else:
                    # we must clone the values, as the tensordict is updated in-place.
                    # otherwise the following code may break:
                    # >>> for i, data in enumerate(collector):
                    # >>>      if i == 0:
                    # >>>          data0 = data
                    # >>>      elif i == 1:
                    # >>>          data1 = data
                    # >>>      else:
                    # >>>          break
                    # >>> assert data0["done"] is not data1["done"]
                    yield tensordict_out.clone()

    def _update_traj_ids(self, tensordict) -> None:
        # we can't use the reset keys because they're gone
        traj_sop = _aggregate_end_of_traj(
            tensordict.get("next"), done_keys=self.env.done_keys
        )
        if traj_sop.any():
            traj_ids = self._tensordict.get(("collector", "traj_ids"))
            traj_ids = traj_ids.clone()
            traj_ids[traj_sop] = traj_ids.max() + torch.arange(
                1, traj_sop.sum() + 1, device=traj_ids.device
            )
            self._tensordict.set(("collector", "traj_ids"), traj_ids)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._tensordict.update(self.env.reset())

        # self._tensordict.fill_(("collector", "step_count"), 0)
        self._tensordict_out.fill_(("collector", "traj_ids"), -1)
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._tensordict)
                else:
                    self.policy(self._tensordict)
                tensordict, tensordict_ = self.env.step_and_maybe_reset(
                    self._tensordict
                )
                self._tensordict = tensordict_.set(
                    "collector", tensordict.get("collector").clone(False)
                )
                tensordicts.append(
                    tensordict.to(self.storing_device, non_blocking=True)
                )

                self._update_traj_ids(tensordict)
                if (
                    self.interruptor is not None
                    and self.interruptor.collection_stopped()
                ):
                    try:
                        torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out[: t + 1],
                        )
                    except RuntimeError:
                        with self._tensordict_out.unlock_():
                            torch.stack(
                                tensordicts,
                                self._tensordict_out.ndim - 1,
                                out=self._tensordict_out[: t + 1],
                            )
                    break
            else:
                try:
                    self._tensordict_out = torch.stack(
                        tensordicts,
                        self._tensordict_out.ndim - 1,
                        out=self._tensordict_out,
                    )
                except RuntimeError:
                    with self._tensordict_out.unlock_():
                        self._tensordict_out = torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out,
                        )
        return self._tensordict_out

    def reset(self, index=None, **kwargs) -> None:
        """Resets the environments to a new initial state."""
        # metadata
        md = self._tensordict.get("collector").clone()
        if index is not None:
            # check that the env supports partial reset
            if prod(self.env.batch_size) == 0:
                raise RuntimeError("resetting unique env with index is not permitted.")
            for reset_key, done_keys in zip(
                self.env.reset_keys, self.env.done_keys_groups
            ):
                _reset = torch.zeros(
                    self.env.full_done_spec[done_keys[0]].shape,
                    dtype=torch.bool,
                    device=self.env.device,
                )
                _reset[index] = 1
                self._tensordict.set(reset_key, _reset)
        else:
            _reset = None
            self._tensordict.zero_()

        self._tensordict.update(self.env.reset(**kwargs))
        md["traj_ids"] = md["traj_ids"] - md["traj_ids"].min()
        self._tensordict["collector"] = md

    def shutdown(self) -> None:
        """Shuts down all workers and/or closes the local environment."""
        if not self.closed:
            self.closed = True
            del self._tensordict, self._tensordict_out
            if not self.env.is_closed:
                self.env.close()
            del self.env
        return

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # an AttributeError will typically be raised if the collector is deleted when the program ends.
            # In the future, insignificant changes to the close method may change the error type.
            # We excplicitely assume that any error raised during closure in
            # __del__ will not affect the program.
            pass

    def state_dict(self) -> OrderedDict:
        """Returns the local state_dict of the data collector (environment and policy).

        Returns:
            an ordered dictionary with fields :obj:`"policy_state_dict"` and
            `"env_state_dict"`.

        """
        from torchrl.envs.batched_envs import _BatchedEnv

        if isinstance(self.env, TransformedEnv):
            env_state_dict = self.env.transform.state_dict()
        elif isinstance(self.env, _BatchedEnv):
            env_state_dict = self.env.state_dict()
        else:
            env_state_dict = OrderedDict()

        if hasattr(self.policy, "state_dict"):
            policy_state_dict = self.policy.state_dict()
            state_dict = OrderedDict(
                policy_state_dict=policy_state_dict,
                env_state_dict=env_state_dict,
            )
        else:
            state_dict = OrderedDict(env_state_dict=env_state_dict)

        state_dict.update({"frames": self._frames, "iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        """Loads a state_dict on the environment and policy.

        Args:
            state_dict (OrderedDict): ordered dictionary containing the fields
                `"policy_state_dict"` and :obj:`"env_state_dict"`.

        """
        strict = kwargs.get("strict", True)
        if strict or "env_state_dict" in state_dict:
            self.env.load_state_dict(state_dict["env_state_dict"], **kwargs)
        if strict or "policy_state_dict" in state_dict:
            self.policy.load_state_dict(state_dict["policy_state_dict"], **kwargs)
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]

    def __repr__(self) -> str:
        env_str = indent(f"env={self.env}", 4 * " ")
        policy_str = indent(f"policy={self.policy}", 4 * " ")
        td_out_str = indent(f"td_out={self._tensordict_out}", 4 * " ")
        string = (
            f"{self.__class__.__name__}("
            f"\n{env_str},"
            f"\n{policy_str},"
            f"\n{td_out_str},"
            f"\nexploration={self.exploration_type})"
        )
        return string


class _MultiDataCollector(DataCollectorBase):
    """Runs a given number of DataCollectors on separate processes.

    Args:
        create_env_fn (List[Callabled]): list of Callables, each returning an
            instance of :class:`~torchrl.envs.EnvBase`.
        policy (Callable, optional): Instance of TensorDictModule class.
            Must accept TensorDictBase object as input.
            If ``None`` is provided, the policy used will be a
            :class:`RandomPolicy` instance with the environment
            ``action_spec``.
        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
        total_frames (int): A keyword-only argument representing the
            total number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
             Endless collectors can be created by passing ``total_frames=-1``.
        device (int, str, torch.device or sequence of such, optional):
            The device on which the policy will be placed.
            If it differs from the input policy device, the
            :meth:`~.update_policy_weights_` method should be queried
            at appropriate times during the training loop to accommodate for
            the lag between parameter configuration at various times.
            If necessary, a list of devices can be passed in which case each
            element will correspond to the designated device of a sub-collector.
            Defaults to ``None`` (i.e. policy is kept on its original device).
        storing_device (int, str, torch.device or sequence of such, optional):
            The device on which the output :class:`tensordict.TensorDict` will
            be stored. For long trajectories, it may be necessary to store the
            data on a different device than the one where the policy and env
            are executed.
            If necessary, a list of devices can be passed in which case each
            element will correspond to the designated storing device of a
            sub-collector.
            Defaults to ``"cpu"``.
        create_env_kwargs (dict, optional): A dictionary with the
            keyword arguments used to create an environment. If a list is
            provided, each of its elements will be assigned to a sub-collector.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span over multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e. no maximum number of steps).
        init_random_frames (int, optional): Number of frames for which the
            policy is ignored before it is called. This feature is mainly
            intended to be used in offline/model-based settings, where a
            batch of random trajectories can be used to initialize training.
            If provided, it will be rounded up to the closest multiple of frames_per_batch.
            Defaults to ``None`` (i.e. no random frames).
        reset_at_each_iter (bool, optional): Whether environments should be reset
            at the beginning of a batch collection.
            Defaults to ``False``.
        postproc (Callable, optional): A post-processing transform, such as
            a :class:`~torchrl.envs.Transform` or a :class:`~torchrl.data.postprocs.MultiStep`
            instance.
            Defaults to ``None``.
        split_trajs (bool, optional): Boolean indicating whether the resulting
            TensorDict should be split according to the trajectories.
            See :func:`~torchrl.collectors.utils.split_trajectories` for more
            information.
            Defaults to ``False``.
        exploration_type (ExplorationType, optional): interaction mode to be used when
            collecting data. Must be one of ``ExplorationType.RANDOM``, ``ExplorationType.MODE`` or
            ``ExplorationType.MEAN``.
            Defaults to ``ExplorationType.RANDOM``
        return_same_td (bool, optional): if ``True``, the same TensorDict
            will be returned at each iteration, with its values
            updated. This feature should be used cautiously: if the same
            tensordict is added to a replay buffer for instance,
            the whole content of the buffer will be identical.
            Default is ``False``.
        reset_when_done (bool, optional): if ``True`` (default), an environment
            that return a ``True`` value in its ``"done"`` or ``"truncated"``
            entry will be reset at the corresponding indices.
        update_at_each_batch (boolm optional): if ``True``, :meth:`~.update_policy_weight_()`
            will be called before (sync) or after (async) each data collection.
            Defaults to ``False``.
        preemptive_threshold (float, optional): a value between 0.0 and 1.0 that specifies the ratio of workers
            that will be allowed to finished collecting their rollout before the rest are forced to end early.
        num_threads (int, optional): number of threads for this process.
            Defaults to the number of workers.
        num_sub_threads (int, optional): number of threads of the subprocesses.
            Should be equal to one plus the number of processes launched within
            each subprocess (or one if a single process is launched).
            Defaults to 1 for safety: if none is indicated, launching multiple
            workers may charge the cpu load too much and harm performance.

    """

    def __init__(
        self,
        create_env_fn: Sequence[Callable[[], EnvBase]],
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ],
        *,
        frames_per_batch: int = 200,
        total_frames: Optional[int] = -1,
        device: DEVICE_TYPING = None,
        storing_device: Optional[Union[DEVICE_TYPING, Sequence[DEVICE_TYPING]]] = None,
        create_env_kwargs: Optional[Sequence[dict]] = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        split_trajs: Optional[bool] = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        exploration_mode=None,
        reset_when_done: bool = True,
        preemptive_threshold: float = None,
        update_at_each_batch: bool = False,
        devices=None,
        storing_devices=None,
        num_threads: int = None,
        num_sub_threads: int = 1,
    ):
        exploration_type = _convert_exploration_type(
            exploration_mode=exploration_mode, exploration_type=exploration_type
        )
        self.closed = True
        if num_threads is None:
            num_threads = len(create_env_fn) + 1  # 1 more thread for this proc
        self.num_sub_threads = num_sub_threads
        self.num_threads = num_threads
        self.create_env_fn = create_env_fn
        self.num_workers = len(create_env_fn)
        self.create_env_kwargs = (
            create_env_kwargs
            if create_env_kwargs is not None
            else [{} for _ in range(self.num_workers)]
        )
        # Preparing devices:
        # We want the user to be able to choose, for each worker, on which
        # device will the policy live and which device will be used to store
        # data. Those devices may or may not match.
        # One caveat is that, if there is only one device for the policy, and
        # if there are multiple workers, sending the same device and policy
        # to be copied to each worker will result in multiple copies of the
        # same policy on the same device.
        # To go around this, we do the copies of the policy in the server
        # (this object) to each possible device, and send to all the
        # processes their copy of the policy.
        if devices is not None:
            if device is not None:
                raise ValueError("Cannot pass both devices and device")
            warnings.warn(
                "`devices` keyword argument will soon be deprecated from multiprocessed collectors. "
                "Please use `device` instead."
            )
            device = devices
        if storing_devices is not None:
            if storing_device is not None:
                raise ValueError("Cannot pass both storing_devices and storing_device")
            warnings.warn(
                "`storing_devices` keyword argument will soon be deprecated from multiprocessed collectors. "
                "Please use `storing_device` instead."
            )
            storing_device = storing_devices

        def device_err_msg(device_name, devices_list):
            return (
                f"The length of the {device_name} argument should match the "
                f"number of workers of the collector. Got len("
                f"create_env_fn)={self.num_workers} and len("
                f"storing_device)={len(devices_list)}"
            )

        if isinstance(device, (str, int, torch.device)):
            device = [torch.device(device) for _ in range(self.num_workers)]
        elif device is None:
            device = [None for _ in range(self.num_workers)]
        elif isinstance(device, Sequence):
            if len(device) != self.num_workers:
                raise RuntimeError(device_err_msg("devices", device))
            device = [torch.device(_device) for _device in device]
        else:
            raise ValueError(
                "devices should be either None, a torch.device or equivalent "
                "or an iterable of devices. "
                f"Found {type(device)} instead."
            )
        self._policy_dict = {}
        self._policy_weights_dict = {}
        self._get_weights_fn_dict = {}

        for i, (_device, create_env, kwargs) in enumerate(
            zip(device, self.create_env_fn, self.create_env_kwargs)
        ):
            if _device in self._policy_dict:
                device[i] = _device
                continue

            if hasattr(create_env, "observation_spec"):
                observation_spec = create_env.observation_spec
            else:
                try:
                    observation_spec = create_env(**kwargs).observation_spec
                except:  # noqa
                    observation_spec = None

            _policy, _device, _get_weight_fn = self._get_policy_and_device(
                policy=policy, device=_device, observation_spec=observation_spec
            )
            self._policy_dict[_device] = _policy
            if isinstance(_policy, nn.Module):
                param_dict = dict(_policy.named_parameters())
                param_dict.update(_policy.named_buffers())
                self._policy_weights_dict[_device] = TensorDict(param_dict, [])
            else:
                self._policy_weights_dict[_device] = TensorDict({}, [])

            self._get_weights_fn_dict[_device] = _get_weight_fn
            device[i] = _device
        self.device = device

        if storing_device is None:
            self.storing_device = self.device
        else:
            if isinstance(storing_device, (str, int, torch.device)):
                self.storing_device = [
                    torch.device(storing_device) for _ in range(self.num_workers)
                ]
            elif isinstance(storing_device, Sequence):
                if len(storing_device) != self.num_workers:
                    raise RuntimeError(
                        device_err_msg("storing_devices", storing_device)
                    )
                self.storing_device = [
                    torch.device(_storing_device) for _storing_device in storing_device
                ]
            else:
                raise ValueError(
                    "storing_devices should be either a torch.device or equivalent or an iterable of devices. "
                    f"Found {type(storing_device)} instead."
                )

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.postprocs = postproc
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.reset_when_done = reset_when_done
        if split_trajs is None:
            split_trajs = False
        elif not self.reset_when_done and split_trajs:
            raise RuntimeError(
                "Cannot split trajectories when reset_when_done is False."
            )
        self.split_trajs = split_trajs
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        self.update_at_each_batch = update_at_each_batch
        self.exploration_type = exploration_type
        self.frames_per_worker = np.inf
        if preemptive_threshold is not None:
            if _is_osx:
                raise NotImplementedError(
                    "Cannot use preemption on OSX due to Queue.qsize() not being implemented on this platform."
                )
            self.preemptive_threshold = np.clip(preemptive_threshold, 0.0, 1.0)
            manager = _InterruptorManager()
            manager.start()
            self.interruptor = manager._Interruptor()
        else:
            self.preemptive_threshold = 1.0
            self.interruptor = None
        self._run_processes()
        self._exclude_private_keys = True
        self._frames = 0
        self._iter = -1

    @property
    def frames_per_batch_worker(self):
        raise NotImplementedError

    def update_policy_weights_(self, policy_weights=None) -> None:
        for _device in self._policy_dict:
            if policy_weights is not None:
                self._policy_weights_dict[_device].apply(lambda x: x.data).update_(
                    policy_weights
                )
            elif self._get_weights_fn_dict[_device] is not None:
                self._policy_weights_dict[_device].update_(
                    self._get_weights_fn_dict[_device]()
                )

    @property
    def _queue_len(self) -> int:
        raise NotImplementedError

    def _run_processes(self) -> None:
        torch.set_num_threads(self.num_threads)
        queue_out = mp.Queue(self._queue_len)  # sends data from proc to main
        self.procs = []
        self.pipes = []
        for i, (env_fun, env_fun_kwargs) in enumerate(
            zip(self.create_env_fn, self.create_env_kwargs)
        ):
            _device = self.device[i]
            _storing_device = self.storing_device[i]
            pipe_parent, pipe_child = mp.Pipe()  # send messages to procs
            if env_fun.__class__.__name__ != "EnvCreator" and not isinstance(
                env_fun, EnvBase
            ):  # to avoid circular imports
                env_fun = CloudpickleWrapper(env_fun)

            kwargs = {
                "pipe_parent": pipe_parent,
                "pipe_child": pipe_child,
                "queue_out": queue_out,
                "create_env_fn": env_fun,
                "create_env_kwargs": env_fun_kwargs,
                "policy": self._policy_dict[_device],
                "max_frames_per_traj": self.max_frames_per_traj,
                "frames_per_batch": self.frames_per_batch_worker,
                "reset_at_each_iter": self.reset_at_each_iter,
                "device": _device,
                "storing_device": _storing_device,
                "exploration_type": self.exploration_type,
                "reset_when_done": self.reset_when_done,
                "idx": i,
                "interruptor": self.interruptor,
            }
            proc = _ProcessNoWarn(
                target=_main_async_collector,
                num_threads=self.num_sub_threads,
                kwargs=kwargs,
            )
            # proc.daemon can't be set as daemonic processes may be launched by the process itself
            try:
                proc.start()
            except _pickle.PicklingError as err:
                if "<lambda>" in str(err):
                    raise RuntimeError(
                        """Can't open a process with doubly cloud-pickled lambda function.
This error is likely due to an attempt to use a ParallelEnv in a
multiprocessed data collector. To do this, consider wrapping your
lambda function in an `torchrl.envs.EnvCreator` wrapper as follows:
`env = ParallelEnv(N, EnvCreator(my_lambda_function))`.
This will not only ensure that your lambda function is cloud-pickled once, but
also that the state dict is synchronised across processes if needed."""
                    ) from err
            pipe_child.close()
            self.procs.append(proc)
            self.pipes.append(pipe_parent)
        for pipe_parent in self.pipes:
            msg = pipe_parent.recv()
            if msg != "instantiated":
                raise RuntimeError(msg)
        self.queue_out = queue_out
        self.closed = False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # an AttributeError will typically be raised if the collector is deleted when the program ends.
            # In the future, insignificant changes to the close method may change the error type.
            # We excplicitely assume that any error raised during closure in
            # __del__ will not affect the program.
            pass

    def shutdown(self) -> None:
        """Shuts down all processes. This operation is irreversible."""
        self._shutdown_main()

    def _shutdown_main(self) -> None:
        if self.closed:
            return
        _check_for_faulty_process(self.procs)
        self.closed = True
        for idx in range(self.num_workers):
            if not self.procs[idx].is_alive():
                continue
            try:
                self.pipes[idx].send((None, "close"))

                if self.pipes[idx].poll(10.0):
                    msg = self.pipes[idx].recv()
                    if msg != "closed":
                        raise RuntimeError(f"got {msg} but expected 'close'")
                else:
                    continue
            except BrokenPipeError:
                continue

        for proc in self.procs:
            exitcode = proc.join(1.0)
            if exitcode is None:
                proc.terminate()
        self.queue_out.close()
        for pipe in self.pipes:
            pipe.close()

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Sets the seeds of the environments stored in the DataCollector.

        Args:
            seed: integer representing the seed to be used for the environment.
            static_seed (bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            Output seed. This is useful when more than one environment is
            contained in the DataCollector, as the seed will be incremented for
            each of these. The resulting seed is the seed of the last
            environment.

        Examples:
            >>> from torchrl.envs import ParallelEnv
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> env_fn = lambda: GymEnv("Pendulum-v1")
            >>> env_fn_parallel = lambda: ParallelEnv(6, env_fn)
            >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
            >>> collector = SyncDataCollector(env_fn_parallel, policy, frames_per_batch=100, total_frames=300)
            >>> out_seed = collector.set_seed(1)  # out_seed = 6

        """
        _check_for_faulty_process(self.procs)
        for idx in range(self.num_workers):
            self.pipes[idx].send(((seed, static_seed), "seed"))
            new_seed, msg = self.pipes[idx].recv()
            if msg != "seeded":
                raise RuntimeError(f"Expected msg='seeded', got {msg}")
            seed = new_seed
        self.reset()
        return seed

    def reset(self, reset_idx: Optional[Sequence[bool]] = None) -> None:
        """Resets the environments to a new initial state.

        Args:
            reset_idx: Optional. Sequence indicating which environments have
                to be reset. If None, all environments are reset.

        """
        _check_for_faulty_process(self.procs)

        if reset_idx is None:
            reset_idx = [True for _ in range(self.num_workers)]
        for idx in range(self.num_workers):
            if reset_idx[idx]:
                self.pipes[idx].send((None, "reset"))
        for idx in range(self.num_workers):
            if reset_idx[idx]:
                j, msg = self.pipes[idx].recv()
                if msg != "reset":
                    raise RuntimeError(f"Expected msg='reset', got {msg}")

    def state_dict(self) -> OrderedDict:
        """Returns the state_dict of the data collector.

        Each field represents a worker containing its own state_dict.

        """
        for idx in range(self.num_workers):
            self.pipes[idx].send((None, "state_dict"))
        state_dict = OrderedDict()
        for idx in range(self.num_workers):
            _state_dict, msg = self.pipes[idx].recv()
            if msg != "state_dict":
                raise RuntimeError(f"Expected msg='state_dict', got {msg}")
            state_dict[f"worker{idx}"] = _state_dict
        state_dict.update({"frames": self._frames, "iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Loads the state_dict on the workers.

        Args:
            state_dict (OrderedDict): state_dict of the form
                ``{"worker0": state_dict0, "worker1": state_dict1}``.

        """
        for idx in range(self.num_workers):
            self.pipes[idx].send((state_dict[f"worker{idx}"], "load_state_dict"))
        for idx in range(self.num_workers):
            _, msg = self.pipes[idx].recv()
            if msg != "loaded":
                raise RuntimeError(f"Expected msg='loaded', got {msg}")
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]


@accept_remote_rref_udf_invocation
class MultiSyncDataCollector(_MultiDataCollector):
    """Runs a given number of DataCollectors on separate processes synchronously.

    .. aafig::

            +----------------------------------------------------------------------+
            |            "MultiSyncDataCollector"                 |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |   "Collector 1" |  "Collector 2"  |  "Collector 3"  |     Main       |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            |                 |       "actor"   |                 |                |
            |                 |                 |                 |                |
            |                       "yield batch of traj 1"------->"collect, train"|
            |                                                     |                |
            | "step" | "step" | "step" | "step" | "step" | "step" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |       "actor"   |        |        |                |
            |                 | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   |                 |       "actor"   |                |
            |                       "yield batch of traj 2"------->"collect, train"|
            |                                                     |                |
            +----------------------------------------------------------------------+

    Envs can be identical or different.

    The collection starts when the next item of the collector is queried,
    and no environment step is computed in between the reception of a batch of
    trajectory and the start of the next collection.
    This class can be safely used with online RL algorithms.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import StepCounter
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> env_maker = lambda: TransformedEnv(GymEnv("Pendulum-v1", device="cpu"), StepCounter(max_steps=50))
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> collector = MultiSyncDataCollector(
        ...     create_env_fn=[env_maker, env_maker],
        ...     policy=policy,
        ...     total_frames=2000,
        ...     max_frames_per_traj=50,
        ...     frames_per_batch=200,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=False,
        ...     devices="cpu",
        ...     storing_devices="cpu",
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)
        >>> collector.shutdown()
        >>> del collector

    """

    __doc__ += _MultiDataCollector.__doc__

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(self):
        if hasattr(self, "out_buffer"):
            del self.out_buffer
        if hasattr(self, "buffers"):
            del self.buffers
        return super().shutdown()

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)

    # for RPC
    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

    @property
    def frames_per_batch_worker(self):
        if self.requested_frames_per_batch % self.num_workers != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch {self.requested_frames_per_batch} is not exactly divisible by the number of collector workers {self.num_workers},"
                f" this results in more frames_per_batch per iteration that requested."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        frames_per_batch_worker = -(
            -self.requested_frames_per_batch // self.num_workers
        )
        return frames_per_batch_worker

    @property
    def _queue_len(self) -> int:
        return self.num_workers

    def iterator(self) -> Iterator[TensorDictBase]:

        self.buffers = {}
        dones = [False for _ in range(self.num_workers)]
        workers_frames = [0 for _ in range(self.num_workers)]
        same_device = None
        self.out_buffer = None

        while not all(dones) and self._frames < self.total_frames:
            _check_for_faulty_process(self.procs)
            if self.update_at_each_batch:
                self.update_policy_weights_()

            for idx in range(self.num_workers):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    msg = "continue_random"
                else:
                    msg = "continue"
                self.pipes[idx].send((None, msg))

            self._iter += 1
            max_traj_idx = None

            if self.interruptor is not None and self.preemptive_threshold < 1.0:
                self.interruptor.start_collection()
                while self.queue_out.qsize() < int(
                    self.num_workers * self.preemptive_threshold
                ):
                    continue
                self.interruptor.stop_collection()
                # Now wait for stragglers to return
                while self.queue_out.qsize() < int(self.num_workers):
                    continue

            for _ in range(self.num_workers):
                new_data, j = self.queue_out.get()
                if j == 0:
                    data, idx = new_data
                    self.buffers[idx] = data
                else:
                    idx = new_data
                workers_frames[idx] = workers_frames[idx] + self.buffers[idx].numel()

                if workers_frames[idx] >= self.total_frames:
                    dones[idx] = True
            # we have to correct the traj_ids to make sure that they don't overlap
            for idx in range(self.num_workers):
                traj_ids = self.buffers[idx].get(("collector", "traj_ids"))
                if max_traj_idx is not None:
                    traj_ids[traj_ids != -1] += max_traj_idx
                    # out_tensordicts_shared[idx].set("traj_ids", traj_ids)
                max_traj_idx = traj_ids.max().item() + 1
                # out = out_tensordicts_shared[idx]
            if same_device is None:
                prev_device = None
                same_device = True
                for item in self.buffers.values():
                    if prev_device is None:
                        prev_device = item.device
                    else:
                        same_device = same_device and (item.device == prev_device)

            if same_device:
                self.out_buffer = torch.cat(
                    list(self.buffers.values()), 0, out=self.out_buffer
                )
            else:
                self.out_buffer = torch.cat(
                    [item.cpu() for item in self.buffers.values()],
                    0,
                    out=self.out_buffer,
                )

            if self.split_trajs:
                out = split_trajectories(self.out_buffer, prefix="collector")
                self._frames += out.get(("collector", "mask")).sum().item()
            else:
                out = self.out_buffer.clone()
                self._frames += prod(out.shape)
            if self.postprocs:
                self.postprocs = self.postprocs.to(out.device)
                out = self.postprocs(out)
            if self._exclude_private_keys:
                excluded_keys = [key for key in out.keys() if key.startswith("_")]
                if excluded_keys:
                    out = out.exclude(*excluded_keys)
            yield out
            del out

        del self.buffers
        # We shall not call shutdown just yet as user may want to retrieve state_dict
        # self._shutdown_main()


@accept_remote_rref_udf_invocation
class MultiaSyncDataCollector(_MultiDataCollector):
    """Runs a given number of DataCollectors on separate processes asynchronously.

    .. aafig::


            +----------------------------------------------------------------------+
            |           "MultiConcurrentCollector"                |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |  "Collector 1"  |  "Collector 2"  |  "Collector 3"  |     "Main"     |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor    | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "yield batch 1" |       "actor"   |                 |"collect, train"|
            |                 |                 |                 |                |
            | "step" | "step" |                 | "yield batch 2" |"collect, train"|
            |        |        |                 |                 |                |
            |        |        | "yield batch 3" |                 |"collect, train"|
            |        |        |                 |                 |                |
            +----------------------------------------------------------------------+

    Environment types can be identical or different.

    The collection keeps on occuring on all processes even between the time
    the batch of rollouts is collected and the next call to the iterator.
    This class can be safely used with offline RL algorithms.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> collector = MultiaSyncDataCollector(
        ...     create_env_fn=[env_maker, env_maker],
        ...     policy=policy,
        ...     total_frames=2000,
        ...     max_frames_per_traj=50,
        ...     frames_per_batch=200,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=False,
        ...     devices="cpu",
        ...     storing_devices="cpu",
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)
        >>> collector.shutdown()
        >>> del collector

    """

    __doc__ += _MultiDataCollector.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_tensordicts = {}
        self.running = False

        if self.postprocs is not None:
            postproc = self.postprocs
            self.postprocs = {}
            for _device in self.storing_device:
                if _device not in self.postprocs:
                    self.postprocs[_device] = deepcopy(postproc).to(_device)

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(self):
        if hasattr(self, "out_tensordicts"):
            del self.out_tensordicts
        return super().shutdown()

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)

    # for RPC
    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

    @property
    def frames_per_batch_worker(self):
        return self.requested_frames_per_batch

    def _get_from_queue(self, timeout=None) -> Tuple[int, int, TensorDictBase]:
        new_data, j = self.queue_out.get(timeout=timeout)
        if j == 0:
            data, idx = new_data
            self.out_tensordicts[idx] = data
        else:
            idx = new_data
        # we clone the data to make sure that we'll be working with a fixed copy
        out = self.out_tensordicts[idx].clone()
        return idx, j, out

    @property
    def _queue_len(self) -> int:
        return 1

    def iterator(self) -> Iterator[TensorDictBase]:
        if self.update_at_each_batch:
            self.update_policy_weights_()

        for i in range(self.num_workers):
            if self.init_random_frames is not None and self.init_random_frames > 0:
                self.pipes[i].send((None, "continue_random"))
            else:
                self.pipes[i].send((None, "continue"))
        self.running = True

        workers_frames = [0 for _ in range(self.num_workers)]
        while self._frames < self.total_frames:
            _check_for_faulty_process(self.procs)
            self._iter += 1
            idx, j, out = self._get_from_queue()

            worker_frames = out.numel()
            if self.split_trajs:
                out = split_trajectories(out, prefix="collector")
            self._frames += worker_frames
            workers_frames[idx] = workers_frames[idx] + worker_frames
            if self.postprocs:
                out = self.postprocs[out.device](out)

            # the function blocks here until the next item is asked, hence we send the message to the
            # worker to keep on working in the meantime before the yield statement
            if (
                self.init_random_frames is not None
                and self._frames < self.init_random_frames
            ):
                msg = "continue_random"
            else:
                msg = "continue"
            self.pipes[idx].send((idx, msg))
            if self._exclude_private_keys:
                excluded_keys = [key for key in out.keys() if key.startswith("_")]
                out = out.exclude(*excluded_keys)
            yield out

        # We don't want to shutdown yet, the user may want to call state_dict before
        # self._shutdown_main()
        self.running = False

    def _shutdown_main(self) -> None:
        if hasattr(self, "out_tensordicts"):
            del self.out_tensordicts
        return super()._shutdown_main()

    def reset(self, reset_idx: Optional[Sequence[bool]] = None) -> None:
        super().reset(reset_idx)
        if self.queue_out.full():
            time.sleep(_TIMEOUT)  # wait until queue is empty
        if self.queue_out.full():
            raise Exception("self.queue_out is full")
        if self.running:
            for idx in range(self.num_workers):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.pipes[idx].send((idx, "continue_random"))
                else:
                    self.pipes[idx].send((idx, "continue"))


@accept_remote_rref_udf_invocation
class aSyncDataCollector(MultiaSyncDataCollector):
    """Runs a single DataCollector on a separate process.

    This is mostly useful for offline RL paradigms where the policy being
    trained can differ from the policy used to collect data. In online
    settings, a regular DataCollector should be preferred. This class is
    merely a wrapper around a MultiaSyncDataCollector where a single process
    is being created.

    Args:
        create_env_fn (Callabled): Callable returning an instance of EnvBase
        policy (Callable, optional): Instance of TensorDictModule class.
            Must accept TensorDictBase object as input.
        total_frames (int): lower bound of the total number of frames returned
            by the collector. In parallel settings, the actual number of
            frames may well be greater than this as the closing signals are
            sent to the workers only once the total number of frames has
            been collected on the server.
        create_env_kwargs (dict, optional): A dictionary with the arguments
            used to create an environment
        max_frames_per_traj: Maximum steps per trajectory. Note that a
            trajectory can span over multiple batches (unless
            reset_at_each_iter is set to True, see below). Once a trajectory
            reaches n_steps, the environment is reset. If the
            environment wraps multiple environments together, the number of
            steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e. no maximum number of steps)
        frames_per_batch (int): Time-length of a batch.
            reset_at_each_iter and frames_per_batch == n_steps are equivalent configurations.
            Defaults to ``200``
        init_random_frames (int): Number of frames for which the policy is ignored before it is called.
            This feature is mainly intended to be used in offline/model-based settings, where a batch of random
            trajectories can be used to initialize training.
            Defaults to ``None`` (i.e. no random frames)
        reset_at_each_iter (bool): Whether or not environments should be reset for each batch.
            default=False.
        postproc (callable, optional): A PostProcessor is an object that will read a batch of data and process it in a
            useful format for training.
            default: None.
        split_trajs (bool): Boolean indicating whether the resulting TensorDict should be split according to the trajectories.
            See utils.split_trajectories for more information.
        device (int, str, torch.device, optional): The device on which the
            policy will be placed. If it differs from the input policy
            device, the update_policy_weights_() method should be queried
            at appropriate times during the training loop to accommodate for
            the lag between parameter configuration at various times.
            Default is `None` (i.e. policy is kept on its original device)
        storing_device (int, str, torch.device, optional): The device on which
            the output TensorDict will be stored. For long trajectories,
            it may be necessary to store the data on a different.
            device than the one where the policy is stored. Default is None.
        update_at_each_batch (bool): if ``True``, the policy weights will be updated every time a batch of trajectories
            is collected.
            default=False

    """

    def __init__(
        self,
        create_env_fn: Callable[[], EnvBase],
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ] = None,
        total_frames: Optional[int] = -1,
        create_env_kwargs: Optional[dict] = None,
        max_frames_per_traj: int | None = None,
        frames_per_batch: int = 200,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        split_trajs: Optional[bool] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        storing_device: Optional[Union[int, str, torch.device]] = None,
        seed: Optional[int] = None,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            create_env_fn=[create_env_fn],
            policy=policy,
            total_frames=total_frames,
            create_env_kwargs=[create_env_kwargs],
            max_frames_per_traj=max_frames_per_traj,
            frames_per_batch=frames_per_batch,
            reset_at_each_iter=reset_at_each_iter,
            init_random_frames=init_random_frames,
            postproc=postproc,
            split_trajs=split_trajs,
            devices=[device] if device is not None else None,
            storing_devices=[storing_device] if storing_device is not None else None,
            **kwargs,
        )

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(self):
        return super().shutdown()

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)


def _main_async_collector(
    pipe_parent: connection.Connection,
    pipe_child: connection.Connection,
    queue_out: queues.Queue,
    create_env_fn: Union[EnvBase, "EnvCreator", Callable[[], EnvBase]],  # noqa: F821
    create_env_kwargs: Dict[str, Any],
    policy: Callable[[TensorDictBase], TensorDictBase],
    max_frames_per_traj: int,
    frames_per_batch: int,
    reset_at_each_iter: bool,
    device: Optional[Union[torch.device, str, int]],
    storing_device: Optional[Union[torch.device, str, int]],
    idx: int = 0,
    exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
    reset_when_done: bool = True,
    verbose: bool = VERBOSE,
    interruptor=None,
) -> None:
    pipe_parent.close()
    # init variables that will be cleared when closing
    tensordict = data = d = data_in = inner_collector = dc_iter = None

    # send the policy to device
    try:
        policy = policy.to(device)
    except Exception:
        if RL_WARNINGS:
            warnings.warn(
                "Couldn't cast the policy onto the desired device on remote process. "
                "If your policy is not a nn.Module instance you can probably ignore this warning."
            )
    inner_collector = SyncDataCollector(
        create_env_fn,
        create_env_kwargs=create_env_kwargs,
        policy=policy,
        total_frames=-1,
        max_frames_per_traj=max_frames_per_traj,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=reset_at_each_iter,
        postproc=None,
        split_trajs=False,
        device=device,
        storing_device=storing_device,
        exploration_type=exploration_type,
        reset_when_done=reset_when_done,
        return_same_td=True,
        interruptor=interruptor,
    )
    if verbose:
        print("Sync data collector created")
    dc_iter = iter(inner_collector)
    j = 0
    pipe_child.send("instantiated")

    has_timed_out = False
    counter = 0
    while True:
        _timeout = _TIMEOUT if not has_timed_out else 1e-3
        if pipe_child.poll(_timeout):
            counter = 0
            data_in, msg = pipe_child.recv()
            if verbose:
                print(f"worker {idx} received {msg}")
        else:
            if verbose:
                print(f"poll failed, j={j}, worker={idx}")
            # default is "continue" (after first iteration)
            # this is expected to happen if queue_out reached the timeout, but no new msg was waiting in the pipe
            # in that case, the main process probably expects the worker to continue collect data
            if has_timed_out:
                counter = 0
                # has_timed_out is True if the process failed to send data, which will
                # typically occur if main has taken another batch (i.e. the queue is Full).
                # In this case, msg is the previous msg sent by main, which will typically be "continue"
                # If it's not the case, it is not expected that has_timed_out is True.
                if msg not in ("continue", "continue_random"):
                    raise RuntimeError(f"Unexpected message after time out: msg={msg}")
            else:
                # if has_timed_out is False, then the time out does not come from the fact that the queue is Full.
                # this means that our process has been waiting for a command from main in vain, while main was not
                # receiving data.
                # This will occur if main is busy doing something else (e.g. computing loss etc).

                counter += _timeout
                if verbose:
                    print(f"worker {idx} has counter {counter}")
                if counter >= (_MAX_IDLE_COUNT * _TIMEOUT):
                    raise RuntimeError(
                        f"This process waited for {counter} seconds "
                        f"without receiving a command from main. Consider increasing the maximum idle count "
                        f"if this is expected via the environment variable MAX_IDLE_COUNT "
                        f"(current value is {_MAX_IDLE_COUNT})."
                        f"\nIf this occurs at the end of a function or program, it means that your collector has not been "
                        f"collected, consider calling `collector.shutdown()` or `del collector` before ending the program."
                    )
                continue
        if msg in ("continue", "continue_random"):
            if msg == "continue_random":
                inner_collector.init_random_frames = float("inf")
            else:
                inner_collector.init_random_frames = -1

            d = next(dc_iter)
            if pipe_child.poll(_MIN_TIMEOUT):
                # in this case, main send a message to the worker while it was busy collecting trajectories.
                # In that case, we skip the collected trajectory and get the message from main. This is faster than
                # sending the trajectory in the queue until timeout when it's never going to be received.
                continue
            if j == 0:
                tensordict = d
                if storing_device is not None and tensordict.device != storing_device:
                    raise RuntimeError(
                        f"expected device to be {storing_device} but got {tensordict.device}"
                    )
                tensordict.share_memory_()
                data = (tensordict, idx)
            else:
                if d is not tensordict:
                    raise RuntimeError(
                        "SyncDataCollector should return the same tensordict modified in-place."
                    )
                data = idx  # flag the worker that has sent its data
            try:
                queue_out.put((data, j), timeout=_TIMEOUT)
                if verbose:
                    print(f"worker {idx} successfully sent data")
                j += 1
                has_timed_out = False
                continue
            except queue.Full:
                if verbose:
                    print(f"worker {idx} has timed out")
                has_timed_out = True
                continue

        elif msg == "update":
            inner_collector.update_policy_weights_()
            pipe_child.send((j, "updated"))
            has_timed_out = False
            continue

        elif msg == "seed":
            data_in, static_seed = data_in
            new_seed = inner_collector.set_seed(data_in, static_seed=static_seed)
            torch.manual_seed(data_in)
            np.random.seed(data_in)
            pipe_child.send((new_seed, "seeded"))
            has_timed_out = False
            continue

        elif msg == "reset":
            inner_collector.reset()
            pipe_child.send((j, "reset"))
            continue

        elif msg == "state_dict":
            state_dict = inner_collector.state_dict()
            # send state_dict to cpu first
            state_dict = recursive_map_to_cpu(state_dict)
            pipe_child.send((state_dict, "state_dict"))
            has_timed_out = False
            continue

        elif msg == "load_state_dict":
            state_dict = data_in
            inner_collector.load_state_dict(state_dict)
            del state_dict
            pipe_child.send((j, "loaded"))
            has_timed_out = False
            continue

        elif msg == "close":
            del tensordict, data, d, data_in
            inner_collector.shutdown()
            del inner_collector, dc_iter
            pipe_child.send("closed")
            if verbose:
                print(f"collector {idx} closed")
            break

        else:
            raise Exception(f"Unrecognized message {msg}")

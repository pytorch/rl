# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from multiprocessing import connection
from multiprocessing.synchronize import Lock as MpLock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import torch

from tensordict import TensorDict, unravel_key
from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.tensordict import LazyStackedTensorDict, TensorDictBase
from torch import multiprocessing as mp
from torchrl._utils import _check_for_faulty_process, VERBOSE
from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.env_creator import get_env_metadata

from torchrl.envs.utils import _set_single_key, _sort_keys

_has_envpool = importlib.util.find_spec("envpool")


def _check_start(fun):
    def decorated_fun(self: _BatchedEnv, *args, **kwargs):
        if self.is_closed:
            self._create_td()
            self._start_workers()
        else:
            if isinstance(self, ParallelEnv):
                _check_for_faulty_process(self._workers)
        return fun(self, *args, **kwargs)

    return decorated_fun


class _dispatch_caller_parallel:
    def __init__(self, attr, parallel_env):
        self.attr = attr
        self.parallel_env = parallel_env

    def __call__(self, *args, **kwargs):
        # remove self from args
        args = [_arg if _arg is not self.parallel_env else "_self" for _arg in args]
        for channel in self.parallel_env.parent_channels:
            channel.send((self.attr, (args, kwargs)))

        results = []
        for channel in self.parallel_env.parent_channels:
            msg, result = channel.recv()
            results.append(result)

        return results

    def __iter__(self):
        # if the object returned is not a callable
        return iter(self.__call__())


class _dispatch_caller_serial:
    def __init__(self, list_callable: List[Callable, Any]):
        self.list_callable = list_callable

    def __call__(self, *args, **kwargs):
        return [_callable(*args, **kwargs) for _callable in self.list_callable]


def lazy_property(prop: property):
    """Converts a property in a lazy property, that will call _set_properties when queried the first time."""
    return property(fget=lazy(prop.fget), fset=prop.fset)


def lazy(fun):
    """Converts a fun in a lazy fun, that will call _set_properties when queried the first time."""

    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        if not self._properties_set:
            self._set_properties()
        return fun(self, *args, **kwargs)

    return new_fun


class _BatchedEnv(EnvBase):
    """Batched environments allow the user to query an arbitrary method / attribute of the environment running remotely.

    Those queries will return a list of length equal to the number of workers containing the
    values resulting from those queries.
        >>> env = ParallelEnv(3, my_env_fun)
        >>> custom_attribute_list = env.custom_attribute
        >>> custom_method_list = env.custom_method(*args)

    Args:
        num_workers: number of workers (i.e. env instances) to be deployed simultaneously;
        create_env_fn (callable or list of callables): function (or list of functions) to be used for the environment
            creation.
            If a single task is used, a callable should be used and not a list of identical callables:
            if a list of callable is provided, the environment will be executed as if multiple, diverse tasks were
            needed, which comes with a slight compute overhead;
        create_env_kwargs (dict or list of dicts, optional): kwargs to be used with the environments being created;
        pin_memory (bool): if True and device is "cpu", calls :obj:`pin_memory` on the tensordicts when created.
        share_individual_td (bool, optional): if ``True``, a different tensordict is created for every process/worker and a lazy
            stack is returned.
            default = None (False if single task);
        shared_memory (bool): whether or not the returned tensordict will be placed in shared memory;
        memmap (bool): whether or not the returned tensordict will be placed in memory map.
        policy_proof (callable, optional): if provided, it'll be used to get the list of
            tensors to return through the :obj:`step()` and :obj:`reset()` methods, such as :obj:`"hidden"` etc.
        device (str, int, torch.device): for consistency, this argument is kept. However this
            argument should not be passed, as the device will be inferred from the environments.
            It is assumed that all environments will run on the same device as a common shared
            tensordict will be used to pass data from process to process. The device can be
            changed after instantiation using :obj:`env.to(device)`.
        allow_step_when_done (bool, optional): if ``True``, batched environments can
            execute steps after a done state is encountered.
            Defaults to ``False``.

    """

    _verbose: bool = VERBOSE
    _excluded_wrapped_keys = [
        "is_closed",
        "parent_channels",
        "batch_size",
        "_dummy_env_str",
    ]

    def __init__(
        self,
        num_workers: int,
        create_env_fn: Union[Callable[[], EnvBase], Sequence[Callable[[], EnvBase]]],
        create_env_kwargs: Union[dict, Sequence[dict]] = None,
        pin_memory: bool = False,
        share_individual_td: Optional[bool] = None,
        shared_memory: bool = True,
        memmap: bool = False,
        policy_proof: Optional[Callable] = None,
        device: Optional[DEVICE_TYPING] = None,
        allow_step_when_done: bool = False,
    ):
        if device is not None:
            raise ValueError(
                "Device setting for batched environment can't be done at initialization. "
                "The device will be inferred from the constructed environment. "
                "It can be set through the `to(device)` method."
            )

        super().__init__(device=None)
        self.is_closed = True
        self._cache_in_keys = None

        self._single_task = callable(create_env_fn) or (len(set(create_env_fn)) == 1)
        if callable(create_env_fn):
            create_env_fn = [create_env_fn for _ in range(num_workers)]
        else:
            if len(create_env_fn) != num_workers:
                raise RuntimeError(
                    f"num_workers and len(create_env_fn) mismatch, "
                    f"got {len(create_env_fn)} and {num_workers}"
                )
            if (
                share_individual_td is False and not self._single_task
            ):  # then it has been explicitly set by the user
                raise ValueError(
                    "share_individual_td must be set to None or True when using multi-task batched environments"
                )
            share_individual_td = True
        create_env_kwargs = {} if create_env_kwargs is None else create_env_kwargs
        if isinstance(create_env_kwargs, dict):
            create_env_kwargs = [
                deepcopy(create_env_kwargs) for _ in range(num_workers)
            ]

        self.policy_proof = policy_proof
        self.num_workers = num_workers
        self.create_env_fn = create_env_fn
        self.create_env_kwargs = create_env_kwargs
        self.pin_memory = pin_memory
        self.share_individual_td = bool(share_individual_td)
        self._share_memory = shared_memory
        self._memmap = memmap
        self.allow_step_when_done = allow_step_when_done
        if self._share_memory and self._memmap:
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        self._batch_size = None
        self._device = None
        self._dummy_env_str = None
        self._seeds = None
        self.__dict__["_input_spec"] = None
        self.__dict__["_output_spec"] = None
        # self._prepare_dummy_env(create_env_fn, create_env_kwargs)
        self._properties_set = False
        self._get_metadata(create_env_fn, create_env_kwargs)

    def _get_metadata(
        self, create_env_fn: List[Callable], create_env_kwargs: List[Dict]
    ):
        if self._single_task:
            # if EnvCreator, the metadata are already there
            meta_data = get_env_metadata(create_env_fn[0], create_env_kwargs[0])
            self.meta_data = meta_data.expand(
                *(self.num_workers, *meta_data.batch_size)
            )
        else:
            n_tasks = len(create_env_fn)
            self.meta_data = []
            for i in range(n_tasks):
                self.meta_data.append(
                    get_env_metadata(create_env_fn[i], create_env_kwargs[i]).clone()
                )
        self._set_properties()

    def update_kwargs(self, kwargs: Union[dict, List[dict]]) -> None:
        """Updates the kwargs of each environment given a dictionary or a list of dictionaries.

        Args:
            kwargs (dict or list of dict): new kwargs to use with the environments

        """
        if isinstance(kwargs, dict):
            for _kwargs in self.create_env_kwargs:
                _kwargs.update(kwargs)
        else:
            for _kwargs, _new_kwargs in zip(self.create_env_kwargs, kwargs):
                _kwargs.update(_new_kwargs)

    def _get_in_keys_to_exclude(self, tensordict):
        if self._cache_in_keys is None:
            self._cache_in_keys = list(
                set(self.input_spec.keys(True)).intersection(
                    tensordict.keys(True, True)
                )
            )
        return self._cache_in_keys

    def _set_properties(self):
        meta_data = self.meta_data
        self._properties_set = True
        if self._single_task:
            self._batch_size = meta_data.batch_size
            device = self._device = meta_data.device

            input_spec = meta_data.specs["input_spec"].to(device)
            output_spec = meta_data.specs["output_spec"].to(device)

            self.action_spec = input_spec["_action_spec"]
            self.state_spec = input_spec["_state_spec"]
            self.observation_spec = output_spec["_observation_spec"]
            self.reward_spec = output_spec["_reward_spec"]
            self.done_spec = output_spec["_done_spec"]

            self._dummy_env_str = meta_data.env_str
            self._env_tensordict = meta_data.tensordict
            self._batch_locked = meta_data.batch_locked
        else:
            self._batch_size = torch.Size([self.num_workers, *meta_data[0].batch_size])
            device = self._device = meta_data[0].device
            # TODO: check that all action_spec and reward spec match (issue #351)

            input_spec = []
            for md in meta_data:
                input_spec.append(md.specs["input_spec"])
            input_spec = torch.stack(input_spec, 0)
            output_spec = []
            for md in meta_data:
                output_spec.append(md.specs["output_spec"])
            output_spec = torch.stack(output_spec, 0)

            self.action_spec = input_spec["_action_spec"]
            self.state_spec = input_spec["_state_spec"]

            self.observation_spec = output_spec["_observation_spec"]
            self.reward_spec = output_spec["_reward_spec"]
            self.done_spec = output_spec["_done_spec"]

            self._dummy_env_str = str(meta_data[0])
            self._env_tensordict = torch.stack(
                [meta_data.tensordict for meta_data in meta_data], 0
            )
            self._batch_locked = meta_data[0].batch_locked

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    batch_size = lazy_property(EnvBase.batch_size)
    device = lazy_property(EnvBase.device)
    input_spec = lazy_property(EnvBase.input_spec)
    output_spec = lazy_property(EnvBase.output_spec)

    def _create_td(self) -> None:
        """Creates self.shared_tensordict_parent, a TensorDict used to store the most recent observations."""
        if self._single_task:
            shared_tensordict_parent = self._env_tensordict.clone()
            if not self._env_tensordict.shape[0] == self.num_workers:
                raise RuntimeError(
                    "batched environment base tensordict has the wrong shape"
                )
        else:
            shared_tensordict_parent = self._env_tensordict.clone()

        if self._single_task:
            self.env_input_keys = sorted(
                list(self.input_spec["_action_spec"].keys(True, True))
                + list(self.state_spec.keys(True, True)),
                key=_sort_keys,
            )
            self.env_output_keys = []
            self.env_obs_keys = []
            for key in self.output_spec["_observation_spec"].keys(True, True):
                self.env_output_keys.append(unravel_key(("next", key)))
                self.env_obs_keys.append(key)
            self.env_output_keys.append(unravel_key(("next", self.reward_key)))
            self.env_output_keys.append(unravel_key(("next", self.done_key)))
        else:
            env_input_keys = set()
            for meta_data in self.meta_data:
                if meta_data.specs["input_spec", "_state_spec"] is not None:
                    env_input_keys = env_input_keys.union(
                        meta_data.specs["input_spec", "_state_spec"].keys(True, True)
                    )
                env_input_keys = env_input_keys.union(
                    meta_data.specs["input_spec", "_action_spec"].keys(True, True)
                )
            env_output_keys = set()
            env_obs_keys = set()
            for meta_data in self.meta_data:
                env_obs_keys = env_obs_keys.union(
                    key
                    for key in meta_data.specs["output_spec"]["_observation_spec"].keys(
                        True, True
                    )
                )
                env_output_keys = env_output_keys.union(
                    unravel_key(("next", key))
                    for key in meta_data.specs["output_spec"]["_observation_spec"].keys(
                        True, True
                    )
                )
            env_output_keys = env_output_keys.union(
                {
                    unravel_key(("next", self.reward_key)),
                    unravel_key(("next", self.done_key)),
                }
            )
            self.env_obs_keys = sorted(env_obs_keys, key=_sort_keys)
            self.env_input_keys = sorted(env_input_keys, key=_sort_keys)
            self.env_output_keys = sorted(env_output_keys, key=_sort_keys)

        self._selected_keys = (
            set(self.env_output_keys)
            .union(self.env_input_keys)
            .union(self.env_obs_keys)
        )
        self._selected_keys.add(self.done_key)
        self._selected_keys.add("_reset")

        self._selected_reset_keys = self.env_obs_keys + [self.done_key] + ["_reset"]
        self._selected_step_keys = self.env_output_keys

        if self._single_task:
            shared_tensordict_parent = shared_tensordict_parent.select(
                *self._selected_keys,
                strict=False,
            )
            self.shared_tensordict_parent = shared_tensordict_parent.to(self.device)
        else:
            # Multi-task: we share tensordict that *may* have different keys
            shared_tensordict_parent = [
                tensordict.select(*self._selected_keys, strict=False).to(self.device)
                for tensordict in shared_tensordict_parent
            ]
            shared_tensordict_parent = torch.stack(
                shared_tensordict_parent,
                0,
            )
            self.shared_tensordict_parent = shared_tensordict_parent

        if self.share_individual_td:
            if not isinstance(self.shared_tensordict_parent, LazyStackedTensorDict):
                self.shared_tensordicts = [
                    td.clone() for td in self.shared_tensordict_parent.unbind(0)
                ]
                self.shared_tensordict_parent = torch.stack(self.shared_tensordicts, 0)
            else:
                # Multi-task: we share tensordict that *may* have different keys
                # LazyStacked already stores this so we don't need to do anything
                self.shared_tensordicts = self.shared_tensordict_parent
            if self._share_memory:
                for td in self.shared_tensordicts:
                    td.share_memory_()
            elif self._memmap:
                for td in self.shared_tensordicts:
                    td.memmap_()
        else:
            if self._share_memory:
                self.shared_tensordict_parent.share_memory_()
                if not self.shared_tensordict_parent.is_shared():
                    raise RuntimeError("share_memory_() failed")
            elif self._memmap:
                self.shared_tensordict_parent.memmap_()
                if not self.shared_tensordict_parent.is_memmap():
                    raise RuntimeError("memmap_() failed")

            self.shared_tensordicts = self.shared_tensordict_parent.unbind(0)
        if self.pin_memory:
            self.shared_tensordict_parent.pin_memory()

    def _start_workers(self) -> None:
        """Starts the various envs."""
        raise NotImplementedError

    def __repr__(self) -> str:
        if self._dummy_env_str is None:
            self._dummy_env_str = self._set_properties()
        return (
            f"{self.__class__.__name__}("
            f"\n\tenv={self._dummy_env_str}, "
            f"\n\tbatch_size={self.batch_size})"
        )

    def close(self) -> None:
        if self.is_closed:
            raise RuntimeError("trying to close a closed environment")
        if self._verbose:
            print(f"closing {self.__class__.__name__}")

        self.__dict__["_input_spec"] = None
        self.__dict__["_output_spec"] = None
        self._properties_set = False
        self.event = None

        self._shutdown_workers()
        self.is_closed = True

    def _shutdown_workers(self) -> None:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]):
        """This method is not used in batched envs."""
        pass

    @lazy
    def start(self) -> None:
        if not self.is_closed:
            raise RuntimeError("trying to start a environment that is not closed.")
        self._create_td()
        self._start_workers()

    def to(self, device: DEVICE_TYPING):
        device = torch.device(device)
        if device == self.device:
            return self
        self._device = device
        self.meta_data = (
            self.meta_data.to(device)
            if self._single_task
            else [meta_data.to(device) for meta_data in self.meta_data]
        )
        if not self.is_closed:
            warn(
                "Casting an open environment to another device requires closing and re-opening it. "
                "This may have unexpected and unwanted effects (e.g. on seeding etc.)"
            )
            # the tensordicts must be re-created on device
            super().to(device)
            self.close()
            self.start()
        else:
            if self.__dict__["_input_spec"] is not None:
                self.__dict__["_input_spec"] = self.__dict__["_input_spec"].to(device)
            if self.__dict__["_output_spec"] is not None:
                self.__dict__["_output_spec"] = self.__dict__["_output_spec"].to(device)
        return self


class SerialEnv(_BatchedEnv):
    """Creates a series of environments in the same process."""

    __doc__ += _BatchedEnv.__doc__

    _share_memory = False

    def _start_workers(self) -> None:
        _num_workers = self.num_workers

        self._envs = []

        for idx in range(_num_workers):
            env = self.create_env_fn[idx](**self.create_env_kwargs[idx])
            self._envs.append(env.to(self.device))
        self.is_closed = False

    @_check_start
    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        for idx, env in enumerate(self._envs):
            state_dict[f"worker{idx}"] = env.state_dict()

        return state_dict

    @_check_start
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        if "worker0" not in state_dict:
            state_dict = OrderedDict(
                **{f"worker{idx}": state_dict for idx in range(self.num_workers)}
            )
        for idx, env in enumerate(self._envs):
            env.load_state_dict(state_dict[f"worker{idx}"])

    @_check_start
    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        self._assert_tensordict_shape(tensordict)
        tensordict_in = tensordict.clone(False)
        for i in range(self.num_workers):
            # shared_tensordicts are locked, and we need to select the keys since we update in-place.
            # There may be unexpected keys, such as "_reset", that we should comfortably ignore here.
            out_td = self._envs[i]._step(tensordict_in[i])
            out_td.update(tensordict_in[i].select(*self.env_input_keys))
            self.shared_tensordicts[i].update_(
                out_td.select(*self.env_input_keys, *self.env_output_keys)
            )
        # We must pass a clone of the tensordict, as the values of this tensordict
        # will be modified in-place at further steps
        if self._single_task:
            out = TensorDict({}, batch_size=self.shared_tensordict_parent.shape)
            for key in self._selected_step_keys:
                _set_single_key(self.shared_tensordict_parent, out, key, clone=True)
        else:
            # strict=False ensures that non-homogeneous keys are still there
            out = self.shared_tensordict_parent.select(
                *self._selected_step_keys, strict=False
            ).clone()
        return out

    def _shutdown_workers(self) -> None:
        if not self.is_closed:
            for env in self._envs:
                env.close()
            del self._envs

    @_check_start
    def set_seed(
        self, seed: Optional[int] = None, static_seed: bool = False
    ) -> Optional[int]:
        for env in self._envs:
            new_seed = env.set_seed(seed, static_seed=static_seed)
            seed = new_seed
        return seed

    @_check_start
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            _reset = tensordict.get("_reset")
            if _reset.shape[-len(self.done_spec.shape) :] != self.done_spec.shape:
                raise RuntimeError(
                    "_reset flag in tensordict should follow env.done_spec"
                )
        else:
            _reset = torch.ones(self.done_spec.shape, dtype=torch.bool)

        for i, _env in enumerate(self._envs):
            if tensordict is not None:
                tensordict_ = tensordict[i]
                if tensordict_.is_empty():
                    tensordict_ = None
            else:
                tensordict_ = None
            if not _reset[i].any():
                # We update the stored tensordict with the value of the "next"
                # key as one may be surprised to receive data that is not up-to-date
                # If we don't do this, the result of calling reset and skipping one env
                # will be that the env will have the data from the previous
                # step at the root (since the shared_tensordict did not go through
                # step_mdp).
                self.shared_tensordicts[i].update_(
                    self.shared_tensordicts[i]["next"].select(
                        *self._selected_reset_keys, strict=False
                    )
                )
                if tensordict_ is not None:
                    self.shared_tensordicts[i].update_(
                        tensordict_.select(*self._selected_reset_keys, strict=False)
                    )
                continue
            _td = _env._reset(tensordict=tensordict_, **kwargs)
            self.shared_tensordicts[i].update_(
                _td.select(*self._selected_keys, strict=False)
            )

        if self._single_task:
            # select + clone creates 2 tds, but we can create one only
            out = TensorDict({}, batch_size=self.shared_tensordict_parent.shape)
            for key in self._selected_reset_keys:
                if key != "_reset":
                    _set_single_key(self.shared_tensordict_parent, out, key, clone=True)
            return out
        else:
            return self.shared_tensordict_parent.select(
                *[key for key in self._selected_reset_keys if key != "_reset"],
                strict=False,
            ).clone()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return super().__getattr__(
                attr
            )  # make sure that appropriate exceptions are raised
        elif attr.startswith("__"):
            raise AttributeError(
                "dispatching built-in private methods is "
                f"not permitted with type {type(self)}. "
                f"Got attribute {attr}."
            )
        else:
            if attr in self._excluded_wrapped_keys:
                raise AttributeError(f"Getting {attr} resulted in an exception")
            try:
                # determine if attr is a callable
                list_attr = [getattr(env, attr) for env in self._envs]
                callable_attr = callable(list_attr[0])
                if callable_attr:
                    if self.is_closed:
                        raise RuntimeError(
                            "Trying to access attributes of closed/non started "
                            "environments. Check that the batched environment "
                            "has been started (e.g. by calling env.reset)"
                        )
                    return _dispatch_caller_serial(list_attr)
                else:
                    return list_attr
            except AttributeError:
                raise AttributeError(
                    f"attribute {attr} not found in " f"{self._dummy_env_str}"
                )

    def to(self, device: DEVICE_TYPING):
        device = torch.device(device)
        if device == self.device:
            return self
        super().to(device)
        if not self.is_closed:
            for env in self._envs:
                env.to(device)
        return self


class ParallelEnv(_BatchedEnv):
    """Creates one environment per process.

    TensorDicts are passed via shared memory or memory map.

    """

    __doc__ += _BatchedEnv.__doc__

    def _start_workers(self) -> None:
        _num_workers = self.num_workers
        ctx = mp.get_context("spawn")

        self.parent_channels = []
        self._workers = []
        if self.device.type == "cuda":
            self.event = torch.cuda.Event()
        else:
            self.event = None
        for idx in range(_num_workers):
            if self._verbose:
                print(f"initiating worker {idx}")
            # No certainty which module multiprocessing_context is
            channel1, channel2 = ctx.Pipe()
            env_fun = self.create_env_fn[idx]
            if env_fun.__class__.__name__ != "EnvCreator":
                env_fun = CloudpickleWrapper(env_fun)

            w = mp.Process(
                target=_run_worker_pipe_shared_mem,
                args=(
                    idx,
                    channel1,
                    channel2,
                    env_fun,
                    self.create_env_kwargs[idx],
                    False,
                    self.env_input_keys,
                    self.device,
                    self.allow_step_when_done,
                ),
            )
            w.daemon = True
            w.start()
            channel2.close()
            self.parent_channels.append(channel1)
            self._workers.append(w)
        for channel1 in self.parent_channels:
            msg = channel1.recv()
            assert msg == "started"

        # send shared tensordict to workers
        for channel, shared_tensordict in zip(
            self.parent_channels, self.shared_tensordicts
        ):
            channel.send(("init", shared_tensordict))
        self.is_closed = False

    @_check_start
    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        for channel in self.parent_channels:
            channel.send(("state_dict", None))
        for idx, channel in enumerate(self.parent_channels):
            msg, _state_dict = channel.recv()
            if msg != "state_dict":
                raise RuntimeError(f"Expected 'state_dict' but received {msg}")
            state_dict[f"worker{idx}"] = _state_dict

        return state_dict

    @_check_start
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        if "worker0" not in state_dict:
            state_dict = OrderedDict(
                **{f"worker{idx}": state_dict for idx in range(self.num_workers)}
            )
        for i, channel in enumerate(self.parent_channels):
            channel.send(("load_state_dict", state_dict[f"worker{i}"]))
        for channel in self.parent_channels:
            msg, _ = channel.recv()
            if msg != "loaded":
                raise RuntimeError(f"Expected 'loaded' but received {msg}")

    @_check_start
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._assert_tensordict_shape(tensordict)
        if self._single_task:
            # this is faster than update_ but won't work for lazy stacks
            for key in self.env_input_keys:
                # self.shared_tensordict_parent.set(
                #     key,
                #     tensordict.get(key),
                #     inplace=True,
                # )
                key = _unravel_key_to_tuple(key)
                self.shared_tensordict_parent._set_tuple(
                    key,
                    tensordict._get_tuple(key, None),
                    inplace=True,
                    validated=True,
                )
        else:
            self.shared_tensordict_parent.update_(
                tensordict.select(*self.env_input_keys, strict=False)
            )
        if self.event is not None:
            self.event.record()
            self.event.synchronize()
        for i in range(self.num_workers):
            self.parent_channels[i].send(("step", None))

        # keys = set()
        for i in range(self.num_workers):
            msg, data = self.parent_channels[i].recv()
            if msg != "step_result":
                raise RuntimeError(
                    f"Expected 'step_result' but received {msg} from worker {i}"
                )
            if data is not None:
                self.shared_tensordicts[i].update_(data)
        # We must pass a clone of the tensordict, as the values of this tensordict
        # will be modified in-place at further steps
        if self._single_task:
            out = TensorDict({}, batch_size=self.shared_tensordict_parent.shape)
            for key in self._selected_step_keys:
                _set_single_key(self.shared_tensordict_parent, out, key, clone=True)
        else:
            # strict=False ensures that non-homogeneous keys are still there
            out = self.shared_tensordict_parent.select(
                *self._selected_step_keys, strict=False
            ).clone()
        return out

    @_check_start
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        cmd_out = "reset"
        if tensordict is not None and "_reset" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            _reset = tensordict.get("_reset")
            if _reset.shape[-len(self.done_spec.shape) :] != self.done_spec.shape:
                raise RuntimeError(
                    "_reset flag in tensordict should follow env.done_spec"
                )
        else:
            _reset = torch.ones(
                self.done_spec.shape, dtype=torch.bool, device=self.device
            )

        for i, channel in enumerate(self.parent_channels):
            if tensordict is not None:
                tensordict_ = tensordict[i]
                if tensordict_.is_empty():
                    tensordict_ = None
            else:
                tensordict_ = None
            if not _reset[i].any():
                # We update the stored tensordict with the value of the "next"
                # key as one may be surprised to receive data that is not up-to-date
                # If we don't do this, the result of calling reset and skipping one env
                # will be that the env will have the data from the previous
                # step at the root (since the shared_tensordict did not go through
                # step_mdp).
                self.shared_tensordicts[i].update_(
                    self.shared_tensordicts[i]["next"].select(
                        *self._selected_reset_keys, strict=False
                    )
                )
                if tensordict_ is not None:
                    self.shared_tensordicts[i].update_(
                        tensordict_.select(*self._selected_reset_keys, strict=False)
                    )
                continue
            out = (cmd_out, tensordict_)
            channel.send(out)

        for i, channel in enumerate(self.parent_channels):
            if not _reset[i].any():
                continue
            cmd_in, data = channel.recv()
            if cmd_in != "reset_obs":
                raise RuntimeError(f"received cmd {cmd_in} instead of reset_obs")
            if data is not None:
                self.shared_tensordicts[i].update_(data)
        if self._single_task:
            # select + clone creates 2 tds, but we can create one only
            out = TensorDict({}, batch_size=self.shared_tensordict_parent.shape)
            for key in self._selected_reset_keys:
                if key != "_reset":
                    _set_single_key(self.shared_tensordict_parent, out, key, clone=True)
            return out
        else:
            return self.shared_tensordict_parent.select(
                *[key for key in self._selected_reset_keys if key != "_reset"],
                strict=False,
            ).clone()

    @_check_start
    def _shutdown_workers(self) -> None:
        if self.is_closed:
            raise RuntimeError(
                "calling {self.__class__.__name__}._shutdown_workers only allowed when env.is_closed = False"
            )
        for i, channel in enumerate(self.parent_channels):
            if self._verbose:
                print(f"closing {i}")
            # try:
            channel.send(("close", None))
            # except:
            #     raise RuntimeError(f"closing {channel} number {i} failed")
            msg, _ = channel.recv()
            if msg != "closing":
                raise RuntimeError(
                    f"Expected 'closing' but received {msg} from worker {i}"
                )

        del self.shared_tensordicts, self.shared_tensordict_parent

        for channel in self.parent_channels:
            channel.close()
        for proc in self._workers:
            proc.join()
        del self._workers
        del self.parent_channels

    @_check_start
    def set_seed(
        self, seed: Optional[int] = None, static_seed: bool = False
    ) -> Optional[int]:
        self._seeds = []
        for channel in self.parent_channels:
            channel.send(("seed", (seed, static_seed)))
            self._seeds.append(seed)
            msg, new_seed = channel.recv()
            if msg != "seeded":
                raise RuntimeError(f"Expected 'seeded' but received {msg}")
            seed = new_seed
        return seed

    def __reduce__(self):
        if not self.is_closed:
            # ParallelEnv contains non-instantiated envs, thus it can be
            # closed and serialized if the environment building functions
            # permit it
            self.close()
        return super().__reduce__()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return super().__getattr__(
                attr
            )  # make sure that appropriate exceptions are raised
        elif attr.startswith("__"):
            raise AttributeError(
                "dispatching built-in private methods is not permitted."
            )
        else:
            if attr in self._excluded_wrapped_keys:
                raise AttributeError(f"Getting {attr} resulted in an exception")
            try:
                # _ = getattr(self._dummy_env, attr)
                if self.is_closed:
                    raise RuntimeError(
                        "Trying to access attributes of closed/non started "
                        "environments. Check that the batched environment "
                        "has been started (e.g. by calling env.reset)"
                    )
                # dispatch to workers
                return _dispatch_caller_parallel(attr, self)
            except AttributeError:
                raise AttributeError(
                    f"attribute {attr} not found in " f"{self._dummy_env_str}"
                )

    def to(self, device: DEVICE_TYPING):
        device = torch.device(device)
        if device == self.device:
            return self
        super().to(device)
        if self._seeds is not None:
            warn(
                "Sending a seeded ParallelEnv to another device requires "
                f"re-seeding it. Re-seeding envs to {self._seeds}."
            )
            self.set_seed(self._seeds[0])
        return self


def _recursively_strip_locks_from_state_dict(state_dict: OrderedDict) -> OrderedDict:
    return OrderedDict(
        **{
            k: _recursively_strip_locks_from_state_dict(item)
            if isinstance(item, OrderedDict)
            else None
            if isinstance(item, MpLock)
            else item
            for k, item in state_dict.items()
        }
    )


def _run_worker_pipe_shared_mem(
    idx: int,
    parent_pipe: connection.Connection,
    child_pipe: connection.Connection,
    env_fun: Union[EnvBase, Callable],
    env_fun_kwargs: Dict[str, Any],
    pin_memory: bool,
    env_input_keys: Dict[str, Any],
    device: DEVICE_TYPING = None,
    allow_step_when_done: bool = False,
    verbose: bool = False,
) -> None:
    if device is None:
        device = torch.device("cpu")
    if device.type == "cuda":
        event = torch.cuda.Event()
    else:
        event = None

    parent_pipe.close()
    pid = os.getpid()
    if not isinstance(env_fun, EnvBase):
        env = env_fun(**env_fun_kwargs)
    else:
        if env_fun_kwargs:
            raise RuntimeError(
                "env_fun_kwargs must be empty if an environment is passed to a process."
            )
        env = env_fun
    env = env.to(device)

    i = -1
    initialized = False

    # make sure that process can be closed
    shared_tensordict = None
    local_tensordict = None

    child_pipe.send("started")

    while True:
        try:
            cmd, data = child_pipe.recv()
        except EOFError as err:
            raise EOFError(f"proc {pid} failed, last command: {cmd}.") from err
        if cmd == "seed":
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            # torch.manual_seed(data)
            # np.random.seed(data)
            new_seed = env.set_seed(data[0], static_seed=data[1])
            child_pipe.send(("seeded", new_seed))

        elif cmd == "init":
            if verbose:
                print(f"initializing {pid}")
            if initialized:
                raise RuntimeError("worker already initialized")
            i = 0
            shared_tensordict = data
            next_shared_tensordict = shared_tensordict.get("next")
            if not (shared_tensordict.is_shared() or shared_tensordict.is_memmap()):
                raise RuntimeError(
                    "tensordict must be placed in shared memory (share_memory_() or memmap_())"
                )
            initialized = True

        elif cmd == "reset":
            if verbose:
                print(f"resetting worker {pid}")
            if not initialized:
                raise RuntimeError("call 'init' before resetting")
            local_tensordict = data
            local_tensordict = env._reset(tensordict=local_tensordict)

            if "_reset" in local_tensordict.keys():
                local_tensordict.del_("_reset")
            if pin_memory:
                local_tensordict.pin_memory()
            shared_tensordict.update_(local_tensordict)
            if event is not None:
                event.record()
                event.synchronize()
            out = ("reset_obs", None)
            child_pipe.send(out)

        elif cmd == "step":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            if local_tensordict is not None:
                for key in env_input_keys:
                    # local_tensordict.set(key, shared_tensordict.get(key))
                    key = _unravel_key_to_tuple(key)
                    local_tensordict._set_tuple(
                        key,
                        shared_tensordict._get_tuple(key, None),
                        inplace=False,
                        validated=True,
                    )
            else:
                local_tensordict = shared_tensordict.clone(recurse=False)
            local_tensordict = env._step(local_tensordict)
            if pin_memory:
                local_tensordict.pin_memory()
            msg = "step_result"
            next_shared_tensordict.update_(local_tensordict.get("next"))
            if event is not None:
                event.record()
                event.synchronize()
            out = (msg, None)
            child_pipe.send(out)

        elif cmd == "close":
            del shared_tensordict, local_tensordict, data
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            env.close()
            del env

            child_pipe.send(("closing", None))
            child_pipe.close()
            if verbose:
                print(f"{pid} closed")
            break

        elif cmd == "load_state_dict":
            env.load_state_dict(data)
            msg = "loaded"
            child_pipe.send((msg, None))

        elif cmd == "state_dict":
            state_dict = _recursively_strip_locks_from_state_dict(env.state_dict())
            msg = "state_dict"
            child_pipe.send((msg, state_dict))

        else:
            err_msg = f"{cmd} from env"
            try:
                attr = getattr(env, cmd)
                if callable(attr):
                    args, kwargs = data
                    args_replace = []
                    for _arg in args:
                        if isinstance(_arg, str) and _arg == "_self":
                            continue
                        else:
                            args_replace.append(_arg)
                    result = attr(*args_replace, **kwargs)
                else:
                    result = attr
            except Exception as err:
                raise AttributeError(
                    f"querying {err_msg} resulted in an error."
                ) from err
            if cmd not in ("to"):
                child_pipe.send(("_".join([cmd, "done"]), result))
            else:
                # don't send env through pipe
                child_pipe.send(("_".join([cmd, "done"]), None))


class MultiThreadedEnvWrapper(_EnvWrapper):
    """Wrapper for envpool-based multithreaded environments."""

    _verbose: bool = False

    def __init__(
        self,
        env: Optional["envpool.python.envpool.EnvPoolMixin"] = None,  # noqa: F821
        **kwargs,
    ):
        if not _has_envpool:
            raise ImportError(
                "envpool python package or one of its dependencies (gym, treevalue) were not found. Please install these dependencies."
            )
        if env is not None:
            kwargs["env"] = env
            self.num_workers = env.config["num_envs"]
            # For synchronous mode batch size is equal to the number of workers
            self.batch_size = torch.Size([self.num_workers])
        super().__init__(**kwargs)

        # Buffer to keep the latest observation for each worker
        # It's a TensorDict when the observation consists of several variables, e.g. "position" and "velocity"
        self.obs: Union[torch.tensor, TensorDict] = self.observation_spec.zero()

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        import envpool

        if not isinstance(env, (envpool.python.envpool.EnvPoolMixin,)):
            raise TypeError("env is not of type 'envpool.python.envpool.EnvPoolMixin'.")

    def _build_env(self, env: "envpool.python.envpool.EnvPoolMixin"):  # noqa: F821
        return env

    def _make_specs(
        self, env: "envpool.python.envpool.EnvPoolMixin"  # noqa: F821
    ) -> None:  # noqa: F821
        from torchrl.envs.libs.gym import set_gym_backend

        with set_gym_backend("gym"):
            self.action_spec = self._get_action_spec()
            output_spec = self._get_output_spec()
            self.observation_spec = output_spec["_observation_spec"]
            self.reward_spec = output_spec["_reward_spec"]
            self.done_spec = output_spec["_done_spec"]

    def _init_env(self) -> Optional[int]:
        pass

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        if tensordict is not None:
            reset_workers = tensordict.get("_reset", None)
        else:
            reset_workers = None
        if reset_workers is not None:
            reset_data = self._env.reset(np.where(reset_workers.cpu().numpy())[0])
        else:
            reset_data = self._env.reset()
        tensordict_out = self._transform_reset_output(reset_data, reset_workers)
        self.is_closed = False
        return tensordict_out

    @torch.no_grad()
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        # Action needs to be moved to CPU and converted to numpy before being passed to envpool
        action = action.to(torch.device("cpu"))
        step_output = self._env.step(action.numpy())
        tensordict_out = self._transform_step_output(step_output)
        return tensordict_out.select().set("next", tensordict_out)

    def _get_action_spec(self) -> TensorSpec:
        # local import to avoid importing gym in the script
        from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

        # Envpool provides Gym-compatible specs as env.spec.action_space and
        # DM_Control-compatible specs as env.spec.action_spec(). We use the Gym ones.

        # Gym specs produced by EnvPool don't contain batch_size, we add it to satisfy checks in EnvBase
        action_spec = _gym_to_torchrl_spec_transform(
            self._env.spec.action_space,
            device=self.device,
            categorical_action_encoding=True,
        )
        action_spec = self._add_shape_to_spec(action_spec)
        return action_spec

    def _get_output_spec(self) -> TensorSpec:
        return CompositeSpec(
            _observation_spec=self._get_observation_spec(),
            _reward_spec=self._get_reward_spec(),
            _done_spec=self._get_done_spec(),
            shape=(self.num_workers,),
            device=self.device,
        )

    def _get_observation_spec(self) -> TensorSpec:
        # local import to avoid importing gym in the script
        from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

        # Gym specs produced by EnvPool don't contain batch_size, we add it to satisfy checks in EnvBase
        observation_spec = _gym_to_torchrl_spec_transform(
            self._env.spec.observation_space,
            device=self.device,
            categorical_action_encoding=True,
        )
        observation_spec = self._add_shape_to_spec(observation_spec)
        if isinstance(observation_spec, CompositeSpec):
            return observation_spec
        return CompositeSpec(
            observation=observation_spec,
            shape=(self.num_workers,),
            device=self.device,
        )

    def _add_shape_to_spec(self, spec: TensorSpec) -> TensorSpec:
        return spec.expand((self.num_workers, *spec.shape))

    def _get_reward_spec(self) -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            device=self.device,
            shape=self.batch_size,
        )

    def _get_done_spec(self) -> TensorSpec:
        return DiscreteTensorSpec(
            2,
            device=self.device,
            shape=self.batch_size,
            dtype=torch.bool,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_workers={self.num_workers}, device={self.device})"

    def _transform_reset_output(
        self,
        envpool_output: Tuple[
            Union["treevalue.TreeValue", np.ndarray], Any  # noqa: F821
        ],
        reset_workers: Optional[torch.Tensor],
    ):
        """Process output of envpool env.reset."""
        import treevalue

        observation, _ = envpool_output
        if reset_workers is not None:
            # Only specified workers were reset - need to set observation buffer values only for them
            if isinstance(observation, treevalue.TreeValue):
                # If observation contain several fields, it will be returned as treevalue.TreeValue.
                # Convert to treevalue.FastTreeValue to allow indexing
                observation = treevalue.FastTreeValue(observation)
            self.obs[reset_workers] = self._treevalue_or_numpy_to_tensor_or_dict(
                observation
            )
        else:
            # All workers were reset - rewrite the whole observation buffer
            self.obs = TensorDict(
                self._treevalue_or_numpy_to_tensor_or_dict(observation), self.batch_size
            )

        obs = self.obs.clone(False)
        obs.update({self.done_key: self.done_spec.zero()})
        return obs

    def _transform_step_output(
        self, envpool_output: Tuple[Any, Any, Any, ...]
    ) -> TensorDict:
        """Process output of envpool env.step."""
        obs, reward, done, *_ = envpool_output

        obs = self._treevalue_or_numpy_to_tensor_or_dict(obs)
        obs.update({self.reward_key: torch.tensor(reward), self.done_key: done})
        self.obs = tensordict_out = TensorDict(
            obs,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _treevalue_or_numpy_to_tensor_or_dict(
        self, x: Union["treevalue.TreeValue", np.ndarray]  # noqa: F821
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Converts observation returned by EnvPool.

        EnvPool step and reset return observation as a numpy array or a TreeValue of numpy arrays, which we convert
        to a tensor or a dictionary of tensors. Currently only supports depth 1 trees, but can easily be extended to
        arbitrary depth if necessary.
        """
        import treevalue

        if isinstance(x, treevalue.TreeValue):
            ret = self._treevalue_to_dict(x)
        elif not isinstance(x, dict):
            ret = {"observation": torch.tensor(x)}
        else:
            ret = x
        return ret

    def _treevalue_to_dict(
        self, tv: "treevalue.TreeValue"  # noqa: F821
    ) -> Dict[str, Any]:
        """Converts TreeValue to a dictionary.

        Currently only supports depth 1 trees, but can easily be extended to arbitrary depth if necessary.
        """
        import treevalue

        return {k[0]: torch.tensor(v) for k, v in treevalue.flatten(tv)}

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            print(
                "MultiThreadedEnvWrapper._set_seed ignored, as setting seed in an existing envorinment is not\
                   supported by envpool. Please create a new environment, passing the seed to the constructor."
            )


class MultiThreadedEnv(MultiThreadedEnvWrapper):
    """Multithreaded execution of environments based on EnvPool.

    An alternative to ParallelEnv based on multithreading. It's faster, as it doesn't require new process spawning, but
    less flexible, as it only supports environments implemented in EnvPool library.
    Currently only supports synchronous execution mode, when the batch size is equal to the number of workers, see
    https://envpool.readthedocs.io/en/latest/content/python_interface.html#batch-size.

    >>> env = MultiThreadedEnv(num_workers=3, env_name="Pendulum-v1")
    >>> env.reset()
    >>> env.rand_step()
    >>> env.rollout(5)
    >>> env.close()

    Args:
        num_workers: number of worker threads to create.
        env_name: name of the environment, corresponding to task_id in EnvPool.
        create_env_kwargs: additional arguments which will be passed to envpool.make.
    """

    def __init__(
        self,
        num_workers: int,
        env_name: str,
        create_env_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.env_name = env_name.replace("ALE/", "")  # Naming convention of EnvPool
        self.num_workers = num_workers
        self.batch_size = torch.Size([num_workers])
        self.create_env_kwargs = create_env_kwargs or {}

        kwargs["num_workers"] = num_workers
        kwargs["env_name"] = self.env_name
        kwargs["create_env_kwargs"] = create_env_kwargs
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        num_workers: int,
        create_env_kwargs: Optional[Dict[str, Any]],
    ) -> Any:
        import envpool

        create_env_kwargs = create_env_kwargs or {}
        env = envpool.make(
            task_id=env_name,
            env_type="gym",
            num_envs=num_workers,
            gym_reset_return_info=True,
            **create_env_kwargs,
        )
        return super()._build_env(env)

    def _set_seed(self, seed: Optional[int]):
        """Library EnvPool only supports setting a seed by recreating the environment."""
        if seed is not None:
            logging.debug("Recreating EnvPool environment to set seed.")
            self.create_env_kwargs["seed"] = seed
            self._env = self._build_env(
                env_name=self.env_name,
                num_workers=self.num_workers,
                create_env_kwargs=self.create_env_kwargs,
            )

    def _check_kwargs(self, kwargs: Dict):
        for arg in ["num_workers", "env_name", "create_env_kwargs"]:
            if arg not in kwargs:
                raise TypeError(f"Expected '{arg}' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, num_workers={self.num_workers}, device={self.device})"

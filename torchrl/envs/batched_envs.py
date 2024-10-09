# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools

import gc

import os
import weakref
from collections import OrderedDict
from copy import copy, deepcopy
from functools import wraps
from multiprocessing import connection
from multiprocessing.synchronize import Lock as MpLock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import torch

from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.utils import _zip_strict
from torch import multiprocessing as mp
from torchrl._utils import (
    _check_for_faulty_process,
    _make_ordinal_device,
    _ProcessNoWarn,
    logger as torchrl_logger,
    VERBOSE,
)
from torchrl.data.tensor_specs import Composite, NonTensor
from torchrl.data.utils import CloudpickleWrapper, contains_lazy_spec, DEVICE_TYPING
from torchrl.envs.common import _do_nothing, _EnvPostInit, EnvBase, EnvMetaData
from torchrl.envs.env_creator import get_env_metadata

# legacy
from torchrl.envs.libs.envpool import (  # noqa: F401
    MultiThreadedEnv,
    MultiThreadedEnvWrapper,
)

from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    _sort_keys,
    _update_during_reset,
    clear_mpi_env_vars,
)


def _check_start(fun):
    def decorated_fun(self: BatchedEnvBase, *args, **kwargs):
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


class _PEnvMeta(_EnvPostInit):
    def __call__(cls, *args, **kwargs):
        serial_for_single = kwargs.pop("serial_for_single", False)
        if serial_for_single:
            num_workers = kwargs.get("num_workers")
            # Remove start method from kwargs
            kwargs.pop("mp_start_method", None)
            if num_workers is None:
                num_workers = args[0]
            if num_workers == 1:
                # We still use a serial to keep the shape unchanged
                return SerialEnv(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class BatchedEnvBase(EnvBase):
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

    Keyword Args:
        create_env_kwargs (dict or list of dicts, optional): kwargs to be used with the environments being created;
        share_individual_td (bool, optional): if ``True``, a different tensordict is created for every process/worker and a lazy
            stack is returned.
            default = None (False if single task);
        shared_memory (bool): whether the returned tensordict will be placed in shared memory;
        memmap (bool): whether the returned tensordict will be placed in memory map.
        policy_proof (callable, optional): if provided, it'll be used to get the list of
            tensors to return through the :obj:`step()` and :obj:`reset()` methods, such as :obj:`"hidden"` etc.
        device (str, int, torch.device): The device of the batched environment can be passed.
            If not, it is inferred from the env. In this case, it is assumed that
            the device of all environments match. If it is provided, it can differ
            from the sub-environment device(s). In that case, the data will be
            automatically cast to the appropriate device during collection.
            This can be used to speed up collection in case casting to device
            introduces an overhead (eg, numpy-based environents etc.): by using
            a ``"cuda"`` device for the batched environment but a ``"cpu"``
            device for the nested environments, one can keep the overhead to a
            minimum.
        num_threads (int, optional): number of threads for this process.
            Should be equal to one plus the number of processes launched within
            each subprocess (or one if a single process is launched).
            Defaults to the number of workers + 1.
            This parameter has no effect for the :class:`~SerialEnv` class.
        num_sub_threads (int, optional): number of threads of the subprocesses.
            Defaults to 1 for safety: if none is indicated, launching multiple
            workers may charge the cpu load too much and harm performance.
            This parameter has no effect for the :class:`~SerialEnv` class.
        serial_for_single (bool, optional): if ``True``, creating a parallel environment
            with a single worker will return a :class:`~SerialEnv` instead.
            This option has no effect with :class:`~SerialEnv`. Defaults to ``False``.
        non_blocking (bool, optional): if ``True``, device moves will be done using the
            ``non_blocking=True`` option. Defaults to ``True``.
        mp_start_method (str, optional): the multiprocessing start method.
            Uses the default start method if not indicated ('spawn' by default in
            TorchRL if not initiated differently before first import).
            To be used only with :class:`~torchrl.envs.ParallelEnv` subclasses.
        use_buffers (bool, optional): whether communication between workers should
            occur via circular preallocated memory buffers. Defaults to ``True`` unless
            one of the environment has dynamic specs.

              .. note:: Learn more about dynamic specs and environments :ref:`here <dynamic_envs>`.

    .. note::
        One can pass keyword arguments to each sub-environments using the following
        technique: every keyword argument in :meth:`~.reset` will be passed to each
        environment except for the ``list_of_kwargs`` argument which, if present,
        should contain a list of the same length as the number of workers with the
        worker-specific keyword arguments stored in a dictionary.
        If a partial reset is queried, the element of ``list_of_kwargs`` corresponding
        to sub-environments that are not reset will be ignored.

    Examples:
        >>> from torchrl.envs import GymEnv, ParallelEnv, SerialEnv, EnvCreator
        >>> make_env = EnvCreator(lambda: GymEnv("Pendulum-v1")) # EnvCreator ensures that the env is sharable. Optional in most cases.
        >>> env = SerialEnv(2, make_env)  # Makes 2 identical copies of the Pendulum env, runs them on the same process serially
        >>> env = ParallelEnv(2, make_env)  # Makes 2 identical copies of the Pendulum env, runs them on dedicated processes
        >>> from torchrl.envs import DMControlEnv
        >>> env = ParallelEnv(2, [
        ...     lambda: DMControlEnv("humanoid", "stand"),
        ...     lambda: DMControlEnv("humanoid", "walk")])  # Creates two independent copies of Humanoid, one that walks one that stands
        >>> rollout = env.rollout(10)  # executes 10 random steps in the environment
        >>> rollout[0]  # data for Humanoid stand
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([10, 21]), device=cpu, dtype=torch.float64, is_shared=False),
                com_velocity: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                extremities: Tensor(shape=torch.Size([10, 12]), device=cpu, dtype=torch.float64, is_shared=False),
                head_height: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                joint_angles: Tensor(shape=torch.Size([10, 21]), device=cpu, dtype=torch.float64, is_shared=False),
                next: TensorDict(
                    fields={
                        com_velocity: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        extremities: Tensor(shape=torch.Size([10, 12]), device=cpu, dtype=torch.float64, is_shared=False),
                        head_height: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                        joint_angles: Tensor(shape=torch.Size([10, 21]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        torso_vertical: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        velocity: Tensor(shape=torch.Size([10, 27]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([10]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                torso_vertical: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                velocity: Tensor(shape=torch.Size([10, 27]), device=cpu, dtype=torch.float64, is_shared=False)},
            batch_size=torch.Size([10]),
            device=cpu,
            is_shared=False)
        >>> rollout[1]  # data for Humanoid walk
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([10, 21]), device=cpu, dtype=torch.float64, is_shared=False),
                com_velocity: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                extremities: Tensor(shape=torch.Size([10, 12]), device=cpu, dtype=torch.float64, is_shared=False),
                head_height: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                joint_angles: Tensor(shape=torch.Size([10, 21]), device=cpu, dtype=torch.float64, is_shared=False),
                next: TensorDict(
                    fields={
                        com_velocity: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        extremities: Tensor(shape=torch.Size([10, 12]), device=cpu, dtype=torch.float64, is_shared=False),
                        head_height: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                        joint_angles: Tensor(shape=torch.Size([10, 21]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        torso_vertical: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        velocity: Tensor(shape=torch.Size([10, 27]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([10]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                torso_vertical: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                velocity: Tensor(shape=torch.Size([10, 27]), device=cpu, dtype=torch.float64, is_shared=False)},
            batch_size=torch.Size([10]),
            device=cpu,
            is_shared=False)
        >>> # serial_for_single to avoid creating parallel envs if not necessary
        >>> env = ParallelEnv(1, make_env, serial_for_single=True)
        >>> assert isinstance(env, SerialEnv)  # serial_for_single allows you to avoid creating parallel envs when not necessary
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
        *,
        create_env_kwargs: Union[dict, Sequence[dict]] = None,
        pin_memory: bool = False,
        share_individual_td: Optional[bool] = None,
        shared_memory: bool = True,
        memmap: bool = False,
        policy_proof: Optional[Callable] = None,
        device: Optional[DEVICE_TYPING] = None,
        allow_step_when_done: bool = False,
        num_threads: int = None,
        num_sub_threads: int = 1,
        serial_for_single: bool = False,
        non_blocking: bool = False,
        mp_start_method: str = None,
        use_buffers: bool = None,
    ):
        super().__init__(device=device)
        self.serial_for_single = serial_for_single
        self.is_closed = True
        self.num_sub_threads = num_sub_threads
        self.num_threads = num_threads
        self._cache_in_keys = None
        self._use_buffers = use_buffers

        self._single_task = callable(create_env_fn) or (len(set(create_env_fn)) == 1)
        if callable(create_env_fn):
            create_env_fn = [create_env_fn for _ in range(num_workers)]
        elif len(create_env_fn) != num_workers:
            raise RuntimeError(
                f"len(create_env_fn) and num_workers mismatch, "
                f"got {len(create_env_fn)} and {num_workers}."
            )

        create_env_kwargs = {} if create_env_kwargs is None else create_env_kwargs
        if isinstance(create_env_kwargs, dict):
            create_env_kwargs = [
                deepcopy(create_env_kwargs) for _ in range(num_workers)
            ]
        elif len(create_env_kwargs) != num_workers:
            raise RuntimeError(
                f"len(create_env_kwargs) and num_workers mismatch, "
                f"got {len(create_env_kwargs)} and {num_workers}."
            )

        self.policy_proof = policy_proof
        self.num_workers = num_workers
        self.create_env_fn = create_env_fn
        self.create_env_kwargs = create_env_kwargs
        self.pin_memory = pin_memory
        if pin_memory:
            raise ValueError("pin_memory for batched envs is deprecated")

        # if share_individual_td is None, we will assess later if the output can be stacked
        self.share_individual_td = share_individual_td
        self._share_memory = shared_memory
        self._memmap = memmap
        self.allow_step_when_done = allow_step_when_done
        if allow_step_when_done:
            raise ValueError("allow_step_when_done is deprecated")
        if self._share_memory and self._memmap:
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        self._batch_size = None
        self._device = (
            _make_ordinal_device(torch.device(device)) if device is not None else device
        )
        self._dummy_env_str = None
        self._seeds = None
        self.__dict__["_input_spec"] = None
        self.__dict__["_output_spec"] = None
        # self._prepare_dummy_env(create_env_fn, create_env_kwargs)
        self._properties_set = False
        self._get_metadata(create_env_fn, create_env_kwargs)
        self._non_blocking = non_blocking
        if mp_start_method is not None and not isinstance(self, ParallelEnv):
            raise TypeError(
                f"Cannot use mp_start_method={mp_start_method} with envs of type {type(self)}."
            )
        self._mp_start_method = mp_start_method

    @property
    def non_blocking(self):
        nb = self._non_blocking
        if nb is None:
            nb = True
            self._non_blocking = nb
        return nb

    @property
    def _sync_m2w(self) -> Callable:
        sync_func = self.__dict__.get("_sync_m2w_value")
        if sync_func is None:
            sync_m2w, sync_w2m = self._find_sync_values()
            self.__dict__["_sync_m2w_value"] = sync_m2w
            self.__dict__["_sync_w2m_value"] = sync_w2m
            return sync_m2w
        return sync_func

    @property
    def _sync_w2m(self) -> Callable:
        sync_func = self.__dict__.get("_sync_w2m_value")
        if sync_func is None:
            sync_m2w, sync_w2m = self._find_sync_values()
            self.__dict__["_sync_m2w_value"] = sync_m2w
            self.__dict__["_sync_w2m_value"] = sync_w2m
            return sync_w2m
        return sync_func

    def _find_sync_values(self):
        """Returns the m2w and w2m sync values, in that order."""
        if not self._use_buffers:
            return _do_nothing, _do_nothing
        # Simplest case: everything is on the same device
        worker_device = self.shared_tensordict_parent.device
        self_device = self.device
        if not self.non_blocking or (
            worker_device == self_device or self_device is None
        ):
            # even if they're both None, there is no device-to-device movement
            return _do_nothing, _do_nothing

        if worker_device is None:
            worker_not_main = False

            def find_all_worker_devices(item):
                nonlocal worker_not_main
                if hasattr(item, "device"):
                    worker_not_main = worker_not_main or (item.device != self_device)

            for td in self.shared_tensordicts:
                td.apply(find_all_worker_devices, filter_empty=True)
            if worker_not_main:
                if torch.cuda.is_available():
                    worker_device = (
                        torch.device("cuda")
                        if self_device.type != "cuda"
                        else torch.device("cpu")
                    )
                elif torch.backends.mps.is_available():
                    worker_device = (
                        torch.device("mps")
                        if self_device.type != "mps"
                        else torch.device("cpu")
                    )
                else:
                    raise RuntimeError("Did not find a valid worker device")
            else:
                worker_device = self_device

        if (
            worker_device is not None
            and worker_device.type == "cuda"
            and self_device is not None
            and self_device.type == "cpu"
        ):
            return _do_nothing, _cuda_sync(worker_device)
        if (
            worker_device is not None
            and worker_device.type == "mps"
            and self_device is not None
            and self_device.type == "cpu"
        ):
            return _mps_sync(worker_device), _mps_sync(worker_device)
        if (
            worker_device is not None
            and worker_device.type == "cpu"
            and self_device is not None
            and self_device.type == "cuda"
        ):
            return _cuda_sync(self_device), _do_nothing
        if (
            worker_device is not None
            and worker_device.type == "cpu"
            and self_device is not None
            and self_device.type == "mps"
        ):
            return _mps_sync(self_device), _mps_sync(self_device)
        return _do_nothing, _do_nothing

    def __getstate__(self):
        out = copy(self.__dict__)
        out["_sync_m2w_value"] = None
        out["_sync_w2m_value"] = None
        return out

    @property
    def _has_dynamic_specs(self):
        return not self._use_buffers

    def _get_metadata(
        self, create_env_fn: List[Callable], create_env_kwargs: List[Dict]
    ):
        if self._single_task:
            # if EnvCreator, the metadata are already there
            meta_data: EnvMetaData = get_env_metadata(
                create_env_fn[0], create_env_kwargs[0]
            )
            self.meta_data = meta_data.expand(
                *(self.num_workers, *meta_data.batch_size)
            )
            if self._use_buffers is not False:
                _use_buffers = not self.meta_data.has_dynamic_specs
                if self._use_buffers and not _use_buffers:
                    warn(
                        "A value of use_buffers=True was passed but this is incompatible "
                        "with the list of environments provided. Turning use_buffers to False."
                    )
                self._use_buffers = _use_buffers
            if self.share_individual_td is None:
                self.share_individual_td = False
        else:
            n_tasks = len(create_env_fn)
            self.meta_data: List[EnvMetaData] = []
            for i in range(n_tasks):
                self.meta_data.append(
                    get_env_metadata(create_env_fn[i], create_env_kwargs[i]).clone()
                )
            if self.share_individual_td is not True:
                share_individual_td = not _stackable(
                    *[meta_data.tensordict for meta_data in self.meta_data]
                )
                if share_individual_td and self.share_individual_td is False:
                    raise ValueError(
                        "share_individual_td=False was provided but share_individual_td must "
                        "be True to accommodate non-stackable tensors."
                    )
                self.share_individual_td = share_individual_td
            _use_buffers = all(
                not metadata.has_dynamic_specs for metadata in self.meta_data
            )
            if self._use_buffers and not _use_buffers:
                warn(
                    "A value of use_buffers=True was passed but this is incompatible "
                    "with the list of environments provided. Turning use_buffers to False."
                )
            self._use_buffers = _use_buffers

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
            if len(kwargs) != self.num_workers:
                raise RuntimeError(
                    f"len(kwargs) and num_workers mismatch, got {len(kwargs)} and {self.num_workers}."
                )
            for _kwargs, _new_kwargs in _zip_strict(self.create_env_kwargs, kwargs):
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

        cls = type(self)

        def _check_for_empty_spec(specs: Composite):
            for subspec in (
                "full_state_spec",
                "full_action_spec",
                "full_done_spec",
                "full_reward_spec",
                "full_observation_spec",
            ):
                for key, spec in reversed(
                    list(specs.get(subspec, default=Composite()).items(True))
                ):
                    if isinstance(spec, Composite) and spec.is_empty():
                        raise RuntimeError(
                            f"The environment passed to {cls.__name__} has empty specs in {key}. Consider using "
                            f"torchrl.envs.transforms.RemoveEmptySpecs to remove the empty specs."
                        )
            return specs

        meta_data = self.meta_data
        self._properties_set = True
        if self._single_task:
            self._batch_size = meta_data.batch_size
            device = meta_data.device
            if self._device is None:
                self._device = device

            input_spec = _check_for_empty_spec(meta_data.specs["input_spec"].to(device))
            output_spec = _check_for_empty_spec(
                meta_data.specs["output_spec"].to(device)
            )

            self.action_spec = input_spec["full_action_spec"]
            self.state_spec = input_spec["full_state_spec"]
            self.observation_spec = output_spec["full_observation_spec"]
            self.reward_spec = output_spec["full_reward_spec"]
            self.done_spec = output_spec["full_done_spec"]

            self._dummy_env_str = meta_data.env_str
            self._env_tensordict = meta_data.tensordict
            if device is None:  # In other cases, the device will be mapped later
                self._env_tensordict.clear_device_()
                device_map = meta_data.device_map

                def map_device(key, value, device_map=device_map):
                    return value.to(device_map[key])

                self._env_tensordict.named_apply(
                    map_device, nested_keys=True, filter_empty=True
                )

            self._batch_locked = meta_data.batch_locked
        else:
            self._batch_size = torch.Size([self.num_workers, *meta_data[0].batch_size])
            devices = set()
            for _meta_data in meta_data:
                device = _meta_data.device
                devices.add(device)
            if self._device is None:
                if len(devices) > 1:
                    raise ValueError(
                        f"The device wasn't passed to {type(self)}, but more than one device was found in the sub-environments. "
                        f"Please indicate a device to be used for collection."
                    )
                device = list(devices)[0]
                self._device = device

            input_spec = []
            for md in meta_data:
                input_spec.append(_check_for_empty_spec(md.specs["input_spec"]))
            input_spec = torch.stack(input_spec, 0)
            output_spec = []
            for md in meta_data:
                output_spec.append(_check_for_empty_spec(md.specs["output_spec"]))
            output_spec = torch.stack(output_spec, 0)

            self.action_spec = input_spec["full_action_spec"]
            self.state_spec = input_spec["full_state_spec"]

            self.observation_spec = output_spec["full_observation_spec"]
            self.reward_spec = output_spec["full_reward_spec"]
            self.done_spec = output_spec["full_done_spec"]

            self._dummy_env_str = str(meta_data[0])
            if self.share_individual_td:
                self._env_tensordict = LazyStackedTensorDict.lazy_stack(
                    [meta_data.tensordict for meta_data in meta_data], 0
                )
            else:
                self._env_tensordict = torch.stack(
                    [meta_data.tensordict for meta_data in meta_data], 0
                )
            self._batch_locked = meta_data[0].batch_locked
        self.has_lazy_inputs = contains_lazy_spec(self.input_spec)

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
        if not self._use_buffers:
            return
        shared_tensordict_parent = self._env_tensordict.clone()
        if self._env_tensordict.shape[0] != self.num_workers:
            raise RuntimeError(
                "batched environment base tensordict has the wrong shape"
            )

        # Non-tensor keys
        non_tensor_keys = []
        for spec in (
            self.full_action_spec,
            self.full_state_spec,
            self.full_observation_spec,
            self.full_reward_spec,
            self.full_done_spec,
        ):
            for key, _spec in spec.items(True, True):
                if isinstance(_spec, NonTensor):
                    non_tensor_keys.append(key)
        self._non_tensor_keys = non_tensor_keys

        if self._single_task:
            self._env_input_keys = sorted(
                list(self.input_spec["full_action_spec"].keys(True, True))
                + list(self.state_spec.keys(True, True)),
                key=_sort_keys,
            )
            self._env_output_keys = []
            self._env_obs_keys = []
            for key in self.output_spec["full_observation_spec"].keys(True, True):
                self._env_output_keys.append(key)
                self._env_obs_keys.append(key)
            self._env_output_keys += self.reward_keys + self.done_keys
        else:
            # this is only possible if _single_task=False
            env_input_keys = set()
            for meta_data in self.meta_data:
                if meta_data.specs["input_spec", "full_state_spec"] is not None:
                    env_input_keys = env_input_keys.union(
                        meta_data.specs["input_spec", "full_state_spec"].keys(
                            True, True
                        )
                    )
                env_input_keys = env_input_keys.union(
                    meta_data.specs["input_spec", "full_action_spec"].keys(True, True)
                )
            env_output_keys = set()
            env_obs_keys = set()
            for meta_data in self.meta_data:
                env_obs_keys = env_obs_keys.union(
                    key
                    for key in meta_data.specs["output_spec"][
                        "full_observation_spec"
                    ].keys(True, True)
                )
                env_output_keys = env_output_keys.union(
                    meta_data.specs["output_spec"]["full_observation_spec"].keys(
                        True, True
                    )
                )
            env_output_keys = env_output_keys.union(self.reward_keys + self.done_keys)
            env_obs_keys = [
                key for key in env_obs_keys if key not in self._non_tensor_keys
            ]
            env_input_keys = [
                key for key in env_input_keys if key not in self._non_tensor_keys
            ]
            env_output_keys = [
                key for key in env_output_keys if key not in self._non_tensor_keys
            ]
            self._env_obs_keys = sorted(env_obs_keys, key=_sort_keys)
            self._env_input_keys = sorted(env_input_keys, key=_sort_keys)
            self._env_output_keys = sorted(env_output_keys, key=_sort_keys)

        reset_keys = self.reset_keys
        self._selected_keys = (
            set(self._env_output_keys)
            .union(self._env_input_keys)
            .union(self._env_obs_keys)
            .union(set(self.done_keys))
        )
        self._selected_keys = self._selected_keys.union(reset_keys)

        # input keys
        self._selected_input_keys = {unravel_key(key) for key in self._env_input_keys}
        # output keys after reset
        self._selected_reset_keys = {
            unravel_key(key) for key in self._env_obs_keys + self.done_keys + reset_keys
        }
        # output keys after reset, filtered
        self._selected_reset_keys_filt = {
            unravel_key(key) for key in self._env_obs_keys + self.done_keys
        }
        # output keys after step
        self._selected_step_keys = {unravel_key(key) for key in self._env_output_keys}

        if not self.share_individual_td:
            shared_tensordict_parent = shared_tensordict_parent.filter_non_tensor_data()
            shared_tensordict_parent = shared_tensordict_parent.select(
                *self._selected_keys,
                *(unravel_key(("next", key)) for key in self._env_output_keys),
                strict=False,
            )
            self.shared_tensordict_parent = shared_tensordict_parent
        else:
            # Multi-task: we share tensordict that *may* have different keys
            shared_tensordict_parent = [
                tensordict.select(
                    *self._selected_keys,
                    *(unravel_key(("next", key)) for key in self._env_output_keys),
                    strict=False,
                ).filter_non_tensor_data()
                for tensordict in shared_tensordict_parent
            ]
            shared_tensordict_parent = LazyStackedTensorDict.lazy_stack(
                shared_tensordict_parent,
                0,
            )
            self.shared_tensordict_parent = shared_tensordict_parent

        if self.share_individual_td:
            if not isinstance(self.shared_tensordict_parent, LazyStackedTensorDict):
                self.shared_tensordicts = [
                    td.clone() for td in self.shared_tensordict_parent.unbind(0)
                ]
                self.shared_tensordict_parent = LazyStackedTensorDict.lazy_stack(
                    self.shared_tensordicts, 0
                )
            else:
                # Multi-task: we share tensordict that *may* have different keys
                # LazyStacked already stores this so we don't need to do anything
                self.shared_tensordicts = self.shared_tensordict_parent
            if self._share_memory:
                self.shared_tensordict_parent.share_memory_()
            elif self._memmap:
                self.shared_tensordict_parent.memmap_()
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
            for td in self.shared_tensordicts:
                td.lock_()

        # we cache all the keys of the shared parent td for future use. This is
        # safe since the td is locked.
        self._cache_shared_keys = set(self.shared_tensordict_parent.keys(True, True))

        self._shared_tensordict_parent_next = self.shared_tensordict_parent.get("next")
        self._shared_tensordict_parent_root = self.shared_tensordict_parent.exclude(
            "next", *self.reset_keys
        )

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
            torchrl_logger.info(f"closing {self.__class__.__name__}")

        self.__dict__["_input_spec"] = None
        self.__dict__["_output_spec"] = None
        self._properties_set = False

        self._shutdown_workers()
        self.is_closed = True
        import torchrl

        num_threads = min(
            torchrl._THREAD_POOL_INIT, torch.get_num_threads() + self.num_workers
        )
        torch.set_num_threads(num_threads)

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
        self._non_blocking = None
        device = _make_ordinal_device(torch.device(device))
        if device == self.device:
            return self
        self._device = device
        self.__dict__["_sync_m2w_value"] = None
        self.__dict__["_sync_w2m_value"] = None
        if self.__dict__["_input_spec"] is not None:
            self.__dict__["_input_spec"] = self.__dict__["_input_spec"].to(device)
        if self.__dict__["_output_spec"] is not None:
            self.__dict__["_output_spec"] = self.__dict__["_output_spec"].to(device)
        return self

    def _reset_proc_data(self, tensordict, tensordict_reset):
        # since we call `reset` directly, all the postproc has been completed
        if tensordict is not None:
            if isinstance(tensordict_reset, LazyStackedTensorDict) and not isinstance(
                tensordict, LazyStackedTensorDict
            ):
                tensordict = LazyStackedTensorDict(*tensordict.unbind(0))
            return _update_during_reset(tensordict_reset, tensordict, self.reset_keys)
        return tensordict_reset

    def add_truncated_keys(self):
        raise RuntimeError(
            "Cannot add truncated keys to a batched environment. Please add these entries to "
            "the nested environments by calling sub_env.add_truncated_keys()"
        )


class SerialEnv(BatchedEnvBase):
    """Creates a series of environments in the same process."""

    __doc__ += BatchedEnvBase.__doc__

    _share_memory = False

    def _start_workers(self) -> None:
        _num_workers = self.num_workers

        self._envs = []
        weakref_set = set()
        for idx in range(_num_workers):
            env = self.create_env_fn[idx](**self.create_env_kwargs[idx])
            # We want to avoid having the same env multiple times
            # so we try to deepcopy it if needed. If we can't, we make
            # the user aware that this isn't a very good idea
            wr = weakref.ref(env)
            if wr in weakref_set:
                try:
                    env = deepcopy(env)
                except Exception:
                    warn(
                        "Deepcopying the env failed within SerialEnv "
                        "but more than one copy of the same env was found. "
                        "This is a dangerous situation if your env keeps track "
                        "of some variables (e.g., state) in-place. "
                        "We'll use the same copy of the environment be beaware that "
                        "this may have important, unwanted issues for stateful "
                        "environments!"
                    )
            weakref_set.add(wr)
            self._envs.append(env)
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
        list_of_kwargs = kwargs.pop("list_of_kwargs", [kwargs] * self.num_workers)
        if kwargs is not list_of_kwargs[0] and kwargs:
            # this means that kwargs had more than one element and that a list was provided
            for elt in list_of_kwargs:
                elt.update(kwargs)
        if tensordict is not None:
            needs_resetting = _aggregate_end_of_traj(
                tensordict, reset_keys=self.reset_keys
            )
            if needs_resetting.ndim > 2:
                needs_resetting = needs_resetting.flatten(1, needs_resetting.ndim - 1)
            if needs_resetting.ndim > 1:
                needs_resetting = needs_resetting.any(-1)
            elif not needs_resetting.ndim:
                needs_resetting = needs_resetting.expand((self.num_workers,))
            tensordict = tensordict.unbind(0)
        else:
            needs_resetting = torch.ones(
                (self.num_workers,), device=self.device, dtype=torch.bool
            )

        out_tds = None
        if not self._use_buffers or self._non_tensor_keys:
            out_tds = [None] * self.num_workers

        tds = []
        for i, _env in enumerate(self._envs):
            if not needs_resetting[i]:
                if out_tds is not None and tensordict is not None:
                    out_tds[i] = tensordict[i].exclude(*self._envs[i].reset_keys)
                continue
            if tensordict is not None:
                tensordict_ = tensordict[i]
                if tensordict_.is_empty():
                    tensordict_ = None
                else:
                    env_device = _env.device
                    if env_device != self.device and env_device is not None:
                        tensordict_ = tensordict_.to(
                            env_device, non_blocking=self.non_blocking
                        )
                    else:
                        tensordict_ = tensordict_.clone(False)
            else:
                tensordict_ = None
            tds.append((i, tensordict_))

        self._sync_m2w()
        for i, tensordict_ in tds:
            _env = self._envs[i]
            _td = _env.reset(tensordict=tensordict_, **list_of_kwargs[i])
            if self._use_buffers:
                try:
                    self.shared_tensordicts[i].update_(
                        _td,
                        keys_to_update=list(self._selected_reset_keys_filt),
                        non_blocking=self.non_blocking,
                    )
                except RuntimeError as err:
                    if "no_grad mode" in str(err):
                        raise RuntimeError(
                            "Cannot update a view of a tensordict when gradients are required. "
                            "To collect gradient across sub-environments, please set the "
                            "share_individual_td argument to True."
                        )
                    raise
            if out_tds is not None:
                out_tds[i] = _td

        device = self.device
        if not self._use_buffers:
            result = LazyStackedTensorDict.maybe_dense_stack(out_tds)
            if result.device != device:
                if device is None:
                    result = result.clear_device_()
                else:
                    result = result.to(device, non_blocking=self.non_blocking)
                    self._sync_w2m()
            return result

        selected_output_keys = self._selected_reset_keys_filt

        # select + clone creates 2 tds, but we can create one only
        def select_and_clone(name, tensor):
            if name in selected_output_keys:
                return tensor.clone()

        out = self.shared_tensordict_parent.named_apply(
            select_and_clone,
            nested_keys=True,
            filter_empty=True,
        )
        if out_tds is not None:
            out.update(
                LazyStackedTensorDict(*out_tds), keys_to_update=self._non_tensor_keys
            )

        if out.device != device:
            if device is None:
                out = out.clear_device_()
            else:
                out = out.to(device, non_blocking=self.non_blocking)
                self._sync_w2m()
        return out

    @_check_start
    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        partial_steps = tensordict.get("_step", None)
        tensordict_save = tensordict
        if partial_steps is not None and partial_steps.all():
            partial_steps = None
        if partial_steps is not None:
            partial_steps = partial_steps.view(tensordict.shape)
            tensordict = tensordict[partial_steps]
            workers_range = partial_steps.nonzero(as_tuple=True)[0].tolist()
            tensordict_in = tensordict
        else:
            workers_range = range(self.num_workers)
            tensordict_in = tensordict.clone(False)
            # if self._use_buffers:
            #     shared_tensordict_parent = self.shared_tensordict_parent

        data_in = []
        for i, td_ in zip(workers_range, tensordict_in):
            # shared_tensordicts are locked, and we need to select the keys since we update in-place.
            # There may be unexpected keys, such as "_reset", that we should comfortably ignore here.
            env_device = self._envs[i].device
            if env_device != self.device and env_device is not None:
                data_in.append(td_.to(env_device, non_blocking=self.non_blocking))
            else:
                data_in.append(td_)

        self._sync_m2w()
        out_tds = None
        if not self._use_buffers or self._non_tensor_keys:
            out_tds = []

        if self._use_buffers:
            next_td = self.shared_tensordict_parent.get("next")
            for i, _data_in in zip(workers_range, data_in):
                out_td = self._envs[i]._step(_data_in)
                next_td[i].update_(
                    out_td,
                    keys_to_update=list(self._env_output_keys),
                    non_blocking=self.non_blocking,
                )
                if out_tds is not None:
                    out_tds.append(out_td)

            # We must pass a clone of the tensordict, as the values of this tensordict
            # will be modified in-place at further steps
            device = self.device

            def select_and_clone(name, tensor):
                if name in self._selected_step_keys:
                    return tensor.clone()

            if partial_steps is not None:
                next_td = TensorDict.lazy_stack([next_td[i] for i in workers_range])
            out = next_td.named_apply(
                select_and_clone, nested_keys=True, filter_empty=True
            )
            if out_tds is not None:
                out.update(
                    LazyStackedTensorDict(*out_tds),
                    keys_to_update=self._non_tensor_keys,
                )

            if out.device != device:
                if device is None:
                    out = out.clear_device_()
                elif out.device != device:
                    out = out.to(device, non_blocking=self.non_blocking)
                    self._sync_w2m()
        else:
            for i, _data_in in zip(workers_range, data_in):
                out_td = self._envs[i]._step(_data_in)
                out_tds.append(out_td)
            out = LazyStackedTensorDict.maybe_dense_stack(out_tds)

        if partial_steps is not None:
            result = out.new_zeros(tensordict_save.shape)
            result[partial_steps] = out
            return result

        return out

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
        device = _make_ordinal_device(torch.device(device))
        if device == self.device:
            return self
        super().to(device)
        if not self.is_closed:
            self._envs = [env.to(device) for env in self._envs]
        return self


class ParallelEnv(BatchedEnvBase, metaclass=_PEnvMeta):
    """Creates one environment per process.

    TensorDicts are passed via shared memory or memory map.

    """

    __doc__ += BatchedEnvBase.__doc__
    __doc__ += """

    .. warning::
      TorchRL's ParallelEnv is quite stringent when it comes to env specs, since
      these are used to build shared memory buffers for inter-process communication.
      As such, we encourage users to first run a check of the env specs with
      :func:`~torchrl.envs.utils.check_env_specs`:

        >>> from torchrl.envs import check_env_specs
        >>> env = make_env()
        >>> check_env_specs(env) # if this passes without error you're good to go!
        >>> penv = ParallelEnv(2, make_env)

      In particular, gym-like envs with info-dict readers may be difficult to
      share across processes if the spec is not properly set, which is hard to
      do automatically. Check :meth:`~torchrl.envs.GymLikeEnv.set_info_dict_reader`
      for more information. Here is a short example:

        >>> from torchrl.envs import GymEnv, set_gym_backend, check_env_specs, TransformedEnv, TensorDictPrimer
        >>> import torch
        >>> env = GymEnv("HalfCheetah-v4")
        >>> env.rollout(3)  # no info registered, this env passes check_env_specs
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([10, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([10, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([10]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([10, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([10]),
            device=cpu,
            is_shared=False)
        >>> check_env_specs(env)  # succeeds!
        >>> env.set_info_dict_reader()  # sets the default info_dict reader
        >>> env.rollout(10)  # because the info_dict is empty at reset time, we're missing the root infos!
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([10, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([10, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_ctrl: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward_run: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        x_position: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                        x_velocity: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([10]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([10, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([10]),
            device=cpu,
            is_shared=False)
        >>> check_env_specs(env)  # This check now fails! We should not use an env constructed like this in a parallel env
        >>> # This ad-hoc fix registers the info-spec for reset. It is wrapped inside `env.auto_register_info_dict()`
        >>> env_fixed = TransformedEnv(env, TensorDictPrimer(env.info_dict_reader[0].info_spec))
        >>> env_fixed.rollout(10)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([10, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([10, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_ctrl: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward_run: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        x_position: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                        x_velocity: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([10]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([10, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                reward_ctrl: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                reward_run: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                x_position: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False),
                x_velocity: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float64, is_shared=False)},
            batch_size=torch.Size([10]),
            device=cpu,
            is_shared=False)
        >>> check_env_specs(env_fixed)  # Succeeds! This env can be used within a parallel env!

        Related classes and methods: :meth:`~torchrl.envs.GymLikeEnv.auto_register_info_dict`
        and :class:`~torchrl.envs.gym_like.default_info_dict_reader`.

    .. warning::
      The choice of the devices where ParallelEnv needs to be executed can
      drastically influence its performance. The rule of thumbs is:

        - If the base environment (backend, e.g., Gym) is executed on CPU, the
          sub-environments should be executed on CPU and the data should be
          passed via shared physical memory.
        - If the base environment is (or can be) executed on CUDA, the sub-environments
          should be placed on CUDA too.
        - If a CUDA device is available and the policy is to be executed on CUDA,
          the ParallelEnv device should be set to CUDA.

      Therefore, supposing a CUDA device is available, we have the following scenarios:

        >>> # The sub-envs are executed on CPU, but the policy is on GPU
        >>> env = ParallelEnv(N, MyEnv(..., device="cpu"), device="cuda")
        >>> # The sub-envs are executed on CUDA
        >>> env = ParallelEnv(N, MyEnv(..., device="cuda"), device="cuda")
        >>> # this will create the exact same environment
        >>> env = ParallelEnv(N, MyEnv(..., device="cuda"))
        >>> # If no cuda device is available
        >>> env = ParallelEnv(N, MyEnv(..., device="cpu"))

    .. warning::
      ParallelEnv disable gradients in all operations (:meth:`~.step`,
      :meth:`~.reset` and :meth:`~.step_and_maybe_reset`) because gradients
      cannot be passed through :class:`multiprocessing.Pipe` objects.
      Only :class:`~torchrl.envs.SerialEnv` will support backpropagation.

    """

    def _start_workers(self) -> None:
        self._timeout = 10.0

        from torchrl.envs.env_creator import EnvCreator

        if self.num_threads is None:
            self.num_threads = max(
                1, torch.get_num_threads() - self.num_workers
            )  # 1 more thread for this proc

        torch.set_num_threads(self.num_threads)

        if self._mp_start_method is not None:
            ctx = mp.get_context(self._mp_start_method)
            proc_fun = ctx.Process
            num_sub_threads = self.num_sub_threads
        else:
            ctx = mp.get_context("spawn")
            proc_fun = functools.partial(
                _ProcessNoWarn,
                num_threads=self.num_sub_threads,
                _start_method=self._mp_start_method,
            )
            num_sub_threads = None

        _num_workers = self.num_workers

        self.parent_channels = []
        self._workers = []
        if self._use_buffers:
            func = _run_worker_pipe_shared_mem
        else:
            func = _run_worker_pipe_direct
        # We look for cuda tensors through the leaves
        # because the shared tensordict could be partially on cuda
        # and some leaves may be inaccessible through get (e.g., LazyStacked)
        has_cuda = [False]

        def look_for_cuda(tensor, has_cuda=has_cuda):
            has_cuda[0] = has_cuda[0] or tensor.is_cuda

        if self._use_buffers:
            self.shared_tensordict_parent.apply(look_for_cuda, filter_empty=True)
        has_cuda = has_cuda[0]
        if has_cuda:
            self.event = torch.cuda.Event()
        else:
            self.event = None
        self._events = [ctx.Event() for _ in range(_num_workers)]
        kwargs = [{"mp_event": self._events[i]} for i in range(_num_workers)]
        with clear_mpi_env_vars():
            for idx in range(_num_workers):
                if self._verbose:
                    torchrl_logger.info(f"initiating worker {idx}")
                # No certainty which module multiprocessing_context is
                parent_pipe, child_pipe = ctx.Pipe()
                env_fun = self.create_env_fn[idx]
                if not isinstance(env_fun, EnvCreator):
                    env_fun = CloudpickleWrapper(env_fun)
                kwargs[idx].update(
                    {
                        "parent_pipe": parent_pipe,
                        "child_pipe": child_pipe,
                        "env_fun": env_fun,
                        "env_fun_kwargs": self.create_env_kwargs[idx],
                        "has_lazy_inputs": self.has_lazy_inputs,
                        "num_threads": num_sub_threads,
                        "non_blocking": self.non_blocking,
                    }
                )
                if self._use_buffers:
                    kwargs[idx].update(
                        {
                            "shared_tensordict": self.shared_tensordicts[idx],
                            "_selected_input_keys": self._selected_input_keys,
                            "_selected_reset_keys": self._selected_reset_keys,
                            "_selected_step_keys": self._selected_step_keys,
                            "_non_tensor_keys": self._non_tensor_keys,
                        }
                    )
                process = proc_fun(target=func, kwargs=kwargs[idx])
                process.daemon = True
                process.start()
                child_pipe.close()
                self.parent_channels.append(parent_pipe)
                self._workers.append(process)

        for parent_pipe in self.parent_channels:
            # use msg as sync point
            parent_pipe.recv()

        # send shared tensordict to workers
        for channel in self.parent_channels:
            channel.send(("init", None))
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
        for event in self._events:
            event.wait(self._timeout)
            event.clear()

    def _step_and_maybe_reset_no_buffers(
        self, tensordict: TensorDictBase
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        partial_steps = tensordict.get("_step", None)
        tensordict_save = tensordict
        if partial_steps is not None and partial_steps.all():
            partial_steps = None
        if partial_steps is not None:
            partial_steps = partial_steps.view(tensordict.shape)
            tensordict = tensordict[partial_steps]
            workers_range = partial_steps.nonzero(as_tuple=True)[0].tolist()
        else:
            workers_range = range(self.num_workers)

        td = tensordict.consolidate(share_memory=True, inplace=True, num_threads=1)
        for i in workers_range:
            # We send the same td multiple times as it is in shared mem and we just need to index it
            # in each process.
            # If we don't do this, we need to unbind it but then the custom pickler will require
            # some extra metadata to be collected.
            self.parent_channels[i].send(("step_and_maybe_reset", (td, i)))

        results = [None] * len(workers_range)

        consumed_indices = []
        events = set(workers_range)
        while len(consumed_indices) < len(workers_range):
            for i in list(events):
                if self._events[i].is_set():
                    results[i] = self.parent_channels[i].recv()
                    self._events[i].clear()
                    consumed_indices.append(i)
                    events.discard(i)

        out_next, out_root = zip(*(future for future in results))
        out = TensorDict.maybe_dense_stack(out_next), TensorDict.maybe_dense_stack(
            out_root
        )
        if partial_steps is not None:
            result = out.new_zeros(tensordict_save.shape)
            result[partial_steps] = out
            return result
        return out

    @torch.no_grad()
    @_check_start
    def step_and_maybe_reset(
        self, tensordict: TensorDictBase
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        if not self._use_buffers:
            # Simply dispatch the input to the workers
            # return self._step_and_maybe_reset_no_buffers(tensordict)
            return super().step_and_maybe_reset(tensordict)

        partial_steps = tensordict.get("_step", None)
        tensordict_save = tensordict
        if partial_steps is not None and partial_steps.all():
            partial_steps = None
        if partial_steps is not None:
            partial_steps = partial_steps.view(tensordict.shape)
            workers_range = partial_steps.nonzero(as_tuple=True)[0].tolist()
            shared_tensordict_parent = TensorDict.lazy_stack(
                [self.shared_tensordict_parent[i] for i in workers_range]
            )
            next_td = TensorDict.lazy_stack(
                [self._shared_tensordict_parent_next[i] for i in workers_range]
            )
            tensordict_ = TensorDict.lazy_stack(
                [self._shared_tensordict_parent_root[i] for i in workers_range]
            )
            if self.shared_tensordict_parent.device is None:
                tensordict = tensordict._fast_apply(
                    lambda x, y: x[partial_steps].to(y.device)
                    if y is not None
                    else x[partial_steps],
                    self.shared_tensordict_parent,
                    default=None,
                    device=None,
                    batch_size=shared_tensordict_parent.shape,
                )
            else:
                tensordict = tensordict[partial_steps].to(
                    self.shared_tensordict_parent.device
                )
        else:
            workers_range = range(self.num_workers)
            shared_tensordict_parent = self.shared_tensordict_parent
            next_td = self._shared_tensordict_parent_next
            tensordict_ = self._shared_tensordict_parent_root

        # We must use the in_keys and nothing else for the following reasons:
        # - efficiency: copying all the keys will in practice mean doing a lot
        #   of writing operations since the input tensordict may (and often will)
        #   contain all the previous output data.
        # - value mismatch: if the batched env is placed within a transform
        #   and this transform overrides an observation key (eg, CatFrames)
        #   the shape, dtype or device may not necessarily match and writing
        #   the value in-place will fail.
        shared_tensordict_parent.update_(
            tensordict,
            keys_to_update=self._env_input_keys,
            non_blocking=self.non_blocking,
        )
        next_td_passthrough = tensordict.get("next", default=None)
        if next_td_passthrough is not None:
            # if we have input "next" data (eg, RNNs which pass the next state)
            # the sub-envs will need to process them through step_and_maybe_reset.
            # We keep track of which keys are present to let the worker know what
            # should be passed to the env (we don't want to pass done states for instance)
            next_td_keys = list(next_td_passthrough.keys(True, True))
            data = [{"next_td_passthrough_keys": next_td_keys} for _ in workers_range]
            shared_tensordict_parent.get("next").update_(
                next_td_passthrough, non_blocking=self.non_blocking
            )
        else:
            # next_td_keys = None
            data = [{} for _ in workers_range]

        if self._non_tensor_keys:
            for i in workers_range:
                data[i]["non_tensor_data"] = tensordict[i].select(
                    *self._non_tensor_keys, strict=False
                )

        self._sync_m2w()
        for i, _data in zip(workers_range, data):
            self.parent_channels[i].send(("step_and_maybe_reset", _data))

        for i in workers_range:
            event = self._events[i]
            event.wait(self._timeout)
            event.clear()

        if self._non_tensor_keys:
            non_tensor_tds = []
            for i in workers_range:
                msg, non_tensor_td = self.parent_channels[i].recv()
                non_tensor_tds.append(non_tensor_td)

        # We must pass a clone of the tensordict, as the values of this tensordict
        # will be modified in-place at further steps
        device = self.device
        if shared_tensordict_parent.device == device:
            next_td = next_td.clone()
            tensordict_ = tensordict_.clone()
        elif device is not None:
            next_td = next_td._fast_apply(
                lambda x: x.to(device, non_blocking=self.non_blocking)
                if x.device != device
                else x.clone(),
                device=device,
                filter_empty=True,
            )
            tensordict_ = tensordict_._fast_apply(
                lambda x: x.to(device, non_blocking=self.non_blocking)
                if x.device != device
                else x.clone(),
                device=device,
                filter_empty=True,
            )
            self._sync_w2m()
        else:
            next_td = next_td.clone().clear_device_()
            tensordict_ = tensordict_.clone().clear_device_()
        tensordict.set("next", next_td)
        if self._non_tensor_keys:
            non_tensor_tds = LazyStackedTensorDict(*non_tensor_tds)
            tensordict.update(
                non_tensor_tds,
                keys_to_update=[("next", key) for key in self._non_tensor_keys],
            )
            tensordict_.update(non_tensor_tds, keys_to_update=self._non_tensor_keys)

        if partial_steps is not None:
            result = tensordict.new_zeros(tensordict_save.shape)
            result_ = tensordict_.new_zeros(tensordict_save.shape)
            result[partial_steps] = tensordict
            result_[partial_steps] = tensordict_
            return result, result_

        return tensordict, tensordict_

    def _step_no_buffers(
        self, tensordict: TensorDictBase
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        partial_steps = tensordict.get("_step", None)
        tensordict_save = tensordict
        if partial_steps is not None and partial_steps.all():
            partial_steps = None
        if partial_steps is not None:
            partial_steps = partial_steps.view(tensordict.shape)
            tensordict = tensordict[partial_steps]
            workers_range = partial_steps.nonzero(as_tuple=True)[0].tolist()
        else:
            workers_range = range(self.num_workers)

        data = tensordict.consolidate(share_memory=True, inplace=True, num_threads=1)
        for i, local_data in zip(workers_range, data.unbind(0)):
            self.parent_channels[i].send(("step", local_data))
        # for i in range(data.shape[0]):
        #     self.parent_channels[i].send(("step", (data, i)))
        out_tds = []
        for i in workers_range:
            channel = self.parent_channels[i]
            self._events[i].wait()
            td = channel.recv()
            out_tds.append(td)
        out = LazyStackedTensorDict.maybe_dense_stack(out_tds)
        if self.device is not None and out.device != self.device:
            out = out.to(self.device, non_blocking=self.non_blocking)
        if partial_steps is not None:
            result = out.new_zeros(tensordict_save.shape)
            result[partial_steps] = out
            return result
        return out

    @torch.no_grad()
    @_check_start
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._use_buffers:
            return self._step_no_buffers(tensordict)
        # We must use the in_keys and nothing else for the following reasons:
        # - efficiency: copying all the keys will in practice mean doing a lot
        #   of writing operations since the input tensordict may (and often will)
        #   contain all the previous output data.
        # - value mismatch: if the batched env is placed within a transform
        #   and this transform overrides an observation key (eg, CatFrames)
        #   the shape, dtype or device may not necessarily match and writing
        #   the value in-place will fail.
        partial_steps = tensordict.get("_step", None)
        tensordict_save = tensordict
        if partial_steps is not None and partial_steps.all():
            partial_steps = None
        if partial_steps is not None:
            partial_steps = partial_steps.view(tensordict.shape)
            workers_range = partial_steps.nonzero(as_tuple=True)[0].tolist()
            shared_tensordict_parent = TensorDict.lazy_stack(
                [self.shared_tensordicts[i] for i in workers_range]
            )
            if self.shared_tensordict_parent.device is None:
                tensordict = tensordict._fast_apply(
                    lambda x, y: x[partial_steps].to(y.device)
                    if y is not None
                    else x[partial_steps],
                    self.shared_tensordict_parent,
                    default=None,
                    device=None,
                    batch_size=shared_tensordict_parent.shape,
                )
            else:
                tensordict = tensordict[partial_steps].to(
                    self.shared_tensordict_parent.device
                )
        else:
            workers_range = range(self.num_workers)
            shared_tensordict_parent = self.shared_tensordict_parent

        shared_tensordict_parent.update_(
            tensordict,
            keys_to_update=list(self._env_input_keys),
            non_blocking=self.non_blocking,
        )
        next_td_passthrough = tensordict.get("next", None)
        if next_td_passthrough is not None:
            # if we have input "next" data (eg, RNNs which pass the next state)
            # the sub-envs will need to process them through step_and_maybe_reset.
            # We keep track of which keys are present to let the worker know what
            # should be passed to the env (we don't want to pass done states for instance)
            next_td_keys = list(next_td_passthrough.keys(True, True))
            data = [
                {"next_td_passthrough_keys": next_td_keys}
                for _ in range(self.num_workers)
            ]
            shared_tensordict_parent.get("next").update_(
                next_td_passthrough, non_blocking=self.non_blocking
            )
        else:
            data = [{} for _ in range(self.num_workers)]

        if self._non_tensor_keys:
            for i in workers_range:
                data[i]["non_tensor_data"] = tensordict[i].select(
                    *self._non_tensor_keys, strict=False
                )

        self._sync_m2w()

        if self.event is not None:
            self.event.record()
            self.event.synchronize()
        for i in workers_range:
            self.parent_channels[i].send(("step", data[i]))

        for i in workers_range:
            event = self._events[i]
            event.wait(self._timeout)
            event.clear()

        if self._non_tensor_keys:
            non_tensor_tds = []
            for i in workers_range:
                msg, non_tensor_td = self.parent_channels[i].recv()
                non_tensor_tds.append(non_tensor_td)

        # We must pass a clone of the tensordict, as the values of this tensordict
        # will be modified in-place at further steps
        next_td = shared_tensordict_parent.get("next")
        device = self.device

        if next_td.device != device and device is not None:

            def select_and_clone(name, tensor):
                if name in self._selected_step_keys:
                    return tensor.to(device, non_blocking=self.non_blocking)

        else:

            def select_and_clone(name, tensor):
                if name in self._selected_step_keys:
                    return tensor.clone()

        out = next_td.named_apply(
            select_and_clone,
            nested_keys=True,
            filter_empty=True,
            device=device,
        )
        if self._non_tensor_keys:
            out.update(
                LazyStackedTensorDict(*non_tensor_tds),
                keys_to_update=self._non_tensor_keys,
            )
        self._sync_w2m()
        if partial_steps is not None:
            result = out.new_zeros(tensordict_save.shape)
            result[partial_steps] = out
            return result
        return out

    def _reset_no_buffers(
        self,
        tensordict: TensorDictBase,
        reset_kwargs_list,
        needs_resetting,
    ) -> Tuple[TensorDictBase, TensorDictBase]:
        if is_tensor_collection(tensordict):
            # tensordict = tensordict.consolidate(share_memory=True, num_threads=1)
            tensordict = tensordict.consolidate(
                share_memory=True, num_threads=1
            ).unbind(0)
        else:
            tensordict = [None] * self.num_workers
        out_tds = [None] * self.num_workers
        for i, (local_data, reset_kwargs) in enumerate(
            zip(tensordict, reset_kwargs_list)
        ):
            if not needs_resetting[i]:
                localtd = local_data
                if localtd is not None:
                    localtd = localtd.exclude(*self.reset_keys)
                out_tds[i] = localtd
                continue
            self.parent_channels[i].send(("reset", (local_data, reset_kwargs)))

        for i, channel in enumerate(self.parent_channels):
            if not needs_resetting[i]:
                continue
            self._events[i].wait()
            td = channel.recv()
            out_tds[i] = td
        result = LazyStackedTensorDict.maybe_dense_stack(out_tds)
        device = self.device
        if device is not None and result.device != device:
            return result.to(self.device, non_blocking=self.non_blocking)
        return result

    @torch.no_grad()
    @_check_start
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:

        list_of_kwargs = kwargs.pop("list_of_kwargs", [kwargs] * self.num_workers)
        if kwargs is not list_of_kwargs[0] and kwargs:
            # this means that kwargs had more than one element and that a list was provided
            for elt in list_of_kwargs:
                elt.update(kwargs)

        if tensordict is not None:
            needs_resetting = _aggregate_end_of_traj(
                tensordict, reset_keys=self.reset_keys
            )
            if needs_resetting.ndim > 2:
                needs_resetting = needs_resetting.flatten(1, needs_resetting.ndim - 1)
            if needs_resetting.ndim > 1:
                needs_resetting = needs_resetting.any(-1)
            elif not needs_resetting.ndim:
                needs_resetting = needs_resetting.expand((self.num_workers,))
        else:
            needs_resetting = torch.ones(
                (self.num_workers,), device=self.device, dtype=torch.bool
            )

        if not self._use_buffers:
            return self._reset_no_buffers(tensordict, list_of_kwargs, needs_resetting)

        outs = []
        for i in range(self.num_workers):
            if tensordict is not None:
                tensordict_ = tensordict[i]
                if tensordict_.is_empty():
                    tensordict_ = None
                elif self.device is not None and self.device.type == "mps":
                    # copy_ fails when moving mps->cpu using copy_
                    # in some cases when a view of an mps tensor is used.
                    # We know the shared tensors are not MPS, so we can
                    # safely assume that the shared tensors are on cpu
                    tensordict_ = tensordict_.to("cpu")
            else:
                tensordict_ = None
            if not needs_resetting[i]:
                # We update the stored tensordict with the value of the "next"
                # key as one may be surprised to receive data that is not up-to-date
                # If we don't do this, the result of calling reset and skipping one env
                # will be that the env will have the data from the previous
                # step at the root (since the shared_tensordict did not go through
                # step_mdp).
                self.shared_tensordicts[i].update_(
                    self.shared_tensordicts[i].get("next"),
                    keys_to_update=list(self._selected_reset_keys),
                    non_blocking=self.non_blocking,
                )
                if tensordict_ is not None:
                    self.shared_tensordicts[i].update_(
                        tensordict_,
                        keys_to_update=list(self._selected_reset_keys),
                        non_blocking=self.non_blocking,
                    )
                continue
            if tensordict_ is not None:
                tdkeys = list(tensordict_.keys(True, True))

                # This way we can avoid calling select over all the keys in the shared tensordict
                def tentative_update(val, other):
                    if other is not None:
                        val.copy_(other, non_blocking=self.non_blocking)
                    return val

                self.shared_tensordicts[i].apply_(
                    tentative_update, tensordict_, default=None
                )
                out = ("reset", (tdkeys, list_of_kwargs[i]))
            else:
                out = ("reset", (False, list_of_kwargs[i]))
            outs.append((i, out))

        self._sync_m2w()

        for i, out in outs:
            self.parent_channels[i].send(out)

        for i, _ in outs:
            event = self._events[i]
            event.wait(self._timeout)
            event.clear()

        workers_nontensor = []
        if self._non_tensor_keys:
            for i, _ in outs:
                msg, non_tensor_td = self.parent_channels[i].recv()
                workers_nontensor.append((i, non_tensor_td))

        selected_output_keys = self._selected_reset_keys_filt
        device = self.device

        if self.shared_tensordict_parent.device != device and device is not None:

            def select_and_clone(name, tensor):
                if name in selected_output_keys:
                    return tensor.to(device, non_blocking=self.non_blocking)

        else:

            def select_and_clone(name, tensor):
                if name in selected_output_keys:
                    return tensor.clone()

        out = self.shared_tensordict_parent.named_apply(
            select_and_clone,
            nested_keys=True,
            filter_empty=True,
            device=device,
        )
        if self._non_tensor_keys:
            workers, nontensor = zip(*workers_nontensor)
            out[torch.tensor(workers)] = LazyStackedTensorDict(*nontensor).select(
                *self._non_tensor_keys
            )
        self._sync_w2m()
        return out

    @_check_start
    def _shutdown_workers(self) -> None:
        try:
            if self.is_closed:
                raise RuntimeError(
                    "calling {self.__class__.__name__}._shutdown_workers only allowed when env.is_closed = False"
                )
            for i, channel in enumerate(self.parent_channels):
                if self._verbose:
                    torchrl_logger.info(f"closing {i}")
                channel.send(("close", None))
            for i in range(self.num_workers):
                self._events[i].wait(self._timeout)
                self._events[i].clear()
            if self._use_buffers:
                del self.shared_tensordicts, self.shared_tensordict_parent

            for channel in self.parent_channels:
                channel.close()
            for proc in self._workers:
                proc.join(timeout=1.0)
        finally:
            for proc in self._workers:
                if proc.is_alive():
                    proc.terminate()
        del self._workers
        del self.parent_channels
        self._cuda_events = None
        self._events = None
        self.event = None

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
                    self.start()
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
        device = _make_ordinal_device(torch.device(device))
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
    parent_pipe: connection.Connection,
    child_pipe: connection.Connection,
    env_fun: Union[EnvBase, Callable],
    env_fun_kwargs: Dict[str, Any],
    mp_event: mp.Event = None,
    shared_tensordict: TensorDictBase = None,
    _selected_input_keys=None,
    _selected_reset_keys=None,
    _selected_step_keys=None,
    _non_tensor_keys=None,
    non_blocking: bool = False,
    has_lazy_inputs: bool = False,
    verbose: bool = False,
    num_threads: int | None = None,  # for fork start method
) -> None:
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    device = shared_tensordict.device
    if device is None or device.type != "cuda":
        # Check if some tensors are shared on cuda
        has_cuda = [False]

        def look_for_cuda(tensor, has_cuda=has_cuda):
            has_cuda[0] = has_cuda[0] or tensor.is_cuda

        shared_tensordict.apply(look_for_cuda, filter_empty=True)
        has_cuda = has_cuda[0]
    else:
        has_cuda = device.type == "cuda"
    if has_cuda:
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
    del env_fun

    i = -1
    import torchrl

    _timeout = torchrl._utils.BATCHED_PIPE_TIMEOUT

    initialized = False

    child_pipe.send("started")
    next_shared_tensordict, root_shared_tensordict = (None,) * 2
    while True:
        try:
            if child_pipe.poll(_timeout):
                cmd, data = child_pipe.recv()
            else:
                raise TimeoutError(
                    f"Worker timed out after {_timeout}s, "
                    f"increase timeout if needed throught the BATCHED_PIPE_TIMEOUT environment variable."
                )
        except EOFError as err:
            raise EOFError(f"proc {pid} failed, last command: {cmd}.") from err
        if cmd == "seed":
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            torch.manual_seed(data[0])
            new_seed = env.set_seed(data[0], static_seed=data[1])
            child_pipe.send(("seeded", new_seed))

        elif cmd == "init":
            if verbose:
                torchrl_logger.info(f"initializing {pid}")
            if initialized:
                raise RuntimeError("worker already initialized")
            i = 0
            next_shared_tensordict = shared_tensordict.get("next")
            root_shared_tensordict = shared_tensordict.exclude("next")
            # TODO: restore this
            # if not (shared_tensordict.is_shared() or shared_tensordict.is_memmap()):
            #     raise RuntimeError(
            #         "tensordict must be placed in shared memory (share_memory_() or memmap_())"
            #     )
            shared_tensordict = shared_tensordict.clone(False).unlock_()

            initialized = True

        elif cmd == "reset":
            if verbose:
                torchrl_logger.info(f"resetting worker {pid}")
            if not initialized:
                raise RuntimeError("call 'init' before resetting")
            # we use 'data' to pass the keys that we need to pass to reset,
            # because passing the entire buffer may have unwanted consequences
            selected_reset_keys, reset_kwargs = data
            cur_td = env.reset(
                tensordict=root_shared_tensordict.select(
                    *selected_reset_keys, strict=False
                )
                if selected_reset_keys
                else None,
                **reset_kwargs,
            )
            shared_tensordict.update_(
                cur_td,
                keys_to_update=list(_selected_reset_keys),
                non_blocking=non_blocking,
            )
            if event is not None:
                event.record()
                event.synchronize()
            mp_event.set()

            if _non_tensor_keys:
                child_pipe.send(
                    ("non_tensor", cur_td.select(*_non_tensor_keys, strict=False))
                )

            del cur_td

        elif cmd == "step":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            # No need to copy here since we don't write in-place
            input = root_shared_tensordict
            if data:
                next_td_passthrough_keys = data.get("next_td_passthrough_keys")
                if next_td_passthrough_keys is not None:
                    input = input.set(
                        "next", next_shared_tensordict.select(*next_td_passthrough_keys)
                    )
                non_tensor_data = data.get("non_tensor_data")
                if non_tensor_data is not None:
                    input.update(non_tensor_data)

            next_td = env._step(input)
            next_shared_tensordict.update_(next_td, non_blocking=non_blocking)
            if event is not None:
                event.record()
                event.synchronize()
            mp_event.set()

            if _non_tensor_keys:
                child_pipe.send(
                    ("non_tensor", next_td.select(*_non_tensor_keys, strict=False))
                )

            del next_td

        elif cmd == "step_and_maybe_reset":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            # We must copy the root shared td here, or at least get rid of done:
            # if we don't `td is root_shared_tensordict`
            # which means that root_shared_tensordict will carry the content of next
            # in the next iteration. When using StepCounter, it will look for an
            # existing done state, find it and consider the env as done by input (not
            # by output) of the step!
            # Caveat: for RNN we may need some keys of the "next" TD so we pass the list
            # through data
            input = root_shared_tensordict
            if data:
                next_td_passthrough_keys = data.get("next_td_passthrough_keys", None)
                if next_td_passthrough_keys is not None:
                    input = input.set(
                        "next", next_shared_tensordict.select(*next_td_passthrough_keys)
                    )
                non_tensor_data = data.get("non_tensor_data", None)
                if non_tensor_data is not None:
                    input.update(non_tensor_data)
            td, root_next_td = env.step_and_maybe_reset(input)
            td_next = td.pop("next")
            next_shared_tensordict.update_(td_next, non_blocking=non_blocking)
            root_shared_tensordict.update_(root_next_td, non_blocking=non_blocking)

            if event is not None:
                event.record()
                event.synchronize()
            mp_event.set()

            if _non_tensor_keys:
                ntd = root_next_td.select(*_non_tensor_keys)
                ntd.set("next", td_next.select(*_non_tensor_keys))
                child_pipe.send(("non_tensor", ntd))

            del td, root_next_td

        elif cmd == "close":
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            env.close()
            del (
                env,
                shared_tensordict,
                data,
                next_shared_tensordict,
                root_shared_tensordict,
            )
            mp_event.set()
            child_pipe.close()
            if verbose:
                torchrl_logger.info(f"{pid} closed")
            gc.collect()
            break

        elif cmd == "load_state_dict":
            env.load_state_dict(data)
            mp_event.set()

        elif cmd == "state_dict":
            state_dict = _recursively_strip_locks_from_state_dict(env.state_dict())
            msg = "state_dict"
            child_pipe.send((msg, state_dict))
            del state_dict

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


def _run_worker_pipe_direct(
    parent_pipe: connection.Connection,
    child_pipe: connection.Connection,
    env_fun: Union[EnvBase, Callable],
    env_fun_kwargs: Dict[str, Any],
    mp_event: mp.Event = None,
    non_blocking: bool = False,
    has_lazy_inputs: bool = False,
    verbose: bool = False,
    num_threads: int | None = None,  # for fork start method
) -> None:
    if num_threads is not None:
        torch.set_num_threads(num_threads)

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
    del env_fun
    for spec in env.output_spec.values(True, True):
        if spec.device is not None and spec.device.type == "cuda":
            has_cuda = True
            break
    else:
        for spec in env.input_spec.values(True, True):
            if spec.device is not None and spec.device.type == "cuda":
                has_cuda = True
                break
        else:
            has_cuda = False
    if has_cuda:
        event = torch.cuda.Event()
    else:
        event = None

    i = -1
    import torchrl

    _timeout = torchrl._utils.BATCHED_PIPE_TIMEOUT

    initialized = False

    child_pipe.send("started")
    while True:
        try:
            if child_pipe.poll(_timeout):
                cmd, data = child_pipe.recv()
            else:
                raise TimeoutError(
                    f"Worker timed out after {_timeout}s, "
                    f"increase timeout if needed throught the BATCHED_PIPE_TIMEOUT environment variable."
                )
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
                torchrl_logger.info(f"initializing {pid}")
            if initialized:
                raise RuntimeError("worker already initialized")
            i = 0

            initialized = True

        elif cmd == "reset":
            if verbose:
                torchrl_logger.info(f"resetting worker {pid}")
            if not initialized:
                raise RuntimeError("call 'init' before resetting")
            # we use 'data' to pass the keys that we need to pass to reset,
            # because passing the entire buffer may have unwanted consequences
            # data, idx, reset_kwargs = data
            # data = data[idx]
            data, reset_kwargs = data
            if data is not None:
                data._fast_apply(
                    lambda x: x.clone() if x.device.type == "cuda" else x, out=data
                )
            cur_td = env.reset(
                tensordict=data,
                **reset_kwargs,
            )
            if event is not None:
                event.record()
                event.synchronize()
            mp_event.set()
            child_pipe.send(
                cur_td.consolidate(share_memory=True, inplace=True, num_threads=1)
            )
            del cur_td

        elif cmd == "step":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            # data, idx = data
            # data = data[idx]
            next_td = env._step(data)
            if event is not None:
                event.record()
                event.synchronize()
            mp_event.set()
            child_pipe.send(
                next_td.consolidate(share_memory=True, inplace=True, num_threads=1)
            )
            del next_td

        elif cmd == "step_and_maybe_reset":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            # data, idx = data
            # data = data[idx]
            data._fast_apply(
                lambda x: x.clone() if x.device.type == "cuda" else x, out=data
            )
            td, root_next_td = env.step_and_maybe_reset(data)

            if event is not None:
                event.record()
                event.synchronize()
            child_pipe.send((td, root_next_td))
            mp_event.set()
            del td, root_next_td

        elif cmd == "close":
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            env.close()
            del (
                env,
                data,
            )
            mp_event.set()
            child_pipe.close()
            if verbose:
                torchrl_logger.info(f"{pid} closed")
            gc.collect()
            break

        elif cmd == "load_state_dict":
            env.load_state_dict(data)
            mp_event.set()

        elif cmd == "state_dict":
            state_dict = _recursively_strip_locks_from_state_dict(env.state_dict())
            msg = "state_dict"
            child_pipe.send((msg, state_dict))
            del state_dict

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


def _filter_empty(tensordict):
    return tensordict.select(*tensordict.keys(True, True))


def _stackable(*tensordicts):
    try:
        ls = LazyStackedTensorDict(*tensordicts, stack_dim=0)
        ls.contiguous()
        return not ls._has_exclusive_keys
    except RuntimeError:
        return False


def _cuda_sync(device):
    return functools.partial(torch.cuda.synchronize, device=device)


def _mps_sync(device):
    return torch.mps.synchronize


# Create an alias for possible imports
_BatchedEnv = BatchedEnvBase

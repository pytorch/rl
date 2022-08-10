# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections import OrderedDict
from copy import deepcopy
from logging import warn
from multiprocessing import connection
from time import sleep
from typing import Callable, Optional, Sequence, Union, Any, List

import torch
from torch import multiprocessing as mp

from torchrl import _check_for_faulty_process
from torchrl.data import TensorDict, TensorSpec
from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs.common import _EnvClass, make_tensordict
from torchrl.envs.env_creator import EnvCreator

__all__ = ["SerialEnv", "ParallelEnv"]


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
        for i, channel in enumerate(self.parallel_env.parent_channels):
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


class _dummy_env_context:
    def __init__(self, fun, kwargs, device):
        self.fun = fun
        self.kwargs = kwargs
        self.device = device

    def __enter__(self):
        self.dummy_env = self.fun(**self.kwargs)
        if self.device is not None:
            self.dummy_env.to(self.device)
        return self.dummy_env

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dummy_env.close()
        del self.dummy_env


class _BatchedEnv(_EnvClass):
    """

    Batched environments allow the user to query an arbitrary method / attribute of the environment running remotely.
    Those queries will return a list of length equal to the number of workers containing the
    values resulting from those queries.
        >>> env = ParallelEnv(3, my_env_fun)
        >>> custom_attribute_list = env.custom_attribute
        >>> custom_method_list = env.custom_method(*args)

    Args:
        num_workers: number of workers (i.e. env instances) to be deployed simultaneously;
        create_env_fn (callable or list of callables): function (or list of functions) to be used for the environment
            creation;
        create_env_kwargs (dict or list of dicts, optional): kwargs to be used with the environments being created;
        env_input_keys (list of str, optional): list of keys that are to be considered policy-output. If the policy has it,
            the attribute policy.out_keys can be used.
            Providing the env_input_keys permit to select which keys to update after the policy is called, which can
            drastically decrease the IO burden when the tensordict is placed in shared memory / memory map.
            env_input_keys will typically contain "action" and if this list is not provided this object
            will look for corresponding keys. When working with stateless models, it is important to include the
            state to be read by the environment. If none is provided, _BatchedEnv will use the `_EnvClass.input_spec`
            keys as indicators of the keys to be sent to the env.
        pin_memory (bool): if True and device is "cpu", calls `pin_memory` on the tensordicts when created.
        selected_keys (list of str, optional): keys that have to be returned by the environment.
            When creating a batch of environment, it might be the case that only some of the keys are to be returned.
            For instance, if the environment returns 'next_pixels' and 'next_vector', the user might only
            be interested in, say, 'next_vector'. By indicating which keys must be returned in the tensordict,
            one can easily control the amount of data occupied in memory (for instance to limit the memory size of a
            replay buffer) and/or limit the amount of data passed from one process to the other;
        excluded_keys (list of str, optional): list of keys to be excluded from the returned tensordicts.
            See selected_keys for more details;
        share_individual_td (bool): if True, a different tensordict is created for every process/worker and a lazy
            stack is returned.
            default = False;
        shared_memory (bool): whether or not the returned tensordict will be placed in shared memory;
        memmap (bool): whether or not the returned tensordict will be placed in memory map.
        policy_proof (callable, optional): if provided, it'll be used to get the list of
            tensors to return through the `step()` and `reset()` methods, such as `"hidden"` etc.
        device (str, int, torch.device): for consistency, this argument is kept. However this
            argument should not be passed, as the device will be inferred from the environments.
            It is assumed that all environments will run on the same device as a common shared
            tensordict will be used to pass data from process to process. The device can be
            changed after instantiation using `env.to(device)`.

    """

    _verbose: bool = False
    _excluded_wrapped_keys = [
        "is_closed",
        "parent_channels",
        "batch_size",
        "_dummy_env_str",
    ]

    def __init__(
        self,
        num_workers: int,
        create_env_fn: Union[
            Callable[[], _EnvClass], Sequence[Callable[[], _EnvClass]]
        ],
        create_env_kwargs: Union[dict, Sequence[dict]] = None,
        env_input_keys: Optional[Sequence[str]] = None,
        pin_memory: bool = False,
        selected_keys: Optional[Sequence[str]] = None,
        excluded_keys: Optional[Sequence[str]] = None,
        share_individual_td: bool = False,
        shared_memory: bool = True,
        memmap: bool = False,
        policy_proof: Optional[Callable] = None,
        device: Optional[DEVICE_TYPING] = None,
    ):
        if device is not None:
            raise ValueError(
                "Device setting for batched environment can't be done at initialization. "
                "The device will be inferred from the constructed environment. "
                "It can be set through the `to(device)` method."
            )

        super().__init__(device=None)
        self.is_closed = True

        if callable(create_env_fn):
            create_env_fn = [create_env_fn for _ in range(num_workers)]
        else:
            if len(create_env_fn) != num_workers:
                raise RuntimeError(
                    f"num_workers and len(create_env_fn) mismatch, "
                    f"got {len(create_env_fn)} and {num_workers}"
                )
        create_env_kwargs = dict() if create_env_kwargs is None else create_env_kwargs
        if isinstance(create_env_kwargs, dict):
            create_env_kwargs = [
                deepcopy(create_env_kwargs) for _ in range(num_workers)
            ]

        self._dummy_env_instance = None
        try:
            self._dummy_env_fun = CloudpickleWrapper(
                create_env_fn[0], **create_env_kwargs[0]
            )
        except RuntimeError as err:
            if isinstance(create_env_fn[0], EnvCreator):
                self._dummy_env_fun = create_env_fn[0]
                self._dummy_env_fun.create_env_kwargs.update(create_env_kwargs[0])
            else:
                raise err

        self.policy_proof = policy_proof
        self.num_workers = num_workers
        self.create_env_fn = create_env_fn
        self.create_env_kwargs = create_env_kwargs
        self.env_input_keys = env_input_keys
        self.pin_memory = pin_memory
        self.selected_keys = selected_keys
        self.excluded_keys = excluded_keys
        self.share_individual_td = share_individual_td
        self._share_memory = shared_memory
        self._memmap = memmap
        if self._share_memory and self._memmap:
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        self._batch_size = None
        self._action_spec = None
        self._observation_spec = None
        self._reward_spec = None
        self._device = None
        self._dummy_env_str = None
        self._seeds = None

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

    def _set_properties(self):
        with self._dummy_env_context as dummy_env:
            self._batch_size = torch.Size([self.num_workers, *dummy_env.batch_size])
            self._action_spec = dummy_env.action_spec
            self._observation_spec = dummy_env.observation_spec
            self._reward_spec = dummy_env.reward_spec
            self._dummy_env_str = str(dummy_env)
            self._device = torch.device(dummy_env.device)

    @property
    def _dummy_env_context(self) -> _dummy_env_context:
        """Returns a context manager that will create a dummy env and delete it afterwards"""
        return _dummy_env_context(
            self._dummy_env_fun, self.create_env_kwargs[0], self._device
        )

    @property
    def _dummy_env(self) -> _EnvClass:
        """Returns a closed dummy environment. This is used to check the type of attributes that will
        be gathered on remote processed.
        """
        if self._dummy_env_instance is None:
            self._dummy_env_instance = self._dummy_env_fun(**self.create_env_kwargs[0])
            self._dummy_env_instance.close()
        return self._dummy_env_instance

    @_dummy_env.setter
    def _dummy_env(self, value: _EnvClass) -> None:
        self._dummy_env_instance = value

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    @property
    def batch_size(self) -> TensorSpec:
        if "_batch_size" not in self.__dir__():
            raise AttributeError("_batch_size is not initialized")
        if self._batch_size is None:
            self._set_properties()
        return self._batch_size

    @property
    def action_spec(self) -> TensorSpec:
        if self._action_spec is None:
            self._set_properties()
        return self._action_spec

    @action_spec.setter
    def action_spec(self, value: TensorSpec) -> None:
        self._action_spec = value

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._set_properties()
        return self._device

    @device.setter
    def device(self, value: DEVICE_TYPING) -> None:
        self.to(value)

    @property
    def observation_spec(self) -> TensorSpec:
        if self._observation_spec is None:
            self._set_properties()
        return self._observation_spec

    @observation_spec.setter
    def observation_spec(self, value: TensorSpec) -> None:
        self._observation_spec = value

    @property
    def reward_spec(self) -> TensorSpec:
        if self._reward_spec is None:
            self._set_properties()
        return self._reward_spec

    @reward_spec.setter
    def reward_spec(self, value: TensorSpec) -> None:
        self._reward_spec = value

    def is_done_set_fn(self, value: bool) -> None:
        self._is_done = value.all()

    def _create_td(self) -> None:
        """Creates self.shared_tensordict_parent, a TensorDict used to store the most recent observations."""
        with self._dummy_env_context as dummy_env:
            shared_tensordict_parent = make_tensordict(
                dummy_env,
                self.policy_proof,
            )

        shared_tensordict_parent = shared_tensordict_parent.expand(
            self.num_workers
        ).clone()

        raise_no_selected_keys = False
        if self.selected_keys is None:
            self.selected_keys = list(shared_tensordict_parent.keys())
            if self.excluded_keys is not None:
                self.selected_keys = set(self.selected_keys).difference(
                    self.excluded_keys
                )
            else:
                raise_no_selected_keys = True
        if self.env_input_keys is not None:
            if not all(
                action_key in self.selected_keys for action_key in self.env_input_keys
            ):
                raise KeyError(
                    "One of the action keys is not part of the selected keys or is part of the excluded keys. Action "
                    "keys need to be part of the selected keys for env.step() to be called."
                )
        else:
            self.env_input_keys = sorted(list(self._dummy_env.input_spec.keys()))
            if not len(self.env_input_keys):
                raise RuntimeError(
                    f"found 0 action keys in {sorted(list(self.selected_keys))}"
                )
        shared_tensordict_parent = shared_tensordict_parent.select(*self.selected_keys)
        self.shared_tensordict_parent = shared_tensordict_parent.to(self.device)

        if self.share_individual_td:
            self.shared_tensordicts = [
                td.clone() for td in self.shared_tensordict_parent.unbind(0)
            ]
            if self._share_memory:
                for td in self.shared_tensordicts:
                    td.share_memory_()
            elif self._memmap:
                for td in self.shared_tensordicts:
                    td.memmap_()
            self.shared_tensordict_parent = torch.stack(self.shared_tensordicts, 0)
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

        if raise_no_selected_keys:
            if self._verbose:
                print(
                    f"\n {self.__class__.__name__}.shared_tensordict_parent is \n{self.shared_tensordict_parent}. \n"
                    f"You can select keys to be synchronised by setting the selected_keys and/or excluded_keys "
                    f"arguments when creating the batched environment."
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
            print(f"closing {self.__class__.__name__}")

        self.action_spec = None
        self.observation_spec = None
        self.reward_spec = None

        self._shutdown_workers()
        self.is_closed = True

    def _shutdown_workers(self) -> None:
        raise NotImplementedError

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
            self.action_spec = self.action_spec.to(device)
            self.reward_spec = self.reward_spec.to(device)
            self.observation_spec = self.observation_spec.to(device)
        return self


class SerialEnv(_BatchedEnv):
    """
    Creates a series of environments in the same process.

    """

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

        tensordict_in = tensordict.select(*self.env_input_keys)
        tensordict_out = []
        for i in range(self.num_workers):
            _tensordict_out = self._envs[i].step(tensordict_in[i])
            tensordict_out.append(_tensordict_out)
        # We must pass a clone of the tensordict, as the values of this tensordict
        # will be modified in-place at further steps
        return torch.stack(tensordict_out, 0).clone()

    def _shutdown_workers(self) -> None:
        if not self.is_closed:
            for env in self._envs:
                env.close()
            del self._envs

    @_check_start
    def set_seed(self, seed: int) -> int:
        for i, env in enumerate(self._envs):
            new_seed = env.set_seed(seed)
            seed = new_seed
        return seed

    @_check_start
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None and "reset_workers" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            reset_workers = tensordict.get("reset_workers")
        else:
            reset_workers = torch.ones(self.num_workers, 1, dtype=torch.bool)

        keys = set()
        for i, _env in enumerate(self._envs):
            if not reset_workers[i]:
                continue
            _td = _env.reset(execute_step=False, **kwargs)
            keys = keys.union(_td.keys())
            self.shared_tensordicts[i].update_(_td)

        return self.shared_tensordict_parent.select(*keys).clone()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
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
                callable_attr = callable(getattr(self._dummy_env, attr))
                list_attr = [getattr(env, attr) for env in self._envs]
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
                    f"attribute {attr} not found in "
                    f"{self._dummy_env.__class__.__name__}"
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
    """
    Creates one environment per process.
    TensorDicts are passed via shared memory or memory map.

    """

    __doc__ += _BatchedEnv.__doc__

    def _start_workers(self) -> None:

        _num_workers = self.num_workers
        ctx = mp.get_context("spawn")

        self.parent_channels = []
        self._workers = []

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
                ),
            )
            w.daemon = True
            w.start()
            channel2.close()
            self.parent_channels.append(channel1)
            self._workers.append(w)

        # send shared tensordict to workers
        for channel, shared_tensordict in zip(
            self.parent_channels, self.shared_tensordicts
        ):
            channel.send(("init", shared_tensordict))
        self.is_closed = False

    @_check_start
    def state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        for idx, channel in enumerate(self.parent_channels):
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

        self.shared_tensordict_parent.update_(tensordict.select(*self.env_input_keys))
        for i in range(self.num_workers):
            self.parent_channels[i].send(("step", None))

        keys = set()
        for i in range(self.num_workers):
            msg, data = self.parent_channels[i].recv()
            if msg != "step_result":
                if msg != "done":
                    raise RuntimeError(
                        f"Expected 'done' but received {msg} from worker {i}"
                    )
            # data is the set of updated keys
            keys = keys.union(data)
        # We must pass a clone of the tensordict, as the values of this tensordict
        # will be modified in-place at further steps
        return self.shared_tensordict_parent.select(*keys).clone()

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
    def set_seed(self, seed: int) -> int:
        self._seeds = []
        for channel in self.parent_channels:
            channel.send(("seed", seed))
            self._seeds.append(seed)
            msg, new_seed = channel.recv()
            if msg != "seeded":
                raise RuntimeError(f"Expected 'seeded' but received {msg}")
            seed = new_seed
        return seed

    @_check_start
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        cmd_out = "reset"
        if tensordict is not None and "reset_workers" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            reset_workers = tensordict.get("reset_workers")
        else:
            reset_workers = torch.ones(self.num_workers, 1, dtype=torch.bool)

        for i, channel in enumerate(self.parent_channels):
            if not reset_workers[i]:
                continue
            channel.send((cmd_out, kwargs))

        keys = set()
        for i, channel in enumerate(self.parent_channels):
            if not reset_workers[i]:
                continue
            cmd_in, new_keys = channel.recv()
            keys = keys.union(new_keys)
            if cmd_in != "reset_obs":
                raise RuntimeError(f"received cmd {cmd_in} instead of reset_obs")
        check_count = 0
        while self.shared_tensordict_parent.get("done").any():
            if check_count == 4:
                raise RuntimeError("Envs have just been reset but some are still done")
            else:
                check_count += 1
                # there might be some delay between writing the shared tensordict
                # and reading the updated value on the main process
                sleep(0.01)
        return self.shared_tensordict_parent.select(*keys).clone()

    def __reduce__(self):
        if not self.is_closed:
            # ParallelEnv contains non-instantiated envs, thus it can be
            # closed and serialized if the environment building functions
            # permit it
            self.close()
        return super().__reduce__()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
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
                _ = getattr(self._dummy_env, attr)
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
                    f"attribute {attr} not found in "
                    f"{self._dummy_env.__class__.__name__}"
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


def _run_worker_pipe_shared_mem(
    idx: int,
    parent_pipe: connection.Connection,
    child_pipe: connection.Connection,
    env_fun: Union[_EnvClass, Callable],
    env_fun_kwargs: dict,
    pin_memory: bool,
    env_input_keys: dict,
    device: DEVICE_TYPING = "cpu",
    verbose: bool = False,
) -> None:
    parent_pipe.close()
    pid = os.getpid()
    if not isinstance(env_fun, _EnvClass):
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
    tensordict = None
    _td = None
    data = None

    reset_keys = None
    step_keys = None

    while True:
        try:
            cmd, data = child_pipe.recv()
        except EOFError as err:
            raise EOFError(
                f"proc {pid} failed, last command: {cmd}. " f"\nErr={str(err)}"
            )
        if cmd == "seed":
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            # torch.manual_seed(data)
            # np.random.seed(data)
            new_seed = env.set_seed(data)
            child_pipe.send(("seeded", new_seed))

        elif cmd == "init":
            if verbose:
                print(f"initializing {pid}")
            if initialized:
                raise RuntimeError("worker already initialized")
            i = 0
            tensordict = data
            if not (tensordict.is_shared() or tensordict.is_memmap()):
                raise RuntimeError(
                    "tensordict must be placed in shared memory (share_memory_() or memmap_())"
                )
            initialized = True

        elif cmd == "reset":
            reset_kwargs = data
            if verbose:
                print(f"resetting worker {pid}")
            if not initialized:
                raise RuntimeError("call 'init' before resetting")
            # _td = tensordict.select("observation").to(env.device).clone()
            _td = env.reset(execute_step=False, **reset_kwargs)
            if reset_keys is None:
                reset_keys = set(_td.keys())
            if pin_memory:
                _td.pin_memory()
            tensordict.update_(_td)
            child_pipe.send(("reset_obs", reset_keys))
            just_reset = True
            if env.is_done:
                raise RuntimeError(
                    f"{env.__class__.__name__}.is_done is {env.is_done} after reset"
                )

        elif cmd == "step":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            _td = tensordict.select(*env_input_keys)
            if env.is_done:
                raise RuntimeError(
                    f"calling step when env is done, just reset = {just_reset}"
                )
            _td = env.step(_td)
            if step_keys is None:
                step_keys = set(_td.keys()) - set(env_input_keys)
            if pin_memory:
                _td.pin_memory()
            tensordict.update_(_td.select(*step_keys))
            if _td.get("done"):
                msg = "done"
            else:
                msg = "step_result"
            data = (msg, step_keys)
            child_pipe.send(data)
            just_reset = False

        elif cmd == "close":
            del tensordict, _td, data
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
            state_dict = env.state_dict()
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
                raise RuntimeError(
                    f"querying {err_msg} resulted in the following error: " f"{err}"
                )
            if cmd not in ("to"):
                child_pipe.send(("_".join([cmd, "done"]), result))
            else:
                # don't send env through pipe
                child_pipe.send(("_".join([cmd, "done"]), None))

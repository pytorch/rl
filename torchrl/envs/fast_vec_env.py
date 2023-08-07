from __future__ import annotations
from torchrl.envs import ParallelEnv

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
from torchrl.envs.vec_env import _recursively_strip_locks_from_state_dict, \
    _check_start


class FastParallelEnv(ParallelEnv):
    _fast_step = True
    _auto_reset = True

    def reset(
        self,
        tensordict: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        raise RuntimeError

    @_check_start
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._single_task:
            # this is faster than update_ but won't work for lazy stacks
            for key in self.env_input_keys:
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
            out = TensorDict({}, batch_size=self.shared_tensordict_parent.shape, device=self.device)
            for key in self._selected_step_keys:
                _set_single_key(self.shared_tensordict_parent, out, key, clone=True)
        else:
            # strict=False ensures that non-homogeneous keys are still there
            out = self.shared_tensordict_parent.select(
                *self._selected_step_keys, strict=False
            ).clone()
        return out.get('next')


    def _split_step(self, tensordict: TensorDictBase, auto_reset=None) -> Tuple[TensorDictBase, TensorDictBase]:
        if auto_reset is None:
            auto_reset = self._auto_reset
        # sanity check
        self._assert_tensordict_shape(tensordict)

        next_tensordict = self._step(tensordict)

        # TODO: Refactor this using reward spec
        reward = next_tensordict.get(self.reward_key)
        # unsqueeze rewards if needed
        # the input tensordict may have more leading dimensions than the batch_size
        # e.g. in model-based contexts.
        batch_size = self.batch_size
        dims = len(batch_size)
        leading_batch_size = (
            next_tensordict.batch_size[:-dims]
            if dims
            else next_tensordict.shape
        )
        expected_reward_shape = torch.Size(
            [*leading_batch_size, *self.reward_spec.shape]
        )
        actual_reward_shape = reward.shape
        if actual_reward_shape != expected_reward_shape:
            reward = reward.view(expected_reward_shape)
            next_tensordict.set(self.reward_key, reward)

        # TODO: Refactor this using done spec
        done = next_tensordict.get(self.done_key)
        # unsqueeze done if needed
        expected_done_shape = torch.Size([*leading_batch_size, *self.done_spec.shape])
        actual_done_shape = done.shape
        if actual_done_shape != expected_done_shape:
            done = done.view(expected_done_shape)
            next_tensordict.set(self.done_key, done)

        if self.run_type_checks:
            for key in self._select_observation_keys(next_tensordict):
                obs = next_tensordict.get(key)
                self.observation_spec.type_check(obs, key)

            if (
                next_tensordict.get(self.reward_key).dtype
                is not self.reward_spec.dtype
            ):
                raise TypeError(
                    f"expected reward.dtype to be {self.reward_spec.dtype} "
                    f"but got {next_tensordict.get(self.reward_key).dtype}"
                )

            if next_tensordict.get(self.done_key).dtype is not self.done_spec.dtype:
                raise TypeError(
                    f"expected done.dtype to be torch.bool but got {next_tensordict.get(self.done_key).dtype}"
                )
        if auto_reset and done.any():
            if self._single_task:
                # select + clone creates 2 tds, but we can create one only
                tensordict = TensorDict(
                    {},
                    batch_size=self.shared_tensordict_parent.shape, device=self.device
                    )
                for key in self._selected_reset_keys:
                    if key != "_reset":
                        _set_single_key(
                            self.shared_tensordict_parent,
                            tensordict,
                            key,
                            clone=True
                            )
            else:
                tensordict = self.shared_tensordict_parent.select(
                    *[key for key in self._selected_reset_keys if
                      key != "_reset"],
                    strict=False,
                ).clone()
        return tensordict, next_tensordict


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

        for channel1 in self.parent_channels:
            msg, data = channel1.recv()
            assert msg == "initialized"

        self.is_closed = False


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
    if pin_memory:
        raise RuntimeError

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
            raise EOFError(
                f"proc {pid} failed, last command: {cmd}."
                ) from err
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
            shared_tensordict = data.clone(False)
            del shared_tensordict['next']
            assert 'reward' not in shared_tensordict.keys()
            if not (
                shared_tensordict.is_shared() or shared_tensordict.is_memmap()):
                raise RuntimeError(
                    "tensordict must be placed in shared memory (share_memory_() or memmap_())"
                )
            initialized = True
            # Reset
            local_tensordict = shared_tensordict.clone(False)
            local_tensordict = env._reset(tensordict=local_tensordict)

            if "_reset" in local_tensordict.keys():
                local_tensordict.del_("_reset")
            shared_tensordict.update_(local_tensordict)
            if event is not None:
                event.record()
                event.synchronize()
            out = ("initialized", None)
            child_pipe.send(out)

        elif cmd == "step":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            for key in env_input_keys:
                key = _unravel_key_to_tuple(key)
                local_tensordict._set_tuple(
                        key,
                        shared_tensordict._get_tuple(key, None),
                        inplace=False,
                        validated=True,
                    )
            _cur, _next = env.step(local_tensordict)
            done = _next.get(env.done_key)
            truncated = _next.get("truncated", default=None)
            if truncated is not None:
                done = done | truncated
            if done.all():
                assert "next" not in shared_tensordict.keys()
                assert "reward" not in shared_tensordict.keys()
                assert "next" not in _cur.keys()
                assert "reward" not in _cur.keys()

                shared_tensordict.update_(_cur)
            elif done.any():
                done.view(_cur.shape)
                shared_tensordict[done] = _cur[done]
            msg = "step_result"
            next_shared_tensordict.update_(_next)
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
            state_dict = _recursively_strip_locks_from_state_dict(
                env.state_dict()
                )
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

import os
from multiprocessing import connection
from typing import Callable, Iterable, Union, Optional

import torch
from torch import multiprocessing as mp

from torchrl.data import TensorDict, TensorSpec
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.data.utils import DEVICE_TYPING, CloudpickleWrapper
from torchrl.envs.common import _EnvClass, make_tensor_dict

__all__ = ["SerialEnv", "ParallelEnv"]


class _BatchedEnv(_EnvClass):
    def __init__(
            self,
            num_workers: int,
            create_env_fn: Union[Callable, Iterable[Callable]],
            create_env_kwargs: dict = None,
            device: DEVICE_TYPING = 'cpu',
            action_keys: Optional[Iterable[str]] = None,
            pin_memory: bool = False,
            selected_keys: Optional[Iterable[str]] = None,
            excluded_keys: Optional[Iterable[str]] = None,
            share_individual_td: bool = False,
            shared_memory: bool = True,
            memmap: bool = False,
    ):
        super().__init__(device=device)
        create_env_kwargs = dict() if create_env_kwargs is None else create_env_kwargs
        if callable(create_env_fn):
            create_env_fn = [create_env_fn for _ in range(num_workers)]
        else:
            assert len(
                create_env_fn) == num_workers, f"num_workers and len(create_env_fn) mismatch, " \
                                               f"got {len(create_env_fn)} and {num_workers}"
        if isinstance(create_env_kwargs, dict):
            create_env_kwargs = [create_env_kwargs for _ in range(num_workers)]
        self._dummy_env = create_env_fn[0](**create_env_kwargs[0])
        self.num_workers = num_workers
        self.create_env_fn = create_env_fn
        self.create_env_kwargs = create_env_kwargs
        self.action_keys = action_keys
        self.pin_memory = pin_memory
        self.selected_keys = selected_keys
        self.excluded_keys = excluded_keys
        self.share_individual_td = share_individual_td
        self._share_memory = shared_memory
        self._memmap = memmap
        assert not (self._share_memory and self._memmap), "memmap and shared memory are mutually exclusive features."

        self.batch_size = torch.Size([self.num_workers, *self._dummy_env.batch_size])
        self._action_spec = self._dummy_env.action_spec
        self._observation_spec = self._dummy_env.observation_spec
        self._reward_spec = self._dummy_env.reward_spec
        self.is_closed = False
        self._dummy_env.close()
        self._create_td()
        self._start_workers()

    @property
    def action_spec(self) -> TensorSpec:
        return self._action_spec

    @property
    def observation_spec(self) -> TensorSpec:
        return self._observation_spec

    @property
    def reward_spec(self) -> TensorSpec:
        return self._reward_spec

    def is_done_set_fn(self, value: bool) -> None:
        self._is_done = value.all()

    def _create_td(self) -> None:
        """
        Creates self.shared_tensor_dict_parent, a TensorDict used to store the most recent observations.

        Returns: None

        """
        shared_tensor_dict_parent = make_tensor_dict(
            self._dummy_env,
            None,
        )

        shared_tensor_dict_parent = shared_tensor_dict_parent.expand(
            self.num_workers
        ).clone()

        raise_no_selected_keys = False
        if self.selected_keys is None:
            self.selected_keys = list(shared_tensor_dict_parent.keys())
            if self.excluded_keys is not None:
                self.selected_keys = set(self.selected_keys) - set(self.excluded_keys)
            else:
                raise_no_selected_keys = True
            if self.action_keys is not None:
                assert all(action_key in self.selected_keys for action_key in self.action_keys), \
                    "One of the action keys is not part of the selected keys or is part of the excluded keys. Action " \
                    "keys need to be part of the selected keys for env.step() to be called."
            else:
                self.action_keys = [key for key in self.selected_keys if key.startswith("action")]
                assert len(self.action_keys), f"found 0 action keys in {sorted(list(self.selected_keys))}"
        shared_tensor_dict_parent = shared_tensor_dict_parent.select(*self.selected_keys)
        self.shared_tensor_dict_parent = shared_tensor_dict_parent.to(
            self.device
        )

        if self.share_individual_td:
            self.shared_tensor_dicts = [td.clone() for td in self.shared_tensor_dict_parent.unbind(0)]
            if self._share_memory:
                for td in self.shared_tensor_dicts:
                    td.share_memory_()
            elif self._memmap:
                for td in self.shared_tensor_dicts:
                    td.memmap_()
            self.shared_tensor_dict_parent = torch.stack(self.shared_tensor_dicts, 0)
        else:
            if self._share_memory:
                self.shared_tensor_dict_parent.share_memory_()
                assert self.shared_tensor_dict_parent.is_shared()
            elif self._memmap:
                self.shared_tensor_dict_parent.memmap_()
                assert self.shared_tensor_dict_parent.is_memmap()
            self.shared_tensor_dicts = self.shared_tensor_dict_parent.unbind(0)

        if raise_no_selected_keys:
            print(f"\n {self.__class__.__name__}.shared_tensor_dict_parent is \n{self.shared_tensor_dict_parent}. \n"
                  f"You can select keys to be synchronised by setting the selected_keys and/or excluded_keys "
                  f"arguments when creating the batched environment.")

    def _start_workers(self) -> None:
        """
        Starts the various envs.

        Returns: None

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n\tenv={self._dummy_env}, \n\tbatch_size={self.batch_size})"

    def __del__(self) -> None:
        if not self.is_closed:
            self.close()


class SerialEnv(_BatchedEnv):
    _share_memory = False

    def _start_workers(self) -> None:
        _num_workers = self.num_workers

        self._envs = []

        for idx in range(_num_workers):
            env = self.create_env_fn[idx](**self.create_env_kwargs[idx])
            self._envs.append(env)

    def _step(self, tensor_dict: TensorDict, ) -> TensorDict:
        self._assert_tensordict_shape(tensor_dict)

        self.shared_tensor_dict_parent.update_(tensor_dict.select(*self.action_keys))
        for i in range(self.num_workers):
            self._envs[i].step(self.shared_tensor_dicts[i])

        return self.shared_tensor_dict_parent

    def _shutdown_workers(self) -> None:
        for env in self._envs:
            env.close()

    def __del__(self) -> None:
        self._shutdown_workers()

    def set_seed(self, seed: int) -> int:
        for i, env in enumerate(self._envs):
            env.set_seed(seed)
            if i < self.num_workers - 1:
                seed = seed + 1
        return seed

    def _reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        if tensor_dict is not None and "reset_workers" in tensor_dict.keys():
            self._assert_tensordict_shape(tensor_dict)
            reset_workers = tensor_dict.get("reset_workers")
        else:
            reset_workers = torch.ones(self.num_workers, 1, dtype=torch.bool)

        keys = set()
        for i, _env in enumerate(self._envs):
            if not reset_workers[i]:
                continue
            _td = _env.reset()
            keys = keys.union(_td.keys())
            self.shared_tensor_dicts[i].update(_td)

        return self.shared_tensor_dict_parent.select(*keys).clone()


class ParallelEnv(_BatchedEnv):

    def _start_workers(self) -> None:
        _num_workers = self.num_workers
        ctx = mp.get_context("spawn")

        self.parent_channels = []
        self._workers = []

        for idx in range(_num_workers):
            print(f"initiating worker {idx}")
            # No certainty which module multiprocessing_context is
            channel1, channel2 = ctx.Pipe()
            w = mp.Process(
                target=_run_worker_pipe_shared_mem,
                args=(idx, channel1, channel2,
                      CloudpickleWrapper(self.create_env_fn[idx]),
                      self.create_env_kwargs[idx],
                      self.pin_memory, self.action_keys),
            )
            w.daemon = True
            w.start()
            channel2.close()
            self.parent_channels.append(channel1)
            self._workers.append(w)

        # send shared tensordict to workers
        for channel, shared_tensor_dict in zip(self.parent_channels, self.shared_tensor_dicts):
            channel.send(("init", shared_tensor_dict))

    def _step(self, tensor_dict: TensorDict) -> TensorDict:
        self._assert_tensordict_shape(tensor_dict)

        self.shared_tensor_dict_parent.update_(
            tensor_dict.select(*self.action_keys)
        )
        for i in range(self.num_workers):
            self.parent_channels[i].send(("step", None))

        keys = set()
        for i in range(self.num_workers):
            cmd, data = self.parent_channels[i].recv()
            if cmd != "step_result":
                assert cmd == "done"
            # data is the set of updated keys
            keys = keys.union(data)
        return self.shared_tensor_dict_parent.select(*keys)

    def _shutdown_workers(self) -> None:
        for i, channel in enumerate(self.parent_channels):
            print(f'closing {i}')
            channel.send(("close", None))
            msg, _ = channel.recv()
            assert msg == "closing"

        for channel in self.parent_channels:
            channel.close()
        for proc in self._workers:
            proc.join()

    def close(self) -> None:
        print(f"closing {self.__class__.__name__}")
        self.is_closed = True
        self._shutdown_workers()

    def set_seed(self, seed: int) -> int:
        for i, channel in enumerate(self.parent_channels):
            channel.send(("seed", seed))
            if i < self.num_workers - 1:
                seed = seed + 1
        for channel in self.parent_channels:
            out, _ = channel.recv()
            assert out == "seeded"
        return seed

    def _reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        cmd_out = "reset"
        if tensor_dict is not None and "reset_workers" in tensor_dict.keys():
            self._assert_tensordict_shape(tensor_dict)
            reset_workers = tensor_dict.get("reset_workers")
        else:
            reset_workers = torch.ones(self.num_workers, 1, dtype=torch.bool)

        for i, channel in enumerate(self.parent_channels):
            if not reset_workers[i]:
                continue
            channel.send((cmd_out, None))

        keys = set()
        for i, channel in enumerate(self.parent_channels):
            if not reset_workers[i]:
                continue
            cmd_in, new_keys = channel.recv()
            keys = keys.union(new_keys)
            assert cmd_in == "reset_obs", f"received cmd {cmd_in} instead of reset_obs"
        assert not self.shared_tensor_dict_parent.get("done").any()
        return self.shared_tensor_dict_parent.select(*keys).clone()


def _run_worker_pipe_shared_mem(
        idx: int, parent_pipe: connection.Connection, child_pipe: connection.Connection, env_fun: Callable,
        env_fun_kwargs: dict, pin_memory: bool, action_keys: dict, verbose: bool = False,
) -> None:
    parent_pipe.close()
    pid = os.getpid()
    env = env_fun(**env_fun_kwargs)
    i = -1
    initialized = False

    while True:
        try:
            cmd, data = child_pipe.recv()
        except EOFError:
            raise EOFError(f"proc {pid} failed, last command: {cmd}")
        if cmd == "close":
            assert initialized, "call 'init' before closing"
            env.close()
            child_pipe.send(("closing", None))
            child_pipe.close()
            if verbose:
                print(f"closing {pid}")
            break

        elif cmd == "seed":
            assert initialized, "call 'init' before closing"
            # torch.manual_seed(data)
            # np.random.seed(data)
            env.set_seed(data)
            child_pipe.send(("seeded", None))

        elif cmd == "init":
            if verbose:
                print(f"initializing {pid}")
            assert not initialized, "worker already initialized"
            i = 0
            tensor_dict = data
            assert tensor_dict.is_shared() or tensor_dict.is_memmap()
            initialized = True

        elif cmd == "reset":
            if verbose:
                print(f'resetting worker {pid}')
            assert initialized, "call 'init' before resetting"
            # _td = tensor_dict.select("observation").to(env.device).clone()
            _td = env.reset()
            keys = set(_td.keys())
            if pin_memory:
                _td.pin_memory()
            tensor_dict.update_(_td)
            child_pipe.send(("reset_obs", keys))
            just_reset = True
            assert not env.is_done, \
                f"{env.__class__.__name__}.is_done is {env.is_done}"

        elif cmd == "step":
            assert initialized, "called 'init' before step"
            i += 1
            _td = tensor_dict.select(*action_keys).to(env.device).clone()
            if env.is_done:
                print(f"Warning: calling step when env is done, just reset = {just_reset}")
                print(f"updated keys: {keys}")
                raise Exception
            _td = env.step(_td)
            keys = set(_td.keys()) - {key for key in action_keys}
            if pin_memory:
                _td.pin_memory()
            tensor_dict.update_(_td.select(*keys))
            if _td.get("done"):
                msg = "done"
            else:
                msg = "step_result"
            data = (msg, keys)
            child_pipe.send(data)
            just_reset = False

import math
import queue
import time
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import connection, queues
from typing import Optional, Callable, Union, List, Iterable

import numpy as np
import torch
from torch import multiprocessing as mp
from torch.utils.data import IterableDataset

from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import ProbabilisticOperator
from .utils import CloudpickleWrapper, split_trajectories

__all__ = ["SyncDataCollector", "aSyncDataCollector", "MultiaSyncDataCollector", "MultiSyncDataCollector"]

from contextlib import contextmanager

from ..data.tensordict.tensordict import _TensorDict, TensorDict
from ..envs.common import _EnvClass

TIMEOUT = 1.0
MIN_TIMEOUT = 1e-3  # should be several orders of magnitude inferior wrt time spent collecting a trajectory


class Timer:
    def __init__(self):
        self.time = time.time()

    def __call__(self):
        return time.time() - self.time


@contextmanager
def timeit():
    # Code to acquire resource, e.g.:
    t = Timer()
    yield t


class RandomPolicy:
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def __call__(self, td):
        return td.set("action", self.action_spec.rand(td.batch_size))


class _DataCollector(IterableDataset):
    def __init__(
            self,
            create_env_fn: Callable,
            policy: Union[ProbabilisticOperator, Callable],
            iterator_len,
            create_env_kwargs: Optional[dict] = None,
            max_steps_per_traj: int = -1,
            frames_per_batch: int = 200,
            reset_at_each_iter=False,
            batcher=None,
            split_trajs=True,
    ):
        raise NotImplementedError

    def _get_policy_and_device(self, policy, device, env=None):
        if policy is None:
            assert env is not None, "env must be provided to _get_policy_and_device if policy is None"
            policy = RandomPolicy(env.action_spec)
        try:
            policy_device = next(policy.parameters()).device
        except:
            policy_device = torch.device(device) if device is not None else torch.device('cpu')
        self.device = torch.device(device) if device is not None else policy_device
        self.get_weights_fn = None
        if policy_device != self.device:
            self.get_weights_fn = policy.state_dict
            assert len(list(policy.parameters())) == 0 or next(policy.parameters()).is_shared(), \
                "Provided policy parameters must be shared."
            policy = deepcopy(policy).requires_grad_(False).to(device)
        self.policy = policy

    def update_policy_weights_(self):
        if self.get_weights_fn is not None:
            self.policy.load_state_dict(self.get_weights_fn())

    def __iter__(self):
        return self.iterator()

    def iterator(self):
        raise NotImplementedError

    @staticmethod
    def set_seed(self, seed):
        raise NotImplementedError


class SyncDataCollector(_DataCollector):
    def __init__(
            self,
            create_env_fn: Callable,
            policy: Optional[Union[ProbabilisticOperator, Callable]] = None,
            total_frames: Optional[int] = -1,
            create_env_kwargs: Optional[dict] = None,
            max_steps_per_traj: int = -1,
            frames_per_batch: int = 200,
            init_random_frames: int = -1,
            reset_at_each_iter: bool = False,
            batcher: Optional[Callable] = None,
            split_trajs: bool = True,
            device: Union[int, str, torch.device] = None,
            passing_device="cpu",
            seed=None,
            pin_memory=False,
    ):
        """
        Generic data collector for RL problems. Requires and environment and a policy.
        Args:
            create_env_fn: Callable, returns an instance of _EnvClass class.
            policy: Instance of ProbabilisticOperator class. Must accept _TensorDict object as input.
            total_frames: lower bound of the total frames returned by the collector.
            create_env_kwargs: Optional. Dictionary of kwargs for create_env_fn.
            max_steps_per_traj: Maximum steps per trajectory. Not that a trajectory can span over multiple batches
                (unless reset_at_each_iter is set to True, see below). Once a trajectory reaches n_steps_max,
                the environment is reset. If the environment wraps multiple environments together, the number of steps
                is tracked for each environment independently.
                default: -1
            frames_per_batch: Time-length of a batch. reset_at_each_iter and frames_per_batch == n_steps_max are
                equivalent configurations.
                default: 200
            reset_at_each_iter: Whether or not environments should be reset for each batch. default=False.
            batcher: A Batcher is an object that will read a batch of data and return it in a useful format for training.
                default: None.
            split_trajs: Boolean indicating whether the resulting TensorDict should be split according to the trajectories.
                See utils.split_trajectories for more information.
            device: The device on which the policy will be placed and where the output TensorDict will be stored.
                If it differs from the input policy device, the update_policy_weights_() method should be queried
                at appropriate times during the training loop to accommodate for the lag between parameter configuration
                at various times.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if create_env_kwargs is None:
            create_env_kwargs = {}
        env = create_env_fn(**create_env_kwargs)
        self.env: _EnvClass = env

        self._get_policy_and_device(policy=policy, device=device, env=env)

        self.env_device = env.device
        if not total_frames > 0:
            total_frames = float("inf")
        self.total_frames = total_frames
        self.reset_at_each_iter = reset_at_each_iter
        self.init_random_frames = init_random_frames
        self.batcher = batcher
        if self.batcher is not None:
            self.batcher.to(self.passing_device)
        self.max_steps_per_traj = max_steps_per_traj
        self.frames_per_batch = frames_per_batch
        self.pin_memory = pin_memory

        self.passing_device = torch.device(passing_device)
        self._tensor_dict = env.reset().to(self.passing_device)
        self._tensor_dict.set("step_count", torch.zeros(*self.env.batch_size, 1, dtype=torch.int))
        self._tensor_dict_out = TensorDict(batch_size=[*self.env.batch_size, self.frames_per_batch],
                                           device=self.passing_device)
        self.split_trajs = split_trajs
        self._td_env = None
        self._td_policy = None

    def set_seed(self, seed):
        return self.env.set_seed(seed)

    def iterator(self, ):
        total_frames = self.total_frames
        i = -1
        self._frames = 0
        while True:
            i += 1
            tensor_dict_out = self.rollout()
            self._frames += tensor_dict_out.numel()
            if self._frames >= total_frames:
                self.env.close()

            if self.split_trajs:
                tensor_dict_out = split_trajectories(tensor_dict_out)
            if self.batcher is not None:
                tensor_dict_out = self.batcher(tensor_dict_out)
            yield tensor_dict_out
            if self._frames >= self.total_frames:
                break

    def _cast_to_policy(self, td: _TensorDict):
        policy_device = self.device
        if hasattr(self.policy, 'in_keys'):
            td = td.select(*self.policy.in_keys)
        if self._td_policy is None:
            self._td_policy = td.to(policy_device)
        else:
            if td.device == torch.device("cpu") and self.pin_memory:
                td.pin_memory()
            self._td_policy.update(td, inplace=True)
        return self._td_policy

    def _cast_to_env(self, td: _TensorDict, dest=None):
        env_device = self.env_device
        if dest is None:
            if self._td_env is None:
                self._td_env = td.to(env_device)
            else:
                self._td_env.update(td, inplace=True)
            return self._td_env
        else:
            dest.update(td, inplace=True)

    def _reset_if_necessary(self):
        done = self._tensor_dict.get("done")
        steps = self._tensor_dict.get("step_count")
        done_or_terminated = done | (steps == self.max_steps_per_traj)
        if done_or_terminated.any():
            if len(self.env.batch_size):
                self._tensor_dict.set("reset_workers", done_or_terminated)
            self.env.reset(tensor_dict=self._tensor_dict)
            if len(self.env.batch_size):
                self._tensor_dict.del_("reset_workers")
            traj_ids = self._tensor_dict.get("traj_ids")
            traj_ids[done_or_terminated] = traj_ids.max() + torch.arange(1, done_or_terminated.sum() + 1,
                                                                         device=traj_ids.device)
            steps[done_or_terminated] = 0

    @torch.no_grad()
    def rollout(self):
        if self.reset_at_each_iter:
            self._tensor_dict.update(self.env.reset())
            self._tensor_dict.fill_("step_count", 0)

        n = self.env.batch_size[0] if len(self.env.batch_size) else 1
        self._tensor_dict.set("traj_ids", torch.arange(n).unsqueeze(-1))

        tensor_dict_out = []
        for t in range(self.frames_per_batch):
            if self._frames < self.init_random_frames:
                self.env.rand_step(self._tensor_dict)
            else:
                td_cast = self._cast_to_policy(self._tensor_dict)
                td_cast = self.policy(td_cast)
                self._cast_to_env(td_cast, self._tensor_dict)
                self.env.step(self._tensor_dict)

            step_count = self._tensor_dict.get("step_count")
            step_count += 1
            tensor_dict_out.append(self._tensor_dict.clone())

            self._reset_if_necessary()
            self._tensor_dict.update(step_tensor_dict(self._tensor_dict))
        # print(f"policy: {t_policy: 4.4f}, step: {t_step: 4.4f}, reset: {t_reset: 4.4f}")
        return torch.stack(tensor_dict_out, len(self.env.batch_size),
                           out=self._tensor_dict_out)  # dim 0 for single env, dim 1 for batch

    def reset(self):
        self._tensor_dict.update(self.env.reset())
        self._tensor_dict.fill_("step_count", 0)


class MultiDataCollector(_DataCollector):
    def __init__(
            self,
            create_env_fn: List[Callable],
            policy: Optional[Union[ProbabilisticOperator, Callable]] = None,
            total_frames: Optional[int] = -1,
            create_env_kwargs: Optional[List[dict]] = None,
            max_steps_per_traj: int = -1,
            frames_per_batch: int = 200,
            init_random_frames: int = -1,
            reset_at_each_iter: bool = False,
            batcher: Optional[Callable] = None,
            split_trajs: bool = True,
            device: Union[int, str, torch.device] = None,
            seed: Optional[int] = None,
            pin_memory: bool = False,
            passing_device: Union[int, str, torch.device] = "cpu",
            update_at_each_batch: bool = True,
    ):
        """
        Runs a number of DataCollectors on separate processes.
        This is mostly useful for offline RL paradigms where the policy being trained can differ from the policy used to
        collect data. In online settings, a regular DataCollector should be preferred.

        Args:
            create_env_fn: list of Callables, each returning an instance of _EnvClass
            create_env_kwargs: A (list of) dictionaries with the arguments used to create an environment
        """
        self.create_env_fn = create_env_fn
        self.num_workers = len(create_env_fn)
        self.create_env_kwargs = create_env_kwargs if create_env_kwargs is not None else [dict() for _ in
                                                                                          range(self.num_workers)]
        self._get_policy_and_device(policy=policy, device=device, env=None)
        self.passing_device = torch.device(passing_device)
        self.total_frames = total_frames if total_frames > 0 else float("inf")
        self.reset_at_each_iter = reset_at_each_iter
        self.batcher = batcher
        if self.batcher is not None:
            self.batcher.to(self.passing_device)
        self.max_steps_per_traj = max_steps_per_traj
        self.frames_per_batch = frames_per_batch
        self.seed = seed
        self.split_trajs = split_trajs
        self.pin_memory = pin_memory
        self.init_random_frames = init_random_frames
        self.update_at_each_batch = update_at_each_batch
        self.frames_per_worker = -(self.total_frames // -self.num_workers)  # ceil(total_frames/num_workers)
        self._run_processes()

    @property
    def _queue_len(self):
        raise NotImplementedError

    def _run_processes(self):
        queue_out = mp.Queue(self._queue_len)  # sends data from proc to main
        self.procs = []
        self.pipes = []
        for i, (env_fun, env_fun_kwargs) in enumerate(zip(self.create_env_fn, self.create_env_kwargs)):
            pipe_parent, pipe_child = mp.Pipe()  # send messages to procs
            args = (
                pipe_parent,
                pipe_child,
                queue_out,
                CloudpickleWrapper(env_fun),
                env_fun_kwargs,
                self.policy,
                self.frames_per_worker,
                self.max_steps_per_traj,
                self.frames_per_batch,
                self.reset_at_each_iter,
                self.device,
                self.passing_device,
                self.seed,
                self.pin_memory,
                i,
            )
            proc = mp.Process(target=main_async_collector, args=args)
            # proc.daemon can't be set as daemonic processes may be launched by the process itself
            proc.start()
            pipe_child.close()
            self.procs.append(proc)
            self.pipes.append(pipe_parent)
        self.queue_out = queue_out

    def shutdown(self):
        self._shutdown_main()

    def _shutdown_main(self):
        for idx in range(self.num_workers):
            self.pipes[idx].send((None, "close"))
            msg = self.pipes[idx].recv()
            assert msg == "closed", f"got {msg}"
        for proc in self.procs:
            proc.join()
        self.queue_out.close()
        for pipe in self.pipes:
            pipe.close()

    # def update_policy_weights_(self):
    # for idx in range(self.num_workers):
    #     self.pipes[idx].send((None, "update"))
    # for idx in range(self.num_workers):
    #     j, msg = self.pipes[idx].recv()
    #     assert msg == "updated", f"got {msg}"
    #     if j < self.iterator_len - 1:
    #         self.pipes[idx].send((idx, "continue"))

    def set_seed(self, seed: int) -> int:
        for idx in range(self.num_workers):
            self.pipes[idx].send((seed, "seed"))
            new_seed, msg = self.pipes[idx].recv()
            assert msg == "seeded", f"got {msg}"
            seed = new_seed
            if idx < self.num_workers - 1:
                seed = seed + 1
        self.reset()
        return seed

    def reset(self, reset_idx: Optional[Iterable[bool]] = None):
        if reset_idx is None:
            reset_idx = [True for _ in range(self.num_workers)]
        for idx in range(self.num_workers):
            if reset_idx[idx]:
                self.pipes[idx].send((None, "reset"))
        for idx in range(self.num_workers):
            if reset_idx[idx]:
                j, msg = self.pipes[idx].recv()
                assert msg == "reset", f"got {msg}"


class MultiSyncDataCollector(MultiDataCollector):
    @property
    def _queue_len(self):
        return self.num_workers

    def iterator(self):
        i = -1
        frames = 0
        out_tensordicts_shared = OrderedDict()
        dones = [False for _ in range(self.num_workers)]
        workers_frames = [0 for _ in range(self.num_workers)]
        while not all(dones) and frames < self.total_frames:
            if self.update_at_each_batch:
                self.update_policy_weights_()

            for idx in range(self.num_workers):
                if frames < self.init_random_frames:
                    msg = "continue_random"
                else:
                    msg = "continue"
                self.pipes[idx].send((None, msg))

            i += 1
            max_traj_idx = None
            for k in range(self.num_workers):
                new_data, j = self.queue_out.get()
                if j == 0:
                    data, idx = new_data
                    out_tensordicts_shared[idx] = data
                else:
                    idx = new_data
                workers_frames[idx] = workers_frames[idx] + math.prod(out_tensordicts_shared[idx].shape)

                if workers_frames[idx] >= self.total_frames:
                    print(f"{idx} is done!")
                    dones[idx] = True
            for idx in range(self.num_workers):
                traj_ids = out_tensordicts_shared[idx].get("traj_ids")
                if max_traj_idx is not None:
                    traj_ids += max_traj_idx
                    # out_tensordicts_shared[idx].set("traj_ids", traj_ids)
                max_traj_idx = traj_ids.max() + 1
                # out = out_tensordicts_shared[idx]
            out = torch.cat([item for key, item in out_tensordicts_shared.items()], 0)
            if self.split_trajs:
                out = split_trajectories(out)
                frames += out.get("mask").sum()
            else:
                frames += math.prod(out.shape)
            if self.batcher is not None:
                out = self.batcher(out)
            yield out

        self._shutdown_main()


class MultiaSyncDataCollector(MultiDataCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_tensordicts = dict()
        self.running = False

    def _get_from_queue(self, timeout=None):
        new_data, j = self.queue_out.get(timeout=timeout)
        if j == 0:
            data, idx = new_data
            self.out_tensordicts[idx] = data
        else:
            idx = new_data
        out = self.out_tensordicts[idx]
        return idx, j, out

    @property
    def _queue_len(self):
        return 1

    def iterator(self):
        for i in range(self.num_workers):
            if self.init_random_frames>0:
                self.pipes[i].send((None, "continue_random"))
            else:
                self.pipes[i].send((None, "continue"))
        self.running = True
        i = -1
        self._frames = 0

        dones = [False for _ in range(self.num_workers)]
        workers_frames = [0 for _ in range(self.num_workers)]
        while not all(dones) and self._frames < self.total_frames:
            i += 1
            idx, j, out = self._get_from_queue()

            if self.split_trajs:
                out = split_trajectories(out)
                worker_frames = out.get("mask").sum()
            else:
                worker_frames = math.prod(out.shape)
            self._frames += worker_frames
            workers_frames[idx] = workers_frames[idx] + worker_frames
            if self.batcher is not None:
                out = self.batcher(out)

            # the function blocks here until the next item is asked, hence we send the message to the
            # worker to keep on working in the meantime before the yield statement
            if workers_frames[idx] < self.frames_per_worker:
                if self._frames<self.init_random_frames:
                    msg = "continue_random"
                else:
                    msg = "continue"
                self.pipes[idx].send((idx, msg))
            else:
                print(f"{idx} is done!")
                dones[idx] = True

            yield out

        self._shutdown_main()
        self.running = False

    # def set_seed(self, seed: Iterable):
    #     super().set_seed(seed)
    #     for idx in range(self.num_workers):
    #         self.pipes[idx].send((idx, "continue"))

    def reset(self, reset_idx: Optional[Iterable[bool]] = None):
        super().reset(reset_idx)
        if self.queue_out.full():
            print('waiting')
            time.sleep(TIMEOUT)  # wait until queue is empty
        assert not self.queue_out.full()
        if self.running:
            for idx in range(self.num_workers):
                if self._frames<self.init_random_frames:
                    self.pipes[idx].send((idx, "continue_random"))
                else:
                    self.pipes[idx].send((idx, "continue"))


class aSyncDataCollector(MultiaSyncDataCollector):
    def __init__(
            self,
            create_env_fn: Callable,
            policy: Optional[Union[ProbabilisticOperator, Callable]] = None,
            total_frames: Optional[int] = -1,
            create_env_kwargs: Optional[dict] = None,
            max_steps_per_traj: int = -1,
            frames_per_batch: int = 200,
            reset_at_each_iter: bool = False,
            batcher: Optional[Callable] = None,
            split_trajs: bool = True,
            device: Union[int, str, torch.device] = None,
            seed: Optional[int] = None,
            pin_memory: bool = False,
    ):
        """
        Runs a DataCollector on a separate process.
        This is mostly useful for offline RL paradigms where the policy being trained can differ from the policy used to
        collect data. In online settings, a regular DataCollector should be preferred.

        Args:
            create_env_fn: Callable that returns an instance of _EnvClass
            create_env_kwargs: A dictionary with the arguments used to create an environment
        """
        super().__init__(
            create_env_fn=[create_env_fn],
            policy=policy,
            total_frames=total_frames,
            create_env_kwargs=[create_env_kwargs],
            max_steps_per_traj=max_steps_per_traj,
            frames_per_batch=frames_per_batch,
            reset_at_each_iter=reset_at_each_iter,
            batcher=batcher,
            split_trajs=split_trajs,
            device=device,
            seed=seed,
            pin_memory=pin_memory,
        )


def main_async_collector(
        pipe_parent: connection.Connection,
        pipe_child: connection.Connection,
        queue_out: queues.Queue,
        create_env_fn: Callable,
        create_env_kwargs: dict,
        policy: Callable,
        frames_per_worker: int,
        max_steps_per_traj: int,
        frames_per_batch: int,
        reset_at_each_iter: bool,
        device: Optional[Union[torch.device, str, int]],
        passing_device: Optional[Union[torch.device, str, int]],
        seed: Union[int, Iterable],
        pin_memory: bool,
        idx: int = 0,
):
    pipe_parent.close()
    dc = SyncDataCollector(
        create_env_fn,
        create_env_kwargs=create_env_kwargs,
        policy=policy,
        total_frames=-1,
        max_steps_per_traj=max_steps_per_traj,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=reset_at_each_iter,
        batcher=None,
        split_trajs=False,
        device=device,
        seed=seed,
        pin_memory=pin_memory,
        passing_device=passing_device,
    )
    print("Sync data collector created")
    dc_iter = iter(dc)
    j = 0
    has_timed_out = False
    while True:
        if j == 0 or pipe_child.poll(TIMEOUT):
            data_in, msg = pipe_child.recv()
        else:
            # default is "continue" (after first iteration)
            # this is expected to happen if queue_out reached the timeout, but no new msg was waiting in the pipe
            # in that case, the main process probably expects the worker to continue collect data
            if has_timed_out:
                assert msg in ("continue", "continue_random")
            else:
                continue
        if msg in ("continue", "continue_random"):
            if msg == "continue_random":
                dc.init_random_frames = float("inf")
            else:
                dc.init_random_frames = -1

            d = next(dc_iter)
            if pipe_child.poll(MIN_TIMEOUT):
                # in this case, main send a message to the worker while it was busy collecting trajectories.
                # In that case, we skip the collected trajectory and get the message from main. This is faster than
                # sending the trajectory in the queue until timeout when it's never going to be received.
                continue
            if j == 0:
                tensor_dict = d
                if passing_device is not None:
                    tensor_dict = tensor_dict.to(passing_device)
                tensor_dict.share_memory_()
                data = (tensor_dict, idx)
            else:
                tensor_dict.update_(d)
                data = idx  # flag the worker that has sent its data
            try:
                queue_out.put((data, j), timeout=TIMEOUT)
                j += 1
                has_timed_out = False
                continue
            except queue.Full:
                has_timed_out = True
                continue
            # pipe_child.send("done")
        elif msg == "update":
            dc.update_policy_weights_()
            pipe_child.send((j, "updated"))
            continue
        elif msg == "seed":
            new_seed = dc.set_seed(data_in)
            torch.manual_seed(data_in)
            np.random.seed(data_in)
            pipe_child.send((new_seed, "seeded"))
            continue
        elif msg == "reset":
            dc.reset()
            pipe_child.send((j, "reset"))
            continue
        elif msg == "close":
            # assert i == iterator_len - 1
            pipe_child.send("closed")
            break
        else:
            raise Exception(f"Unrecognized message {msg}")

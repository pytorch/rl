from __future__ import annotations

import collections
import time
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from queue import Empty

import torch

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl import logger as torchrl_logger
from torchrl._utils import (
    _check_for_faulty_process,
    accept_remote_rref_udf_invocation,
    RL_WARNINGS,
)
from torchrl.collectors._base import _make_legacy_metaclass
from torchrl.collectors._constants import _MAX_IDLE_COUNT, _TIMEOUT
from torchrl.collectors._multi_base import _MultiCollectorMeta, MultiCollector
from torchrl.collectors.utils import split_trajectories


@accept_remote_rref_udf_invocation
class MultiSyncCollector(MultiCollector):
    """Runs a given number of DataCollectors on separate processes synchronously.

    .. aafig::

            +----------------------------------------------------------------------+
            |            "MultiSyncCollector"                 |                |
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
    This class can be safely used with online RL sota-implementations.

    .. note::
        Python requires multiprocessed code to be instantiated within a main guard:

            >>> from torchrl.collectors import MultiSyncCollector
            >>> if __name__ == "__main__":
            ...     # Create your collector here
            ...     collector = MultiSyncCollector(...)

        See https://docs.python.org/3/library/multiprocessing.html for more info.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> from torchrl.collectors import MultiSyncCollector
        >>> if __name__ == "__main__":
        ...     env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        ...     policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        ...     collector = MultiSyncCollector(
        ...         create_env_fn=[env_maker, env_maker],
        ...         policy=policy,
        ...         total_frames=2000,
        ...         max_frames_per_traj=50,
        ...         frames_per_batch=200,
        ...         init_random_frames=-1,
        ...         reset_at_each_iter=False,
        ...         device="cpu",
        ...         storing_device="cpu",
        ...         cat_results="stack",
        ...     )
        ...     for i, data in enumerate(collector):
        ...         if i == 2:
        ...             print(data)
        ...             break
        ...     collector.shutdown()
        ...     del collector
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

    """

    __doc__ += MultiCollector.__doc__

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        if not close_env:
            raise RuntimeError(
                f"Cannot shutdown {type(self).__name__} collector without environment being closed."
            )
        if hasattr(self, "out_buffer"):
            del self.out_buffer
        if hasattr(self, "buffers"):
            del self.buffers
        try:
            return super().shutdown(timeout=timeout)
        except Exception as e:
            if raise_on_error:
                raise e
            else:
                pass

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
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        **kwargs,
    ) -> None:
        if "policy_weights" in kwargs:
            warnings.warn(
                "`policy_weights` is deprecated. Use `policy_or_weights` instead.",
                DeprecationWarning,
            )
            policy_or_weights = kwargs.pop("policy_weights")

        super().update_policy_weights_(
            policy_or_weights=policy_or_weights, worker_ids=worker_ids, **kwargs
        )

    def frames_per_batch_worker(self, *, worker_idx: int | None = None) -> int:
        if worker_idx is not None and isinstance(self._frames_per_batch, Sequence):
            return self._frames_per_batch[worker_idx]
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
        cat_results = self.cat_results
        if cat_results is None:
            cat_results = "stack"

        self.buffers = [None for _ in range(self.num_workers)]
        dones = [False for _ in range(self.num_workers)]
        workers_frames = [0 for _ in range(self.num_workers)]
        same_device = None
        self.out_buffer = None
        preempt = self.interruptor is not None and self.preemptive_threshold < 1.0

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

            if preempt:
                self.interruptor.start_collection()
                while self.queue_out.qsize() < int(
                    self.num_workers * self.preemptive_threshold
                ):
                    continue
                self.interruptor.stop_collection()
                # Now wait for stragglers to return
                while self.queue_out.qsize() < int(self.num_workers):
                    continue

            recv = collections.deque()
            t0 = time.time()
            while len(recv) < self.num_workers and (
                (time.time() - t0) < (_TIMEOUT * _MAX_IDLE_COUNT)
            ):
                for _ in range(self.num_workers):
                    try:
                        new_data, j = self.queue_out.get(timeout=_TIMEOUT)
                        recv.append((new_data, j))
                    except (TimeoutError, Empty):
                        _check_for_faulty_process(self.procs)
            if (time.time() - t0) > (_TIMEOUT * _MAX_IDLE_COUNT):
                try:
                    self.shutdown()
                finally:
                    raise RuntimeError(
                        f"Failed to gather all collector output within {_TIMEOUT * _MAX_IDLE_COUNT} seconds. "
                        f"Increase the MAX_IDLE_COUNT environment variable to bypass this error."
                    )

            for _ in range(self.num_workers):
                new_data, j = recv.popleft()
                use_buffers = self._use_buffers
                if self.replay_buffer is not None:
                    idx = new_data
                    workers_frames[idx] = workers_frames[
                        idx
                    ] + self.frames_per_batch_worker(worker_idx=idx)
                    continue
                elif j == 0 or not use_buffers:
                    try:
                        data, idx = new_data
                        self.buffers[idx] = data
                        if use_buffers is None and j > 0:
                            self._use_buffers = False
                    except TypeError:
                        if use_buffers is None:
                            self._use_buffers = True
                            idx = new_data
                        else:
                            raise
                else:
                    idx = new_data

                if preempt:
                    # mask buffers if cat, and create a mask if stack
                    if cat_results != "stack":
                        buffers = [None] * self.num_workers
                        for worker_idx, buffer in enumerate(self.buffers):
                            # Skip pre-empted envs:
                            if buffer is None:
                                continue
                            valid = buffer.get(("collector", "traj_ids")) != -1
                            if valid.ndim > 2:
                                valid = valid.flatten(0, -2)
                            if valid.ndim == 2:
                                valid = valid.any(0)
                            buffers[worker_idx] = buffer[..., valid]
                    else:
                        for buffer in filter(lambda x: x is not None, self.buffers):
                            with buffer.unlock_():
                                buffer.set(
                                    ("collector", "mask"),
                                    buffer.get(("collector", "traj_ids")) != -1,
                                )
                        buffers = self.buffers
                else:
                    buffers = self.buffers

                # Skip frame counting if this worker didn't send data this iteration
                # (happens when reusing buffers or on first iteration with some workers)
                if self.buffers[idx] is None:
                    continue

                workers_frames[idx] = workers_frames[idx] + buffers[idx].numel()

                if workers_frames[idx] >= self.total_frames:
                    dones[idx] = True

            if self.replay_buffer is not None:
                yield
                self._frames += sum(
                    self.frames_per_batch_worker(worker_idx=worker_idx)
                    for worker_idx in range(self.num_workers)
                )
                continue

            # we have to correct the traj_ids to make sure that they don't overlap
            # We can count the number of frames collected for free in this loop
            n_collected = 0
            for idx in range(self.num_workers):
                buffer = buffers[idx]
                if buffer is None:
                    continue
                traj_ids = buffer.get(("collector", "traj_ids"))
                if preempt:
                    if cat_results == "stack":
                        mask_frames = buffer.get(("collector", "traj_ids")) != -1
                        n_collected += mask_frames.sum().cpu()
                    else:
                        n_collected += traj_ids.numel()
                else:
                    n_collected += traj_ids.numel()

            if same_device is None:
                prev_device = None
                same_device = True
                for item in filter(lambda x: x is not None, self.buffers):
                    if prev_device is None:
                        prev_device = item.device
                    else:
                        same_device = same_device and (item.device == prev_device)

            if cat_results == "stack":
                stack = (
                    torch.stack if self._use_buffers else TensorDict.maybe_dense_stack
                )
                if same_device:
                    self.out_buffer = stack(
                        [item for item in buffers if item is not None], 0
                    )
                else:
                    self.out_buffer = stack(
                        [item.cpu() for item in buffers if item is not None], 0
                    )
            else:
                if self._use_buffers is None:
                    torchrl_logger.warning(
                        "use_buffer not specified and not yet inferred from data, assuming `True`."
                    )
                elif not self._use_buffers:
                    raise RuntimeError(
                        "Cannot concatenate results with use_buffers=False"
                    )
                try:
                    if same_device:
                        self.out_buffer = torch.cat(
                            [item for item in buffers if item is not None], cat_results
                        )
                    else:
                        self.out_buffer = torch.cat(
                            [item.cpu() for item in buffers if item is not None],
                            cat_results,
                        )
                except RuntimeError as err:
                    if (
                        preempt
                        and cat_results != -1
                        and "Sizes of tensors must match" in str(err)
                    ):
                        raise RuntimeError(
                            "The value provided to cat_results isn't compatible with the collectors outputs. "
                            "Consider using `cat_results=-1`."
                        )
                    raise

            # TODO: why do we need to do cat inplace and clone?
            if self.split_trajs:
                out = split_trajectories(self.out_buffer, prefix="collector")
            else:
                out = self.out_buffer
            if cat_results in (-1, "stack"):
                out.refine_names(*[None] * (out.ndim - 1) + ["time"])

            self._frames += n_collected

            if self.postprocs:
                self.postprocs = (
                    self.postprocs.to(out.device)
                    if hasattr(self.postprocs, "to")
                    else self.postprocs
                )
                out = self.postprocs(out)
            if self._exclude_private_keys:
                excluded_keys = [key for key in out.keys() if key.startswith("_")]
                if excluded_keys:
                    out = out.exclude(*excluded_keys)
            yield out
            del out

        del self.buffers
        self.out_buffer = None
        # We shall not call shutdown just yet as user may want to retrieve state_dict
        # self._shutdown_main()

    # for RPC
    def receive_weights(self, policy_or_weights: TensorDictBase | None = None):
        return super().receive_weights(policy_or_weights)

    # for RPC
    def _receive_weights_scheme(self):
        return super()._receive_weights_scheme()


_LegacyMultiSyncMeta = _make_legacy_metaclass(_MultiCollectorMeta)


class MultiSyncDataCollector(MultiSyncCollector, metaclass=_LegacyMultiSyncMeta):
    """Deprecated version of :class:`~torchrl.collectors.MultiSyncCollector`."""

    ...

from __future__ import annotations

import time
import warnings
from collections import defaultdict, OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from queue import Empty

import torch

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl._utils import _check_for_faulty_process, accept_remote_rref_udf_invocation
from torchrl.collectors._constants import _MAX_IDLE_COUNT, _TIMEOUT
from torchrl.collectors._multi_base import _MultiDataCollector
from torchrl.collectors.utils import split_trajectories


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

    The collection keeps on occurring on all processes even between the time
    the batch of rollouts is collected and the next call to the iterator.
    This class can be safely used with offline RL sota-implementations.

    .. note:: Python requires multiprocessed code to be instantiated within a main guard:

            >>> from torchrl.collectors import MultiaSyncDataCollector
            >>> if __name__ == "__main__":
            ...     # Create your collector here

        See https://docs.python.org/3/library/multiprocessing.html for more info.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> from torchrl.collectors import MultiaSyncDataCollector
        >>> if __name__ == "__main__":
        ...     env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        ...     policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        ...     collector = MultiaSyncDataCollector(
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

    __doc__ += _MultiDataCollector.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_tensordicts = defaultdict(lambda: None)
        self.running = False

        if self.postprocs is not None and self.replay_buffer is None:
            postproc = self.postprocs
            self.postprocs = {}
            for _device in self.storing_device:
                if _device not in self.postprocs:
                    if hasattr(postproc, "to"):
                        postproc = deepcopy(postproc).to(_device)
                    self.postprocs[_device] = postproc

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
        if hasattr(self, "out_tensordicts"):
            del self.out_tensordicts
        if not close_env:
            raise RuntimeError(
                f"Cannot shutdown {type(self).__name__} collector without environment being closed."
            )
        return super().shutdown(timeout=timeout, raise_on_error=raise_on_error)

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

    def frames_per_batch_worker(self, worker_idx: int | None = None) -> int:
        return self.requested_frames_per_batch

    def _get_from_queue(self, timeout=None) -> tuple[int, int, TensorDictBase]:
        new_data, j = self.queue_out.get(timeout=timeout)
        use_buffers = self._use_buffers
        if self.replay_buffer is not None:
            idx = new_data
        elif j == 0 or not use_buffers:
            try:
                data, idx = new_data
                self.out_tensordicts[idx] = data
                if use_buffers is None and j > 0:
                    use_buffers = self._use_buffers = False
            except TypeError:
                if use_buffers is None:
                    use_buffers = self._use_buffers = True
                    idx = new_data
                else:
                    raise
        else:
            idx = new_data
        out = self.out_tensordicts[idx]
        if not self.replay_buffer and (j == 0 or use_buffers):
            # we clone the data to make sure that we'll be working with a fixed copy
            out = out.clone()
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
            self._iter += 1
            counter = 0
            while True:
                try:
                    idx, j, out = self._get_from_queue(timeout=_TIMEOUT)
                    break
                except (TimeoutError, Empty):
                    counter += _TIMEOUT
                    _check_for_faulty_process(self.procs)
                if counter > (_TIMEOUT * _MAX_IDLE_COUNT):
                    raise RuntimeError(
                        f"Failed to gather all collector output within {_TIMEOUT * _MAX_IDLE_COUNT} seconds. "
                        f"Increase the MAX_IDLE_COUNT environment variable to bypass this error."
                    )
            if self.replay_buffer is None:
                worker_frames = out.numel()
                if self.split_trajs:
                    out = split_trajectories(out, prefix="collector")
            else:
                worker_frames = self.frames_per_batch_worker()
            self._frames += worker_frames
            workers_frames[idx] = workers_frames[idx] + worker_frames
            if out is not None and self.postprocs:
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
            if out is not None and self._exclude_private_keys:
                excluded_keys = [key for key in out.keys() if key.startswith("_")]
                out = out.exclude(*excluded_keys)
            yield out

        # We don't want to shutdown yet, the user may want to call state_dict before
        # self._shutdown_main()
        self.running = False

    def _shutdown_main(self, *args, **kwargs) -> None:
        if hasattr(self, "out_tensordicts"):
            del self.out_tensordicts
        return super()._shutdown_main(*args, **kwargs)

    def reset(self, reset_idx: Sequence[bool] | None = None) -> None:
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

    def _receive_weights_scheme(self):
        return super()._receive_weights_scheme()

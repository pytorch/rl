# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Callable, Dict, Iterator, List, OrderedDict, Union

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    DataCollectorBase,
    DEFAULT_EXPLORATION_TYPE,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.collectors.utils import _NON_NN_POLICY_WEIGHTS, split_trajectories
from torchrl.envs.common import EnvBase
from torchrl.envs.env_creator import EnvCreator


RAY_ERR = None
try:
    import ray
    from ray._private.services import get_node_ip_address

    _has_ray = True
except ImportError as err:
    _has_ray = False
    RAY_ERR = err

DEFAULT_RAY_INIT_CONFIG = {
    "address": None,
    "num_cpus": None,
    "num_gpus": None,
    "resources": None,
    "object_store_memory": None,
    "local_mode": False,
    "ignore_reinit_error": False,
    "include_dashboard": None,
    "dashboard_host": "127.0.0.1",
    "dashboard_port": None,
    "job_config": None,
    "configure_logging": True,
    "logging_level": "info",
    "logging_format": None,
    "log_to_driver": True,
    "namespace": None,
    "runtime_env": None,
    "storage": None,
}

DEFAULT_REMOTE_CLASS_CONFIG = {
    "num_cpus": 1,
    "num_gpus": 0.2,
    "memory": 2 * 1024**3,
}


def print_remote_collector_info(self):
    """Prints some information about the remote collector."""
    s = (
        f"Created remote collector with in machine "
        f"{get_node_ip_address()} using gpus {ray.get_gpu_ids()}"
    )
    # torchrl_logger.warning(s)
    torchrl_logger.info(s)


@classmethod
def as_remote(cls, remote_config):
    """Creates an instance of a remote ray class.

    Args:
        cls (Python Class): class to be remotely instantiated.
        remote_config (dict): the quantity of CPU cores to reserve for this class.

    Returns:
        A function that creates ray remote class instances.
    """
    remote_collector = ray.remote(**remote_config)(cls)
    remote_collector.is_remote = True
    return remote_collector


class RayCollector(DataCollectorBase):
    """Distributed data collector with `Ray <https://docs.ray.io/>`_ backend.

    This Python class serves as a ray-based solution to instantiate and coordinate multiple
    data collectors in a distributed cluster. Like TorchRL non-distributed collectors, this
    collector is an iterable that yields TensorDicts until a target number of collected
    frames is reached, but handles distributed data collection under the hood.

    The class dictionary input parameter "ray_init_config" can be used to provide the kwargs to
    call Ray initialization method ray.init(). If "ray_init_config" is not provided, the default
    behavior is to autodetect an existing Ray cluster or start a new Ray instance locally if no
    existing cluster is found. Refer to Ray documentation for advanced initialization kwargs.

    Similarly, dictionary input parameter "remote_configs" can be used to specify the kwargs for
    ray.remote() when called to create each remote collector actor, including collector compute
    resources.The sum of all collector resources should be available in the cluster. Refer to Ray
    documentation for advanced configuration of the ray.remote() method. Default kwargs are:

    >>> kwargs = {
    ...     "num_cpus": 1,
    ...     "num_gpus": 0.2,
    ...     "memory": 2 * 1024 ** 3,
    ... }


    The coordination between collector instances can be specified as "synchronous" or "asynchronous".
    In synchronous coordination, this class waits for all remote collectors to collect a rollout,
    concatenates all rollouts into a single TensorDict instance and finally yields the concatenated
    data. On the other hand, if the coordination is to be carried out asynchronously, this class
    provides the rollouts as they become available from individual remote collectors.

    Args:
        create_env_fn (Callable or List[Callabled]): list of Callables, each returning an
            instance of :class:`~torchrl.envs.EnvBase`.
        policy (Callable): Policy to be executed in the environment.
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
            If ``None`` is provided, the policy used will be a
            :class:`~torchrl.collectors.RandomPolicy` instance with the environment
            ``action_spec``.
            Accepted policies are usually subclasses of :class:`~tensordict.nn.TensorDictModuleBase`.
            This is the recommended usage of the collector.
            Other callables are accepted too:
            If the policy is not a ``TensorDictModuleBase`` (e.g., a regular :class:`~torch.nn.Module`
            instances) it will be wrapped in a `nn.Module` first.
            Then, the collector will try to assess if these
            modules require wrapping in a :class:`~tensordict.nn.TensorDictModule` or not.
            - If the policy forward signature matches any of ``forward(self, tensordict)``,
              ``forward(self, td)`` or ``forward(self, <anything>: TensorDictBase)`` (or
              any typing with a single argument typed as a subclass of ``TensorDictBase``)
              then the policy won't be wrapped in a :class:`~tensordict.nn.TensorDictModule`.
            - In all other cases an attempt to wrap it will be undergone as such: ``TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)``.

    Keyword Args:
        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
        total_frames (int, Optional): lower bound of the total number of frames returned by the collector.
            The iterator will stop once the total number of frames equates or exceeds the total number of
            frames passed to the collector. Default value is -1, which mean no target total number of frames
            (i.e. the collector will run indefinitely).
        device (int, str or torch.device, optional): The generic device of the
            collector. The ``device`` args fills any non-specified device: if
            ``device`` is not ``None`` and any of ``storing_device``, ``policy_device`` or
            ``env_device`` is not specified, its value will be set to ``device``.
            Defaults to ``None`` (No default device).
            Lists of devices are supported.
        storing_device (int, str or torch.device, optional): The *remote* device on which
            the output :class:`~tensordict.TensorDict` will be stored.
            If ``device`` is passed and ``storing_device`` is ``None``, it will
            default to the value indicated by ``device``.
            For long trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``None`` (the output tensordict isn't on a specific device,
            leaf tensors sit on the device where they were created).
            Lists of devices are supported.
        env_device (int, str or torch.device, optional): The *remote* device on which
            the environment should be cast (or executed if that functionality is
            supported). If not specified and the env has a non-``None`` device,
            ``env_device`` will default to that value. If ``device`` is passed
            and ``env_device=None``, it will default to ``device``. If the value
            as such specified of ``env_device`` differs from ``policy_device``
            and one of them is not ``None``, the data will be cast to ``env_device``
            before being passed to the env (i.e., passing different devices to
            policy and env is supported). Defaults to ``None``.
            Lists of devices are supported.
        policy_device (int, str or torch.device, optional): The *remote* device on which
            the policy should be cast.
            If ``device`` is passed and ``policy_device=None``, it will default
            to ``device``. If the value as such specified of ``policy_device``
            differs from ``env_device`` and one of them is not ``None``,
            the data will be cast to ``policy_device`` before being passed to
            the policy (i.e., passing different devices to policy and env is
            supported). Defaults to ``None``.
            Lists of devices are supported.
        create_env_kwargs (dict, optional): Dictionary of kwargs for
            ``create_env_fn``.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span across multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e., no maximum number of steps).
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
            collecting data. Must be one of ``torchrl.envs.utils.ExplorationType.DETERMINISTIC``,
            ``torchrl.envs.utils.ExplorationType.RANDOM``, ``torchrl.envs.utils.ExplorationType.MODE``
            or ``torchrl.envs.utils.ExplorationType.MEAN``.
        collector_class (Python class): a collector class to be remotely instantiated. Can be
            :class:`~torchrl.collectors.SyncDataCollector`,
            :class:`~torchrl.collectors.MultiSyncDataCollector`,
            :class:`~torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these.
            Defaults to :class:`~torchrl.collectors.SyncDataCollector`.
        collector_kwargs (dict or list, optional): a dictionary of parameters to be passed to the
            remote data-collector. If a list is provided, each element will
            correspond to an individual set of keyword arguments for the
            dedicated collector.
        num_workers_per_collector (int): the number of copies of the
            env constructor that is to be used on the remote nodes.
            Defaults to 1 (a single env per collector).
            On a single worker node all the sub-workers will be
            executing the same environment. If different environments need to
            be executed, they should be dispatched across worker nodes, not
            subnodes.
        ray_init_config (dict, Optional): kwargs used to call ray.init().
        remote_configs (list of dicts, Optional): ray resource specs for each remote collector.
            A single dict can be provided as well, and will be used in all collectors.
        num_collectors (int, Optional): total number of collectors to be instantiated.
        sync (bool): if ``True``, the resulting tensordict is a stack of all the
            tensordicts collected on each node. If ``False`` (default), each
            tensordict results from a separate node in a "first-ready,
            first-served" fashion.
        update_after_each_batch (bool, optional): if ``True``, the weights will
            be updated after each collection. For ``sync=True``, this means that
            all workers will see their weights updated. For ``sync=False``,
            only the worker from which the data has been gathered will be
            updated.
            Defaults to ``False``, i.e. updates have to be executed manually
            through
            ``torchrl.collectors.distributed.RayDistributedCollector.update_policy_weights_()``
        max_weight_update_interval (int, optional): the maximum number of
            batches that can be collected before the policy weights of a worker
            is updated.
            For sync collections, this parameter is overwritten by ``update_after_each_batch``.
            For async collections, it may be that one worker has not seen its
            parameters being updated for a certain time even if ``update_after_each_batch``
            is turned on.
            Defaults to -1 (no forced update).

    Examples:
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.collectors.collectors import SyncDataCollector
        >>> from torchrl.collectors.distributed import RayCollector
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> distributed_collector = RayCollector(
        ...     create_env_fn=[env_maker],
        ...     policy=policy,
        ...     collector_class=SyncDataCollector,
        ...     max_frames_per_traj=50,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=-False,
        ...     collector_kwargs={
        ...         "device": "cpu",
        ...         "storing_device": "cpu",
        ...     },
        ...     num_collectors=1,
        ...     total_frames=10000,
        ...     frames_per_batch=200,
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
    """

    def __init__(
        self,
        create_env_fn: Union[Callable, EnvBase, List[Callable], List[EnvBase]],
        policy: Callable[[TensorDict], TensorDict],
        *,
        frames_per_batch: int,
        total_frames: int = -1,
        device: torch.device | List[torch.device] = None,
        storing_device: torch.device | List[torch.device] = None,
        env_device: torch.device | List[torch.device] = None,
        policy_device: torch.device | List[torch.device] = None,
        max_frames_per_traj=-1,
        init_random_frames=-1,
        reset_at_each_iter=False,
        postproc=None,
        split_trajs=False,
        exploration_type=DEFAULT_EXPLORATION_TYPE,
        collector_class: Callable[[TensorDict], TensorDict] = SyncDataCollector,
        collector_kwargs: Union[Dict, List[Dict]] = None,
        num_workers_per_collector: int = 1,
        sync: bool = False,
        ray_init_config: Dict = None,
        remote_configs: Union[Dict, List[Dict]] = None,
        num_collectors: int = None,
        update_after_each_batch=False,
        max_weight_update_interval=-1,
    ):
        if remote_configs is None:
            remote_configs = DEFAULT_REMOTE_CLASS_CONFIG

        if ray_init_config is None:
            ray_init_config = DEFAULT_RAY_INIT_CONFIG

        if collector_kwargs is None:
            collector_kwargs = {}

        # Make sure input parameters are consistent
        def check_consistency_with_num_collectors(param, param_name, num_collectors):
            """Checks that if param is a list, it has length num_collectors."""
            if isinstance(param, list):
                if len(param) != num_collectors:
                    raise ValueError(
                        f"Inconsistent RayDistributedCollector parameters, {param_name} is a list of length "
                        f"{len(param)} but the specified number of collectors is {num_collectors}."
                    )
            else:
                param = [param] * num_collectors
            return param

        if num_collectors:
            create_env_fn = check_consistency_with_num_collectors(
                create_env_fn, "create_env_fn", num_collectors
            )
            collector_kwargs = check_consistency_with_num_collectors(
                collector_kwargs, "collector_kwargs", num_collectors
            )
            remote_configs = check_consistency_with_num_collectors(
                remote_configs, "remote_config", num_collectors
            )

        def check_list_length_consistency(*lists):
            """Checks that all input lists have the same length.

            If any non-list input is given, it is converted to a list
            of the same length as the others by repeating the same
            element multiple times.
            """
            lengths = set()
            new_lists = []
            for lst in lists:
                if isinstance(lst, list):
                    lengths.add(len(lst))
                    new_lists.append(lst)
                else:
                    new_lst = [lst] * max(lengths)
                    new_lists.append(new_lst)
                    lengths.add(len(new_lst))
            if len(lengths) > 1:
                raise ValueError(
                    "Inconsistent RayDistributedCollector parameters. create_env_fn, "
                    "collector_kwargs and remote_configs are lists of different length."
                )
            else:
                return new_lists

        out_lists = check_list_length_consistency(
            create_env_fn, collector_kwargs, remote_configs
        )
        create_env_fn, collector_kwargs, remote_configs = out_lists
        num_collectors = len(create_env_fn)

        for i in range(len(create_env_fn)):
            if not isinstance(create_env_fn[i], (EnvBase, EnvCreator)):
                create_env_fn[i] = EnvCreator(create_env_fn[i])

        # If ray available, try to connect to an existing Ray cluster or start one and connect to it.
        if not _has_ray:
            raise RuntimeError(
                "ray library not found, unable to create a DistributedCollector. "
            ) from RAY_ERR
        ray.init(**ray_init_config)
        if not ray.is_initialized():
            raise RuntimeError("Ray could not be initialized.")

        # Define collector_class, monkey patch it with as_remote and print_remote_collector_info methods
        if collector_class == "async":
            collector_class = MultiaSyncDataCollector
        elif collector_class == "sync":
            collector_class = MultiSyncDataCollector
        elif collector_class == "single":
            collector_class = SyncDataCollector
        collector_class.as_remote = as_remote
        collector_class.print_remote_collector_info = print_remote_collector_info

        self._local_policy = policy
        if isinstance(self._local_policy, nn.Module):
            policy_weights = TensorDict.from_module(self._local_policy)
            policy_weights = policy_weights.data.lock_()
        else:
            warnings.warn(_NON_NN_POLICY_WEIGHTS)
            policy_weights = TensorDict(lock=True)
        self.policy_weights = policy_weights
        self.collector_class = collector_class
        self.collected_frames = 0
        self.split_trajs = split_trajs
        self.total_frames = total_frames
        self.num_collectors = num_collectors
        self.update_after_each_batch = update_after_each_batch
        self.max_weight_update_interval = max_weight_update_interval
        self.collector_kwargs = (
            collector_kwargs if collector_kwargs is not None else [{}]
        )
        self.device = device
        self.storing_device = storing_device
        self.env_device = env_device
        self.policy_device = policy_device
        self._batches_since_weight_update = [0 for _ in range(self.num_collectors)]
        self._sync = sync

        if self._sync:
            if frames_per_batch % self.num_collectors != 0:
                raise RuntimeError(
                    f"Cannot dispatch {frames_per_batch} frames across {self.num_collectors}. "
                    f"Consider using a number of frames per batch that is divisible by the number of workers."
                )
            self._frames_per_batch_corrected = frames_per_batch // self.num_collectors
        else:
            self._frames_per_batch_corrected = frames_per_batch

        # update collector kwargs
        for i, collector_kwarg in enumerate(self.collector_kwargs):
            collector_kwarg["max_frames_per_traj"] = max_frames_per_traj
            collector_kwarg["init_random_frames"] = (
                init_random_frames // self.num_collectors
            )
            if not self._sync and init_random_frames > 0:
                warnings.warn(
                    "async distributed data collection with init_random_frames > 0 "
                    "may have unforeseen consequences as we do not control that once "
                    "non-random data is being collected all nodes are returning non-random data. "
                    "If this is a feature that you feel should be fixed, please raise an issue on "
                    "torchrl's repo."
                )
            collector_kwarg["reset_at_each_iter"] = reset_at_each_iter
            collector_kwarg["exploration_type"] = exploration_type
            collector_kwarg["split_trajs"] = False
            collector_kwarg["frames_per_batch"] = self._frames_per_batch_corrected
            collector_kwarg["device"] = self.device[i]
            collector_kwarg["storing_device"] = self.storing_device[i]
            collector_kwarg["env_device"] = self.env_device[i]
            collector_kwarg["policy_device"] = self.policy_device[i]

        self.postproc = postproc

        # Create remote instances of the collector class
        self._remote_collectors = []
        if self.num_collectors > 0:
            self.add_collectors(
                create_env_fn,
                num_workers_per_collector,
                policy,
                collector_kwargs,
                remote_configs,
            )

        # Print info of all remote workers
        pending_samples = [
            e.print_remote_collector_info.remote() for e in self.remote_collectors()
        ]
        ray.wait(pending_samples)

    @property
    def num_workers(self):
        return self.num_collectors

    @property
    def device(self) -> List[torch.device]:
        return self._device

    @property
    def storing_device(self) -> List[torch.device]:
        return self._storing_device

    @property
    def env_device(self) -> List[torch.device]:
        return self._env_device

    @property
    def policy_device(self) -> List[torch.device]:
        return self._policy_device

    @device.setter
    def device(self, value):
        if isinstance(value, (tuple, list)):
            self._device = value
        else:
            self._device = [value] * self.num_collectors

    @storing_device.setter
    def storing_device(self, value):
        if isinstance(value, (tuple, list)):
            self._storing_device = value
        else:
            self._storing_device = [value] * self.num_collectors

    @env_device.setter
    def env_device(self, value):
        if isinstance(value, (tuple, list)):
            self._env_device = value
        else:
            self._env_device = [value] * self.num_collectors

    @policy_device.setter
    def policy_device(self, value):
        if isinstance(value, (tuple, list)):
            self._policy_device = value
        else:
            self._policy_device = [value] * self.num_collectors

    @staticmethod
    def _make_collector(cls, env_maker, policy, other_params):
        """Create a single collector instance."""
        collector = cls(
            env_maker,
            policy,
            total_frames=-1,
            **other_params,
        )
        return collector

    def add_collectors(
        self,
        create_env_fn,
        num_envs,
        policy,
        collector_kwargs,
        remote_configs,
    ):
        """Creates and adds a number of remote collectors to the set."""
        for env_maker, other_params, remote_config in zip(
            create_env_fn, collector_kwargs, remote_configs
        ):
            cls = self.collector_class.as_remote(remote_config).remote
            collector = self._make_collector(
                cls,
                [env_maker] * num_envs
                if self.collector_class is not SyncDataCollector
                else env_maker,
                policy,
                other_params,
            )
            self._remote_collectors.extend([collector])

    def local_policy(self):
        """Returns local collector."""
        return self._local_policy

    def remote_collectors(self):
        """Returns list of remote collectors."""
        return self._remote_collectors

    def stop_remote_collectors(self):
        """Stops all remote collectors."""
        for _ in range(len(self._remote_collectors)):
            collector = self.remote_collectors().pop()
            # collector.__ray_terminate__.remote()  # This will kill the actor but let pending tasks finish
            ray.kill(
                collector
            )  # This will interrupt any running tasks on the actor, causing them to fail immediately

    def iterator(self):
        if self._sync:
            data = self._sync_iterator()
        else:
            data = self._async_iterator()

        if self.split_trajs:
            data = split_trajectories(data)
        if self.postproc is not None:
            data = self.postproc(data)

        return data

    def _sync_iterator(self) -> Iterator[TensorDictBase]:
        """Collects one data batch per remote collector in each iteration."""
        while self.collected_frames < self.total_frames:
            if self.update_after_each_batch:
                self.update_policy_weights_()
            else:
                for j in range(self.num_collectors):
                    self._batches_since_weight_update[j] += 1

            # Ask for batches to all remote workers.
            pending_tasks = [e.next.remote() for e in self.remote_collectors()]

            # Wait for all rollouts
            samples_ready = []
            while len(samples_ready) < self.num_collectors:
                samples_ready, samples_not_ready = ray.wait(
                    pending_tasks, num_returns=len(pending_tasks)
                )

            # Retrieve and concatenate Tensordicts
            out_td = []
            for r in pending_tasks:
                rollouts = ray.get(r)
                ray.internal.free(
                    r
                )  # should not be necessary, deleted automatically when ref count is down to 0
                out_td.append(rollouts)
            if len(rollouts.batch_size):
                out_td = torch.stack(out_td)
            else:
                out_td = torch.cat(out_td)

            self.collected_frames += out_td.numel()

            yield out_td

            if self.max_weight_update_interval > -1:
                for j in range(self.num_collectors):
                    rank = j + 1
                    if (
                        self._batches_since_weight_update[j]
                        > self.max_weight_update_interval
                    ):
                        self.update_policy_weights_(rank)

        self.shutdown()

    def _async_iterator(self) -> Iterator[TensorDictBase]:
        """Collects a data batch from a single remote collector in each iteration."""
        pending_tasks = {}
        for index, collector in enumerate(self.remote_collectors()):
            future = collector.next.remote()
            pending_tasks[future] = index

        while self.collected_frames < self.total_frames:
            if not len(list(pending_tasks.keys())) == len(self.remote_collectors()):
                raise RuntimeError("Missing pending tasks, something went wrong")

            # Wait for first worker to finish
            wait_results = ray.wait(list(pending_tasks.keys()))
            future = wait_results[0][0]
            collector_index = pending_tasks.pop(future)
            collector = self.remote_collectors()[collector_index]

            # Retrieve single rollouts
            out_td = ray.get(future)
            ray.internal.free(
                [future]
            )  # should not be necessary, deleted automatically when ref count is down to 0
            self.collected_frames += out_td.numel()

            yield out_td

            for j in range(self.num_collectors):
                self._batches_since_weight_update[j] += 1
            if self.update_after_each_batch:
                self.update_policy_weights_(worker_rank=collector_index + 1)
            elif self.max_weight_update_interval > -1:
                for j in range(self.num_collectors):
                    rank = j + 1
                    if (
                        self._batches_since_weight_update[j]
                        > self.max_weight_update_interval
                    ):
                        self.update_policy_weights_(rank)

            # Schedule a new collection task
            future = collector.next.remote()
            pending_tasks[future] = collector_index

        # Wait for the in-process collections tasks to finish.
        refs = list(pending_tasks.keys())
        ray.wait(refs, num_returns=len(refs))

        # Cancel the in-process collections tasks
        # for ref in refs:
        #     ray.cancel(
        #         object_ref=ref,
        #         force=False,
        #     )

        self.shutdown()

    def update_policy_weights_(self, worker_rank=None) -> None:
        """Updates the weights of the worker nodes.

        Args:
            worker_rank (int, optional): if provided, only this worker weights
                will be updated.
        """
        # Update agent weights
        policy_weights_local_collector_ref = ray.put(self.policy_weights.detach())

        if worker_rank is None:
            for index, e in enumerate(self.remote_collectors()):
                e.update_policy_weights_.remote(policy_weights_local_collector_ref)
                self._batches_since_weight_update[index] = 0
        else:
            self.remote_collectors()[worker_rank - 1].update_policy_weights_.remote(
                policy_weights_local_collector_ref
            )
            self._batches_since_weight_update[worker_rank - 1] = 0

    def set_seed(self, seed: int, static_seed: bool = False) -> List[int]:
        """Calls parent method for each remote collector iteratively and returns final seed."""
        for collector in self.remote_collectors():
            seed = ray.get(object_refs=collector.set_seed.remote(seed, static_seed))
        return seed

    def state_dict(self) -> List[OrderedDict]:
        """Calls parent method for each remote collector and returns a list of results."""
        futures = [
            collector.state_dict.remote() for collector in self.remote_collectors()
        ]
        results = ray.get(object_refs=futures)
        return results

    def load_state_dict(
        self, state_dict: Union[OrderedDict, List[OrderedDict]]
    ) -> None:
        """Calls parent method for each remote collector."""
        if isinstance(state_dict, OrderedDict):
            state_dicts = [state_dict]
        if len(state_dict) == 1:
            state_dicts = state_dict * len(self.remote_collectors())
        for collector, state_dict in zip(self.remote_collectors(), state_dicts):
            collector.load_state_dict.remote(state_dict)

    def shutdown(self):
        """Finishes processes started by ray.init()."""
        self.stop_remote_collectors()
        ray.shutdown()

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string

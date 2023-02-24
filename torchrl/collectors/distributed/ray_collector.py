import logging
from typing import Callable, Dict, Iterator, List, OrderedDict, Union

import torch
from ray._private.services import get_node_ip_address
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    _DataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.envs import EnvBase, EnvCreator


logger = logging.getLogger(__name__)

RAY_ERR = None
try:
    import ray

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
    # logger.warning(s)
    print(s)


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


class RayDistributedCollector(_DataCollector):
    """Distributed data collector with Ray (https://docs.ray.io/) backend.

    This Python class serves as a ray-based solution to instantiate and coordinate multiple
    data collectors in a distributed cluster. Like TorchRL non-distributed collectors, this
    collector is an iterable that yields TensorDicts until a target number of collected
    frames is reached, but handles distributed data collection under the hood.

    The class dictionary input parameter "ray_init_config" can be used to provide the kwargs to
    call Ray initialization method ray.init(). If "ray_init_config" is not provided, the default
    behaviour is to autodetect an existing Ray cluster or start a new Ray instance locally if no
    existing cluster is found. Refer to Ray documentation for advanced initialization kwargs.

    Similarly, dictionary input parameter "remote_configs" can be used to specify the kwargs for
    ray.remote() when called to created each remote collector actor, including collector compute
    resources.The sum of all collector resources should be available in the cluster. Refer to Ray
    documentation for advanced configuration of the ray.remote() method. Default kwargs are:

    {
        "num_cpus": 1,
        "num_gpus": 0.2,
        "memory": 2 * 1024 ** 3,
    }


    The coordination between collector instances can be specified as "synchronous" or "asynchronous".
    In synchronous coordination, this class waits for all remote collectors to collect a rollout,
    concatenates all rollouts into a single TensorDict instance and finally yields the concatenated
    data. On the other hand, if the coordination is to be carried out asynchronously, this class
    provides the rollouts as they become available from individual remote collectors.

    Args:
        env_makers (list of callables or EnvBase instances): a list of the
            same length as the number of nodes to be launched. A single callable
            can be provides as well, and will be used in all collectors.
        policy (Callable[[TensorDict], TensorDict]): a callable that populates
            the tensordict with an `"action"` field.
        collector_class (Python class): a collector class to be remotely instantiated. Can be
            :class:`torchrl.collectors.SyncDataCollector`,
            :class:`torchrl.collectors.MultiSyncDataCollector`,
            :class:`torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these.
            Defaults to :class:`torchrl.collectors.SyncDataCollector`.
        collector_kwargs (list of dicts): kwargs to instantiate each collector_class.
            A single dict can be provides as well, and will be used in all collectors.
        num_workers_per_collector (int): the number of copies of the
            env constructor that is to be used on the remote nodes.
            Defaults to 1 (a single env per collector).
            On a single worker node all the sub-workers will be
            executing the same environment. If different environments need to
            be executed, they should be dispatched across worker nodes, not
            subnodes.
        ray_init_config (dict, Optional): kwargs used to call ray.init().
        remote_configs (list of dicts, Optional): ray resource specs for each remote collector.
            A single dict can be provides as well, and will be used in all collectors.
        num_collectors (int, Optional): total number of collectors to be instantiated.
        total_frames (int, Optional): lower bound of the total number of frames returned by the collector.
            The iterator will stop once the total number of frames equates or exceeds the total number of
            frames passed to the collector. Default value is -1, which mean no target total number of frames
            (i.e. the collector will run indefinitely).
        sync (bool): if ``True``, the resulting tensordict is a stack of all the
            tensordicts collected on each node. If ``False`` (default), each
            tensordict results from a separate node in a "first-ready,
            first-served" fashion.
        storing_device (torch.device, optional): if specified, collected tensordicts will be moved
            to this devices before returning them to the user.
        update_after_each_batch (bool, optional): if ``True``, the weights will
            be updated after each collection. For ``sync=True``, this means that
            all workers will see their weights updated. For ``sync=False``,
            only the worker from which the data has been gathered will be
            updated.
            Defaults to ``False``, ie. updates have to be executed manually
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
        >>> from torchrl.collectors.distributed.ray_collector import RayDistributedCollector
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> distributed_collector = RayDistributedCollector(
        ...     env_makers=[env_maker],
        ...     policy=policy,
        ...     collector_class=SyncDataCollector,
        ...     collector_kwargs={
        ...         "max_frames_per_traj": 50,
        ...         "init_random_frames": -1,
        ...         "reset_at_each_iter": False,
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
        env_makers: Union[Callable, EnvBase, List[Callable], List[EnvBase]],
        policy: Callable[[TensorDict], TensorDict],
        frames_per_batch: int,
        total_frames: int = -1,
        collector_class: Callable[[TensorDict], TensorDict] = SyncDataCollector,
        collector_kwargs: Union[Dict, List[Dict]] = None,
        num_workers_per_collector: int = 1,
        sync: bool = False,
        ray_init_config: Dict = None,
        remote_configs: Union[Dict, List[Dict]] = None,
        num_collectors: int = None,
        storing_device: torch.device = "cpu",
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
            env_makers = check_consistency_with_num_collectors(
                env_makers, "env_makers", num_collectors
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
                    "Inconsistent RayDistributedCollector parameters. env_makers, "
                    "collector_kwargs and remote_configs are lists of different length."
                )
            else:
                return new_lists

        out_lists = check_list_length_consistency(
            env_makers, collector_kwargs, remote_configs
        )
        env_makers, collector_kwargs, remote_configs = out_lists
        num_collectors = len(env_makers)

        for i in range(len(env_makers)):
            if not isinstance(env_makers[i], (EnvBase, EnvCreator)):
                env_makers[i] = EnvCreator(env_makers[i])

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
        self.collector_class = collector_class
        self.collected_frames = 0
        self.total_frames = total_frames
        self.num_collectors = num_collectors
        self.update_after_each_batch = update_after_each_batch
        self.max_weight_update_interval = max_weight_update_interval
        self.collector_kwargs = collector_kwargs if collector_kwargs is not None else {}
        self.storing_device = storing_device
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

        # Create remote instances of the collector class
        self._remote_collectors = []
        if self.num_collectors > 0:
            self.add_collectors(
                env_makers,
                num_workers_per_collector,
                policy,
                frames_per_batch,
                collector_kwargs,
                remote_configs,
            )

        # Print info of all remote workers
        pending_samples = [
            e.print_remote_collector_info.remote() for e in self.remote_collectors()
        ]
        ray.wait(object_refs=pending_samples)

    @staticmethod
    def _make_collector(cls, env_maker, policy, frames_per_batch, other_params):
        """Create a single collector instance."""
        collector = cls(
            env_maker,
            policy,
            total_frames=-1,
            frames_per_batch=frames_per_batch,
            split_trajs=False,
            **other_params,
        )
        return collector

    def add_collectors(
        self,
        env_makers,
        num_envs,
        policy,
        frames_per_batch,
        collector_kwargs,
        remote_configs,
    ):
        """Creates and adds a number of remote collectors to the set."""
        for env_maker, other_params, remote_config in zip(
            env_makers, collector_kwargs, remote_configs
        ):
            cls = self.collector_class.as_remote(remote_config).remote
            collector = self._make_collector(
                cls,
                [env_maker] * num_envs
                if self.collector_class is not SyncDataCollector
                else env_maker,
                policy,
                frames_per_batch,
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
            return self._sync_iterator()
        else:
            return self._async_iterator()

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
                    object_refs=pending_tasks, num_returns=len(pending_tasks)
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

            yield out_td.to(self.storing_device)

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
            wait_results = ray.wait(object_refs=list(pending_tasks.keys()))
            future = wait_results[0][0]
            collector_index = pending_tasks.pop(future)
            collector = self.remote_collectors()[collector_index]

            # Retrieve single rollouts
            out_td = ray.get(future)
            ray.internal.free(
                [future]
            )  # should not be necessary, deleted automatically when ref count is down to 0
            self.collected_frames += out_td.numel()

            yield out_td.to(self.storing_device)

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
        ray.wait(object_refs=refs, num_returns=len(refs))

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
        policy_weights_local_collector = {
            "policy_state_dict": self.local_policy().state_dict()
        }
        policy_weights_local_collector_ref = ray.put(policy_weights_local_collector)

        if worker_rank is None:
            for index, e in enumerate(self.remote_collectors()):
                e.load_state_dict.remote(
                    **{
                        "state_dict": policy_weights_local_collector_ref,
                        "strict": False,
                    }
                )
                self._batches_since_weight_update[index] = 0
        else:
            self.remote_collectors()[worker_rank - 1].load_state_dict.remote(
                **{"state_dict": policy_weights_local_collector_ref, "strict": False}
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

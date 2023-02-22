import logging
from abc import ABC
from typing import Dict, Iterator, OrderedDict, List
import torch
from torch.utils.data import IterableDataset
from ray._private.services import get_node_ip_address
from tensordict.tensordict import TensorDictBase
from torchrl.envs import EnvBase, EnvCreator
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    _DataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)


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
    "memory": 5 * 1024 ** 3,
    "object_store_memory": 2 * 1024 ** 3
}


def print_remote_collector_info(self):
    """Prints some information about the remote collector."""
    s = f"Created remote collector in machine {get_node_ip_address()} using gpus {ray.get_gpu_ids()}"
    # logger.warning(s)
    print(s)


@classmethod
def as_remote(cls,
              num_cpus=None,
              num_gpus=None,
              memory=None,
              object_store_memory=None,
              resources=None):
    """
    Creates an instance of a remote ray class.

    Args:
        cls (Python Class): class to be remotely instantiated.
        num_cpus (int): the quantity of CPU cores to reserve for this class.
        num_gpus (float): the quantity of GPUs to reserve for this class.
        memory (int): the heap memory quota for this class (in bytes).
        object_store_memory (int): the object store memory quota for this class (in bytes).
        resources (Dict[str, float]): the default resources required by the class creation task.

    Returns:
        A function that creates ray remote class instances.
    """
    remote_collector = ray.remote(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        object_store_memory=object_store_memory,
        resources=resources)(cls)
    remote_collector.is_remote = True
    return remote_collector


class RayDistributedCollector(IterableDataset, _DataCollector, ABC):
    """
    Distributed data collector with Ray (https://docs.ray.io/) backend.

    This Python class serves as a ray-based solution to instantiate and coordinate multiple
    data collectors in a distributed cluster. Like TorchRL non-distributed collector, this
    collector is an iterable that yields TensorDict until a target number of collected
    frames is reached and handles distributed data collection under the hood.

    The coordination between collector instances can be specified as synchronous or asynchronous.
    In synchronous coordination, this class waits for all remote collectors to collect a rollout,
    concatenates all rollouts into a single TensorDict instance and finally yields the  concatenated
    data. On the other hand, if the coordination is to be carried out asynchronously, this class
    provides a seamless the rollouts as they become available from individual remote collectors.

    Args:
        collector_class (Python Class): class to be remotely instantiated.
        collector_params (dict): collector_class kwargs.
        ray_init_config (dict, Optional): kwargs used to call ray.init().
        remote_config (dict, Optional): ray resource specs for the remote collectors.
        num_collectors (int, Optional): total number of collectors to be instantiated.
        total_frames (int, Optional): lower bound of the total number of frames returned by the collector.
            The iterator will stop once the total number of frames equates or exceeds the total number of
            frames passed to the collector. Default value is -1, which mean no target total number of frame.
        coordination (str): coordination pattern between collector instances (should be 'sync' or 'async')
    """

    def __init__(self,
                 policy,
                 collector_class,
                 collector_params,
                 ray_init_config=None,
                 remote_config=None,
                 num_collectors=1,
                 total_frames=-1,
                 coordination="sync",  # "sync" or "async"
                 ):

        if remote_config is None:
            remote_config = DEFAULT_REMOTE_CLASS_CONFIG

        if ray_init_config is None:
            ray_init_config = DEFAULT_RAY_INIT_CONFIG

        if not _has_ray:
            raise RuntimeError(
                f"ray library not found, unable to create a DistributedCollector. "
            ) from RAY_ERR

        if coordination not in ("sync", "async"):
            raise ValueError(f"Coordination input parameter in CollectorSet has to be sync or async.")

        # Connect to an existing Ray cluster or start one and connect to it.
        ray.init(**ray_init_config)
        if not ray.is_initialized():
            raise RuntimeError(f"Ray could not be initialized.")

        # Monkey patching as_remote and print_remote_collector_info to collector class
        collector_class.as_remote = as_remote
        collector_class.print_remote_collector_info = print_remote_collector_info
        self.collector_class = collector_class

        self.collected_frames = 0
        self.total_frames = total_frames
        self.collector_class = collector_class
        self.collector_params = collector_params
        self.num_collectors = num_collectors
        self.remote_config = remote_config
        self.coordination = coordination

        # Create a local instance of the collector class.
        # self._local_collector = self._make_collector(
        #     self.collector_class, collector_params)
        # self.local_collector().is_remote = False

        # Probably we just need a copy of the policy!
        self.local_policy = policy

        # Create remote instances of the collector class
        self._remote_collectors = []
        if self.num_collectors > 0:
            self.add_collectors(self.num_collectors, collector_params)

        # Print info of all remote workers
        pending_samples = [e.print_remote_collector_info.remote() for e in self.remote_collectors()]
        ray.wait(object_refs=pending_samples)

    @staticmethod
    def _make_collector(cls, collector_params):
        """Create a single collector instance."""
        w = cls(**collector_params)
        return w

    def add_collectors(self, num_collectors, collector_params):
        """Creates and adds a number of remote collectors to the set."""
        cls = self.collector_class.as_remote(**self.remote_config).remote
        self._remote_collectors.extend(
            [self._make_collector(cls, collector_params) for _ in range(num_collectors)])

    def local_collector(self):
        """Returns local collector."""
        return self._local_collector

    def remote_collectors(self):
        """Returns list of remote collectors."""
        return self._remote_collectors

    def stop_remote_collectors(self):
        """Stops all remote collectors."""
        for collector in self.remote_collectors():
            # collector.__ray_terminate__.remote()  # This will kill the actor but let pending tasks finish
            ray.kill(collector)  # This will interrupt any running tasks on the actor, causing them to fail immediately

    def __iter__(self) -> Iterator[TensorDictBase]:
        if self.coordination == "sync":
            return self.sync_iterator()
        else:
            return self.async_iterator()

    def sync_iterator(self) -> Iterator[TensorDictBase]:

        while self.collected_frames < self.total_frames:

            # Broadcast agent weights
            # self.local_collector().update_policy_weights_()
            # policy_weights_local_collector = {"policy_state_dict": self._local_collector.policy.state_dict()}
            policy_weights_local_collector = {"policy_state_dict": self.local_policy.state_dict()}
            policy_weights_local_collector_ref = ray.put(policy_weights_local_collector)
            for e in self.remote_collectors():
                e.load_state_dict.remote(**{"state_dict": policy_weights_local_collector_ref, "strict": False})

            # Ask for batches to all remote workers.
            pending_tasks = [e._iterator_step.remote() for e in self.remote_collectors()]

            # Wait for all rollouts
            samples_ready = []
            while len(samples_ready) < self.num_collectors:
                samples_ready, samples_not_ready = ray.wait(
                    object_refs=pending_tasks, num_returns=len(pending_tasks))

            # Retrieve and concatenate Tensordicts
            out_td = []
            for r in pending_tasks:
                rollouts = ray.get(r)
                ray.internal.free(r)  # should not be necessary, deleted automatically when ref count is down to 0
                out_td.append(rollouts)
            if len(pending_tasks[0].batch_size):
                out_td = torch.stack(out_td)
            else:
                out_td = torch.cat(out_td)

            self.collected_frames += out_td.numel()

            yield out_td

        self.stop_remote_collectors()
        ray.shutdown()

    def async_iterator(self) -> Iterator[TensorDictBase]:

        pending_tasks = {}
        for w in self.remote_collectors():
            future = w._iterator_step.remote()
            pending_tasks[future] = w

        while self.collected_frames < self.total_frames:

            if not len(list(pending_tasks.keys())) == len(self.remote_collectors()):
                raise RuntimeError("Missing pending tasks, something went wrong")

            # Wait for first worker to finish
            wait_results = ray.wait(object_refs=list(pending_tasks.keys()))
            future = wait_results[0][0]
            w = pending_tasks.pop(future)

            # Retrieve single rollouts
            out_td = ray.get(future)
            ray.internal.free(future)  # should not be necessary, deleted automatically when ref count is down to 0
            self.collected_frames += out_td.numel()

            # Update agent weights
            # self.local_collector().update_policy_weights_()
            # policy_weights_local_collector = {"policy_state_dict": self._local_collector.policy.state_dict()}
            policy_weights_local_collector = {"policy_state_dict": self.local_policy.state_dict()}
            policy_weights_local_collector_ref = ray.put(policy_weights_local_collector)
            w.load_state_dict.remote(**{"state_dict": policy_weights_local_collector_ref, "strict": False})

            # Schedule a new collection task
            future = w._iterator_step.remote()
            pending_tasks[future] = w

            yield out_td

        # Wait for the in-process collections tasks to finish.
        refs = list(pending_tasks.keys())
        ray.wait(object_refs=refs, num_returns=len(refs))

        # Cancel the in-process collections tasks
        # for ref in refs:
        #     ray.cancel(
        #         object_ref=ref,
        #         force=False,
        #     )

        self.stop_remote_collectors()
        ray.shutdown()

    def set_seed(self, seed: int, static_seed: bool = False) -> List[int]:
        """Adds a set_seed(seed, static_seed) method call for each remote collector."""
        futures = [collector.set_seed.remote(seed, static_seed) for collector in self.remote_collectors()]
        results = ray.get(object_refs=futures)
        return results

    def state_dict(self) -> List[OrderedDict]:
        """Adds a state_dict() method call for each remote collector."""
        futures = [collector.state_dict.remote() for collector in self.remote_collectors()]
        results = ray.get(object_refs=futures)
        return results

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Adds a load_state_dict(state_dict) method call for each remote collector."""
        for collector in self.remote_collectors():
            collector.load_state_dict.remote(state_dict)

    def shutdown(self):
        """Finishes processes started by ray.init()."""
        ray.shutdown()

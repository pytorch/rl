import logging
from typing import Dict, Iterator, OrderedDict, List, Union
import torch
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
    "memory": 2 * 1024 ** 3,
}


def print_remote_collector_info(self):
    """Prints some information about the remote collector."""
    s = f"Created remote collector in machine {get_node_ip_address()} using gpus {ray.get_gpu_ids()}"
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

    Similarly, dictionary input parameter "remote_config" can be used to specify the kwargs for
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
        collector_class (Python Class): class to be remotely instantiated.
        collector_kwargs (dict): collector_class kwargs.
        ray_init_config (dict, Optional): kwargs used to call ray.init().
        remote_config (dict, Optional): ray resource specs for the remote collectors.
        num_collectors (int, Optional): total number of collectors to be instantiated.
        total_frames (int, Optional): lower bound of the total number of frames returned by the collector.
            The iterator will stop once the total number of frames equates or exceeds the total number of
            frames passed to the collector. Default value is -1, which mean no target total number of frames
            (i.e. the collector will run indefinitely).
        coordination (str): coordination pattern between collector instances (should be 'sync' or 'async')

    Examples:

    """

    def __init__(self,
                 policy,
                 collector_class,
                 collector_kwargs,
                 ray_init_config=None,
                 remote_config=None,
                 num_collectors=1,
                 total_frames=-1,
                 coordination="synchronous",
                 ):

        if remote_config is None:
            remote_config = DEFAULT_REMOTE_CLASS_CONFIG

        if ray_init_config is None:
            ray_init_config = DEFAULT_RAY_INIT_CONFIG

        if not _has_ray:
            raise RuntimeError(
                f"ray library not found, unable to create a DistributedCollector. "
            ) from RAY_ERR

        if coordination not in ("synchronous", "asynchronous"):
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
        self.collector_params = collector_kwargs
        self.num_collectors = num_collectors
        self.remote_config = remote_config
        self.coordination = coordination
        self.local_policy = policy

        # Create remote instances of the collector class
        self._remote_collectors = []
        if self.num_collectors > 0:
            self.add_collectors(self.num_collectors, collector_kwargs)

        # Print info of all remote workers
        pending_samples = [e.print_remote_collector_info.remote() for e in self.remote_collectors()]
        ray.wait(object_refs=pending_samples)

    @staticmethod
    def _make_collector(cls, collector_params):
        """Create a single collector instance."""
        collector = cls(**collector_params)
        return collector

    def add_collectors(self, num_collectors, collector_params):
        """Creates and adds a number of remote collectors to the set."""
        cls = self.collector_class.as_remote(**self.remote_config).remote
        self._remote_collectors.extend(
            [self._make_collector(cls, collector_params) for _ in range(num_collectors)])

    def local_policy(self):
        """Returns local collector."""
        return self.local_policy

    def remote_collectors(self):
        """Returns list of remote collectors."""
        return self._remote_collectors

    def stop_remote_collectors(self):
        """Stops all remote collectors."""
        for collector in self.remote_collectors():
            # collector.__ray_terminate__.remote()  # This will kill the actor but let pending tasks finish
            ray.kill(collector)  # This will interrupt any running tasks on the actor, causing them to fail immediately

    def iterator(self):
        if self.coordination == "synchronous":
            return self._sync_iterator()
        else:
            return self._async_iterator()

    def _sync_iterator(self) -> Iterator[TensorDictBase]:

        while self.collected_frames < self.total_frames:

            # Broadcast agent weights
            policy_weights_local_collector = {"policy_state_dict": self.local_policy.state_dict()}
            policy_weights_local_collector_ref = ray.put(policy_weights_local_collector)
            for e in self.remote_collectors():
                e.load_state_dict.remote(**{"state_dict": policy_weights_local_collector_ref, "strict": False})

            # Ask for batches to all remote workers.
            pending_tasks = [e.next.remote() for e in self.remote_collectors()]

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
            if len(rollouts.batch_size):
                out_td = torch.stack(out_td)
            else:
                out_td = torch.cat(out_td)

            self.collected_frames += out_td.numel()

            yield out_td

        self.shutdown()

    def _async_iterator(self) -> Iterator[TensorDictBase]:

        pending_tasks = {}
        for w in self.remote_collectors():
            future = w.next.remote()
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

            yield out_td

            # Update agent weights
            policy_weights_local_collector = {"policy_state_dict": self.local_policy.state_dict()}
            policy_weights_local_collector_ref = ray.put(policy_weights_local_collector)
            w.load_state_dict.remote(**{"state_dict": policy_weights_local_collector_ref, "strict": False})

            # Schedule a new collection task
            future = w.next.remote()
            pending_tasks[future] = w

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

    def set_seed(self, seed: int, static_seed: bool = False) -> List[int]:
        """Calls set_seed(seed, static_seed) method for each remote collector and results a list of results."""
        for collector in self.remote_collectors():
            seed = ray.get(object_refs=collector.set_seed.remote(seed, static_seed))
        return seed

    def state_dict(self) -> List[OrderedDict]:
        """Calls state_dict() method for each remote collector and returns a list of results."""
        futures = [collector.state_dict.remote() for collector in self.remote_collectors()]
        results = ray.get(object_refs=futures)
        return results

    def load_state_dict(self, state_dict: Union[OrderedDict, List[OrderedDict]]) -> None:
        """Calls load_state_dict(state_dict) method for each remote collector."""
        if isinstance(state_dict, OrderedDict):
            state_dict = [state_dict]
        if len(state_dict) == 1:
            state_dict = state_dict * len(self.remote_collectors())
        for collector, state_dict in zip(self.remote_collectors(), state_dict):
            collector.load_state_dict.remote(state_dict)

    def shutdown(self):
        """Finishes processes started by ray.init()."""
        self.stop_remote_collectors()
        ray.shutdown()

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string

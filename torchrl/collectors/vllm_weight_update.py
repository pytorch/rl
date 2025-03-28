import torch
import threading

from torchrl.collectors.weight_update import RemoteWeightUpdaterBase
from torchrl.collectors.weight_update import LocalWeightUpdaterBase


VLLM_ERR = None
try:
    import vllm
    from vllm.worker.worker import Worker

    _has_vllm = True
except ImportError as err:
    _has_vllm = False
    VLLM_ERR = err

# These utilities are copied from vLLM's example code.
def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


if _has_vllm:
    # I should use worker_extension_cls arg and not inherit from worker,
    # but that is only available on main and not vLLM 0.7.3
    class WorkerExtension(Worker):
        """
        The class for vLLM's worker to inherit from.
        By defining an extension class, the code can work no matter what is
        the underlying worker class. This way, the code can be compatible
        with both vLLM V0 and V1.
        NOTE: we define this class in a separate module, and the main module
        should pass the full qualified name as `worker_extension_cls` argument.
        """

        def init_weight_update_group(self, master_address, master_port,
                                    rank_offset, world_size):
            from vllm.distributed.parallel_state import get_world_group
            # rank = get_world_group().rank + rank_offset
            rank = rank_offset
            self.model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )
            self.version = torch.tensor([0], device="cuda")

        def update_weight(self, name, dtype, shape):
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            self.model_update_group.broadcast(weight,
                                              src=0,
                                              stream=torch.cuda.current_stream())
            
            self.model_runner.model.load_weights(weights=[(name, weight)])

            del weight
    
        def update_policy_version(self):
            self.model_update_group.broadcast(self.version,
                                              src=0,
                                              stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            # print(f"{self=} {self.model_runner.model=}")
            self.policy_version = self.version
        
        def check_weights_changed(self):
            """
            Check if the weights are updated to 0.
            """
            weights_updated = True
            for name, p in self.model_runner.model.named_parameters():
                weights_updated = weights_updated and torch.allclose(
                    p, torch.zeros_like(p))
            return weights_updated
else:
    class WorkerExtension:
        pass


class vLLMHFLocalWeightUpdater(LocalWeightUpdaterBase):
    def __init__(self, master_address, master_port, model_metadata):
        print(f"{master_address=}, {master_port=}")
        self.master_address = master_address
        self.master_port = master_port
        self.model_metadata = model_metadata
        self.initialized_group = None

    def _get_server_weights(self):
        return None

    def _get_local_weights(self):
        # We don't implement this because we let vLLM's update_weights API handle everything for now
        return None
    
    def _maybe_map_weights(self, server_weights, local_weights):
        # vLLM update_weights function handles the mapping from huggingface
        # so we don't implement this for now
        return None
    
    def _update_local_weights(self, local_weights, mapped_weights):
        llm = self.collector.policy["generate"].module
        if self.initialized_group is None:
            weight_sync_world_size = llm.llm_engine.parallel_config.tensor_parallel_size + 1
            llm.collective_rpc(
                "init_weight_update_group",
                args=(self.master_address, self.master_port, 1, weight_sync_world_size)
            )
            self.initialized_group = True
        
        for k, (dtype, shape) in self.model_metadata.items():
            llm.collective_rpc(
                "update_weight",
                args=(k, dtype, shape)
            )
    
        llm.collective_rpc("update_policy_version")
        print("done local update_weight")

class ReadWriteLock:
    """ A lock object that allows many simultaneous "read locks", but
    only one "write lock." """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        """ Acquire a read lock. Blocks only if a thread has
        acquired the write lock. """
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """ Release a read lock. """
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """ Acquire a write lock. Blocks until there are no
        acquired read or write locks. """
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """ Release a write lock. """
        self._read_ready.release()

class vLLMRemoteWeightUpdaterBase(RemoteWeightUpdaterBase):
    def __init__(self, vllm_master_addresses, vllm_master_ports):
        super().__init__()
        from transformers import AutoModel
        self.vllm_master_addresses = vllm_master_addresses
        self.vllm_master_ports = vllm_master_ports
        # state_dict = dict()
        # for k, (dtype, shape)  in model_metadata.items():
        #     self.state_dict[k] = torch.zeros(shape, dtype=dtype, device="cuda")
        # self.state_dict = state_dict()
        # self.state_dict_lock = ReadWriteLock()
        self.vllm_comm_groups = dict()
        self.vllm_weight_versions = dict()
        # self.version = -1

    def register_model_metadata(self, model_metadata):
        self.model_metadata = model_metadata
        self.state_dict = dict()
        for k, (dtype, shape)  in model_metadata.items():
            self.state_dict[k] = torch.zeros(shape, dtype=dtype, device="cuda")
        self.state_dict_lock = ReadWriteLock()
        self.version = 0
        self.version_tensor = torch.tensor([0], device="cuda")

    def acquire_state_dict_lock(self):
        self.state_dict_lock.acquire_write()
    
    def release_state_dict_lock(self):
        self.version += 1
        self.version_tensor += 1
        torch.cuda.synchronize()
        self.state_dict_lock.release_write()
    
    def all_worker_ids(self):
        return [i for i in range(len(self.collector._remote_collectors))]
        
    def _get_server_weights(self):
        return self.state_dict
    
    def _maybe_map_weights(self, server_weights):
        return server_weights
    
    def _skip_update(self, worker_id):
        if self.version == 0:
            return True
        if worker_id not in self.vllm_weight_versions:
            return False
        if self.vllm_weight_versions[worker_id] == self.version:
            print(f"skipping update for {worker_id=}, {self.version=}, {self.vllm_weight_versions[worker_id]=}")
            return True
        return False
    
    def _init_model_update_group(self, worker_id):
        # here again, I want to grab the tp size from the vLLM worker... :(
        # llm.llm_engine.parallel_config.tensor_parallel_size
        vllm_tp_size = 1
        weight_sync_world_size = vllm_tp_size + 1
        print("before stateless_init_process_group")
        model_update_group = stateless_init_process_group(
            self.vllm_master_addresses[worker_id],
            self.vllm_master_ports[worker_id],
            0,
            weight_sync_world_size,
            torch.device("cuda:0"),
        )
        print("after stateless_init_process_group")
        self.vllm_comm_groups[worker_id] = model_update_group

    def _sync_weights_with_worker(
        self, worker_id: int, server_weights
    ):
        print(f"in _sync_weights_with_worker {worker_id}", flush=True)
        self.collector._remote_collectors[worker_id].update_policy_weights_.remote()
        if worker_id not in self.vllm_comm_groups:
            print("init model update group")
            self._init_model_update_group(worker_id)
        print("done init model update group")
        self.state_dict_lock.acquire_read()
        for i, k in enumerate(server_weights.keys()):
            # if i == 0:
            #     print(f"{server_weights[k][0]=}")
            self.vllm_comm_groups[worker_id].broadcast(server_weights[k], src=0, stream=torch.cuda.current_stream())
        self.vllm_comm_groups[worker_id].broadcast(self.version_tensor, src=0, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        print(f"_sync_weights_with_worker done broadcast {worker_id} {self.version=}")
        self.vllm_weight_versions[worker_id] = self.version
        self.state_dict_lock.release_read()
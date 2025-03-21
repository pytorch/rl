import ray

from argparse import ArgumentParser
from functools import partial

import torch
from datasets import load_dataset
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.collectors.weight_update import RayRemoteWeightUpdater
from transformers import AutoTokenizer, AutoModel
from vllm import LLM

from vllm.utils import get_ip, get_open_port

from vllm.worker.worker import Worker

from torchrl.collectors.distributed import RayCollector
from torchrl.envs import LLMEnv
from torchrl.modules import from_vllm

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--steps_per_batch", type=int, default=16)
parser.add_argument("--optim_batch_size", type=int, default=4)

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


# I should use worker_extension_cls arg and not inherit from worker,
# but that is only available on main and not 0.7.3
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
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
    
    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


def make_policy():
    inference_model = LLM(
        "facebook/opt-125m",
        enforce_eager=True,
        # change to worker_extension_cls when available in stable release
        worker_cls=WorkerExtension,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy = from_vllm(
        inference_model, tokenizer=tokenizer, from_text=False, generate=True, return_log_probs=True, generate_kwargs={"temperature": 0.0})
    return policy


def make_env(dataset, batch_size):
    dataset = load_dataset(dataset, "main")
    train_dataset = dataset["train"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Env
    dataloader = DataLoader(  # noqa: TOR401
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    env = LLMEnv.from_dataloader(
        dataloader=dataloader,
        tokenizer=tokenizer,
        str2str=True,
        batch_size=(args.batch_size * args.repeats,),
        repeats=args.repeats, )
    return env


def collate_fn(batch):
    batch = torch.stack([TensorDict.from_dict(_batch) for _batch in batch])
    batch.rename_key_("question", "text")
    return batch

@ray.remote(num_cpus=1, num_gpus=1)
class TrainerActor:
    def __init__(self, env_vars):
        import os
        import torch
        import torch.distributed
        from torch.distributed._composable.fsdp import fully_shard

        torch.cuda.set_device(torch.device('cuda', 0))
    
        print(os.environ["CUDA_VISIBLE_DEVICES"])

        for var in env_vars:
            os.environ[var] = str(env_vars[var])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", device_id=torch.device('cuda:0'))
            print("initialized process group")

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print(world_size, rank)
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # self.param_server_comm_group = None
        # if self.rank == 0:
        #     self.param_server_comm_group = torch.distributed.new_group(ranks=[0, self.world_size - 1], use_local_synchronization=True)

        # hold back one rank for the parameter server
        self.fsdp_group = torch.distributed.new_group(ranks=list(range(self.world_size - 1)))
        self.comm_group = torch.distributed.new_group(ranks=[0, 2])
        self.device_mesh = torch.distributed.device_mesh.DeviceMesh.from_group(self.fsdp_group, device_type="cuda") 

        self.model = AutoModel.from_pretrained("facebook/opt-125m").cuda()

        fully_shard(self.model, mesh=self.device_mesh)
    
    def register_parameter_server(self, param_server):
        # assert self.rank == 0
        self.param_server = param_server
    
    def send_weights_to_param_server(self):
        # assert(hasattr(self, "param_server"))
        for k, v in self.model.state_dict().items():
            replicated_v = v.full_tensor()
            # dst is global rank, can switch to group_dst arg if not 2.5.1
            if self.rank == 0:
                # print(f"sending {k}, {replicated_v.nbytes}")
                handle = self.param_server.receive_from_train.remote(k)
                torch.distributed.send(replicated_v, dst=2)
            # ray.get(handle)
    
    def zero_(self):
        sd = self.model.state_dict()
        for k, v in sd.items():
            sd[k] = v.data.zero_()
    
    def train(self):
        import time
        for _ in range(1):
            # actually run train loop
            # ...
            self.zero_()
            torch.distributed.barrier(group=self.fsdp_group)
            print("done barrier!")
            # if self.rank == 0:
            #     print("starting send weights")
            self.send_weights_to_param_server()
            torch.distributed.barrier(group=self.fsdp_group)


from torchrl.collectors.weight_update import RemoteWeightUpdaterBase

@ray.remote(num_cpus=1, num_gpus=1)
class vLLMParameterServer(RemoteWeightUpdaterBase):
    def __init__(self, env_vars):
        import os
        import torch
        import torch.distributed

        torch.cuda.set_device(torch.device('cuda', 0))

        for var in env_vars:
            os.environ[var] = str(env_vars[var])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", device_id=torch.device('cuda:0'))
            print("initialized process group")

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print(world_size, rank)
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        assert self.rank == self.world_size - 1

        self.fsdp_group = torch.distributed.new_group(ranks=list(range(self.world_size - 1)))
        self.comm_group = torch.distributed.new_group(ranks=[0, 2])

        # self.param_server_trainer_comm_group = torch.distributed.new_group(ranks=[0, self.world_size - 1], use_local_synchronization=True)
        
        self.param_server_vllm_comm_groups = dict()

        # Having the state_dict fit on one GPU will not scale
        self.state_dict = AutoModel.from_pretrained("facebook/opt-125m").cuda().eval().state_dict()

        self.lock = torch.multiprocessing.Lock()
        self.version = 0

        print(self.state_dict.keys())

    def receive_from_train(self, k):
        # with self.lock:
            # src is global rank, an change to group_src once not 2.5.1
        # print(f"receiving {k}")
        torch.distributed.recv(self.state_dict[k], src=0)
        # self.version += 1
        # print(f"received {k} {self.state_dict[k].flatten()[0]}")
    
    def _init_model_update_group(self, worker_id):
        print("in init model update group", worker_id)
        master_address, master_port = get_ip(), get_open_port()
        print(master_address, master_port)
        # FIXME!!!! This needs to be grabbed from each remote collector
        vllm_tp_size = 1
        weight_sync_world_size = vllm_tp_size + 1
        print("calling collective_rpc")
        self.collector._remote_collectors[worker_id].call_policy_method.remote(
            "collective_rpc",
            ("init_weight_update_group",),
            {'args': (master_address, master_port, 1, weight_sync_world_size)}
        )
        print("done collective_rpc")
        model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            0,
            weight_sync_world_size,
            torch.device("cuda:0"),
        )
        print("done stateless init process group")
        self.param_server_vllm_comm_groups[worker_id] = model_update_group


    def _sync_weights_with_worker(
        self, worker_id: int, server_weights
    ):
        if worker_id not in self.param_server_vllm_comm_groups:
            self._init_model_update_group(worker_id)
        handles = []
        for i, (k, v) in enumerate(server_weights.items()):
            handle = self.collector._remote_collectors[worker_id].call_policy_method.remote(
                "collective_rpc",
                ("update_weight",),
                {'args': (k, v.dtype, v.shape)}
            )
            handles.append(handle)
            # self.collector._remote_collectors[worker_id].collective_rpc.remote("update_weight", args=(k, v.dtype, v.shape))
            self.param_server_vllm_comm_groups[worker_id].broadcast(server_weights[k], src=0, stream=torch.cuda.current_stream())
        handle = self.collector._remote_collectors[worker_id].call_policy_method.remote(
            "collective_rpc",
            ("check_weights_changed",),
            {},
        )

        print(f"weights changed {ray.get(handle)}")
        # probably no need barrier because subsequent gpu work should be serialized
        # self._batches_since_weight_update[worker_id] = 0
    
    def _get_server_weights(self):
        print("in _get_server_weights")
        with self.lock:
            return self.state_dict
    
    def _maybe_map_weights(self, server_weights):
        # This is making a design choice that weight mapping always happens on the parameter servver
        # I don't think we should make this design choice so early.
        return server_weights
    
    def all_worker_ids(self):
        return [0]

    def _skip_update(self, worker_id: int) -> bool:
        pass
    
    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.state_dict.items():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


def _create_trainer_group(worker_cls, param_server_cls, world_size: int):
    addr, port = get_ip(), get_open_port()
    trainer_workers = []
    fsdp_world_size = world_size - 1
    for i in range(fsdp_world_size):
        env_vars = {
            "RANK": str(i),
            "WORLD_SIZE": world_size,
            "MASTER_ADDR": str(addr),
            "MASTER_PORT": str(port),
        }
        worker = worker_cls.remote(env_vars)
        trainer_workers.append(worker)
    
    env_vars = {
        "RANK": str(world_size - 1),
        "WORLD_SIZE": world_size,
        "MASTER_ADDR": str(addr),
        "MASTER_PORT": str(port),
    }
    parameter_server = param_server_cls.remote(env_vars)
    trainer_workers[0].register_parameter_server.remote(parameter_server)
    trainer_workers[1].register_parameter_server.remote(parameter_server)
    return trainer_workers, parameter_server


if __name__ == "__main__":
    args = parser.parse_args()

    remote_configs = {
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 2 * 1024**3,
    }

    ray.init(num_cpus=4, num_gpus=4)

    trainer_workers, parameter_server = _create_trainer_group(TrainerActor, vLLMParameterServer, 3)

    handles = []
    for trainer_worker in trainer_workers:
        handles.append(trainer_worker.train.remote())
    

    print(f"param server weights updated {ray.get(parameter_server.check_weights_changed.remote())}")

    make_env_parsed = partial(make_env, batch_size=args.batch_size, dataset=args.dataset)
    collector = RayCollector(
        [make_env_parsed],
        policy_factory=make_policy,
        frames_per_batch=40,
        total_frames=200,
        remote_configs=remote_configs,
        remote_weights_updater=parameter_server,
        update_after_each_batch=True,
    )
    print("done collector init")

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    for i, data in enumerate(collector):
        print(tokenizer.decode(data["tokens"][0].squeeze()))
        print(tokenizer.decode(data["tokens_response"][0].squeeze()))
        if i == 1:
            break
    collector.shutdown()
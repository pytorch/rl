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

from torchrl.collectors.distributed import RayCollector
from torchrl.envs import LLMEnv
from torchrl.modules import from_vllm

from torchrl.collectors.vllm_weight_update import vLLMHFLocalWeightUpdater, vLLMRemoteWeightUpdaterBase, WorkerExtension

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--steps_per_batch", type=int, default=16)
parser.add_argument("--optim_batch_size", type=int, default=4)


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
    def __init__(self, model, env_vars):
        import os
        import torch
        import torch.distributed
        from torch.distributed._composable.fsdp import fully_shard

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


        # hold back one rank for the parameter server
        self.fsdp_group = torch.distributed.new_group(ranks=list(range(self.world_size - 1)))
        self.device_mesh = torch.distributed.device_mesh.DeviceMesh.from_group(self.fsdp_group, device_type="cuda") 

        self.model = AutoModel.from_pretrained(model).cuda()

        fully_shard(self.model, mesh=self.device_mesh)
    
    def register_parameter_server(self, param_server):
        assert self.rank == 0
        self.param_server = param_server
    
    def send_weights_to_param_server(self):
        if self.rank == 0:
            ray.get(self.param_server.acquire_state_dict_lock.remote())
            self.param_server.receive_from_trainer.remote()
        for k, v in self.model.state_dict().items():
            replicated_v = v.full_tensor()
            if self.rank == 0:
                # dst is global rank, can switch to group_dst arg if not 2.5.1
                torch.distributed.send(replicated_v, dst=2)
        if self.rank == 0:
            ray.get(self.param_server.release_state_dict_lock.remote())
    
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
            self.send_weights_to_param_server()
            torch.distributed.barrier(group=self.fsdp_group)


@ray.remote(num_cpus=1, num_gpus=1)
class vLLMParameterServer(vLLMRemoteWeightUpdaterBase):
    def __init__(self, model, vllm_master_address, vllm_master_port, env_vars):
        super().__init__(model, vllm_master_address, vllm_master_port)
        import os
        import torch
        import torch.distributed

        torch.cuda.set_device(torch.device('cuda', 0))

        for var in env_vars:
            os.environ[var] = str(env_vars[var])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", device_id=torch.device('cuda:0'))

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        assert self.rank == self.world_size - 1

        self.fsdp_group = torch.distributed.new_group(ranks=list(range(self.world_size - 1)))
    
    def receive_from_trainer(self):
        for k, v in self.state_dict.items():
            torch.distributed.recv(v, src=0)

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



def _create_trainer_group(
    worker_cls,
    param_server_cls,
    world_size: int,
    vllm_master_address,
    vllm_master_port,
    model,
):
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
        worker = worker_cls.remote(model, env_vars)
        trainer_workers.append(worker)
    
    env_vars = {
        "RANK": str(world_size - 1),
        "WORLD_SIZE": world_size,
        "MASTER_ADDR": str(addr),
        "MASTER_PORT": str(port),
    }
    parameter_server = param_server_cls.remote(model, vllm_master_address, vllm_master_port, env_vars)
    trainer_workers[0].register_parameter_server.remote(parameter_server)
    return trainer_workers, parameter_server


if __name__ == "__main__":
    args = parser.parse_args()

    remote_configs = {
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 2 * 1024**3,
    }

    model = "facebook/opt-125m"

    ray.init(num_cpus=5, num_gpus=5)

    vllm_addresses = [get_ip()] * 2
    vllm_ports = [get_open_port() for i in range(2)]
    print(vllm_ports)

    trainer_workers, parameter_server = _create_trainer_group(
                                            TrainerActor,
                                            vLLMParameterServer,
                                            3,
                                            vllm_addresses,
                                            vllm_ports,
                                            model,
                                        )

    handles = []
    for trainer_worker in trainer_workers:
        handles.append(trainer_worker.train.remote())

    model_metadata = ray.get(parameter_server.get_model_metadata.remote())
    local_weight_updaters = [
        vLLMHFLocalWeightUpdater(vllm_master_address, vllm_update_port, model_metadata) for
        vllm_master_address, vllm_update_port in zip(vllm_addresses, vllm_ports)
    ]

    make_env_parsed = partial(make_env, batch_size=args.batch_size, dataset=args.dataset)
    collector = RayCollector(
        [make_env_parsed, make_env_parsed],
        policy_factory=make_policy,
        frames_per_batch=40,
        total_frames=200,
        remote_configs=remote_configs,
        remote_weight_updater=parameter_server,
        num_collectors=2,
        collector_kwargs=[
            {
                "local_weight_updater": local_weight_updaters[0],
            },
            {
                "local_weight_updater": local_weight_updaters[1],
            }
        ],
        update_after_each_batch=True,
    )
    print("done collector init")

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    for i, data in enumerate(collector):
        print(tokenizer.decode(data["tokens"][0].squeeze()))
        print(tokenizer.decode(data["tokens_response"][0].squeeze()))
        if i == 3:
            break
    collector.shutdown()
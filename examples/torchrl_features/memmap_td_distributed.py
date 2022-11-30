# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import configargparse
import torch
import torch.distributed.rpc as rpc
from tensordict import TensorDict
from tensordict.memmap import set_transfer_ownership

parser = configargparse.ArgumentParser()
parser.add_argument("--world_size", default=2, type=int)
parser.add_argument("--rank", default=-1, type=int)
parser.add_argument("--task", default=1, type=int)
parser.add_argument("--rank_var", default="SLURM_JOB_ID", type=str)
parser.add_argument(
    "--master_addr",
    type=str,
    default="localhost",
    help="""Address of master, will default to localhost if not provided.
    Master must be able to accept network traffic on the address + port.""",
)
parser.add_argument(
    "--master_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
    provided. Master must be able to accept network traffic on the host and port.""",
)
parser.add_argument("--memmap", action="store_true")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--shared_mem", action="store_true")

AGENT_NAME = "main"
OBSERVER_NAME = "worker{}"


def get_tensordict():
    return tensordict


def tensordict_add():
    tensordict.set_("a", tensordict.get("a") + 1)
    tensordict.set("b", torch.zeros(*SIZE))
    if tensordict.is_memmap():
        td = tensordict.clone().apply_(set_transfer_ownership)
        return td
    return tensordict


def tensordict_add_noreturn():
    tensordict.set_("a", tensordict.get("a") + 1)
    tensordict.set("b", torch.zeros(*SIZE))


SIZE = (32, 50, 3, 84, 84)

if __name__ == "__main__":
    args = parser.parse_args()
    rank = args.rank
    if rank < 0:
        rank = int(os.environ[args.rank_var])
    print("rank: ", rank)
    world_size = args.world_size

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    str_init_method = "tcp://localhost:10000"
    options = rpc.TensorPipeRpcBackendOptions(
        _transports=["uv"], num_worker_threads=16, init_method=str_init_method
    )

    if rank == 0:
        # rank0 is the trainer
        rpc.init_rpc(
            AGENT_NAME,
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )

        if args.task == 0:
            time.sleep(1)
            t0 = time.time()
            for w in range(1, args.world_size):
                fut0 = rpc.rpc_async(f"worker{w}", get_tensordict, args=())
                fut0.wait()
                fut1 = rpc.rpc_async(f"worker{w}", tensordict_add, args=())
                tensordict2 = fut1.wait()
                tensordict2.clone()
            print("time: ", time.time() - t0)
        elif args.task == 1:
            time.sleep(1)
            t0 = time.time()
            waiters = [
                rpc.remote(f"worker{w}", get_tensordict, args=())
                for w in range(1, args.world_size)
            ]
            td = torch.stack([waiter.to_here() for waiter in waiters], 0).contiguous()
            print("time: ", time.time() - t0)

            t0 = time.time()
            waiters = [
                rpc.remote(f"worker{w}", tensordict_add, args=())
                for w in range(1, args.world_size)
            ]
            td = torch.stack([waiter.to_here() for waiter in waiters], 0).contiguous()
            print("time: ", time.time() - t0)
            assert (td[:, 3].get("a") == 1).all()
            assert (td[:, 3].get("b") == 0).all()

        elif args.task == 2:
            time.sleep(1)
            t0 = time.time()
            # waiters = [rpc.rpc_async(f"worker{w}", get_tensordict, args=()) for w in range(1, args.world_size)]
            waiters = [
                rpc.remote(f"worker{w}", get_tensordict, args=())
                for w in range(1, args.world_size)
            ]
            # td = torch.stack([waiter.wait() for waiter in waiters], 0).clone()
            td = torch.stack([waiter.to_here() for waiter in waiters], 0)
            print("time to receive objs: ", time.time() - t0)
            t0 = time.time()
            if args.memmap:
                waiters = [
                    rpc.remote(f"worker{w}", tensordict_add_noreturn, args=())
                    for w in range(1, args.world_size)
                ]
                print("temp t: ", time.time() - t0)
                [
                    waiter.to_here() for waiter in waiters
                ]  # the previous stack will track the original files
                print("temp t: ", time.time() - t0)
            else:
                waiters = [
                    rpc.remote(f"worker{w}", tensordict_add, args=())
                    for w in range(1, args.world_size)
                ]
                print("temp t: ", time.time() - t0)
                td = torch.stack([waiter.to_here() for waiter in waiters], 0)
                print("temp t: ", time.time() - t0)
            assert (td[:, 3].get("a") == 1).all()
            assert (td[:, 3].get("b") == 0).all()
            print("time to receive updates: ", time.time() - t0)

        elif args.task == 3:
            time.sleep(1)
            t0 = time.time()
            waiters = [
                rpc.remote(f"worker{w}", get_tensordict, args=())
                for w in range(1, args.world_size)
            ]
            td = torch.stack([waiter.to_here() for waiter in waiters], 0)
            print("time to receive objs: ", time.time() - t0)
            t0 = time.time()
            waiters = [
                rpc.remote(f"worker{w}", tensordict_add, args=())
                for w in range(1, args.world_size)
            ]
            print("temp t: ", time.time() - t0)
            td = torch.stack([waiter.to_here() for waiter in waiters], 0)
            print("temp t: ", time.time() - t0)
            if args.memmap:
                print(td[0].get("a").filename)
                print(td[0].get("a").file)
                print(td[0].get("a")._has_ownership)

            print("time to receive updates: ", time.time() - t0)
            assert (td[:, 3].get("a") == 1).all()
            assert (td[:, 3].get("b") == 0).all()
            print("time to read one update: ", time.time() - t0)

    else:

        global tensordict
        # other ranks are the observer
        tensordict = TensorDict(
            {
                "a": torch.zeros(*SIZE),
                "b": torch.randn(*SIZE),
            },
            batch_size=SIZE[:1],
        )
        if args.memmap:
            tensordict.memmap_()
            if rank == 1:
                print(tensordict.get("a").filename)
                print(tensordict.get("a").file)
        if args.shared_mem:
            tensordict.share_memory_()
        elif args.cuda:
            tensordict = tensordict.cuda()
        rpc.init_rpc(
            OBSERVER_NAME.format(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )

    rpc.shutdown()

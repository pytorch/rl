# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import configargparse
import torch
import torch.distributed.rpc as rpc
from tensordict import MemmapTensor

parser = configargparse.ArgumentParser()
parser.add_argument("--rank", default=-1, type=int)
parser.add_argument("--world_size", default=2, type=int)
parser.add_argument("--tensortype", default="memmap", type=str)

AGENT_NAME = "main"
OBSERVER_NAME = "worker{}"

str_init_method = "tcp://localhost:10000"
options = rpc.TensorPipeRpcBackendOptions(
    _transports=["uv"], num_worker_threads=16, init_method=str_init_method
)

global tensor


def send_tensor(t):
    global tensor
    tensor = t
    print(tensor)


def op_on_tensor(idx):
    tensor[idx] += 1
    if isinstance(tensor, torch.Tensor):
        return tensor


if __name__ == "__main__":
    args = parser.parse_args()
    rank = args.rank
    world_size = args.world_size
    tensortype = args.tensortype

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if rank == 0:
        rpc.init_rpc(
            AGENT_NAME,
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        # create tensor
        tensor = torch.zeros(10000, 10000)
        if tensortype == "memmap":
            tensor = MemmapTensor(tensor)
        elif tensortype == "tensor":
            pass
        else:
            raise NotImplementedError

        #  send tensor
        w = 1
        fut0 = rpc.remote(f"worker{w}", send_tensor, args=(tensor,))
        fut0.to_here()

        #  execute
        t0 = time.time()
        idx = 10
        for i in range(100):
            fut1 = rpc.remote(f"worker{w}", op_on_tensor, args=(idx,))
            tensor_out = fut1.to_here()

            if tensortype == "memmap":
                assert (tensor[idx] == i + 1).all()
            else:
                assert (tensor_out[idx] == i + 1).all()
        print(f"{tensortype}, time spent: {time.time() - t0: 4.4f}")

    else:
        rpc.init_rpc(
            OBSERVER_NAME.format(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )

    rpc.shutdown()

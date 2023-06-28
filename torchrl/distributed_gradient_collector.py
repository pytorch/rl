# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Distributed gradient-collector using torch.distributed backend."""


import os
import logging
import warnings
from datetime import timedelta
from torch.utils.data import IterableDataset
from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional
from torch.optim import Optimizer, Adam


import torch
from torch import multiprocessing as mp, nn
from torchrl._utils import VERBOSE
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    DataCollectorBase,
    DEFAULT_EXPLORATION_TYPE,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.objectives import LossModule
from torchrl.collectors.utils import split_trajectories
from torchrl.envs import EnvBase, EnvCreator

TCP_PORT = os.environ.get("TCP_PORT", "10003")
MAX_TIME_TO_CONNECT = 1000


def get_params_and_grad(loss_module: LossModule):
    params = TensorDict(dict(loss_module.named_parameters()), [])

    def set_grad(p):
        p.grad = torch.zeros_like(p.data)
        return p

    params.apply(set_grad)
    grad = params.apply(lambda p: p.grad)
    return params, grad


def _run_gradient_worker(
        rank: int,
        world_size: int,
        loss_module: LossModule,
        rank0_ip: str,
        tcpport: str,
        backend: str = "gloo",
        verbose: bool = True,
):
    """Run a gradient worker."""
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = str(tcpport)

    if verbose:
        print(
            f"Rank0 IP address: '{rank0_ip}' \ttcp port: '{tcpport}', backend={backend}."
        )
        print(
            f"node with rank {rank} with world_size {world_size} -- launching distributed"
        )
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(MAX_TIME_TO_CONNECT),
        init_method=f"tcp://{rank0_ip}:{tcpport}",
    )

    loss_module = create_model()  # TODO: testing only
    params, grad = get_params_and_grad(loss_module)

    # while True:
    for _ in range(10):

        # pretend we compute something here
        grad.apply_(lambda x: torch.ones_like(x))
        grad.reduce(0)  # send grads to server, operation is SUM

        # receive latest params
        params.irecv(src=0)
        print(f"agent {rank} received params from server")


class DistributedGradientCollector:
    """Distributed gradient collector with torch.distributed backend..

        This Python class serves as a solution to instantiate and coordinate multiple
    gradient workers in a distributed cluster. This class is an iterable that yields
    TensorDicts with gradients until a target number of collected frames is reached.
    """

    _VERBOSE = VERBOSE  # for debugging

    def __init__(
        self,
        loss_module: LossModule,
        *,
        backend="gloo",
        launcher="mp",  # For now, only support multiprocessing
        tcp_port=None,
    ):

        self.num_workers = 2
        self.backend = backend
        self.loss_module = loss_module
        if tcp_port is None:
            self.tcp_port = os.environ.get("TCP_PORT", TCP_PORT)
        else:
            self.tcp_port = str(tcp_port)

        self._init_workers()

    def _init_workers(self):
        # Init both the local and remote workers
        IPAddr = "localhost"
        if self._VERBOSE:
            print("Server IP address:", IPAddr)
        self.IPAddr = IPAddr
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = str(self.tcp_port)

        self.jobs = []
        for i in range(self.num_workers):
            if self._VERBOSE:
                print("Submitting job")
            job = self._init_worker_dist_mp(i)
            if self._VERBOSE:
                print("job launched")
            self.jobs.append(job)
        self._init_master_dist(self.num_workers + 1, self.backend)

    def _init_master_dist(
        self,
        world_size,
        backend,
    ):
        if self._VERBOSE:
            print(
                f"launching main node with tcp port '{self.tcp_port}' and "
                f"IP '{self.IPAddr}'. rank: 0, world_size: {world_size}, backend={backend}."
            )
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = str(self.tcp_port)

        TCP_PORT = self.tcp_port
        torch.distributed.init_process_group(
            backend,
            rank=0,
            world_size=world_size,
            timeout=timedelta(MAX_TIME_TO_CONNECT),
            init_method=f"tcp://{self.IPAddr}:{TCP_PORT}",
        )

        loss_module = create_model()  # TODO: testing only
        self.params, self.grad = get_params_and_grad(loss_module)

        if self._VERBOSE:
            print("main initiated!", end="\t")

    def _init_worker_dist_mp(self, i):
        TCP_PORT = self.tcp_port
        job = mp.Process(
            target=_run_gradient_worker,
            args=(
                i + 1,
                self.num_workers + 1,
                self.loss_module,
                self.IPAddr,
                int(TCP_PORT),
                self.backend,
                self._VERBOSE,
            ),
        )
        job.start()
        return job

    def _get_params_and_grads(self, model):
        # Equivalent to creating the local storage where grads will be sent
        pass

    def iterator(self):

        for _ in range(10):
            # collect gradients from workers
            print(f"server received grads from agents")
            self.grad.reduce(0, op=torch.distributed.ReduceOp.SUM)  # see reduce doc to see what ops are supported
            self.grad.apply_(lambda x: x / 2)  # average grads
            print(self.grad['weight'])

            yield self.grad

            # update params and send updated version to workers
            self.params.apply_(lambda p, g: p.data.copy_(g), self.grad)
            self.params.isend(dst=1)
            self.params.isend(dst=2)

    def _iterator_dist(self):
        if self._VERBOSE:
            print("iterating...")

    def compute_gradients(self, data: TensorDict):
        # Split the data, send it to the workers, receiving grads and averaging them
        pass

    def update_policy_weights_(self, worker_rank=None) -> None:
        """Updates the weights of the worker nodes.

        Args:
            worker_rank (int, optional): if provided, only this worker weights
                will be updated.
        """

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        if self._VERBOSE:
            print("gradient collector shut down")


if __name__ == "__main__":
    def create_model():
        model = nn.Linear(3, 4)
        model.weight.data.fill_(0.0)
        model.bias.data.fill_(0.0)
        return model

    grad_collector = DistributedGradientCollector(
        loss_module=None,  # TODO: testing only
    )

    for grads in grad_collector:
        import ipdb; ipdb.set_trace()

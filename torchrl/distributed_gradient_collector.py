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
from torchrl.objectives.value.advantages import ValueEstimatorBase
from torchrl.collectors.utils import split_trajectories
from torchrl.envs import EnvBase, EnvCreator

TCP_PORT = os.environ.get("TCP_PORT", "10003")
MAX_TIME_TO_CONNECT = 1000


def create_model():
    model = nn.Linear(3, 4)
    model.weight.data.fill_(0.0)
    model.bias.data.fill_(0.0)
    return model


def get_weights_and_grad(loss_module: LossModule):
    params = TensorDict(dict(loss_module.named_parameters()), [])

    def set_grad(p):
        p.grad = torch.zeros_like(p.data)
        return p

    params.apply(set_grad)

    weights = params.apply(lambda p: p.data)
    grad = params.apply(lambda p: p.grad)

    return weights, grad


def _run_gradient_worker(
        rank: int,
        world_size: int,
        model: nn.Module,
        collector: DataCollectorBase,
        # data_buffer: ReplayBuffer,
        loss_module: LossModule,
        value_estimator: ValueEstimatorBase,
        rank0_ip: str,
        tcpport: str,
        backend: str = "gloo",
        verbose: bool = True,
):
    data_buffer = None

    """Run a gradient worker."""
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = str(tcpport)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO: how to handle this?

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

    weigths, grad = get_weights_and_grad(model)
    # weigths, grad = get_weights_and_grad(loss_module)  # TODO: Would this work?

    for data in collector:

        data_view = data.reshape(-1)  # TODO: how do we handle this for rnn?

        with torch.no_grad():
            data_view = value_estimator(data_view)

        # # Add to replay buffer
        # data_buffer.extend(data_view)
        #
        # # Sample batch from replay buffer
        # mini_batch = data_buffer.sample().to(device)
        mini_batch = data_view.to(device)

        # Compute loss
        loss = loss_module(mini_batch)
        loss_sum = sum([item for key, item in loss.items() if key.startswith("loss")])

        # Backprop loss
        loss_sum.backward()

        grad.apply_(lambda x: torch.ones_like(x))
        grad.reduce(0)  # send grads to server, operation is SUM

        # receive latest params
        weigths.irecv(src=0)
        print(f"agent {rank} received params from server")


class DistributedGradientCollector:
    """Distributed gradient collector with torch.distributed backend..

        This Python class serves as a solution to instantiate and coordinate multiple
    gradient workers in a distributed cluster. This class is an iterable that yields
    TensorDicts with gradients until a target number of collected frames is reached.
    """

    _iterator = None
    _VERBOSE = True  # VERBOSE  # for debugging

    def __init__(
        self,
        model: nn.Module,
        num_workers: int,
        *,
        collector: DataCollectorBase = None,
        loss_module: LossModule = None,
        data_buffer: ReplayBuffer = None,
        value_estimator: ValueEstimatorBase,
        backend="gloo",
        launcher="mp",  # For now, only support multiprocessing
        tcp_port=None,
    ):

        self.collector = collector
        self.loss_module = loss_module
        self.data_buffer = data_buffer
        self.value_estimator = value_estimator

        self.num_workers = num_workers
        self.backend = backend
        self.model = model
        if tcp_port is None:
            self.tcp_port = os.environ.get("TCP_PORT", TCP_PORT)
        else:
            self.tcp_port = str(tcp_port)

        self._init_workers()

        # Local collector and buffer can probably be deleted at this point

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

        self.weights, self.grad = get_weights_and_grad(self.model)

        if self._VERBOSE:
            print("main initiated!", end="\t")

    def _init_worker_dist_mp(self, i):
        TCP_PORT = self.tcp_port
        job = mp.Process(
            target=_run_gradient_worker,
            args=(
                i + 1,
                self.num_workers + 1,
                self.model,
                self.collector,
                # self.data_buffer,
                self.value_estimator,
                self.loss_module,
                self.IPAddr,
                int(TCP_PORT),
                self.backend,
                self._VERBOSE,
            ),
        )
        job.start()
        return job

    def __iter__(self) -> Iterator[TensorDictBase]:
        return self.iterator()

    def __next__(self):
        try:
            if self._iterator is None:

                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            out.clear_device_()
            return out
        except StopIteration:
            return None

    def iterator(self):

        while True:

            # collect gradients from workers
            print(f"server received grads from agents")
            self.grad.reduce(0, op=torch.distributed.ReduceOp.SUM)  # see reduce doc to see what ops are supported
            self.grad.apply_(lambda x: x / self.num_workers)  # average grads

            yield self.grad

            self.grad.zero_()

            # TODO: when to stop?

    def _iterator_dist(self):
        if self._VERBOSE:
            print("iterating...")

    def compute_gradients(self, data: TensorDict):
        # Split the data, send it to the workers, receiving grads and averaging them
        pass

    def update_policy_weights_(self, weights, worker_rank=None) -> None:
        """Updates weights and send updated version to worker nodes

        Args:
            worker_rank (int, optional): if provided, only this worker weights
                will be updated.
        """
        self.weights.update_(weights)
        for i in range(self.num_workers):
            if worker_rank is None or worker_rank == i:
                self.weights.isend(dst=i + 1)

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        if self._VERBOSE:
            print("gradient collector shut down")


if __name__ == "__main__":

    model = create_model()
    model.weight.data.fill_(0.0)
    model.bias.data.fill_(0.0)
    weights, _ = get_weights_and_grad(model)

    grad_collector = DistributedGradientCollector(
        model=model,
        num_workers=2,
    )

    for grads in grad_collector:
        print(grads["weight"])
        print(weights["weight"])
        grad_collector.update_policy_weights_(weights)

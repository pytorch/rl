import logging
import warnings
from torch.utils.data import IterableDataset
from typing import Callable, Dict, Iterator, List, OrderedDict, Union, Optional
from torch.optim import Optimizer, Adam


import torch
import torch.nn as nn
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


class GradientWorker:
    """Distributed gradient collector.

        This Python class serves as a solution to instantiate and coordinate multiple
    gradient collectors in a distributed cluster. Like TorchRL GradientCollector class,
    this class is an iterable that yields TensorDicts with gradients until a target number
    of collected frames is reached, and handles both data collection and gradient computation
    under the hood.

    The coordination between GradientCollector instances can be specified as "synchronous" or
    "asynchronous". In synchronous coordination, this class waits for all remote collectors
    to finish gradient computation, then averages the gradients from all workers and finally
    yields the averaged gradients. On the other hand, if the coordination is to be carried out
    asynchronously, this class provides gradients as they become available from individual
    remote GradientCollector's.
    """
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


        This Python class serves as a ray-based solution to instantiate and coordinate multiple
    data collectors in a distributed cluster. Like TorchRL non-distributed collectors, this
    collector is an iterable that yields TensorDicts until a target number of collected
    frames is reached, but handles distributed data collection under the hood.


    This class is an iterable that yields model gradients until a target number of collected
    frames is reached.
    """
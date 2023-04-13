import os
import uuid

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import EnvCreator, ParallelEnv, RewardScaling, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)
def make_env(
    parallel=False,
    obs_norm_sd=None,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(
                lambda: GymEnv(
                    "CartPole-v1",
                    from_pixels=True,
                    pixels_only=True,
                    device=device,
                )
            ),
        )
    else:
        base_env = GymEnv(
            "CartPole-v1",
            from_pixels=True,
            pixels_only=True,
            device=device,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        ),
    )
    return env


def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    return obs_norm_sd

class DuelingCnnDQNet(nn.Module):
    """Dueling CNN Q-network.

    Presented in https://arxiv.org/abs/1511.06581

    Args:
        out_features (int): number of features for the advantage network
        out_features_value (int): number of features for the value network
        cnn_kwargs (dict, optional): kwargs for the feature network.
            Default is

            >>> cnn_kwargs = {
            ...     'num_cells': [32, 64, 64],
            ...     'strides': [4, 2, 1],
            ...     'kernels': [8, 4, 3],
            ... }

        mlp_kwargs (dict, optional): kwargs for the advantage and value network.
            Default is

            >>> mlp_kwargs = {
            ...     "depth": 1,
            ...     "activation_class": nn.ELU,
            ...     "num_cells": 512,
            ...     "bias_last_layer": True,
            ... }

        device (Optional[DEVICE_TYPING]): device to create the module on.
    """

    def __init__(
        self,
        out_features: int,
        out_features_value: int = 1,
        cnn_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
        device: Optional[DEVICE_TYPING] = None,
    ):
        super().__init__()

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        _cnn_kwargs = {
            "num_cells": [32, 64, 64],
            "strides": [4, 2, 1],
            "kernel_sizes": [8, 4, 3],
        }
        _cnn_kwargs.update(cnn_kwargs)
        self.features = ConvNet(device=device, **_cnn_kwargs)

        _mlp_kwargs = {
            "depth": 1,
            "activation_class": nn.ELU,
            "num_cells": 512,
            "bias_last_layer": True,
        }
        mlp_kwargs = mlp_kwargs if mlp_kwargs is not None else {}
        _mlp_kwargs.update(mlp_kwargs)
        self.out_features = out_features
        self.out_features_value = out_features_value
        self.advantage = MLP(out_features=out_features, device=device, **_mlp_kwargs)
        self.value = MLP(out_features=out_features_value, device=device, **_mlp_kwargs)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and isinstance(
                layer.bias, torch.Tensor
            ):
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

def make_model(dummy_env):
    cnn_kwargs = {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [6, 4, 3],
        "strides": [2, 2, 1],
        "activation_class": nn.ELU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ELU,
    }
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)
    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # we wrap our actor in an EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )

    return actor, actor_explore

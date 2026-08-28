# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import torch
import tqdm

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.libs.openml import OpenMLEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import DistributionalQValueActor, EGreedyModule, MLP, QValueActor
from torchrl.objectives import DistributionalDQNLoss, DQNLoss


class SyntheticBanditEnv(EnvBase):
    """Small offline contextual-bandit environment for smoke tests."""

    def __init__(
        self,
        batch_size,
        n_features: int = 8,
        n_actions: int = 4,
        dataset_size: int = 4096,
        device: torch.device | str = "cpu",
    ):
        batch_size = torch.Size(batch_size)
        device = torch.device(device)
        self.n_features = n_features
        self.n_actions = n_actions
        self.dataset_size = dataset_size
        self.rng = torch.Generator(device=device).manual_seed(0)
        self.features = torch.randn(
            dataset_size, n_features, device=device, generator=self.rng
        )
        weights = torch.randn(n_features, n_actions, device=device, generator=self.rng)
        self.outcomes = (self.features @ weights).argmax(-1)
        super().__init__(device=device, batch_size=batch_size)
        self.observation_spec = Composite(
            {
                "observation": Unbounded(
                    shape=(*batch_size, n_features), device=self.device
                ),
                "y": Categorical(n_actions, shape=batch_size, device=self.device),
            },
            shape=batch_size,
            device=self.device,
        )
        self.action_spec = Categorical(n_actions, shape=batch_size, device=self.device)
        self.reward_spec = Unbounded(shape=(*batch_size, 1), device=self.device)

    def _sample(self) -> TensorDict:
        index = torch.randint(
            self.dataset_size,
            self.batch_size,
            generator=self.rng,
            device=self.device,
        )
        return TensorDict(
            {
                "observation": self.features[index],
                "y": self.outcomes[index],
            },
            self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase | None = None) -> TensorDict:
        return self._sample()

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        reward = (tensordict["action"] == tensordict["y"]).float().unsqueeze(-1)
        done = torch.ones_like(reward, dtype=torch.bool)
        return TensorDict(
            {
                "done": done,
                "reward": reward,
                **tensordict.select(*self.observation_spec.keys()),
            },
            self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            seed = 0
        self.rng = torch.Generator(device=self.device).manual_seed(seed)


parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--n_steps", type=int, default=10000, help="number of steps")
parser.add_argument(
    "--eps_greedy", type=float, default=0.1, help="epsilon-greedy parameter"
)
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
parser.add_argument("--n_cells", type=int, default=128, help="number of cells")
parser.add_argument(
    "--distributional", action="store_true", help="enable distributional Q-learning"
)
parser.add_argument(
    "--dataset",
    default="adult_onehot",
    choices=[
        "adult_num",
        "adult_onehot",
        "mushroom_num",
        "mushroom_onehot",
        "covertype",
        "shuttle",
        "magic",
        "synthetic",
    ],
    help="OpenML dataset",
)

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    batch_size = args.batch_size
    n_steps = args.n_steps
    eps_greedy = args.eps_greedy
    lr = args.lr
    wd = args.wd
    n_cells = args.n_cells
    distributional = args.distributional
    dataset = args.dataset

    if dataset == "synthetic":
        env = SyntheticBanditEnv(batch_size=[batch_size])
    else:
        env = OpenMLEnv(dataset, batch_size=[batch_size])
    n_actions = env.action_spec.space.n
    if distributional:
        # does not really make sense since the value is either 0 or 1 and hopefully we
        # should always predict 1
        nbins = 2
        model = MLP(
            out_features=(nbins, n_actions),
            depth=3,
            num_cells=n_cells,
            activation_class=nn.Tanh,
        )
        actor = DistributionalQValueActor(
            model, support=torch.arange(2), action_space="categorical"
        )
        actor(env.reset())
        loss = DistributionalDQNLoss(
            actor,
        )
        loss.make_value_estimator(gamma=0.9)
    else:
        model = MLP(
            out_features=n_actions, depth=3, num_cells=n_cells, activation_class=nn.Tanh
        )
        actor = QValueActor(model, action_space="categorical")
        actor(env.reset())
        loss = DQNLoss(actor, loss_function="smooth_l1", action_space=env.action_spec)
        loss.make_value_estimator(gamma=0.0)
    policy = TensorDictSequential(
        actor,
        EGreedyModule(
            eps_init=eps_greedy,
            eps_end=0.0,
            annealing_num_steps=n_steps,
            spec=env.action_spec,
        ),
    )
    optim = torch.optim.Adam(loss.parameters(), lr, weight_decay=wd)

    pbar = tqdm.tqdm(range(n_steps))

    init_r = None
    init_loss = None
    for i in pbar:
        with set_exploration_type(ExplorationType.RANDOM):
            data = env.step(policy(env.reset()))
        loss_vals = loss(data)
        loss_val = sum(
            value for key, value in loss_vals.items() if key.startswith("loss")
        )
        loss_val.backward()
        optim.step()
        optim.zero_grad()
        if i % 10 == 0:
            test_data = env.step(policy(env.reset()))
            if init_r is None:
                init_r = test_data["next", "reward"].sum() / env.numel()
            if init_loss is None:
                init_loss = loss_val.detach().item()
            pbar.set_description(
                f"reward: {test_data['next', 'reward'].sum() / env.numel(): 4.4f} (init={init_r: 4.4f}), "
                f"training reward {data['next', 'reward'].sum() / env.numel() : 4.4f}, "
                f"loss {loss_val: 4.4f} (init: {init_loss: 4.4f})"
            )
        policy[1].step()

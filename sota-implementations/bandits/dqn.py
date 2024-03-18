# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import tqdm

from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.envs.libs.openml import OpenMLEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import DistributionalQValueActor, EGreedyModule, MLP, QValueActor
from torchrl.objectives import DistributionalDQNLoss, DQNLoss

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

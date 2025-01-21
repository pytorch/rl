# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Using the SyncDataCollector with Different Device Combinations
==============================================================

TorchRL's SyncDataCollector allows you to specify the devices on which different components of the data collection
process are executed. This example demonstrates how to use the collector with various device combinations.


Understanding Device Precedence
-------------------------------

When creating a SyncDataCollector, you can specify the devices for the environment (env_device), policy (policy_device),
and data collection (device). The device argument serves as a default value for any unspecified devices. However, if you
provide env_device or policy_device, they take precedence over the device argument for their respective components.

For example:

- If you set device="cuda", all components will be executed on the CUDA device unless you specify otherwise.
- If you set env_device="cpu" and device="cuda", the environment will be executed on the CPU, while the policy and data
  collection will be executed on the CUDA device.

Keeping Policy Parameters in Sync
---------------------------------

When using a policy with buffers or other attributes that are not automatically updated when moving the policy's
parameters to a different device, it's essential to keep the policy's parameters in sync between the main workspace and
the collector.

To do this, call update_policy_weights_() anytime the policy's parameters (and buffers!) are updated. This ensures that
the policy used by the collector has the same parameters as the policy in the main workspace.

Example Use Cases
-----------------

This script demonstrates the SyncDataCollector with the following device combinations:

- Collector on CUDA
- Collector on CPU
- Mixed collector: policy on CUDA, env untouched (ie, unmarked CPU, env.device == None)
- Mixed collector: policy on CUDA, env on CPU (env.device == "cpu")
- Mixed collector: all on CUDA, except env on CPU.

For each configuration, we run a DQN algorithm and check that it converges.
By following this example, you can learn how to use the SyncDataCollector with different device combinations and ensure
that your policy's parameters are kept in sync.

"""

import logging
import time

import torch.cuda
import torch.nn as nn
import torch.optim as optim

from tensordict.nn import TensorDictSequential as TDSeq

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import Compose, GymEnv, RewardSum, StepCounter, TransformedEnv
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate


logging.basicConfig(level=logging.INFO)
my_logger = logging.getLogger(__name__)

ENV_NAME = "CartPole-v1"

INIT_RND_STEPS = 5_120
FRAMES_PER_BATCH = 128
BUFFER_SIZE = 100_000

GAMMA = 0.98
OPTIM_STEPS = 10
BATCH_SIZE = 128

SOFTU_EPS = 0.99
LR = 0.02


class Net(nn.Module):
    def __init__(self, obs_size: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        orig_shape_unbatched = len(x.shape) == 1
        if orig_shape_unbatched:
            x = x.unsqueeze(0)

        out = self.net(x)

        if orig_shape_unbatched:
            out = out.squeeze(0)
        return out


def make_env(env_name: str):
    return TransformedEnv(GymEnv(env_name), Compose(StepCounter(), RewardSum()))


if __name__ == "__main__":

    for env_device, policy_device, device in (
        (None, None, "cuda"),
        (None, None, "cpu"),
        (None, "cuda", None),
        ("cpu", "cuda", None),
        ("cpu", None, "cuda"),
        # These configs don't run because the collector needs to know that the policy is on CUDA
        #  This is not true for the env which has specs that are associated with a device, we can
        #  automatically transfer the data. The policy does not, in general, have a spec indicating
        #  what the input and output devices are, so this must be told to the collector.
        #        (None, None, None),
        #        ("cpu", None, None),
    ):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        env = make_env(ENV_NAME)
        env.set_seed(0)

        n_obs = env.observation_spec["observation"].shape[-1]
        n_act = env.action_spec.shape[-1]

        net = Net(n_obs, n_act).to(device="cuda:0")
        agent = QValueActor(net, spec=env.action_spec.to("cuda:0"))

        # policy_explore has buffers on CPU - we will need to call collector.update_policy_weights_()
        #  to sync them during data collection.
        policy_explore = EGreedyModule(env.action_spec)
        agent_explore = TDSeq(agent, policy_explore)

        collector = SyncDataCollector(
            env,
            agent_explore,
            frames_per_batch=FRAMES_PER_BATCH,
            init_random_frames=INIT_RND_STEPS,
            device=device,
            env_device=env_device,
            policy_device=policy_device,
        )
        exp_buffer = ReplayBuffer(
            storage=LazyTensorStorage(BUFFER_SIZE, device="cuda:0")
        )

        loss = DQNLoss(
            value_network=agent, action_space=env.action_spec, delay_value=True
        )
        loss.make_value_estimator(gamma=GAMMA)
        target_updater = SoftUpdate(loss, eps=SOFTU_EPS)
        optimizer = optim.Adam(loss.parameters(), lr=LR)

        total_count = 0
        total_episodes = 0
        t0 = time.time()
        for i, data in enumerate(collector):
            # Check the data devices
            if device is None:
                assert data["action"].device == torch.device("cuda:0")
                assert data["observation"].device == torch.device("cpu")
                assert data["done"].device == torch.device("cpu")
            elif device == "cpu":
                assert data["action"].device == torch.device("cpu")
                assert data["observation"].device == torch.device("cpu")
                assert data["done"].device == torch.device("cpu")
            else:
                assert data["action"].device == torch.device("cuda:0")
                assert data["observation"].device == torch.device("cuda:0")
                assert data["done"].device == torch.device("cuda:0")

            exp_buffer.extend(data)
            max_length = exp_buffer["next", "step_count"].max()
            max_reward = exp_buffer["next", "episode_reward"].max()
            if len(exp_buffer) > INIT_RND_STEPS:
                for _ in range(OPTIM_STEPS):
                    optimizer.zero_grad()
                    sample = exp_buffer.sample(batch_size=BATCH_SIZE)

                    loss_vals = loss(sample)
                    loss_vals["loss"].backward()
                    optimizer.step()

                    agent_explore[1].step(data.numel())
                    target_updater.step()

                    total_count += data.numel()
                    total_episodes += data["next", "done"].sum()

                if i % 10 == 0:
                    my_logger.info(
                        f"Step: {i}, max. count / epi reward: {max_length} / {max_reward}."
                    )
            collector.update_policy_weights_()
            if max_length > 200:
                t1 = time.time()
                my_logger.info(f"SOLVED in {t1 - t0}s!! MaxLen: {max_length}!")
                my_logger.info(f"With {max_reward} Reward!")
                my_logger.info(f"In {total_episodes} Episodes!")
                my_logger.info(f"Using devices {(env_device, policy_device, device)}")
                break
        else:
            raise RuntimeError(
                f"Failed to converge with config {(env_device, policy_device, device)}"
            )

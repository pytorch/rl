import tqdm
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
import torchrl
from torchrl.envs.transforms.transforms import Compose, RewardSum, RewardScaling
from torchrl.objectives import DDPGLoss
from torchrl.data import ReplayBuffer, ListStorage, TensorDictReplayBuffer
from torchrl.modules import (
    ConvNet,
    EGreedyWrapper,
    LSTMModule,
    MLP,
    QValueModule,
    TanhDelta,
)
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn
import torch
from torchrl.envs.transforms import DoubleToFloat, TransformedEnv
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


# Config
seed = 0
gamma = 0.99
tau = 0.005
frames_per_batch = 200  # Frames sampled each sampling iteration
batch_size = 200  # in terms of rollout size
utd = 200  # update to data ratio
max_steps = 100
n_iters = 500  # Number of sampling/training iterations
warmup_frames = 100  # To prefill replay buffer
total_frames = frames_per_batch * n_iters
memory_size = frames_per_batch * 100
lr = 0.001

torch.manual_seed(0)


def env():
    env = GymEnv("Pendulum-v1", from_pixels=False)
    env = TransformedEnv(
        env, Compose(DoubleToFloat(in_keys=["observation"]), RewardSum(), RewardScaling(loc=0.0, scale=0.1))
    )
    return env


proof_env = env()
obs_size = proof_env.observation_spec["observation"].shape[0]
act_size = proof_env.action_spec.shape[0]

# Nets


class CriticNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + act_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        nn.init.normal_(self.net[-1].weight, 0, 0.001)
        self.net[-1].bias.data.zero_()

    def forward(self, observation, action):
        return self.net(torch.cat([observation, action], dim=-1))


class ActorNet(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        nn.init.normal_(self.net[-1].weight, 0, 0.001)
        self.net[-1].bias.data.zero_()

    def forward(self, observation):
        return self.net(observation)


actor_mod = Mod(ActorNet(obs_size), in_keys=["observation"], out_keys=["param"])
actor = ProbabilisticActor(
    module=actor_mod,
    spec=proof_env.action_spec,
    in_keys=["param"],
    distribution_class=TanhDelta,
    distribution_kwargs={
        "min": proof_env.action_spec.space.minimum,
        "max": proof_env.action_spec.space.maximum,
    },
    return_log_prob=False,
)
policy = AdditiveGaussianWrapper(
    actor, annealing_num_steps=int(total_frames), sigma_end=0.1, spec=proof_env.action_spec,
)

critic = ValueOperator(CriticNet(obs_size, act_size), in_keys=["observation", "action"])

collector = torchrl.collectors.SyncDataCollector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)
buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(memory_size), batch_size=batch_size)

loss_module = DDPGLoss(
    actor_network=actor,
    value_network=critic,
    delay_value=True,
)
loss_module.make_value_estimator(gamma=gamma)
target_net_updater = SoftUpdate(loss_module, eps=1 - tau)

actor_opt = torch.optim.Adam(actor.parameters(), lr)
critic_opt = torch.optim.Adam(critic.parameters(), lr)

pbar = tqdm.tqdm(total=total_frames)
for i, tensordict in enumerate(collector):
    pbar.update(tensordict.numel())
    buffer.extend(tensordict)
    for j in range(utd):
        data = buffer.sample()
        loss = loss_module(data)
        critic_opt.zero_grad()
        actor_opt.zero_grad()
        loss["loss_actor"].backward()
        loss["loss_value"].backward()
        critic_opt.step()
        actor_opt.step()
        pbar.set_description(
            "Epoch {}: episode reward {:.2f} actor loss {:.2f}, critic loss {:.2f}".format(
                i,
                tensordict['next']["episode_reward"][tensordict['next']['done']].mean(),
                loss["loss_actor"].item(),
                loss["loss_value"].item(),
            )
        )
        target_net_updater.step()
    policy.step(tensordict.numel())
    collector.update_policy_weights_()

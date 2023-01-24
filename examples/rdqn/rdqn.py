import torch
from torch import nn
import torchrl
from torchrl.envs.libs.gym import GymEnv
from torchrl.objectives import DQNLoss
from torchrl.modules import QValueActor, EGreedyWrapper, LSTMNet
from tensordict import TensorDict
from tensordict.tensordict import pad_sequence_td, pad
from tensordict.nn import TensorDictSequential, TensorDictModule
from torchrl.data import ReplayBuffer
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.storages import ListStorage


def pad_to_size(datas):
    """Collate_fn that pads to the longest sequence in the sampled batch"""
    maxlen = max([data.shape[-2] for data in datas])
    padded = [pad(data, [0, 0, 0, maxlen - data.shape[1]]) for data in datas]
    return torch.cat(padded, dim=0)



# We will have two examples:
# one will use the built-in LSTMNet
# the other will use a custom recurrent model (GRU)
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def rollout(self, x, rstate=None):
        # Depending on collector/rollout type, we will either have
        # Dims[feature] or Dims[num_envs, feature]
        # During inference x is dim == 3 but must be 2 for nn.GRU
        assert x.dim() in [1, 2]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if rstate is not None:
            if rstate.dim() == 1:
                rstate = rstate.unsqueeze(0)
            if rstate.dim() == 2:
                rstate = rstate.unsqueeze(1)
        x, rstate = self.gru(x, rstate)
        x = x.squeeze()
        rstate = rstate.squeeze()
        return x, rstate

    def train_forward(self, x, old_rstate):
        # Dims: [replay_size, batch, time, feature]
        assert x.dim() == 4
        replay_size, batch, time, feature = x.shape
        x = x.reshape(replay_size * batch, time, feature)
        x, rstate = self.gru(x, None)
        x = x.reshape(replay_size, batch, time, feature)
        return x, old_rstate

    def forward(self, x, rstate=None):
        # During rollouts x is dim == 1 but must be 3 for nn.GRU
        if x.dim() in [1, 2]:
            x, rstate = self.rollout(x, rstate)
        elif x.dim() == 4:
            x, rstate = self.train_forward(x, rstate)
        return x, rstate


def main():
    # Create env
    env = GymEnv("CartPole-v1", from_pixels=False)
    env_make = lambda: GymEnv("CartPole-v1", from_pixels=False)
    """
    model = LSTMNet(env.action_spec.ndim, {"input_size": 32, "hidden_size": 32}, {"out_features": 32})
    train_policy = QValueActor(module=model, spec=env.action_spec, in_keys=["observation"])
    rollout_policy = EGreedyWrapper(train_policy)
    """
    
    # Build the models and policies
    # This estimates the Markov state 
    preprocessor = TensorDictModule(
        nn.Sequential(
            nn.LazyLinear(32),
            nn.Mish(),
            nn.Linear(32, 32)
        ),
        in_keys=["observation"],
        out_keys=["hidden"],
    )
    backbone = TensorDictModule(
        GRU(32, 32),
        in_keys=["hidden", "rstate"],
        out_keys=["mstate", "rstate"]
    )
    """
    backbone = TensorDictModule(
        nn.Sequential(
            nn.Linear(32, 32),
        ),
        in_keys=["hidden"],
        out_keys=["mstate"]
    )
    """
    q = QValueActor(
        nn.Linear(32, env.action_spec.ndim),
        in_keys=["mstate"],
        spec=env.action_spec,
    )
    greedy_q = EGreedyWrapper(q, spec=env.action_spec, eps_init=0.2)
    train_policy = TensorDictSequential(
        preprocessor,
        backbone,
        q,
    )
    rollout_policy = TensorDictSequential(
        preprocessor,
        backbone,
        greedy_q
    )

    loss_fn = DQNLoss(q, gamma=0.99)

    buffer = ReplayBuffer(storage=ListStorage(20), collate_fn=pad_to_size)

    collector = SyncDataCollector(
        env_make,
        policy=rollout_policy,
        max_frames_per_traj=-1,
        init_random_frames=10_000,
        reset_at_each_iter=True,
        total_frames=1_000_000,
        frames_per_batch=2_000,
        split_trajs=True
    )
    opt = torch.optim.Adam(train_policy.parameters(), lr=1e-4)

    for data in collector:
        buffer.add(data)
        train_data = buffer.sample(4)
        train_policy(train_data)
        mean_creward = data["reward"].sum(-2).mean()
        unpadded = train_data[train_data["mask"]]
        loss = loss_fn(unpadded)["loss"]
        loss.backward()
        opt.step()
        greedy_q.step(1)
        opt.zero_grad()
        print(
            f"loss: {loss.item():.2f}, "
            f"reward: {mean_creward.item():.2f}, "
            f"eps: {greedy_q.eps.item():.2f}"
        )
        collector.update_policy_weights_()

if __name__ == "__main__":
    main()
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import tensordict.nn
import torch
import tqdm
from tensordict.nn import TensorDictSequential as TDSeq, TensorDictModule as TDMod, \
    ProbabilisticTensorDictModule as TDProb, ProbabilisticTensorDictSequential as TDProbSeq
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from torchrl.collectors import SyncDataCollector

from torchrl.envs import ChessEnv, Tokenizer
from torchrl.modules import MLP
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement

tensordict.nn.set_composite_lp_aggregate(False)

num_epochs = 10
batch_size = 256
frames_per_batch = 2048

env = ChessEnv(include_legal_moves=True, include_fen=True)

# tokenize the fen - assume max 70 elements
transform = Tokenizer(in_keys=["fen"], out_keys=["fen_tokenized"], max_length=70)

env = env.append_transform(transform)
n = env.action_spec.n
print(env.rollout(10000))

# Embedding layer for the legal moves
embedding_moves = nn.Embedding(num_embeddings=n + 1, embedding_dim=64)

# Embedding for the fen
embedding_fen = nn.Embedding(num_embeddings=transform.tokenizer.vocab_size, embedding_dim=64)

backbone = MLP(out_features=512, num_cells=[512] * 8, activation_class=nn.ReLU)

actor_head = nn.Linear(512, env.action_spec.n)
actor_head.bias.data.fill_(0)

critic_head = nn.Linear(512, 1)
critic_head.bias.data.fill_(0)

prob = TDProb(in_keys=["logits", "mask"], out_keys=["action"], distribution_class=MaskedCategorical, return_log_prob=True)

def make_mask(idx):
    mask = idx.new_zeros((*idx.shape[:-1], n + 1), dtype=torch.bool)
    return mask.scatter_(-1, idx, torch.ones_like(idx, dtype=torch.bool))[..., :-1]

actor = TDProbSeq(
    TDMod(
        make_mask,
        in_keys=["legal_moves"], out_keys=["mask"]),
    TDMod(embedding_moves, in_keys=["legal_moves"], out_keys=["embedded_legal_moves"]),
    TDMod(embedding_fen, in_keys=["fen_tokenized"], out_keys=["embedded_fen"]),
    TDMod(lambda *args: torch.cat([arg.view(*arg.shape[:-2], -1) for arg in args], dim=-1), in_keys=["embedded_legal_moves", "embedded_fen"],
          out_keys=["features"]),
    TDMod(backbone, in_keys=["features"], out_keys=["hidden"]),
    TDMod(actor_head, in_keys=["hidden"], out_keys=["logits"]),
    prob,
)
critic = TDSeq(
    TDMod(critic_head, in_keys=["hidden"], out_keys=["state_value"]),
)


print(env.rollout(3, actor))
# loss
loss = ClipPPOLoss(actor, critic)

optim = Adam(loss.parameters())

gae = GAE(value_network=TDSeq(*actor[:-2], critic), gamma=0.99, lmbda=0.95, shifted=True)

# Create a data collector
collector = SyncDataCollector(
    create_env_fn=env,
    policy=actor,
    frames_per_batch=frames_per_batch,
    total_frames=1_000_000,
)

replay_buffer0 = ReplayBuffer(storage=LazyTensorStorage(max_size=collector.frames_per_batch//2), batch_size=batch_size, sampler=SamplerWithoutReplacement())
replay_buffer1 = ReplayBuffer(storage=LazyTensorStorage(max_size=collector.frames_per_batch//2), batch_size=batch_size, sampler=SamplerWithoutReplacement())

for data in tqdm.tqdm(collector):
    data = data.filter_non_tensor_data()
    print('data', data[0::2])
    for i in range(num_epochs):
        replay_buffer0.empty()
        replay_buffer1.empty()
        with torch.no_grad():
            # player 0
            data0 = gae(data[0::2])
            # player 1
            data1 = gae(data[1::2])
            if i == 0:
                print('win rate for 0', data0["next", "reward"].sum() / data["next", "done"].sum().clamp_min(1e-6))
                print('win rate for 1', data1["next", "reward"].sum() / data["next", "done"].sum().clamp_min(1e-6))

            replay_buffer0.extend(data0)
            replay_buffer1.extend(data1)

        n_iter = collector.frames_per_batch//(2 * batch_size)
        for (d0, d1) in tqdm.tqdm(zip(replay_buffer0, replay_buffer1, strict=True), total=n_iter):
            loss_vals = (loss(d0) + loss(d1)) / 2
            loss_vals.sum(reduce=True).backward()
            gn = clip_grad_norm_(loss.parameters(), 100.0)
            optim.step()
            optim.zero_grad()

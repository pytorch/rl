import torch
import tqdm
from tensordict import TensorDict
from tensordict.nn import (
    set_skip_existing,
    skip_existing,
    TensorDictModule as Mod,
    TensorDictModuleBase as Base,
    TensorDictSequential as Seq,
)
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictReplayBuffer,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import (
    Compose,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    step_mdp,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss


env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True),
    Compose(
        ToTensorImage(),
        GrayScale(),
        Resize(84, 84),
        StepCounter(),
        InitTracker(),
        ObservationNorm(standard_normal=True, in_keys=["pixels"]),
    ),
)
env.transform[-1].init_stats(1, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])

feature = Mod(
    ConvNet(
        num_cells=[32, 32, 64],
        squeeze_output=True,
        aggregator_class=nn.AdaptiveAvgPool2d,
        aggregator_kwargs={"output_size": (1, 1)},
    ),
    in_keys=["pixels"],
    out_keys=["embed"],
)
n_cells = feature(env.reset())["embed"].shape[-1]
lstm = nn.LSTM(input_size=n_cells, hidden_size=128, batch_first=True)
lstm = LSTMModule(
    lstm,
    in_keys=["embed", "hidden_0", "hidden_1"],
    out_keys=["embed", ("next", "hidden_0"), ("next", "hidden_1")],
)
mlp = Mod(
    MLP(
        out_features=2,
        num_cells=[
            64,
        ],
    ),
    in_keys=["embed"],
    out_keys=["action_value"],
)
qval = QValueModule("one_hot")
stoch_policy = Seq(feature, lstm, mlp, qval)
stoch_policy = EGreedyWrapper(
    stoch_policy, annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
)
policy = Seq(feature, lstm.set_temporal_mode(True), mlp, qval)

# init lazy modules
stoch_policy(env.reset())

loss_fn = DQNLoss(policy, action_space="one_hot")
optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

collector = SyncDataCollector(
    env, stoch_policy, frames_per_batch=50, total_frames=5_000_000
)
rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(200_000), batch_size=4)

utd = 16
pbar = tqdm.tqdm(total=5_000_000)
for data in collector:
    pbar.update(data.numel())
    # it is important to pass data that is not flattened
    rb.extend(data)
    for _ in range(utd):
        s = rb.sample()
        loss_vals = loss_fn(s)
        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()
    pbar.set_description(
        f"steps: {data['step_count'].max()}, loss_val: {loss_vals['loss'].item(): 4.4f}"
    )
    stoch_policy.step(data.numel())

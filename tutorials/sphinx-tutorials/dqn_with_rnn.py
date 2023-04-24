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
    RewardScaling,
    step_mdp,
    StepCounter,
    TensorDictPrimer,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")

env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True, device=device),
    Compose(
        ToTensorImage(),
        TensorDictPrimer(
            {
                "hidden_0": UnboundedContinuousTensorSpec(shape=(1, 128)),
                "hidden_1": UnboundedContinuousTensorSpec(shape=(1, 128)),
            }
        ),
        GrayScale(),
        Resize(84, 84),
        StepCounter(),
        InitTracker(),
        RewardScaling(loc=0.0, scale=0.1),
        ObservationNorm(standard_normal=True, in_keys=["pixels"]),
    ),
)
env.transform[-1].init_stats(1000, reduce_dim=[0, 1, 2], cat_dim=0, keep_dims=[0])
td = env.reset()

feature = Mod(
    ConvNet(
        num_cells=[32, 32, 64],
        squeeze_output=True,
        aggregator_class=nn.AdaptiveAvgPool2d,
        aggregator_kwargs={"output_size": (1, 1)},
        device=device,
    ),
    in_keys=["pixels"],
    out_keys=["embed"],
)
n_cells = feature(env.reset())["embed"].shape[-1]
print(n_cells)
lstm = nn.LSTM(input_size=n_cells, hidden_size=128, batch_first=True, device=device)
lstm = LSTMModule(
    lstm,
    in_keys=["embed", "hidden_0", "hidden_1"],
    out_keys=["embed", ("next", "hidden_0"), ("next", "hidden_1")],
)
mlp = MLP(
    out_features=2,
    num_cells=[
        64,
    ],
    device=device,
)
mlp[-1].bias.data.fill_(0.0)
mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])
qval = QValueModule(action_space=env.action_spec)

stoch_policy = Seq(feature, lstm, mlp, qval)

stoch_policy = EGreedyWrapper(
    stoch_policy, annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
)

policy = Seq(feature, lstm.set_temporal_mode(True), mlp, qval)

policy(env.reset())

loss_fn = DQNLoss(policy, action_space="one_hot", delay_value=True)
optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
with torch.no_grad():
    td = stoch_policy(env.reset())


collector = SyncDataCollector(
    env, stoch_policy, frames_per_batch=50, total_frames=1_000_000
    # env, stoch_policy, frames_per_batch=50, total_frames=1000,
)
rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(200_000), batch_size=4, prefetch=10)
updater = SoftUpdate(loss_fn, eps=0.95)
updater.init_()

utd = 16
pbar = tqdm.tqdm(total=1_000_000)
longest = 0

traj_lens = []
for i, data in enumerate(collector):
    step_counts = data["next", "step_count"][data["next", "done"].squeeze(-1)]
    if step_counts.numel():
        traj_lens += step_counts.tolist()
    if i == 0:
        print("data:", data)
    pbar.update(data.numel())
    # it is important to pass data that is not flattened
    rb.extend(data.unsqueeze(0).to_tensordict().cpu())  # .exclude("hidden_0", "hidden_1"))
    for k in range(utd):
        s = rb.sample().to(device)
        if k == 0 and i == 0:
            print("sample:", s)
        loss_vals = loss_fn(s)
        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()
    longest = max(longest, data['step_count'].max().item())
    pbar.set_description(
        f"steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
    )
    stoch_policy.step(data.numel())
    updater.step()

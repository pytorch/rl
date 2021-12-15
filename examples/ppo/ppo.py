import math
import time
import uuid
from argparse import ArgumentParser

import torch
import tqdm
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from torchrl.collectors import sync_sync_collector
from torchrl.data.tensor_specs import *
from torchrl.data.transforms import (
    TransformedEnv,
    Resize,
    Compose,
    GrayScale,
    ToTensorImage,
    CatFrames,
    ObservationNorm,
    RewardScaling,
    FiniteTensorDictCheck,
    DoubleToFloat,
    CatTensors, )
from torchrl.envs import GymEnv, RetroEnv, DMControlEnv
from torchrl.modules.recipes import make_actor_critic_model
from torchrl.objectives.costs.ppo import ClipPPOLoss, KLPENPPOLoss
from torchrl.objectives.returns.gae import GAE
from torchrl.record.recorder import VideoRecorder, TensorDictRecorder

parser = ArgumentParser()

parser.add_argument("--library", type=str, default="dm_control")
parser.add_argument("--env_name", type=str, default="cheetah")
parser.add_argument("--env_task", type=str, default="run")

parser.add_argument("--record_video", action="store_true")
# parser.add_argument('--record_steps', default=250, type=int)
parser.add_argument('--record_interval', default=100, type=int)

parser.add_argument("--from_vector", action="store_true")
parser.add_argument("--exp_name", type=str, default="")

parser.add_argument('--loss_class', default='clip', type=str)
parser.add_argument('--training_sub_steps', default=10, type=int)
parser.add_argument('--frames_per_batch', default=250, type=int, help="how long a batch is")
parser.add_argument('--max_steps_per_traj', default=250, type=int,
                    help="after how many steps is a local env reset on a worker")
parser.add_argument('--num_workers', default=32, type=int)
parser.add_argument('--num_collectors', default=4, type=int)
parser.add_argument('--total_frames', default=50000000, type=int)
parser.add_argument('--frame_skip', default=4, type=int)

parser.add_argument('--init_env_steps', default=200, type=int)

parser.add_argument('--lamda', default=0.95, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--grad_clip_norm', default=1.0, type=float)

parser.add_argument("--DEBUG", action="store_true", )
parser.add_argument("--collector_device", type=str, default="cpu")

LOSS_DICT = {
    'clip': ClipPPOLoss,
    'kl': KLPENPPOLoss,
}

DEFAULT_POLICY_DIST_MAP = {
    BoundedTensorSpec: "tanh-normal",
    OneHotDiscreteTensorSpec: "categorical",
    BinaryDiscreteTensorSpec: "binomial",
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

env_library_map = {
    "gym": GymEnv,
    "retro": RetroEnv,
    "dm_control": DMControlEnv,
}

if __name__ == "__main__":
    mp.set_start_method("spawn")

    args = parser.parse_args()
    env_name = args.env_name
    env_library = env_library_map[args.library]
    env_task = args.env_task
    T = args.frames_per_batch
    K = args.num_workers
    N = args.total_frames // T // K if not args.DEBUG else 100
    exp_name = args.exp_name

    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"device is {device}")


    def make_transformed_env(video_tag="", writer=None, stats=None, reward_stats=None):
        env_kwargs = {
            'envname': env_name,
            "device": "cpu",
            'frame_skip': args.frame_skip,
            'from_pixels': not args.from_vector or len(video_tag),
        }
        if env_library is DMControlEnv:
            env_kwargs.update({'taskname': env_task})
        env = env_library(
            **env_kwargs
        )
        keys = env.reset().keys()
        transforms = []
        if not args.from_vector:
            transforms += [
                ToTensorImage(),
                Resize(84, 84),
                GrayScale(),
                CatFrames(),
                ObservationNorm(loc=-1.0, scale=2.0, keys=["next_observation_pixels"]),
            ]
        if reward_stats is None:
            rs = RewardScaling(0.0, 1.0)
        else:
            rs = RewardScaling(**reward_stats)
        transforms += [
            rs,
            FiniteTensorDictCheck(),
        ]
        if env_library is DMControlEnv:
            selected_keys = ["next_" + key for key in keys if key.startswith("observation") and "pixels" not in key]
            if args.from_vector:
                if stats is None:
                    stats = {"loc": 0.0, "scale": 1.0}
                transforms += [
                    CatTensors(keys=selected_keys,
                               out_key="next_observation_vector"),
                    ObservationNorm(**stats, keys=["next_observation_vector"], standard_normal=True),
                    DoubleToFloat(keys=["action", "next_observation_vector", "reward"]),
                    # DMControl requires double-precision
                ]
            else:
                transforms += [
                    CatTensors(keys=selected_keys,
                               out_key="next_observation_vector"),
                    DoubleToFloat(keys=["action", "next_observation_pixels", "reward"]),
                    # DMControl requires double-precision
                ]
        if len(video_tag):
            transforms = [
                VideoRecorder(
                    writer=writer,
                    tag=f"{video_tag}_{env_name}_video",
                ),
                TensorDictRecorder(f"{video_tag}_{env_name}"),
                *transforms,
            ]
        env = TransformedEnv(env, Compose(*transforms), )
        return env


    env = make_transformed_env()
    env_specs = env.specs  # TODO: use env.sepcs
    action_spec = env_specs["action_spec"]
    actor_critic_model = make_actor_critic_model(action_spec, save_dist_params=args.loss_class == "kl").to(device)
    actor_model = actor_critic_model.get_policy_operator()
    critic_model = actor_critic_model.get_value_operator()

    with torch.no_grad():
        td = env.reset()
        td_device = td.to(device)
        td_device = td_device.unsqueeze(0)
        td_device = actor_critic_model(td_device)  # for init
        td = td_device.squeeze(0).to("cpu")
        t0 = time.time()
        env.step(td)  # for sanity check

        stats = None
        td = env.rollout(n_steps=args.init_env_steps)
        if args.from_vector:
            stats = {"loc": td.get("observation_vector").mean(0), "scale": td.get("observation_vector").std(0)}
        reward_stats = {"loc": td.get("reward").mean(), "scale": td.get("reward").std()}

    loss_module = LOSS_DICT[args.loss_class](actor=actor_model, critic=critic_model)
    advantage = GAE(args.gamma, args.lamda, critic=critic_model, average_rewards=True)

    training_sub_steps = args.training_sub_steps

    assert args.num_workers >= args.num_collectors
    collector = sync_sync_collector(
        make_transformed_env,
        env_kwargs={"stats": stats, "reward_stats": reward_stats},
        policy=actor_critic_model,
        max_steps_per_traj=args.max_steps_per_traj,
        frames_per_batch=T,
        total_frames=args.total_frames,
        num_collectors=args.num_collectors,
        num_env_per_collector=args.num_workers // args.num_collectors,
        device=args.collector_device,
    )
    log_dir = "/".join(["ppo_logging", exp_name, str(uuid.uuid1())])
    if args.record_video:
        video_tag = log_dir + "/"
    else:
        video_tag = ""
    writer = SummaryWriter(log_dir)
    env_record = make_transformed_env(stats=stats, video_tag=video_tag, writer=writer)

    params = list(actor_critic_model.parameters())
    optim = OPTIMIZERS[args.optimizer](params, lr=args.lr, weight_decay=args.wd)
    pbar = tqdm.tqdm(total=args.total_frames)
    frame_count = 0
    init_rewards_noexplore = None  # for progress bar

    for k, batch_of_trajectories in enumerate(collector):

        batch_of_trajectories = batch_of_trajectories.to(device)
        n_frames_per_batch = math.prod(batch_of_trajectories.batch_size)
        frame_count += n_frames_per_batch
        batches = torch.ones(batch_of_trajectories.batch_size).nonzero()
        batches = batches[torch.randperm(batches.shape[0])]
        batches = batches.split(batches.shape[0] // training_sub_steps, 0)
        loss_module.reset()

        for j in range(training_sub_steps):
            batch_idx = batches[j]
            with torch.no_grad():
                _batch_of_trajs = advantage(batch_of_trajectories)
            _batch_of_trajs = _batch_of_trajs[batch_idx.unbind(-1)]
            loss = loss_module(_batch_of_trajs).mean()
            loss.backward()
            try:
                gv = torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm, error_if_nonfinite=True)
                optim.step()
                optim.zero_grad()
            except:
                print('infinte grad, skipping')

        reward_avg = batch_of_trajectories.get('reward').sum(1).mean()
        pbar.set_description(
            f"grad norm: {gv:4.4f}, loss={loss:4.4f}, reward={reward_avg :4.4f}")
        pbar.update(n_frames_per_batch)
        if (k % args.record_interval) == 0:
            print("recording")
            with torch.no_grad():
                actor_model.eval()
                td_record = env_record.rollout(
                    policy=actor_model, n_steps=args.max_steps_per_traj, explore=False
                )
                actor_model.train()
            print("dumping")
            env_record.transform.dump()
            rewards_noexplore = td_record.get("reward").sum(-2).mean().item()
            if init_rewards_noexplore is None:
                init_rewards_noexplore = rewards_noexplore
            writer.add_scalar("loss", loss.item(), frame_count)
            writer.add_scalar("reward_avg", reward_avg, frame_count)
            writer.add_scalar("rewards_noexplore", rewards_noexplore, frame_count)
            writer.add_scalar("grad norm", gv, frame_count)
            torch.save(
                {
                    "actor_critic_model": actor_critic_model.state_dict(),
                },
                log_dir + "/ppo_nets.t",
            )
        collector.update_policy_weights_()

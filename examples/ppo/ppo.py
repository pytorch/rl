import datetime
import math
import os.path
import time
import uuid
from argparse import ArgumentParser

import numpy as np
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

parser.add_argument("--env_library", type=str, default="dm_control", choices=["gym", "dm_control"],
                    help="env_library used for the simulated environment. Default=dm_control")
parser.add_argument("--env_name", type=str, default="cheetah",
                    help="name of the environment to be created. Default=cheetah")
parser.add_argument("--env_task", type=str, default="run",
                    help="task (if any) for the environment. Default=run")

parser.add_argument("--record_video", action="store_true",
                    help="whether a video of the task should be rendered during logging.")
# parser.add_argument('--record_steps', default=250, type=int)
parser.add_argument('--record_interval', default=100, type=int)

parser.add_argument("--from_vector", action="store_true",
                    help="whether the environment output should be pixels (default) or the state vector(s).")
parser.add_argument("--exp_name", type=str, default="",
                    help="experiment name. Used for logging directory. "
                         "A date and uuid will be joined to account for multiple experiments with the same name.")

parser.add_argument("--loss_class", type=str, default="clip", choices=["clip", "kl"],
                    help="PPO loss class, either clip or kl. Default=clip")
parser.add_argument('--training_sub_steps', default=10, type=int,
                    help="number of optimization steps on a batch of rollouts before another round of rollout "
                         "is collected.")
parser.add_argument("--frames_per_batch", type=int, default=250,
                    help="number of steps executed in the environment per collection."
                         "This value represents how many steps will the data collector execute and return in *each*"
                         "environment that has been created in between two rounds of optimization "
                         "(see the training_sub_steps above). ")
parser.add_argument("--max_frames_per_traj", type=int, default=250,
                    help="Number of steps before a reset of the environment is called (if it has not been flagged "
                         "as done).")
parser.add_argument("--num_workers", type=int, default=16,
                    help="Number of workers used for data collection. ")
parser.add_argument("--env_per_collector", default=4, type=int,
                    help="Number of environments per collector. If the env_per_collector is in the range: "
                         "1<env_per_collector<=num_workers, then the collector runs"
                         "ceil(num_workers/env_per_collector) in parallel and executes the policy steps synchronously "
                         "for each of these parallel wrappers. If env_per_collector=num_workers, no parallel wrapper is created.")
parser.add_argument("--total_frames", type=int, default=50000000,
                    help="total number of frames collected for training. Does not account for frame_skip and should"
                         "be corrected accordingly. Default=50e6.")
parser.add_argument("--frame_skip", type=int, default=4,
                    help="frame_skip for the environment. Note that this value does NOT impact the buffer size,"
                         "maximum steps per trajectory, frames per batch or any other factor in the algorithm,"
                         "e.g. if the total number of frames that has to be computed is 50e6 and the frame skip is 4,"
                         "the actual number of frames retrieved will be 200e6. Default=4.")

parser.add_argument("--init_env_steps", type=int, default=250,
                    help="number of random steps to compute normalizing constants")

parser.add_argument('--lamda', default=0.95, type=float,
                    help="lambda factor in GAE (using 'lambda' as attribute is prohibited in python, "
                         "hence the mispelling)")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Decay factor for return computation. Default=0.99.")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate used for the optimizer. Default=2e-4.")
parser.add_argument("--wd", type=float, default=1e-4,
                    help="Weight-decay to be used with the optimizer. Default=1e-4.")
parser.add_argument('--optimizer', default='adam', type=str, choices=["adam"],
                    help="Optimizer to be used. Default=adam")
parser.add_argument("--grad_clip_norm", type=float, default=100.0,
                    help="value at which the total gradient norm should be clipped. Default=100.0")

parser.add_argument("--collector_device", type=str, default="cpu",
                    help="device on which the data collector should store the trajectories to be passed to this script."
                         "If the collector device differs from the policy device (cuda:0 if available), then the "
                         "weights of the collector policy are synchronized with collector.update_policy_weights_().")

parser.add_argument("--seed", type=int, default=42,
                    help="seed used for the environment, pytorch and numpy.")

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_name = args.env_name
    env_library = env_library_map[args.env_library]
    env_task = args.env_task
    T = args.frames_per_batch
    K = args.num_workers
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
    env.set_seed(args.seed)
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

    collector = sync_sync_collector(
        make_transformed_env,
        env_kwargs={"stats": stats, "reward_stats": reward_stats},
        policy=actor_critic_model,
        max_steps_per_traj=args.max_frames_per_traj,
        frames_per_batch=T,
        total_frames=args.total_frames,
        num_collectors=- args.num_workers // -args.env_per_collector,
        num_env_per_collector=args.env_per_collector,
        device=args.collector_device,
        passing_device=args.collector_device,
    )
    collector.set_seed(args.seed)

    log_dir = "/".join(
        ["dqn_logging", exp_name, str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")), str(uuid.uuid1())])
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
        n_frames_per_batch = batch_of_trajectories.numel()
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
                    policy=actor_model,
                    n_steps=args.max_frames_per_traj,
                    explore=False
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
                os.path.join(log_dir, "ppo_nets.t"),
            )
        collector.update_policy_weights_()

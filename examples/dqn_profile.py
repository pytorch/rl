import argparse
import os
import sys
import uuid

import numpy as np
import torch
import tqdm
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from torchrl.envs import GymEnv, RetroEnv
from torchrl.collectors import (
    ParallelDataCollector,
    ParallelDataCollectorQueue,
)
from torchrl.data import MultiStep
from torchrl.data.replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
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
)
from torchrl.modules import EGreedyWrapper, make_dqn_actor, NoisyLinear, \
    reset_noise
from torchrl.objectives import DQNLoss, DoubleDQNLoss, DistributionalDQNLoss, DistributionalDoubleDQNLoss
from torchrl.record.recorder import VideoRecorder, TensorDictRecorder

if sys.platform == "darwin":
    os.system("defaults write org.python.python ApplePersistenceIgnoreState NO")

import time

parser = argparse.ArgumentParser()
# parser.add_argument("--env_name", type=str, default="Pong-v0")
parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
parser.add_argument("--env_type", type=str, default="gym")
parser.add_argument("--video_file", action="store_true")
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--loss", type=str, default="double")
parser.add_argument(
    "--loss_type", type=str, default="l2", choices=["l1", "l2", "smooth_l1"]
)
parser.add_argument("--collector_device", type=str, default="cpu")
parser.add_argument("--value_network_update_interval", type=int, default=100)
parser.add_argument("--steps_per_collection", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--buffer_size", type=int, default=1000)
parser.add_argument("--frame_skip", type=int, default=4)

parser.add_argument("--total_frames", type=int, default=5000)
parser.add_argument("--annealing_frames", type=int, default=1000)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--traj_len", type=int, default=-1)
parser.add_argument("--frames_per_batch", type=int, default=50)

parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--grad_clip_norm", type=float, default=100.0)

parser.add_argument("--reset_at_each_iter", action="store_true")
parser.add_argument("--use_queue", action="store_true")
parser.add_argument("--DEBUG", action="store_true")
parser.add_argument("--pass_tensors_through_queues", action="store_true")

parser.add_argument("--noisy", action="store_true")
parser.add_argument("--distributional", action="store_true")
parser.add_argument("--atoms", type=int, default=51)
parser.add_argument("--multi_step", action="store_true")
parser.add_argument("--prb", action="store_true")
parser.add_argument("--n_steps_return", type=int, default=3)


env_type_map = {
    "gym": GymEnv,
    "retro": RetroEnv,
}


# def make():
#     print(f"making env on {os.getpid()}")
#     return gym.make(env_name)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    args = parser.parse_args()
    env_name = args.env_name
    env_type = env_type_map[args.env_type]
    exp_name = args.exp_name
    T = args.frames_per_batch
    K = args.num_workers
    N = args.total_frames // T // K if not args.DEBUG else 100
    annealing_num_steps = args.annealing_frames // T // K
    print(f"number of cpus: {mp.cpu_count()}")
    print(f"traj len: {T}, workers: {K}, iterator length: {N}")
    steps_per_collection = args.steps_per_collection
    gamma = args.gamma
    
    loss_kwargs = {}
    if args.distributional:
        if args.loss == "single":
            loss_class = DistributionalDQNLoss
        elif args.loss == "double":
            loss_class = DistributionalDoubleDQNLoss
        else:
            raise NotImplementedError
    else:
        loss_kwargs.update({'loss_type':args.loss_type})
        if args.loss == "single":
            loss_class = DQNLoss
        elif args.loss == "double":
            loss_class = DoubleDQNLoss
        else:
            raise NotImplementedError

    def make_transformed_env(video_tag="", writer=None):
        env = env_type(
            env_name, device="cpu", frame_skip=args.frame_skip, dtype=np.float32,
        )
        transforms = [
            ToTensorImage(),
            Resize(84, 84),
            GrayScale(),
            CatFrames(),
            ObservationNorm(loc=-1.0, scale=2.0),
            RewardScaling(0.0, 1.05),  # just for fun
            FiniteTensorDictCheck(),
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
        env = TransformedEnv(env, Compose(*transforms),)
        return env

    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"device is {device}")
    # Create an example environment
    env = make_transformed_env()
    env_specs = env.env.specs  # TODO: use env.sepcs
    linear_layer_class = torch.nn.Linear if not args.noisy else NoisyLinear
    # Create the actor network. For dqn, the actor is actually a Q-value-network. make_actor will figure out that
    value_network = make_dqn_actor(
        env_specs=env_specs,
        atoms=args.atoms if args.distributional else None,
        net_kwargs={
            "cnn_kwargs": {
                "bias_last_layer": True,
                "depth": None,
                "num_cells": [32, 64, 64] if not args.DEBUG else [32,],
                "kernel_sizes": [8, 4, 3] if not args.DEBUG else [8,],
                "strides": [4, 2, 1] if not args.DEBUG else [16,],
            },
            "mlp_kwargs": {
                "num_cells": 512 if not args.DEBUG else 16, 
                "layer_class": linear_layer_class},
        }
    ).to(device)

    value_network_explore = EGreedyWrapper(value_network, annealing_num_steps=annealing_num_steps).to(device)

    with torch.no_grad():
        td = env.reset()
        td_device = td.to(device)
        td_device = td_device.unsqueeze(0)
        td_device = value_network(td_device)  # for init
        td = td_device.squeeze(0).to("cpu")
        t0 = time.time()
        env.step(td)

    value_network_explore = value_network_explore.share_memory()

    try:
        env.close()
    except:
        pass
    del td
    
    if args.use_queue:
        collector_class = ParallelDataCollectorQueue
    else:
        collector_class = ParallelDataCollector
    if args.multi_step:
        ms = MultiStep(gamma=gamma, n_steps_max=args.n_steps_return )
    else:
        ms = None
    collector = collector_class(
        make_transformed_env,
        policy=value_network_explore,
        iterator_len=N,
        n_steps_max=-1,
        frames_per_batch=T,
        n_traj_per_batch=K,
        num_workers=args.num_workers,
        pass_tensors_through_queues=args.pass_tensors_through_queues,
        device=args.collector_device,
        reset_at_each_iter=args.reset_at_each_iter,
        postproc = ms
    )

    if not args.prb:
        buffer = ReplayBuffer(args.buffer_size, args.batch_size)
    else:
        buffer = PrioritizedReplayBuffer(args.buffer_size, args.batch_size)

    pbar = tqdm.tqdm(enumerate(collector), total=N)
    optim = torch.optim.Adam(
        params=value_network.parameters(), lr=args.lr, weight_decay=args.wd
    )
    init_reward = None
    init_rewards_noexplore = None

    frame_count = 0
    optim_count = 0
    t = 0.0
    loss_fn = None
    gv = 0.0
    loss = torch.zeros(1)

    log_dir = "/".join(["dqn_logging", exp_name, str(uuid.uuid1())])
    if args.video_file:
        video_tag = log_dir + "/"
    else:
        video_tag = ""
    writer = SummaryWriter(log_dir)
    torch.save(args, log_dir+"/args.t")

    env_record = make_transformed_env(video_tag=video_tag, writer=writer)
    t_optim = 0.0
    t_collection = 0.0
    _t_collection = time.time()

    if loss_fn is None:
        loss_fn = loss_class(
            value_network=value_network, gamma=gamma, **loss_kwargs
        )

    if args.loss == "double":
        print(f"optim_count: {optim_count}, updating target network")
        loss_fn.copy()

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=10,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/'.join([log_dir, 'profile'])),
            with_stack=True
    ) as profiler:
        for i, b in pbar:
            _t_collection = time.time() - _t_collection
            t_collection = t_collection * 0.9 + _t_collection * 0.1
            reward_avg = (b.masked_select(b.get("mask").squeeze(-1)).get("reward")).mean().item()
            if i > 0:
                if init_reward is None:
                    init_reward = reward_avg
                pbar.set_description(
                    f"reward (expl): {reward_avg:4.4f} (init={init_reward:4.2f}), "
                    f"reward (no expl): {rewards_noexplore:4.4f} (init={init_rewards_noexplore:4.2f}), "
                    f"loss: {loss.mean():4.4f}, "
                    f"gn: {gv:4.2f}, "
                    f"frames: {frame_count}, "
                    f"optims: {optim_count}, "
                    f"eps: {value_network_explore.eps.item():4.2f}, "
                    f"#rb: {buffer.size}, "
                    f"optim: {t_optim:4.2f}s, collection: {t_collection:4.2f}s"
                )

            _t_optim = time.time()

            # Split rollouts in single events
            b = b.masked_select(b.get("mask").squeeze(-1))

            frame_count += len(b)
            # Add single events to buffer
            buffer.add(b)

            value_network.apply(reset_noise)
            for j in range(steps_per_collection):
                # logging
                if (optim_count % 10000) == 0:
                    print("recording")
                    with torch.no_grad():
                        value_network.eval()
                        td_record = env_record.rollout(
                            policy=value_network, n_steps=10000, explore=False
                        )
                        value_network.train()
                    print("dumping")
                    env_record.transform.dump()
                    rewards_noexplore = td_record.get("reward").mean().item()
                    if init_rewards_noexplore is None:
                        init_rewards_noexplore = rewards_noexplore
                    writer.add_scalar("loss", loss.item(), optim_count)
                    writer.add_scalar("reward_avg", reward_avg, optim_count)
                    writer.add_scalar("rewards_noexplore", rewards_noexplore, optim_count)
                    torch.save(
                        {
                            "net": loss_fn.value_network.state_dict(),
                            "target_net": loss_fn.target_value_network.state_dict(),
                        },
                        log_dir+"/dqn_nets.t",
                    )

                optim_count += 1
                # Sample from buffer
                td = buffer.sample()

                # Train value network
                loss = loss_fn(td)
                loss.backward()
                gv = torch.nn.utils.clip_grad.clip_grad_norm_(
                    value_network.parameters(), args.grad_clip_norm
                )
                optim.step()
                optim.zero_grad()

                if (
                    args.loss == "double"
                    and (optim_count % args.value_network_update_interval) == 0
                ):
                    print(f"optim_count: {optim_count}, updating target network")
                    loss_fn.copy()

            value_network_explore.step()
            _t_optim = time.time() - _t_optim
            t_optim = t_optim * 0.9 + _t_optim * 0.1
            _t_collection = time.time()
            profiler.step()

import argparse
import os
import sys
import uuid

import numpy as np
import torch
import tqdm
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.data import MultiStep
# from torchrl.data.replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer, TensorDictPrioritizedReplayBuffer
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
from torchrl.envs import GymEnv, RetroEnv, ParallelEnv
from torchrl.modules import EGreedyWrapper, make_dqn_actor, NoisyLinear, \
    reset_noise
from torchrl.objectives import DQNLoss, DoubleDQNLoss, DistributionalDQNLoss, DistributionalDoubleDQNLoss, SoftUpdate, \
    HardUpdate
from torchrl.record.recorder import VideoRecorder, TensorDictRecorder

if sys.platform == "darwin":
    os.system("defaults write org.python.python ApplePersistenceIgnoreState NO")

import time

parser = argparse.ArgumentParser()
# parser.add_argument("--env_name", type=str, default="Pong-v0")
parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
parser.add_argument("--env_type", type=str, default="gym")
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--loss", type=str, default="double")
parser.add_argument("--soft_update", action="store_true")
parser.add_argument(
    "--loss_type", type=str, default="l2", choices=["l1", "l2", "smooth_l1"]
)
parser.add_argument("--collector_device", type=str, default="cpu")
parser.add_argument("--value_network_update_interval", type=int, default=1000)
parser.add_argument("--steps_per_collection", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--buffer_size", type=int, default=1000000)
parser.add_argument("--frame_skip", type=int, default=4)

parser.add_argument("--total_frames", type=int, default=50000000)
parser.add_argument("--annealing_frames", type=int, default=1000000)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--traj_len", type=int, default=-1)
parser.add_argument("--frames_per_batch", type=int, default=200)

parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--grad_clip_norm", type=float, default=100.0)

parser.add_argument("--reset_at_each_iter", action="store_true")
parser.add_argument("--concurrent", action="store_true")
parser.add_argument("--DEBUG", action="store_true")

parser.add_argument("--noisy", action="store_true")
parser.add_argument("--distributional", action="store_true")
parser.add_argument("--atoms", type=int, default=51)
parser.add_argument("--multi_step", action="store_true")
parser.add_argument("--prb", action="store_true")
parser.add_argument("--n_steps_return", type=int, default=3)

parser.add_argument("--share_individual_td", action="store_true")
parser.add_argument("--memmap", action="store_true")
parser.add_argument("--env_per_collector", default=1, type=int)
parser.add_argument("--collector_update_interval", type=int, default=32)

parser.add_argument("--seed", type=int, default=42)

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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_name = args.env_name
    env_type = env_type_map[args.env_type]
    exp_name = args.exp_name
    T = args.frames_per_batch
    K = args.num_workers
    N = args.total_frames // T // K if not args.DEBUG else 100
    annealing_num_steps = args.annealing_frames // T // args.env_per_collector
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
        loss_kwargs.update({'loss_type': args.loss_type})
        if args.loss == "single":
            loss_class = DQNLoss
        elif args.loss == "double":
            loss_class = DoubleDQNLoss
        else:
            raise NotImplementedError


    def make_transformed_env(video_tag="", writer=None, catframes=True):
        env = env_type(
            env_name, device="cpu", frame_skip=args.frame_skip, dtype=np.float32,
        )
        transforms = [
            ToTensorImage(keys=["next_observation_pixels"]),
            Resize(84, 84, keys=["next_observation_pixels"]),
            GrayScale(keys=["next_observation_pixels"]),
        ]
        if catframes:
            transforms += [
                CatFrames(keys=["next_observation_pixels"]),
                ObservationNorm(loc=-1.0, scale=2.0, keys=["next_observation_pixels"], observation_spec_key="pixels"),
            ]
        transforms += [
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
        env = TransformedEnv(env, Compose(*transforms), )
        return env


    def make_parallel_env(num_workers=args.env_per_collector, **kwargs):
        kwargs['catframes'] = False
        return TransformedEnv(
            ParallelEnv(num_workers=num_workers, create_env_fn=make_transformed_env, create_env_kwargs=kwargs,
                        device='cpu', share_individual_td=args.share_individual_td,
                        # pin_memory=torch.cuda.device_count() > 0,
                        shared_memory=not args.memmap,
                        memmap=args.memmap),
            Compose(
                CatFrames(keys=["next_observation_pixels"]),
                ObservationNorm(loc=-1.0, scale=2.0, keys=["next_observation_pixels"], observation_spec_key="pixels"),
            ))


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
    value_model = make_dqn_actor(
        env_specs=env_specs,
        atoms=args.atoms if args.distributional else None,
        net_kwargs={
            "cnn_kwargs": {
                "bias_last_layer": True,
                "depth": None,
                "num_cells": [32, 64, 64] if not args.DEBUG else [32, ],
                "kernel_sizes": [8, 4, 3] if not args.DEBUG else [8, ],
                "strides": [4, 2, 1] if not args.DEBUG else [16, ],
            },
            "mlp_kwargs": {
                "num_cells": 512 if not args.DEBUG else 16,
                "layer_class": linear_layer_class},
        },
        in_key="observation_pixels",
    ).to(device)

    value_model_explore = EGreedyWrapper(value_model, annealing_num_steps=annealing_num_steps).to(device)

    with torch.no_grad():
        td = env.reset()
        td_device = td.to(device)
        td_device = td_device.unsqueeze(0)
        td_device = value_model(td_device)  # for init
        td = td_device.squeeze(0).to("cpu")
        t0 = time.time()
        env.step(td)

    value_model_explore = value_model_explore.share_memory()
    print(f'value model: {value_model_explore}')

    try:
        env.close()
    except:
        pass
    del td

    if args.concurrent:
        collector_class = MultiaSyncDataCollector
        if args.env_per_collector == 1:
            env_fn = [make_transformed_env for _ in range(args.num_workers)]
            seed = list(range(args.seed, args.seed + args.num_workers))
        else:
            assert args.num_workers >= args.env_per_collector
            env_fn = [make_parallel_env for _ in range(- (args.num_workers // -args.env_per_collector))]
            seed = [list(range(args.env_per_collector*i, args.env_per_collector*(i+1))) for i in range(len(env_fn))]
    else:
        collector_class = SyncDataCollector
        assert args.num_workers == args.env_per_collector
        env_fn = make_parallel_env
        seed = list(range(args.seed, args.seed + args.num_workers))

    if args.multi_step:
        ms = MultiStep(gamma=gamma, n_steps_max=args.n_steps_return, device=args.collector_device)
    else:
        ms = None

    collector = collector_class(
        create_env_fn=env_fn,
        policy=value_model_explore,
        iterator_len=-1,
        max_steps_per_traj=-1,
        frames_per_batch=T,
        reset_at_each_iter=args.reset_at_each_iter,
        batcher=ms,
        device=args.collector_device,
    )
    collector.set_seed(seed)

    if not args.prb:
        buffer = ReplayBuffer(args.buffer_size, collate_fn=lambda x: torch.stack(x, 0), pin_memory=device != "cpu")
    else:
        buffer = TensorDictPrioritizedReplayBuffer(args.buffer_size, alpha=0.7, beta=0.5,
                                                   collate_fn=lambda x: torch.stack(x, 0),
                                                   pin_memory=device != "cpu")

    pbar = tqdm.tqdm(total=args.total_frames)
    optim = torch.optim.Adam(
        params=value_model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    init_reward = None
    init_rewards_noexplore = None

    frame_count = 0
    optim_count = 0
    t = 0.0
    gv = 0.0
    loss = torch.zeros(1)

    log_dir = "/".join(["dqn_logging", exp_name, str(uuid.uuid1())])
    if args.record_video:
        video_tag = log_dir + "/"
    else:
        video_tag = ""
    writer = SummaryWriter(log_dir)
    torch.save(args, log_dir + "/args.t")

    env_record = make_transformed_env(video_tag=video_tag, writer=writer)
    t_optim = 0.0
    t_collection = 0.0
    _t_collection = time.time()

    loss_module = loss_class(
        value_network=value_model, gamma=gamma, **loss_kwargs
    )
    target_net_updater = None
    if args.loss == "double":
        if args.soft_update:
            target_net_updater = SoftUpdate(loss_module)
        else:
            target_net_updater = HardUpdate(loss_module, args.value_network_update_interval)

    if args.loss == "double":
        print(f"optim_count: {optim_count}, updating target network")
        target_net_updater.init_()

    for i, b in enumerate(collector):
        pbar.update(b.numel())
        if (i + 1) % args.collector_update_interval == 0:
            collector.update_policy_weights_()
        _t_collection = time.time() - _t_collection
        t_collection = t_collection * 0.9 + _t_collection * 0.1
        reward_avg = (b.get("reward")[b.get('mask')]).mean().item()
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
                f"eps: {value_model_explore.eps.item():4.2f}, "
                f"#rb: {len(buffer)}, "
                f"optim: {t_optim:4.2f}s, collection: {t_collection:4.2f}s"
            )

        _t_optim = time.time()

        # Split rollouts in single events
        b = b.cpu().masked_select(b.get("mask").squeeze(-1))

        frame_count += b.shape[0]
        # Add single events to buffer
        buffer.extend(b)

        value_model.apply(reset_noise)
        for j in range(steps_per_collection):
            # logging
            if (optim_count % 10000) == 0:
                print("recording")
                with torch.no_grad():
                    value_model.eval()
                    td_record = env_record.rollout(
                        policy=value_model, n_steps=10000, explore=False
                    )
                    value_model.train()
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
                        "net": loss_module.value_network.state_dict(),
                        "target_net": loss_module.target_value_network.state_dict(),
                    },
                    log_dir + "/dqn_nets.t",
                )

            optim_count += 1
            # Sample from buffer
            td = buffer.sample(args.batch_size)

            # Train value network
            loss = loss_module(td)
            if args.prb:
                buffer.update_priority(td)

            loss.backward()
            gv = torch.nn.utils.clip_grad.clip_grad_norm_(
                value_model.parameters(), args.grad_clip_norm
            )
            optim.step()
            optim.zero_grad()
            if target_net_updater is not None:
                target_net_updater.step()

        value_model_explore.step()
        _t_optim = time.time() - _t_optim
        t_optim = t_optim * 0.9 + _t_optim * 0.1
        _t_collection = time.time()
        if frame_count >= args.total_frames:
            break

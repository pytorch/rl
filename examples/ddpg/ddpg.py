import argparse
import os
import sys
import uuid

import torch
import tqdm
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from torchrl.collectors import aSyncDataCollector, SyncDataCollector
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
    DoubleToFloat,
    CatTensors,
)
from torchrl.envs import GymEnv, RetroEnv, DMControlEnv, ParallelEnv
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper, make_ddpg_actor, NoisyLinear, \
    reset_noise
from torchrl.objectives import DDPGLoss, DoubleDDPGLoss, SoftUpdate, \
    HardUpdate
from torchrl.record.recorder import VideoRecorder, TensorDictRecorder

if sys.platform == "darwin":
    os.system("defaults write org.python.python ApplePersistenceIgnoreState NO")

import time

parser = argparse.ArgumentParser()
# parser.add_argument("--env_name", type=str, default="Pong-v0")
parser.add_argument("--library", type=str, default="dm_control")
parser.add_argument("--env_name", type=str, default="cheetah")
parser.add_argument("--env_task", type=str, default="run")

parser.add_argument("--record_video", action="store_true")
parser.add_argument("--from_vector", action="store_true")
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--loss", type=str, default="double")
parser.add_argument("--soft_update", action="store_true")
parser.add_argument(
    "--loss_type", type=str, default="l2", choices=["l1", "l2", "smooth_l1"]
)
parser.add_argument("--collector_device", type=str, default="cpu")
parser.add_argument("--value_network_update_interval", type=int, default=1000)
parser.add_argument("--steps_per_collection", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--buffer_size", type=int, default=1000000)
parser.add_argument("--frame_skip", type=int, default=4)
parser.add_argument("--max_steps_per_traj", type=int, default=250)

parser.add_argument("--log_interval", type=int, default=1000)
parser.add_argument("--record_steps", type=int, default=1000)

parser.add_argument("--total_frames", type=int, default=50000000)
parser.add_argument("--annealing_frames", type=int, default=1000000)
parser.add_argument("--num_workers", type=int, default=16)
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
parser.add_argument("--init_random_frames", type=int, default=5000)

parser.add_argument("--init_env_steps", type=int, default=250, help="number of random steps to compute"
                                                                    "normalizing constants")

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
        raise NotImplementedError
    else:
        loss_kwargs.update({'loss_type': args.loss_type})
        if args.loss == "single":
            loss_class = DDPGLoss
        elif args.loss == "double":
            loss_class = DoubleDDPGLoss
        else:
            raise NotImplementedError


    def make_transformed_env(video_tag="", writer=None, stats=None):
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
        transforms += [
            RewardScaling(0.0, 1.05),  # just for fun
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
                    DoubleToFloat(keys=["action", "reward"]),  # DMControl requires double-precision
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


    def make_parallel_env(**kwargs):
        return ParallelEnv(num_workers=args.num_workers, create_env_fn=make_transformed_env, create_env_kwargs=kwargs)


    if torch.cuda.device_count() > 0:
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"device is {device}")
    # Create an example environment
    env = make_transformed_env()
    env_specs = env.specs  # TODO: use env.sepcs
    linear_layer_class = torch.nn.Linear if not args.noisy else NoisyLinear
    # Create the actor network. For dqn, the actor is actually a Q-value-network. make_actor will figure out that

    actor_model, value_model = make_ddpg_actor(
        env_specs=env_specs,
        atoms=args.atoms if args.distributional else None,
        from_pixels=not args.from_vector,
        actor_net_kwargs={
            "mlp_net_kwargs": {
                "layer_class": linear_layer_class},
        },
        value_net_kwargs={
            "mlp_net_kwargs": {
                "layer_class": linear_layer_class},
        },
    )
    actor_model = actor_model.to(device)
    value_model = value_model.to(device)

    actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
        actor_model,
        annealing_num_steps=annealing_num_steps).to(device)

    with torch.no_grad():
        td = env.reset()
        td_device = td.to(device)
        td_device = td_device.unsqueeze(0)
        td_device = actor_model(td_device)  # for init
        value_model(td_device)  # for init
        td = td_device.squeeze(0).to("cpu")
        t0 = time.time()
        env.step(td)  # for sanity check

        stats = None
        if args.from_vector:
            td = env.rollout(n_steps=args.init_env_steps)
            stats = {"loc": td.get("observation_vector").mean(0), "scale": td.get("observation_vector").std(0)}

    actor_model_explore = actor_model_explore.share_memory()

    # get rid of env
    try:
        env.close()
    except:
        pass
    del td

    if args.concurrent:
        collector_class = aSyncDataCollector
    else:
        collector_class = SyncDataCollector
    if args.multi_step:
        ms = MultiStep(gamma=gamma, n_steps_max=args.n_steps_return, device=args.collector_device)
    else:
        ms = None
    collector = collector_class(
        create_env_fn=make_parallel_env,
        create_env_kwargs={"stats": stats},
        policy=actor_model_explore,
        iterator_len=N,
        max_steps_per_traj=args.max_steps_per_traj,
        frames_per_batch=T,
        batcher=ms,
        device=args.collector_device,
    )

    if not args.prb:
        buffer = ReplayBuffer(args.buffer_size, collate_fn=lambda x: torch.stack(x, 0), pin_memory=device != "cpu")
    else:
        buffer = TensorDictPrioritizedReplayBuffer(args.buffer_size, alpha=0.7, beta=0.5,
                                                   collate_fn=lambda x: torch.stack(x, 0),
                                                   pin_memory=device != "cpu")

    pbar = tqdm.tqdm(enumerate(collector), total=N)
    params = list(actor_model.parameters()) + list(value_model.parameters())
    optim = torch.optim.Adam(
        params=params,
        lr=args.lr, weight_decay=args.wd
    )
    init_reward = None
    init_rewards_noexplore = None

    frame_count = 0
    optim_count = 0
    t = 0.0
    gv = 0.0
    # loss = torch.zeros(1)

    log_dir = "/".join(["ddpg_logging", exp_name, str(uuid.uuid1())])
    if args.record_video:
        video_tag = log_dir + "/"
    else:
        video_tag = ""
    writer = SummaryWriter(log_dir)
    torch.save(args, log_dir + "/args.t")

    env_record = make_transformed_env(video_tag=video_tag, writer=writer, stats=stats)
    td_test = env_record.rollout(None, 100)
    # print(td_test.get("observation_vector").mean(), td_test.get("observation_vector").std())
    t_optim = 0.0
    t_collection = 0.0
    _t_collection = time.time()

    loss_module = loss_class(
        actor_model, value_model, gamma=gamma, **loss_kwargs
    )
    target_net_updater = None
    if args.loss == "double":
        if args.soft_update:
            target_net_updater = SoftUpdate(loss_module, 1 - 1 / args.value_network_update_interval)
        else:
            target_net_updater = HardUpdate(loss_module, args.value_network_update_interval)

    if args.loss == "double":
        print(f"optim_count: {optim_count}, updating target network")
        target_net_updater.init_()
    for i, b in pbar:
        collector.update_policy_weights_()
        _t_collection = time.time() - _t_collection
        t_collection = t_collection * 0.9 + _t_collection * 0.1
        reward_avg = (b.masked_select(b.get("mask").squeeze(-1)).get("reward")).mean().item()
        if optim_count > 0:
            if init_reward is None:
                init_reward = reward_avg
            pbar.set_description(
                f"reward (expl): {reward_avg:4.4f} (init={init_reward:4.2f}), "
                f"reward (no expl): {rewards_noexplore:4.4f} (init={init_rewards_noexplore:4.2f}), "
                f"loss: {loss.mean():4.4f}, "
                f"gn: {gv:4.2f}, "
                f"frames: {frame_count}, "
                f"optims: {optim_count}, "
                f"eps: {actor_model_explore.eps.item():4.2f}, "
                f"#rb: {len(buffer)}, "
                f"optim: {t_optim:4.2f}s, collection: {t_collection:4.2f}s"
            )

        _t_optim = time.time()

        # Split rollouts in single events
        b = b[b.get("mask").squeeze(-1)]

        frame_count += b.shape[0]

        # Add single events to buffer
        buffer.extend(b)

        if frame_count > args.init_random_frames:
            actor_model.apply(reset_noise)
            value_model.apply(reset_noise)

            for j in range(steps_per_collection):
                # logging
                if optim_count > 0 and ((optim_count % args.log_interval) == 0 or optim_count == 1):
                    print("recording")
                    with torch.no_grad():
                        actor_model.eval()
                        td_record = env_record.rollout(
                            policy=actor_model, n_steps=args.record_steps, explore=False
                        )
                        actor_model.train()
                    print("dumping")
                    env_record.transform.dump()
                    rewards_noexplore = td_record.get("reward").mean().item()
                    if init_rewards_noexplore is None:
                        init_rewards_noexplore = rewards_noexplore
                    writer.add_scalar("loss", loss.item(), frame_count)
                    writer.add_scalar("loss_actor", loss_actor.item(), frame_count)
                    writer.add_scalar("loss_value", loss_value.item(), frame_count)
                    writer.add_scalar("reward_avg", reward_avg, frame_count)
                    writer.add_scalar("rewards_noexplore", rewards_noexplore, frame_count)
                    writer.add_scalar("grad norm", gv, frame_count)
                    torch.save(
                        {
                            "net": loss_module.value_network.state_dict(),
                            "target_net": loss_module.target_value_network.state_dict(),
                        },
                        log_dir + "/ddpg_nets.t",
                    )

                optim_count += 1
                # Sample from buffer
                td = buffer.sample(args.batch_size)

                # Train value network
                loss_actor, loss_value = loss_module(td)
                loss = loss_actor.mean() + loss_value.mean()
                if args.prb:
                    buffer.update_priority(td)
                loss.backward()
                gv = torch.nn.utils.clip_grad.clip_grad_norm_(
                    params, args.grad_clip_norm
                )
                optim.step()
                optim.zero_grad()
                if target_net_updater is not None:
                    target_net_updater.step()

            actor_model_explore.step()
        _t_optim = time.time() - _t_optim
        t_optim = t_optim * 0.9 + _t_optim * 0.1
        _t_collection = time.time()

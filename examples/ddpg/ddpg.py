import argparse
import datetime
import os
import sys
import uuid

import torch
import tqdm
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from torchrl.collectors import sync_async_collector, sync_sync_collector
from torchrl.data import MultiStep
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
parser.add_argument("--env_library", type=str, default="dm_control", choices=["dm_control", "gym"],
                    help="env_library used for the simulated environment. Default=dm_control")
parser.add_argument("--env_name", type=str, default="cheetah",
                    help="name of the environment to be created. Default=cheetah")
parser.add_argument("--env_task", type=str, default="run",
                    help="task (if any) for the environment. Default=run")

parser.add_argument("--record_video", action="store_true",
                    help="whether a video of the task should be rendered during logging.")
parser.add_argument("--from_vector", action="store_true",
                    help="whether the environment output should be pixels (default) or the state vector(s).")
parser.add_argument("--exp_name", type=str, default="",
                    help="experiment name. Used for logging directory. "
                         "A date and uuid will be joined to account for multiple experiments with the same name.")
parser.add_argument("--loss", type=str, default="double", choices=["double", "single"],
                    help="whether double or single DDPG loss should be used. Default=double")
parser.add_argument("--soft_update", action="store_true",
                    help="whether soft-update should be used with double DDPG loss.")
parser.add_argument(
    "--loss_function", type=str, default="smooth_l1", choices=["l1", "l2", "smooth_l1"],
    help="loss function for the value network. Either one of l1, l2 or smooth_l1 (default)."
)
parser.add_argument("--collector_device", type=str, default="cpu",
                    help="device on which the data collector should store the trajectories to be passed to this script."
                         "If the collector device differs from the policy device (cuda:0 if available), then the "
                         "weights of the collector policy are synchronized with collector.update_policy_weights_().")
parser.add_argument("--value_network_update_interval", type=int, default=1000,
                    help="how often the target value network weights are updated (in number of updates)."
                         "If soft-updates are used, the value is translated into a moving average decay by using the "
                         "formula decay=1-1/args.value_network_update_interval. Default=1000")
parser.add_argument("--optim_steps_per_collection", type=int, default=200,
                    help="Number of optimization steps in between two collection of data. See frames_per_batch below."
                         "Default=200")
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size of the TensorDict retrieved from the replay buffer. Default=64.")
parser.add_argument("--buffer_size", type=int, default=1000000,
                    help="buffer size, in number of frames stored. Default=1e6")
parser.add_argument("--frame_skip", type=int, default=4,
                    help="frame_skip for the environment. Note that this value does NOT impact the buffer size,"
                         "maximum steps per trajectory, frames per batch or any other factor in the algorithm,"
                         "e.g. if the total number of frames that has to be computed is 50e6 and the frame skip is 4,"
                         "the actual number of frames retrieved will be 200e6. Default=4.")

parser.add_argument("--max_frames_per_traj", type=int, default=250,
                    help="Number of steps before a reset of the environment is called (if it has not been flagged as "
                         "done before). Default=250.")
parser.add_argument("--frames_per_batch", type=int, default=200,
                    help="number of steps executed in the environment per collection."
                         "This value represents how many steps will the data collector execute and return in *each*"
                         "environment that has been created in between two rounds of optimization "
                         "(see the optim_steps_per_collection above). "
                         "On the one hand, a low value will enhance the data throughput between processes in async "
                         "settings, which can make the accessing of data a computational bottleneck. "
                         "High values will on the other hand lead to greater tensor sizes in memory and disk to be "
                         "written and read at each global iteration. One should look at the number of frames per second"
                         "in the log to assess the efficiency of the configuration.")

parser.add_argument("--log_interval", type=int, default=1000,
                    help="logging interval, in terms of optimization steps. Default=1000.")
parser.add_argument("--record_steps", type=int, default=250,
                    help="maximum number of steps used for the recorded environment. Default=250.")

parser.add_argument("--total_frames", type=int, default=50000000,
                    help="total number of frames collected for training. Does not account for frame_skip and should"
                         "be corrected accordingly. Default=50e6.")
parser.add_argument("--annealing_frames", type=int, default=1000000,
                    help="Number of frames used for annealing of the OrnsteinUhlenbeckProcess. Default=1e6.")
parser.add_argument("--num_workers", type=int, default=16,
                    help="Number of workers used for data collection. ")
parser.add_argument("--env_per_collector", default=1, type=int,
                    help="Number of environments per collector. If the env_per_collector is in the range: "
                         "1<env_per_collector<=num_workers, then the collector runs"
                         "ceil(num_workers/env_per_collector) in parallel and executes the policy steps synchronously "
                         "for each of these parallel wrappers. If env_per_collector=num_workers, no parallel wrapper is created.")

parser.add_argument("--gamma", type=float, default=0.99,
                    help="Decay factor for return computation. Default=0.99.")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate used for the optimizer. Default=2e-4.")
parser.add_argument("--wd", type=float, default=0.0,
                    help="Weight-decay to be used with the optimizer. Default=0.0.")
parser.add_argument("--grad_clip_norm", type=float, default=100.0,
                    help="value at which the total gradient norm should be clipped. Default=100.0")

parser.add_argument("--async_collection", action="store_true",
                    help="whether data collection should be done asynchrously. Asynchrounous data collection means "
                         "that the data collector will keep on running the environment with the previous weights "
                         "configuration while the optimization loop is being done. If the algorithm is trained "
                         "synchronously, data collection and optimization will occur iteratively, not concurrently.")

parser.add_argument("--noisy", action="store_true",
                    help="whether to use NoisyLinearLayers in the value network.")
parser.add_argument("--distributional", action="store_true",
                    help="whether a distributional loss should be used (TODO: not implemented yet).")
parser.add_argument("--atoms", type=int, default=51,
                    help="number of atoms used for the distributional loss (TODO)")
parser.add_argument("--multi_step", action="store_true",
                    help="whether or not multi-step rewards should be used.")
parser.add_argument("--n_steps_return", type=int, default=3,
                    help="If multi_step is set to True, this value defines the number of steps to look ahead for the "
                         "reward computation.")
parser.add_argument("--prb", action="store_true",
                    help="whether a Prioritized replay buffer should be used instead of a more basic circular one.")
parser.add_argument("--init_random_frames", type=int, default=5000,
                    help="Initial number of random frames used before the policy is being used. Default=5000.")

parser.add_argument("--init_env_steps", type=int, default=250,
                    help="number of random steps to compute normalizing constants")

env_library_map = {
    "gym": GymEnv,
    "retro": RetroEnv,
    "dm_control": DMControlEnv,
}

if __name__ == "__main__":
    mp.set_start_method("spawn")

    args = parser.parse_args()
    env_name = args.env_name
    env_library = env_library_map[args.env_library]
    env_task = args.env_task

    exp_name = args.exp_name
    T = args.frames_per_batch
    K = args.num_workers
    annealing_frames = args.annealing_frames
    print(f"number of cpus: {mp.cpu_count()}")
    print(f"traj len: {T}, workers: {K}")
    optim_steps_per_collection = args.optim_steps_per_collection
    gamma = args.gamma

    loss_kwargs = {}
    if args.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({'loss_function': args.loss_function})
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
        return ParallelEnv(
            num_workers=args.env_per_collector,
            create_env_fn=make_transformed_env,
            create_env_kwargs=kwargs)


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
        annealing_num_steps=annealing_frames).to(device)

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

    if args.async_collection:
        collector_helper = sync_async_collector
    else:
        collector_helper = sync_sync_collector

    if args.multi_step:
        ms = MultiStep(gamma=gamma, n_steps_max=args.n_steps_return, device=args.collector_device)
    else:
        ms = None

    collector_helper_kwargs = {
        "env_fns": make_transformed_env if args.env_per_collector == 1 else make_parallel_env,
        "env_kwargs": {},
        "policy": actor_model_explore,
        "max_steps_per_traj": args.max_frames_per_traj,
        "frames_per_batch": T,
        "total_frames": args.total_frames,
        "batcher": ms,
        "num_env_per_collector": 1,  # we already took care of building the make_parallel_env function above
        "num_collectors": - args.num_workers // -args.env_per_collector,
        "passing_device": args.collector_device,
        "device": args.collector_device,
    }

    collector = collector_helper(**collector_helper_kwargs)

    if not args.prb:
        buffer = ReplayBuffer(args.buffer_size, collate_fn=lambda x: torch.stack(x, 0), pin_memory=device != "cpu")
    else:
        buffer = TensorDictPrioritizedReplayBuffer(args.buffer_size, alpha=0.7, beta=0.5,
                                                   collate_fn=lambda x: torch.stack(x, 0),
                                                   pin_memory=device != "cpu")

    pbar = tqdm.tqdm(total=args.total_frames)
    params = list(actor_model.parameters()) + list(value_model.parameters())
    optim = torch.optim.Adam(
        params=params,
        lr=args.lr, weight_decay=args.wd
    )
    init_reward = None
    init_rewards_noexplore = None

    frame_count = 0
    optim_count = 0
    gv = 0.0

    log_dir = "/".join(
        ["ddpg_logging", exp_name, str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")), str(uuid.uuid1())])
    if args.record_video:
        video_tag = log_dir + "/"
    else:
        video_tag = ""
    writer = SummaryWriter(log_dir)
    torch.save(args, log_dir + "/args.t")

    env_record = make_transformed_env(video_tag=video_tag, writer=writer, stats=stats)
    td_test = env_record.rollout(None, 100)
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
        print(f"optim_count: {optim_count}, updating target network")
        target_net_updater.init_()
    else:
        assert not args.soft_update, "soft-update is supposed to be used with double DDPG loss. " \
                                     "Consider using --loss=double or discarding the soft_update flag."
    i = -1
    for b in collector:
        i += 1
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

        frame_count += b.numel()

        # Add single events to buffer
        buffer.extend(b)

        pbar.update(b.numel())

        if frame_count > args.init_random_frames:
            actor_model.apply(reset_noise)
            value_model.apply(reset_noise)

            for j in range(optim_steps_per_collection):
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
                td = buffer.sample(args.batch_size).contiguous()

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
            for _ in range(b.numel()):
                # 1 step per frame
                actor_model_explore.step()
        _t_optim = time.time() - _t_optim
        t_optim = t_optim * 0.9 + _t_optim * 0.1
        _t_collection = time.time()

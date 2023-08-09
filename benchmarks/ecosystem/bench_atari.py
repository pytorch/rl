# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Logs Gym Async env data collection speed with a simple policy.
#
import time

from argparse import ArgumentParser
from typing import Dict, List

import numpy as np

import torch

from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl._utils import timeit

parser = ArgumentParser()
parser.add_argument(
    "--env_name",
    choices=["ALE/Pong-v5"],
    default="ALE/Pong-v5",
)
parser.add_argument("--n_envs", default=4, type=int)
parser.add_argument("--log_sep", default=200, type=int)
parser.add_argument("--preemptive_threshold", default=0.7, type=float)
parser.add_argument("--total_frames", default=100_000, type=int)
parser.add_argument("--device", default="auto")
parser.add_argument(
    "--fpb", "--frames-per-batch", "--frames_per_batch", default=200, type=int
)
parser.add_argument(
    "--run",
    choices=["collector", "collector_preempt", "sb3", "penv", "tianshou"],
    default="penv",
)
parser.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard", "tb"])

env_maps = {
    "ALE/Pong-v5": {
        "distribution": Categorical,
        "key": ["logits"],
        "out_features": 6,
    },
}
if __name__ == "__main__":
    # Parallel environments
    args = parser.parse_args()
    env_name = args.env_name
    n_envs = args.n_envs
    out_features = env_maps[env_name]["out_features"]
    dist_class = env_maps[env_name]["distribution"]
    dist_key = env_maps[env_name]["key"]
    log_sep = args.log_sep
    fpb = args.fpb
    total_frames = args.total_frames
    run = args.run
    fps_list = []

    if args.device == "auto":
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda:0")
        )
    elif "," in args.device:
        # cuda:0,cuda:1 is interpreted as ["cuda:0", "cuda:1"]
        devices = args.device.split(",")
        device = [torch.device(device) for device in devices]
    else:
        device = torch.device(args.device)
    if args.logger == "wandb":
        from torchrl.record.loggers.wandb import WandbLogger

        Logger = WandbLogger
        logger_kwargs = {"project": "benchmark"}
    elif args.logger in ("tensorboard", "tb"):
        from torchrl.record.loggers.tensorboard import TensorboardLogger

        Logger = TensorboardLogger
        logger_kwargs = {}

    if run == "tianshou":
        import warnings

        from tianshou.data import Batch
        from tianshou_atari_wrapper import DQN, DQNPolicy, make_atari_env

        warnings.filterwarnings("ignore")
        net = DQN(12, 84, 84, out_features, device).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        # define policy
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
        )

        env = make_atari_env(
            env_name,
            num_envs=n_envs,
            scale=0,
            frame_stack=4,
        )

        logger = Logger(exp_name=f"tianshou-{env_name}", **logger_kwargs)

        obs, _ = env.reset()
        i = 0
        model_time = 0
        env_time = 0
        frames = 0
        cur_frames = 0
        while frames < total_frames:
            i += 1
            with timeit("policy"):
                t0 = time.time()
                action = policy(Batch(obs=obs)).act.cpu().numpy()
                t1 = time.time()
            with timeit("step"):
                obs, rewards, term, dones, info = env.step(action)
                if np.sum(dones):
                    env.reset(np.where(dones)[0])
                t2 = time.time()

            frames += len(dones)
            cur_frames += len(dones)

            model_time += t1 - t0
            env_time += t2 - t1

            if i % log_sep == 0:
                logger.log_scalar(
                    "model step fps", cur_frames / model_time, step=frames
                )
                logger.log_scalar("env step", cur_frames / env_time, step=frames)

                fps = cur_frames / (env_time + model_time)
                logger.log_scalar("total", fps, step=frames)
                logger.log_scalar("frames", frames)
                env_time = 0
                model_time = 0
                cur_frames = 0
                if i > 0:
                    # skip first
                    fps_list.append(fps)

            # vec_env.render("human")
        if args.logger == "wandb":
            logger.experiment.finish()
        env.close()
        del env
        del policy

    elif run == "sb3":

        from stable_baselines3 import A2C
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

        vec_env = make_atari_env(
            "PongNoFrameskip-v4", n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv
        )
        vec_env = VecFrameStack(vec_env, n_stack=4)

        model = A2C("CnnPolicy", vec_env, verbose=0)
        print("policy", model.policy)

        logger = Logger(exp_name=f"sb3-{env_name}", **logger_kwargs)

        obs = vec_env.reset()
        i = 0
        model_time = 0
        env_time = 0
        frames = 0
        cur_frames = 0
        while frames < total_frames:
            i += 1

            with timeit("policy"):
                t0 = time.time()
                action, _states = model.predict(obs)
                t1 = time.time()
            with timeit("step"):
                obs, rewards, dones, info = vec_env.step(action)
                t2 = time.time()

            frames += len(dones)
            cur_frames += len(dones)

            model_time += t1 - t0
            env_time += t2 - t1

            if i % log_sep == 0:
                logger.log_scalar(
                    "model step fps", cur_frames / model_time, step=frames
                )
                logger.log_scalar("env step", cur_frames / env_time, step=frames)

                fps = cur_frames / (env_time + model_time)
                if i > 0:
                    fps_list.append(fps)
                logger.log_scalar("total", fps, step=frames)
                logger.log_scalar("frames", frames)
                env_time = 0
                model_time = 0
                cur_frames = 0

            # vec_env.render("human")

        if args.logger == "wandb":
            logger.experiment.finish()
        vec_env.close()
        del vec_env
        del model

    elif run == "penv":
        from torchrl.envs import (
            CatFrames,
            Compose,
            DoubleToFloat,
            EnvCreator,
            GrayScale,
            ParallelEnv,
            Resize,
            ToTensorImage,
            TransformedEnv,
        )
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import ConvNet, MLP, ProbabilisticActor, TanhNormal

        # reproduce the actor
        backbone = nn.Sequential(
            ConvNet(
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                activation_class=nn.ReLU,
                num_cells=[32, 64, 64],
                device=device,
            ),
            MLP(out_features=out_features, num_cells=[512]),
        )
        module = TensorDictModule(
            backbone,
            in_keys=["pixels"],
            out_keys=dist_key,
        )
        actor = ProbabilisticActor(
            module, in_keys=dist_key, distribution_class=dist_class
        ).to(device)

        def make_env():
            return GymEnv(
                env_name, frame_skip=4, categorical_action_encoding=True, device=device
            )

        env = TransformedEnv(
            ParallelEnv(n_envs, EnvCreator(make_env)),
            Compose(
                ToTensorImage(),
                Resize(84, 84),
                GrayScale(),
                CatFrames(N=4, dim=-3, in_keys=["pixels"]),
            ),
        )

        logger = Logger(exp_name=f"torchrl-penv-{env_name}", **logger_kwargs)

        prev_t = time.time()
        frames = 0
        cur = 0
        fpb = fpb // env.batch_size.numel()
        i = 0
        with torch.no_grad():
            while frames < total_frames:
                if i == 1:
                    timeit.erase()
                data = env.rollout(fpb, actor, break_when_any_done=False)
                # data = env._single_rollout(fpb, actor, break_when_any_done=False)
                frames += data.numel()
                cur += data.numel()
                if i % 20 == 0:
                    t = time.time()
                    fps = cur / (t - prev_t)
                    if i > 0:
                        fps_list.append(fps)
                    logger.log_scalar("total", fps, step=frames)
                    logger.log_scalar("frames", frames)
                    prev_t = t
                    cur = 0
                i += 1
        if args.logger == "wandb":
            logger.experiment.finish()
        del env, actor, logger, module, backbone

    elif run == "collector":
        from torchrl.collectors import MultiaSyncDataCollector
        from torchrl.envs import (
            CatFrames,
            Compose,
            DoubleToFloat,
            EnvCreator,
            GrayScale,
            ParallelEnv,
            Resize,
            ToTensorImage,
            TransformedEnv,
        )
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import ConvNet, MLP, ProbabilisticActor

        # reproduce the actor
        backbone = nn.Sequential(
            ConvNet(
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                activation_class=nn.ReLU,
                num_cells=[32, 64, 64],
                device=device,
            ),
            MLP(out_features=out_features, num_cells=[512]),
        )
        module = TensorDictModule(
            backbone,
            in_keys=["pixels"],
            out_keys=dist_key,
        )
        actor = ProbabilisticActor(
            module, in_keys=dist_key, distribution_class=dist_class
        )

        def make_env():
            return TransformedEnv(
                GymEnv(
                    env_name,
                    frame_skip=4,
                    categorical_action_encoding=True,
                    device=device,
                ),
                Compose(
                    ToTensorImage(),
                    Resize(84, 84),
                    GrayScale(),
                    CatFrames(N=4, dim=-3, in_keys=["pixels"]),
                ),
            )

        # round up fpb
        fpb = -(fpb // -n_envs) * n_envs
        collector = MultiaSyncDataCollector(
            n_envs * [make_env],
            actor,
            total_frames=total_frames,
            frames_per_batch=fpb,
            device="cuda:0" if torch.cuda.device_count() else "cpu",
            storing_device="cpu",
        )

        logger = Logger(exp_name=f"torchrl-async-{env_name}", **logger_kwargs)

        prev_t = time.time()
        frames = 0
        cur = 0
        for i, data in enumerate(collector):
            frames += data.numel()
            cur += data.numel()
            if i % 20 == 0:
                t = time.time()
                fps = cur / (t - prev_t)
                if i > 0:
                    fps_list.append(fps)
                logger.log_scalar("total", fps, step=frames)
                logger.log_scalar("frames", frames)
                prev_t = t
                cur = 0

        if args.logger == "wandb":
            logger.experiment.finish()
        collector.shutdown()
        del collector, actor, logger, module, backbone

    elif run == "collector_preempt":
        from torchrl.collectors import MultiSyncDataCollector
        from torchrl.envs import (
            CatFrames,
            Compose,
            EnvCreator,
            GrayScale,
            ParallelEnv,
            Resize,
            ToTensorImage,
            TransformedEnv,
        )
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import ConvNet, MLP, ProbabilisticActor

        # reproduce the actor
        backbone = nn.Sequential(
            ConvNet(
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                activation_class=nn.ReLU,
                num_cells=[32, 64, 64],
                device=device,
            ),
            MLP(out_features=out_features, num_cells=[512]),
        )
        module = TensorDictModule(
            backbone,
            in_keys=["pixels"],
            out_keys=dist_key,
        )
        actor = ProbabilisticActor(
            module, in_keys=dist_key, distribution_class=dist_class
        )

        def make_env():
            return TransformedEnv(
                GymEnv(
                    env_name,
                    frame_skip=4,
                    categorical_action_encoding=True,
                    device=device,
                ),
                Compose(
                    ToTensorImage(),
                    Resize(84, 84),
                    GrayScale(),
                    CatFrames(N=4, dim=-3, in_keys=["pixels"]),
                ),
            )

        # round up fpb
        fpb = -(fpb // -n_envs) * n_envs
        collector = MultiSyncDataCollector(
            n_envs * [make_env],
            actor,
            total_frames=total_frames,
            frames_per_batch=fpb,
            device="cuda:0" if torch.cuda.device_count() else "cpu",
            storing_device="cpu",
            preemptive_threshold=args.preemptive_threshold,
        )

        logger = Logger(exp_name=f"torchrl-async-{env_name}", **logger_kwargs)

        prev_t = time.time()
        frames = 0
        cur = 0
        for i, data in enumerate(collector):
            frames += data.numel()
            cur += data.numel()
            if i % 20 == 0:
                t = time.time()
                fps = cur / (t - prev_t)
                if i > 0:
                    fps_list.append(fps)
                logger.log_scalar("total", fps, step=frames)
                logger.log_scalar("frames", frames)
                prev_t = t
                cur = 0

        if args.logger == "wandb":
            logger.experiment.finish()
        collector.shutdown()
        del collector, actor, logger, module, backbone

    print("\n\n", "=" * 20, "\n" + "fps:", np.mean(fps_list))
    timeit.print()

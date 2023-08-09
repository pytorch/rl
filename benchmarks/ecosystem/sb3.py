# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Logs Gym Async env data collection speed with a simple policy.
#
import time

from argparse import ArgumentParser

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl._utils import timeit
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal
from torchrl.record.loggers.wandb import WandbLogger

parser = ArgumentParser()
parser.add_argument("--env_name", default="CartPole-v1")
parser.add_argument("--n_envs", default=4, type=int)
parser.add_argument("--log_sep", default=200, type=int)
parser.add_argument("--total_frames", default=100_000, type=int)
parser.add_argument("--run", choices=["collector", "sb3", "penv"], default="penv")

env_maps = {
    "CartPole-v1": {
        "in_features": 4,
        "out_features": 2,
        "distribution": Categorical,
        "key": ["logits"],
    },
    "Pendulum-v1": {
        "in_features": 3,
        "out_features": 2,
        "distribution": TanhNormal,
        "key": ["loc", "scale"],
    },
}
if __name__ == "__main__":
    # Parallel environments
    args = parser.parse_args()
    env_name = args.env_name
    n_envs = args.n_envs
    in_features = env_maps[env_name]["in_features"]
    out_features = env_maps[env_name]["out_features"]
    dist_class = env_maps[env_name]["distribution"]
    dist_key = env_maps[env_name]["key"]
    log_sep = args.log_sep
    fpb = log_sep * n_envs
    total_frames = args.total_frames
    run = args.run
    device = (
        torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device("cuda:0")
    )

    if run == "sb3":
        vec_env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

        model = PPO("MlpPolicy", vec_env, verbose=0)
        print("policy", model.policy)

        logger = WandbLogger(exp_name=f"sb3-{env_name}", project="benchmark")

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
                logger.log_scalar(
                    "total", cur_frames / (env_time + model_time), step=frames
                )
                logger.log_scalar("frames", frames)
                env_time = 0
                model_time = 0
                cur_frames = 0

            # vec_env.render("human")

        logger.experiment.finish()
        vec_env.close()
        del vec_env
        del model

    elif run == "penv":
        # reproduce the actor
        backbone = MLP(
            in_features=in_features,
            out_features=out_features,
            depth=2,
            num_cells=64,
            activation_class=nn.Tanh,
            device=device,
        )
        if dist_class is TanhNormal:
            backbone = nn.Sequential(backbone, NormalParamExtractor())
        module = TensorDictModule(
            backbone,
            in_keys=["observation"],
            out_keys=dist_key,
        )
        actor = ProbabilisticActor(
            module, in_keys=dist_key, distribution_class=dist_class
        )

        def make_env():
            return GymEnv(env_name, categorical_action_encoding=True, device=device)

        env = ParallelEnv(n_envs, EnvCreator(make_env))

        logger = WandbLogger(exp_name=f"torchrl-penv-{env_name}", project="benchmark")

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
                    logger.log_scalar("total", cur / (t - prev_t), step=frames)
                    logger.log_scalar("frames", frames)
                    prev_t = t
                    cur = 0
                i += 1
        del env

    elif run == "collector":
        # reproduce the actor
        backbone = MLP(
            in_features=in_features,
            out_features=out_features,
            depth=2,
            num_cells=64,
            activation_class=nn.Tanh,
            device=device,
        )
        if dist_class is TanhNormal:
            backbone = nn.Sequential(backbone, NormalParamExtractor())
        module = TensorDictModule(
            backbone,
            in_keys=["observation"],
            out_keys=dist_key,
        )
        actor = ProbabilisticActor(
            module, in_keys=dist_key, distribution_class=dist_class
        )
        collector = MultiaSyncDataCollector(
            n_envs
            * [
                lambda: GymEnv(
                    env_name, categorical_action_encoding=True,
                    device=device,
                )
            ],
            actor,
            total_frames=total_frames,
            frames_per_batch=fpb,
            storing_device=device,
            device=device,
        )

        logger = WandbLogger(exp_name=f"torchrl-async-{env_name}", project="benchmark")

        prev_t = time.time()
        frames = 0
        cur = 0
        for i, data in enumerate(collector):
            frames += data.numel()
            cur += data.numel()
            if i % 20 == 0:
                t = time.time()
                logger.log_scalar("total", cur / (t - prev_t), step=frames)
                logger.log_scalar("frames", frames)
                prev_t = t
                cur = 0

    timeit.print()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Logs Gym Async env data collection speed with a simple policy.
#
import time
import torch
from torch import nn
from torch.distributions import Categorical

from tensordict.nn import TensorDictModule
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, MLP
from torchrl.record.loggers.wandb import WandbLogger

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--env_name', default="CartPole-v1")
parser.add_argument('--n_envs', default=4, type=int)
parser.add_argument('--in_features', default=4, type=int)
parser.add_argument('--out_features', default=2, type=int)
parser.add_argument('--log_sep', default=200, type=int)
parser.add_argument('--total_frames', default=100_000, type=int)
parser.add_argument('--run', choices=['collector', 'sb3', 'penv'],default='penv')

if __name__ == "__main__":
    # Parallel environments
    args = parser.parse_args()
    env_name = args.env_name
    n_envs = args.n_envs
    in_features = args.in_features
    out_features = args.out_features
    log_sep = args.log_sep
    fpb = log_sep * n_envs
    total_frames = args.total_frames
    run = args.run
    device = torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device("cuda")

    if run == 'sb3':
        vec_env = make_vec_env(env_name, n_envs=n_envs)

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

            model_time += t1-t0
            env_time += t2-t1

            if i % log_sep == 0:
                logger.log_scalar("model step fps", cur_frames/model_time, step=frames)
                logger.log_scalar("env step", cur_frames/env_time, step=frames)
                logger.log_scalar("total", cur_frames/(env_time + model_time), step=frames)
                logger.log_scalar('frames', frames)
                env_time = 0
                model_time = 0
                cur_frames = 0

            # vec_env.render("human")

        logger.experiment.finish()
        vec_env.close()
        del vec_env
        del model

    elif run == 'penv':
        # reproduce the actor
        module = TensorDictModule(MLP(in_features=in_features, out_features=out_features, depth=2, num_cells=64, activation_class=nn.Tanh, device=device), in_keys=['observation'], out_keys=['logits'])
        actor = ProbabilisticActor(module, in_keys=["logits"], distribution_class=Categorical)
        def make_env():
            return GymEnv(env_name, categorical_action_encoding=True, device=device)
        env = ParallelEnv(n_envs, EnvCreator(make_env))

        logger = WandbLogger(exp_name=f"torchrl-penv-{env_name}", project="benchmark")

        prev_t = time.time()
        frames = 0
        cur = 0
        fpb = fpb//env.batch_size.numel()
        i = 0
        with torch.no_grad():
            while frames < total_frames:
                data = env.rollout(fpb, actor, break_when_any_done=False)
                # data = env._single_rollout(fpb, actor, break_when_any_done=False)
                frames += data.numel()
                cur += data.numel()
                if i % 20 == 0:
                    t = time.time()
                    logger.log_scalar("total", cur / (t-prev_t), step=frames)
                    logger.log_scalar('frames', frames)
                    prev_t = t
                    cur = 0
                i += 1
        del env

    elif run == 'collector':
        module = TensorDictModule(MLP(in_features=in_features, out_features=out_features, depth=2, num_cells=64, activation_class=nn.Tanh, device=device), in_keys=['observation'], out_keys=['logits'])
        actor = ProbabilisticActor(module, in_keys=["logits"], distribution_class=Categorical)
        collector = MultiaSyncDataCollector(n_envs * [lambda: GymEnv(env_name, categorical_action_encoding=True, device=device)], actor, total_frames=total_frames, frames_per_batch=fpb)

        logger = WandbLogger(exp_name=f"torchrl-async-{env_name}", project="benchmark")

        prev_t = time.time()
        frames = 0
        cur = 0
        for i, data in enumerate(collector):
            frames += data.numel()
            cur += data.numel()
            if i % 20 == 0:
                t = time.time()
                logger.log_scalar("total", cur / (t-prev_t), step=frames)
                logger.log_scalar('frames', frames)
                prev_t = t
                cur = 0

    timeit.print()
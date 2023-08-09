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

import torch

from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl._utils import timeit
from torchrl.modules import TanhNormal

parser = ArgumentParser()
parser.add_argument("--env_name", choices=["Pendulum-v1", "CartPole-v1", "HalfCheetah-v4", "LunarLander-v2"], default="CartPole-v1")
parser.add_argument("--n_envs", default=4, type=int)
parser.add_argument("--log_sep", default=200, type=int)
parser.add_argument("--preemptive_threshold", default=0.7, type=float)
parser.add_argument("--total_frames", default=100_000, type=int)
parser.add_argument("--device", default="auto")
parser.add_argument("--fpb", "--frames-per-batch", "--frames_per_batch", default=200, type=int)
parser.add_argument(
    "--run", choices=["collector", "collector_preempt", "sb3", "penv", "tianshou"], default="penv"
)
parser.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard", "tb"])

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
    "HalfCheetah-v4": {
        "in_features": 17,
        "out_features": 6,
        "distribution": TanhNormal,
        "key": ["loc", "scale"],
    },
    "LunarLander-v2": {
        "in_features": 8,
        "out_features": 4,
        "distribution": Categorical,
        "key": ["logits"],
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
    fpb = args.fpb
    total_frames = args.total_frames
    run = args.run
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

        import gym
        import numpy as np

        from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
        from tianshou.env import SubprocVectorEnv
        from tianshou.policy import BasePolicy
        from tianshou.utils.net.common import Net
        from tianshou.utils.net.continuous import Actor as ContActor
        from tianshou.utils.net.discrete import Actor as DiscreteActor

        class REINFORCEPolicy(BasePolicy):
            """Implementation of REINFORCE algorithm."""

            def __init__(
                self,
                model: torch.nn.Module,
                optim: torch.optim.Optimizer,
            ):
                super().__init__()
                self.actor = model
                self.optim = optim
                # action distribution
                self.dist_fn = dist_class

            def forward(self, batch: Batch) -> Batch:
                """Compute action over the given batch data."""
                preds, _ = self.actor(batch.obs)
                if self.dist_fn == TanhNormal:
                    preds = preds.chunk(2, dim=-1)
                    preds = [preds[0], preds[1].exp()]
                    dist = self.dist_fn(*preds)
                else:
                    dist = self.dist_fn(preds)
                act = dist.sample()
                return Batch(act=act, dist=dist)

            def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices) -> Batch:
                """Compute the discounted returns for each transition."""
                returns, _ = self.compute_episodic_return(
                    batch, buffer, indices, gamma=0.99, gae_lambda=1.0
                )
                batch.returns = returns
                return batch

            def learn(
                self, batch: Batch, batch_size: int, repeat: int
            ) -> Dict[str, List[float]]:
                """Perform the back-propagation."""
                logging_losses = []
                for _ in range(repeat):
                    for minibatch in batch.split(batch_size, merge_last=True):
                        self.optim.zero_grad()
                        result = self(minibatch)
                        dist = result.dist
                        act = to_torch_as(minibatch.act, result.act)
                        ret = to_torch(
                            minibatch.returns, torch.float, result.act.device
                        )
                        log_prob = (
                            dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                        )
                        loss = -(log_prob * ret).mean()
                        loss.backward()
                        self.optim.step()
                        logging_losses.append(loss.item())
                return {"loss": logging_losses}

        warnings.filterwarnings("ignore")
        if env_maps[env_name]["backbone"] == "mlp":
            net = Net(in_features, activation=nn.Tanh, hidden_sizes=[64, 64], device=device)
        if dist_class == Categorical:
            actor = DiscreteActor(net, out_features, device=device)
        else:
            actor = ContActor(net, out_features, device=device)
        # this is useless but Tianshou requires the policy to have an optimizer associated
        optim = torch.optim.Adam(actor.parameters(), lr=0.0003)
        policy = REINFORCEPolicy(actor, optim)

        env = SubprocVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(n_envs)])

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

            # vec_env.render("human")
        if args.logger == "wandb":
            logger.experiment.finish()
        env.close()
        del env
        del actor, policy

    elif run == "sb3":

        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        vec_env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

        model = PPO("MlpPolicy", vec_env, verbose=0)
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
        from torchrl.envs import EnvCreator, ParallelEnv
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import MLP, ProbabilisticActor, TanhNormal

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
        from torchrl.envs import EnvCreator
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import MLP, ProbabilisticActor, TanhNormal

        # reproduce the actor
        backbone = MLP(
            in_features=in_features,
            out_features=out_features,
            depth=2,
            num_cells=64,
            activation_class=nn.Tanh,
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
        # round up fpb
        fpb = -(fpb // -n_envs) * n_envs
        collector = MultiaSyncDataCollector(
            n_envs
            * [
                lambda: GymEnv(
                    env_name,
                    categorical_action_encoding=True,
                )
            ],
            actor,
            total_frames=total_frames,
            frames_per_batch=fpb,
            device=device,
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
                logger.log_scalar("total", fps, step=frames)
                logger.log_scalar("frames", frames)
                prev_t = t
                cur = 0

        if args.logger == "wandb":
            logger.experiment.finish()
        collector.shutdown()
        del collector, actor, logger, module, backbone

    elif run == "collector":
        from torchrl.collectors import MultiaSyncDataCollector
        from torchrl.envs import EnvCreator
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import MLP, ProbabilisticActor, TanhNormal

        # reproduce the actor
        backbone = MLP(
            in_features=in_features,
            out_features=out_features,
            depth=2,
            num_cells=64,
            activation_class=nn.Tanh,
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
        # round up fpb
        fpb = -(fpb // -n_envs) * n_envs
        collector = MultiaSyncDataCollector(
            n_envs
            * [
                lambda: GymEnv(
                    env_name,
                    categorical_action_encoding=True,
                )
            ],
            actor,
            total_frames=total_frames,
            frames_per_batch=fpb,
            device=device,
        )

        logger = Logger(exp_name=f"torchrl-preemptive-{env_name}", **logger_kwargs)

        prev_t = time.time()
        frames = 0
        cur = 0
        for i, data in enumerate(collector):
            frames += data.numel()
            cur += data.numel()
            if i % 20 == 0:
                t = time.time()
                fps = cur / (t - prev_t)
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
        from torchrl.envs import EnvCreator
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import MLP, ProbabilisticActor, TanhNormal

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
        # round up fpb
        fpb = -(fpb // -n_envs) * n_envs
        collector = MultiSyncDataCollector(
            n_envs
            * [
                lambda: GymEnv(
                    env_name,
                    categorical_action_encoding=True,
                    device=device,
                )
            ],
            actor,
            total_frames=total_frames,
            frames_per_batch=fpb,
            storing_device=device,
            device=device,
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
                logger.log_scalar("total", fps, step=frames)
                logger.log_scalar("frames", frames)
                prev_t = t
                cur = 0

        if args.logger == "wandb":
            logger.experiment.finish()
        collector.shutdown()
        del collector, actor, logger, module, backbone

    print("\n\n", "=" * 20, "\n" + "fps:", fps)
    timeit.print()

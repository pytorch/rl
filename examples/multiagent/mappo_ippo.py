import time

import numpy as np
import torch

import wandb
from models.mlp import MultiAgentMLP
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers import generate_exp_name
from torchrl.record.loggers.wandb import WandbLogger


def rendering_callback(env, td):
    env.frames.append(env_test.render(mode="rgb_array", agent_index_focus=None))


if __name__ == "__main__":
    # Device
    training_device = "cpu" if not torch.has_cuda else "cuda:0"
    vmas_device = "cpu"

    # Seeding
    seed = 0
    torch.manual_seed(seed)

    # Log
    log = True

    # Sampling
    frames_per_batch = 60_000  # Frames sampled each sampling iteration
    max_steps = 200
    vmas_envs = frames_per_batch // max_steps
    n_iters = 500  # Number of sampling/training iterations
    total_frames = frames_per_batch * n_iters

    scenario_name = "balance"
    env_config = {
        "n_agents": 3,
    }

    config = {
        # RL
        "gamma": 0.9,
        "seed": seed,
        # PPO
        "lmbda": 0.9,
        "entropy_eps": 0,
        "clip_epsilon": 0.2,
        # Sampling,
        "frames_per_batch": frames_per_batch,
        "max_steps": max_steps,
        "vmas_envs": vmas_envs,
        "n_iters": n_iters,
        "total_frames": total_frames,
        "vmas_device": vmas_device,
        # Training
        "num_epochs": 45,  # optimization steps per batch of data collected
        "minibatch_size": 4096,  # size of minibatches used in each epoch
        "lr": 5e-5,
        "max_grad_norm": 40.0,
        "training_device": training_device,
        # Evaluation
        "evaluation_interval": 20,
        "evaluation_episodes": 5,
    }

    model_config = {
        "shared_parameters": True,
        "centralised_critic": True,  # MAPPO if True, IPPO if False
    }

    # Create env and env_test
    env = VmasEnv(
        scenario=scenario_name,
        num_envs=vmas_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=vmas_device,
        seed=seed,
        # Scenario kwargs
        **env_config,
    )
    env_test = VmasEnv(
        scenario=scenario_name,
        num_envs=1,
        continuous_actions=True,
        max_steps=max_steps,
        device=vmas_device,
        seed=seed,
        # Scenario kwargs
        **env_config,
    )
    env_config.update({"n_agents": env.n_agents, "scenario_name": scenario_name})

    # Policy
    actor_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=model_config["shared_parameters"],
            device=training_device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum[0, ...],
            "max": env.action_spec.space.maximum[0, ...],
        },
        return_log_prob=True,
    )

    # Critic
    module = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=model_config["centralised_critic"],
        share_params=model_config["shared_parameters"],
        device=training_device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=["observation"],
    )

    value_module(policy(env.reset().to(training_device)))

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=training_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config["minibatch_size"],
        collate_fn=lambda x: x,  # Make it not clone when sampling
    )

    # Loss
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        advantage_key="advantage",
        clip_epsilon=config["clip_epsilon"],
        entropy_coef=config["entropy_eps"],
        normalize_advantage=False,
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=config["gamma"], lmbda=config["lmbda"]
    )
    optim = torch.optim.Adam(loss_module.parameters(), config["lr"])

    # Logging
    if log:
        config.update({"model": model_config, "env": env_config})
        model_name = (
            ("Het" if not model_config["shared_parameters"] else "")
            + ("MA" if model_config["centralised_critic"] else "I")
            + "PPO"
        )
        logger = WandbLogger(
            exp_name=generate_exp_name(env_config["scenario_name"], model_name),
            project=f"torchrl_{env_config['scenario_name']}",
            group=model_name,
            save_code=True,
            config=config,
        )
        wandb.run.log_code(".")

    total_time = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        print(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start
        print(f"Sampling took {sampling_time}")

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_params.detach(),
                target_params=loss_module.target_critic_params,
            )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(config["num_epochs"]):
            for _ in range(frames_per_batch // config["minibatch_size"]):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config["max_grad_norm"]
                )
                training_tds[-1]["grad_norm"] = total_norm.mean()

                optim.step()
                optim.zero_grad()

        training_time = time.time() - training_start
        print(f"Training took: {training_time}")

        # More logs
        if log:
            training_tds = torch.stack(training_tds)
            logger.experiment.log(
                {
                    f"train/learner/{key}": value.mean().item()
                    for key, value in training_tds.items()
                },
                commit=False,
            )
            if "info" in tensordict_data.keys():
                logger.experiment.log(
                    {
                        f"train/info/{key}": value.mean().item()
                        for key, value in tensordict_data["info"].items()
                    },
                    commit=False,
                )
            iteration_time = sampling_time + training_time
            total_time += iteration_time
            logger.experiment.log(
                {
                    "train/reward/reward_min": tensordict_data["next", "reward"]
                    .min()
                    .item(),
                    "train/reward/reward_mean": tensordict_data["next", "reward"]
                    .mean()
                    .item(),
                    "train/reward/reward_max": tensordict_data["next", "reward"]
                    .max()
                    .item(),
                    "train/sampling_time": sampling_time,
                    "train/training_time": training_time,
                    "train/iteration_time": iteration_time,
                    "train/total_time": total_time,
                    "train/training_iteration": i,
                },
                commit=False,
            )

        if (
            config["evaluation_episodes"] > 0
            and i % config["evaluation_interval"] == 0
            and log
        ):
            evaluation_start = time.time()
            with torch.no_grad():
                rollouts = []
                for _ in range(config["evaluation_episodes"] - 1):
                    rollouts.append(
                        env_test.rollout(
                            max_steps=max_steps,
                            policy=policy,
                            auto_cast_to_device=True,
                        )
                    )

                env_test.frames = []
                rollouts.append(
                    env_test.rollout(
                        max_steps=max_steps,
                        policy=policy,
                        callback=rendering_callback,
                        auto_cast_to_device=True,
                    )
                )
                vid = np.transpose(env_test.frames, (0, 3, 1, 2))
                logger.experiment.log(
                    {
                        "eval/video": wandb.Video(
                            vid, fps=1 / env_test.world.dt, format="mp4"
                        ),
                    },
                    commit=False,
                ),
                evaluation_time = time.time() - evaluation_start
                print(f"Evaluation took: {evaluation_time}")

                logger.experiment.log(
                    {
                        "eval/episode_reward_mean": sum(
                            [td["next", "reward"].sum(1).mean() for td in rollouts]
                        )
                        / len(rollouts),
                        "eval/episode_len_mean": sum(
                            [td.batch_size[1] for td in rollouts]
                        )
                        / len(rollouts),
                        "eval/evaluation_time": evaluation_time,
                    },
                    commit=False,
                )
        if log:
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()

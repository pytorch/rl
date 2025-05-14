# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GAIL Example.

This is a self-contained example of an offline GAIL training script.

The helper functions for gail are coded in the gail_utils.py and helper functions for ppo in ppo_utils.

"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from gail_utils import log_metrics, make_gail_discriminator, make_offline_replay_buffer
from ppo_utils import eval_model, make_env, make_ppo_models
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, GAILLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    set_gym_backend(cfg.env.backend).set()

    device = cfg.gail.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)
    num_mini_batches = (
        cfg.ppo.collector.frames_per_batch // cfg.ppo.loss.mini_batch_size
    )
    total_network_updates = (
        (cfg.ppo.collector.total_frames // cfg.ppo.collector.frames_per_batch)
        * cfg.ppo.loss.ppo_epochs
        * num_mini_batches
    )

    # Create logger
    exp_name = generate_exp_name("Gail", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="gail_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create models (check utils_mujoco.py)
    actor, critic = make_ppo_models(
        cfg.env.env_name, compile=cfg.compile.compile, device=device
    )

    # Create data buffer
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.ppo.collector.frames_per_batch,
            device=device,
            compilable=cfg.compile.compile,
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.ppo.loss.mini_batch_size,
        compilable=cfg.compile.compile,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.ppo.loss.gamma,
        lmbda=cfg.ppo.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.ppo.loss.clip_epsilon,
        loss_critic_type=cfg.ppo.loss.loss_critic_type,
        entropy_coef=cfg.ppo.loss.entropy_coef,
        critic_coef=cfg.ppo.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=torch.tensor(cfg.ppo.optim.lr, device=device), eps=1e-5
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=torch.tensor(cfg.ppo.optim.lr, device=device), eps=1e-5
    )
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=cfg.ppo.collector.frames_per_batch,
        total_frames=cfg.ppo.collector.total_frames,
        device=device,
        max_frames_per_traj=-1,
        compile_policy={"mode": compile_mode} if compile_mode is not None else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create Discriminator
    discriminator = make_gail_discriminator(cfg, collector.env, device)

    # Create loss
    discriminator_loss = GAILLoss(
        discriminator,
        use_grad_penalty=cfg.gail.use_grad_penalty,
        gp_lambda=cfg.gail.gp_lambda,
    )

    # Create optimizer
    discriminator_optim = torch.optim.Adam(
        params=discriminator.parameters(), lr=cfg.gail.lr
    )

    # Create test environment
    logger_video = cfg.logger.video
    test_env = make_env(cfg.env.env_name, device, from_pixels=logger_video)
    if logger_video:
        test_env = test_env.append_transform(
            VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    test_env.eval()
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)

    def update(data, expert_data, num_network_updates=num_network_updates):
        # Add collector data to expert data
        expert_data.set(
            discriminator_loss.tensor_keys.collector_action,
            data["action"][: expert_data.batch_size[0]],
        )
        expert_data.set(
            discriminator_loss.tensor_keys.collector_observation,
            data["observation"][: expert_data.batch_size[0]],
        )
        d_loss = discriminator_loss(expert_data)

        # Backward pass
        d_loss.get("loss").backward()
        discriminator_optim.step()
        discriminator_optim.zero_grad(set_to_none=True)

        # Compute discriminator reward
        with torch.no_grad():
            data = discriminator(data)
        d_rewards = -torch.log(1 - data["d_logits"] + 1e-8)

        # Set discriminator rewards to tensordict
        data.set(("next", "reward"), d_rewards)

        # Update PPO
        for _ in range(cfg_loss_ppo_epochs):
            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffer
            data_buffer.empty()
            data_buffer.extend(data_reshape)

            for batch in data_buffer:
                optim.zero_grad(set_to_none=True)

                # Linearly decrease the learning rate and clip epsilon
                alpha = torch.ones((), device=device)
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                (actor_loss + critic_loss).backward()

                # Update the networks
                optim.step()
        return {"dloss": d_loss, "alpha": alpha}

    if cfg.compile.compile:
        update = compile_with_warmup(update, warmup=2, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.ppo.collector.total_frames)

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.ppo.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.ppo.optim.anneal_lr
    cfg_optim_lr = cfg.ppo.optim.lr
    cfg_loss_anneal_clip_eps = cfg.ppo.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.ppo.loss.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes

    total_iter = len(collector)
    collector_iter = iter(collector)
    for i in range(total_iter):

        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collection"):
            data = next(collector_iter)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        with timeit("rb - sample expert"):
            # Get expert data
            expert_data = replay_buffer.sample()
            expert_data = expert_data.to(device)

        with timeit("update"):
            torch.compiler.cudagraph_mark_step_begin()
            metadata = update(data, expert_data)
        d_loss = metadata["dloss"]
        alpha = metadata["alpha"]

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]

            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        metrics_to_log.update(
            {
                "train/discriminator_loss": d_loss["loss"],
                "train/lr": alpha * cfg_optim_lr,
                "train/clip_epsilon": (
                    alpha * cfg_loss_clip_epsilon
                    if cfg_loss_anneal_clip_eps
                    else cfg_loss_clip_epsilon
                ),
            }
        )

        # evaluation
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                actor.eval()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                metrics_to_log.update(
                    {
                        "eval/reward": test_rewards.mean(),
                    }
                )
                actor.train()
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, i)

    pbar.close()


if __name__ == "__main__":
    main()

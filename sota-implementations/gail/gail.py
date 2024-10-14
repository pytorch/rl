# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GAIL Example.

This is a self-contained example of an offline GAIL training script.

The helper functions for gail are coded in the gail_utils.py and helper functions for ppo in ppo_utils.

"""
import hydra
import numpy as np
import torch
import tqdm

from gail_utils import log_metrics, make_gail_discriminator, make_offline_replay_buffer
from ppo_utils import eval_model, make_env, make_ppo_models
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, GAILLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger


@hydra.main(config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
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
    actor, critic = make_ppo_models(cfg.env.env_name)
    actor, critic = actor.to(device), critic.to(device)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=cfg.ppo.collector.frames_per_batch,
        total_frames=cfg.ppo.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.ppo.collector.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.ppo.loss.mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.ppo.loss.gamma,
        lmbda=cfg.ppo.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
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
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.ppo.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.ppo.optim.lr, eps=1e-5)

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

    # Training loop
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=cfg.ppo.collector.total_frames)

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.ppo.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.ppo.optim.anneal_lr
    cfg_optim_lr = cfg.ppo.optim.lr
    cfg_loss_anneal_clip_eps = cfg.ppo.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.ppo.loss.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes

    for i, data in enumerate(collector):

        log_info = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Update discriminator
        # Get expert data
        expert_data = replay_buffer.sample()
        expert_data = expert_data.to(device)
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
        discriminator_optim.zero_grad()
        d_loss.get("loss").backward()
        discriminator_optim.step()

        # Compute discriminator reward
        with torch.no_grad():
            data = discriminator(data)
        d_rewards = -torch.log(1 - data["d_logits"] + 1e-8)

        # Set discriminator rewards to tensordict
        data.set(("next", "reward"), d_rewards)

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )
        # Update PPO
        for _ in range(cfg_loss_ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffer
            data_buffer.extend(data_reshape)

            for _, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                actor_loss.backward()
                critic_loss.backward()

                # Update the networks
                actor_optim.step()
                critic_optim.step()
                actor_optim.zero_grad()
                critic_optim.zero_grad()

        log_info.update(
            {
                "train/actor_loss": actor_loss.item(),
                "train/critic_loss": critic_loss.item(),
                "train/discriminator_loss": d_loss["loss"].item(),
                "train/lr": alpha * cfg_optim_lr,
                "train/clip_epsilon": (
                    alpha * cfg_loss_clip_epsilon
                    if cfg_loss_anneal_clip_eps
                    else cfg_loss_clip_epsilon
                ),
            }
        )

        # evaluation
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                actor.eval()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                log_info.update(
                    {
                        "eval/reward": test_rewards.mean(),
                    }
                )
                actor.train()
        if logger is not None:
            log_metrics(logger, log_info, i)

    pbar.close()


if __name__ == "__main__":
    main()

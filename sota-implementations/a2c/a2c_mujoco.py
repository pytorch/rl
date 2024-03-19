# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
from torchrl._utils import logger as torchrl_logger


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import time

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import A2CLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_mujoco import eval_model, make_env, make_ppo_models

    # Define paper hyperparameters
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        cfg.collector.total_frames // cfg.collector.frames_per_batch
    ) * num_mini_batches

    # Create models (check utils_mujoco.py)
    actor, critic = make_ppo_models(cfg.env.env_name)
    actor, critic = actor.to(device), critic.to(device)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    loss_module = A2CLoss(
        actor_network=actor,
        critic_network=critic,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr)

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("A2C", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="a2c",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create test environment
    test_env = make_env(cfg.env.env_name, device)
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    sampling_start = time.time()
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Get training rewards and lengths
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

        losses = TensorDict({}, batch_size=[num_mini_batches])
        training_start = time.time()

        # Compute GAE
        with torch.no_grad():
            data = adv_module(data)
        data_reshape = data.reshape(-1)

        # Update the data buffer
        data_buffer.extend(data_reshape)

        for k, batch in enumerate(data_buffer):

            # Get a data batch
            batch = batch.to(device)

            # Linearly decrease the learning rate and clip epsilon
            alpha = 1.0
            if cfg.optim.anneal_lr:
                alpha = 1 - (num_network_updates / total_network_updates)
                for group in actor_optim.param_groups:
                    group["lr"] = cfg.optim.lr * alpha
                for group in critic_optim.param_groups:
                    group["lr"] = cfg.optim.lr * alpha
            num_network_updates += 1

            # Forward pass A2C loss
            loss = loss_module(batch)
            losses[k] = loss.select(
                "loss_critic", "loss_objective"  # , "loss_entropy"
            ).detach()
            critic_loss = loss["loss_critic"]
            actor_loss = loss["loss_objective"]  # + loss["loss_entropy"]

            # Backward pass
            actor_loss.backward()
            critic_loss.backward()

            # Update the networks
            actor_optim.step()
            critic_optim.step()
            actor_optim.zero_grad()
            critic_optim.zero_grad()

        # Get training losses
        training_time = time.time() - training_start
        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg.optim.lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if ((i - 1) * frames_in_batch) // cfg.logger.test_interval < (
                i * frames_in_batch
            ) // cfg.logger.test_interval:
                actor.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg.logger.num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "test/reward": test_rewards.mean(),
                        "test/eval_time": eval_time,
                    }
                )
                actor.train()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()

    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()

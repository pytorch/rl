"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the on Atari Environments.
"""
import hydra


@hydra.main(config_path=".", config_name="config_atari", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import time

    import numpy as np
    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_atari import make_parallel_env, make_ppo_models

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Correct for frame_skip
    frame_skip = 4
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    mini_batch_size = cfg.loss.mini_batch_size // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip

    # Create models (check utils_atari.py)
    actor, critic, critic_head = make_ppo_models(cfg.env.env_name)
    actor, critic, critic_head = (
        actor.to(device),
        critic.to(device),
        critic_head.to(device),
    )

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_parallel_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    # Create logger
    exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
    logger = get_logger(cfg.logger.backend, logger_name="ppo", experiment_name=exp_name)

    # Create test environment
    test_env = make_parallel_env(cfg.env.env_name, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )

    for data in collector:

        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Train loging
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            logger.log_scalar(
                "reward_train", episode_rewards.mean().item(), collected_frames
            )

        # Apply episodic end of life
        data["done"].copy_(data["end_of_life"])
        data["next", "done"].copy_(data["next", "end_of_life"])

        losses = TensorDict({}, batch_size=[cfg.loss.ppo_epochs, num_mini_batches])
        for j in range(cfg.loss.ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffer
            data_buffer.extend(data_reshape)

            for i, batch in enumerate(data_buffer):

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1 - (num_network_updates / total_network_updates)
                if cfg.optim.anneal_lr:
                    for g in optim.param_groups:
                        g["lr"] = cfg.optim.lr * alpha
                if cfg.loss.anneal_clip_epsilon:
                    loss_module.clip_epsilon.copy_(cfg.loss.clip_epsilon * alpha)
                num_network_updates += 1

                # Get a data batch
                batch = batch.to(device)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, i] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )

                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=cfg.optim.max_grad_norm
                )

                # Update the networks
                optim.step()
                optim.zero_grad()

        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses.items():
            logger.log_scalar(key, value.item(), collected_frames)
        logger.log_scalar("lr", alpha * cfg.optim.lr, collected_frames)
        logger.log_scalar(
            "clip_epsilon", alpha * cfg.loss.clip_epsilon, collected_frames
        )

        # Test logging
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_in_batch) // test_interval < (
                collected_frames // test_interval
            ):
                actor.eval()
                test_rewards = []
                for _ in range(cfg.logger.num_test_episodes):
                    td_test = test_env.rollout(
                        policy=actor,
                        auto_reset=True,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                        max_steps=10_000_000,
                    )
                    reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                    test_rewards = np.append(test_rewards, reward.cpu().numpy())
                    del td_test
                logger.log_scalar("reward_test", test_rewards.mean(), collected_frames)
                actor.train()

        collector.update_policy_weights_()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()

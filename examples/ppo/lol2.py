# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the on MuJoCo Environments.
"""


def main():

    import time

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
    from utils_mujoco import eval_model, make_env, make_ppo_models


# Define paper hyperparameters
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "Walker2d-v3"
    frames_per_batch = 2048
    mini_batch_size = 64
    total_frames = 1_000_000
    record_interval = 1_000_000  # check final performance
    gamma = 0.99
    gae_lambda = 0.95
    lr = 3e-4
    ppo_epochs = 10
    critic_coef = 0.25
    entropy_coef = 0.0
    clip_epsilon = 0.2
    loss_critic_type = "l2"
    logger_backend = "wandb"
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (total_frames // frames_per_batch) * ppo_epochs * num_mini_batches

    # Create models (check utils_mujoco.py)
    actor, critic = make_ppo_models(env_name)
    actor, critic = actor.to(device), critic.to(device)

    # Create collector
    collector = SyncDataCollector(
        make_env(env_name, device),
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    adv_module = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=clip_epsilon,
        loss_critic_type=loss_critic_type,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
        normalize_advantage=True,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr, eps=1e-5)

    exp_name = generate_exp_name("PPO", f"Atari_Schulman17_{env_name}")
    logger = get_logger(logger_backend, logger_name="ppo", experiment_name=exp_name)

    # Create test environment
    test_env = make_env(env_name, device)
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for data in collector:

        log_info = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

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

        losses = TensorDict({}, batch_size=[ppo_epochs, num_mini_batches])
        for j in range(ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffer
            data_buffer.extend(data_reshape)

            for i, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1 - (num_network_updates / total_network_updates)
                for g in actor_optim.param_groups:
                    g['lr'] = lr * alpha
                for g in critic_optim.param_groups:
                    g['lr'] = lr * alpha
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, i] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
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

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})

        for key, value in log_info.items():
            logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")

if __name__ == "__main__":
    main()
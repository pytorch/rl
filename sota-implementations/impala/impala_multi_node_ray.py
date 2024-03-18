# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the IMPALA Algorithm
results from Espeholt et al. 2018 for the on Atari Environments.
"""
import hydra
from torchrl._utils import logger as torchrl_logger


@hydra.main(config_path="", config_name="config_multi_node_ray", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import time

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from torchrl.collectors import SyncDataCollector
    from torchrl.collectors.distributed import RayCollector
    from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import A2CLoss
    from torchrl.objectives.value import VTrace
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils import eval_model, make_env, make_ppo_models

    device = torch.device(cfg.local_device)

    # Correct for frame_skip
    frame_skip = 4
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip

    # Extract other config parameters
    batch_size = cfg.loss.batch_size  # Number of rollouts per batch
    num_workers = (
        cfg.collector.num_workers
    )  # Number of parallel workers collecting rollouts
    lr = cfg.optim.lr
    anneal_lr = cfg.optim.anneal_lr
    sgd_updates = cfg.loss.sgd_updates
    max_grad_norm = cfg.optim.max_grad_norm
    num_test_episodes = cfg.logger.num_test_episodes
    total_network_updates = (
        total_frames // (frames_per_batch * batch_size)
    ) * cfg.loss.sgd_updates

    # Create models (check utils.py)
    actor, critic = make_ppo_models(cfg.env.env_name)
    actor, critic = actor.to(device), critic.to(device)

    # Create collector
    ray_init_config = {
        "address": cfg.ray_init_config.address,
        "num_cpus": cfg.ray_init_config.num_cpus,
        "num_gpus": cfg.ray_init_config.num_gpus,
        "resources": cfg.ray_init_config.resources,
        "object_store_memory": cfg.ray_init_config.object_store_memory,
        "local_mode": cfg.ray_init_config.local_mode,
        "ignore_reinit_error": cfg.ray_init_config.ignore_reinit_error,
        "include_dashboard": cfg.ray_init_config.include_dashboard,
        "dashboard_host": cfg.ray_init_config.dashboard_host,
        "dashboard_port": cfg.ray_init_config.dashboard_port,
        "job_config": cfg.ray_init_config.job_config,
        "configure_logging": cfg.ray_init_config.configure_logging,
        "logging_level": cfg.ray_init_config.logging_level,
        "logging_format": cfg.ray_init_config.logging_format,
        "log_to_driver": cfg.ray_init_config.log_to_driver,
        "namespace": cfg.ray_init_config.namespace,
        "runtime_env": cfg.ray_init_config.runtime_env,
        "storage": cfg.ray_init_config.storage,
    }
    remote_config = {
        "num_cpus": cfg.remote_worker_resources.num_cpus,
        "num_gpus": cfg.remote_worker_resources.num_gpus
        if torch.cuda.device_count()
        else 0,
        "memory": cfg.remote_worker_resources.memory,
    }
    collector = RayCollector(
        create_env_fn=[make_env(cfg.env.env_name, device)] * num_workers,
        policy=actor,
        collector_class=SyncDataCollector,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        max_frames_per_traj=-1,
        ray_init_config=ray_init_config,
        remote_configs=remote_config,
        sync=False,
        update_after_each_batch=True,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch * batch_size),
        sampler=sampler,
        batch_size=frames_per_batch * batch_size,
    )

    # Create loss and adv modules
    adv_module = VTrace(
        gamma=cfg.loss.gamma,
        value_network=critic,
        actor_network=actor,
        average_adv=False,
    )
    loss_module = A2CLoss(
        actor_network=actor,
        critic_network=critic,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
    )
    loss_module.set_keys(done="eol", terminated="eol")

    # Create optimizer
    optim = torch.optim.RMSprop(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
        alpha=cfg.optim.alpha,
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name(
            "IMPALA", f"{cfg.logger.exp_name}_{cfg.env.env_name}"
        )
        logger = get_logger(
            cfg.logger.backend,
            logger_name="impala",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create test environment
    test_env = make_env(cfg.env.env_name, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    accumulator = []
    start_time = sampling_start = time.time()
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "terminated"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        if len(accumulator) < batch_size:
            accumulator.append(data)
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, collected_frames)
            continue

        losses = TensorDict({}, batch_size=[sgd_updates])
        training_start = time.time()
        for j in range(sgd_updates):

            # Create a single batch of trajectories
            stacked_data = torch.stack(accumulator, dim=0).contiguous()
            stacked_data = stacked_data.to(device, non_blocking=True)

            # Compute advantage
            with torch.no_grad():
                stacked_data = adv_module(stacked_data)

            # Add to replay buffer
            for stacked_d in stacked_data:
                stacked_data_reshape = stacked_d.reshape(-1)
                data_buffer.extend(stacked_data_reshape)

            for batch in data_buffer:

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = lr * alpha
                num_network_updates += 1

                # Get a data batch
                batch = batch.to(device, non_blocking=True)

                # Forward pass loss
                loss = loss_module(batch)
                losses[j] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )

                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=max_grad_norm
                )

                # Update the networks
                optim.step()
                optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if ((i - 1) * frames_in_batch * frame_skip) // test_interval < (
                i * frames_in_batch * frame_skip
            ) // test_interval:
                actor.eval()
                eval_start = time.time()
                test_reward = eval_model(
                    actor, test_env, num_episodes=num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_reward,
                        "eval/time": eval_time,
                    }
                )
                actor.train()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()
        accumulator = []

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()

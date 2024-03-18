# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DQN: Reproducing experimental results from Mnih et al. 2015 for the
Deep Q-Learning Algorithm on Atari Environments.
"""
import tempfile
import time

import hydra
import torch.nn
import torch.optim
import tqdm
from tensordict.nn import TensorDictSequential
from torchrl._utils import logger as torchrl_logger

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_atari import eval_model, make_dqn_model, make_env


@hydra.main(config_path="", config_name="config_atari", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    device = torch.device(cfg.device)

    # Correct for frame_skip
    frame_skip = 4
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    init_random_frames = cfg.collector.init_random_frames // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip

    # Make the components
    model = make_dqn_model(cfg.env.env_name, frame_skip)
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=model.spec,
    )
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    ).to(device)

    # Create the collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, frame_skip, device),
        policy=model_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        init_random_frames=init_random_frames,
    )

    # Create the replay buffer
    if cfg.buffer.scratch_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        scratch_dir = tempdir.name
    else:
        scratch_dir = cfg.buffer.scratch_dir
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyMemmapStorage(
            max_size=cfg.buffer.buffer_size,
            scratch_dir=scratch_dir,
        ),
        batch_size=cfg.buffer.batch_size,
    )

    # Create the loss module
    loss_module = DQNLoss(
        value_network=model,
        loss_function="l2",
        delay_value=True,
    )
    loss_module.set_keys(done="end-of-life", terminated="end-of-life")
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("DQN", f"Atari_mnih15_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="dqn",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create the test environment
    test_env = make_env(cfg.env.env_name, frame_skip, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    sampling_start = time.time()
    num_updates = cfg.loss.num_updates
    max_grad = cfg.optim.max_grad_norm
    num_test_episodes = cfg.logger.num_test_episodes
    q_losses = torch.zeros(num_updates, device=device)
    pbar = tqdm.tqdm(total=total_frames)
    for data in collector:

        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel() * frame_skip
        collected_frames += current_frames
        greedy_module.step(current_frames)
        replay_buffer.extend(data)

        # Get and log training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            log_info.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    "train/episode_length": episode_length_mean,
                }
            )

        if collected_frames < init_random_frames:
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue

        # optimization steps
        training_start = time.time()
        for j in range(num_updates):

            sampled_tensordict = replay_buffer.sample()
            sampled_tensordict = sampled_tensordict.to(device)

            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(loss_module.parameters()), max_norm=max_grad
            )
            optimizer.step()
            target_net_updater.step()
            q_losses[j].copy_(q_loss.detach())

        training_time = time.time() - training_start

        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_values": (data["action_value"] * data["action"]).sum().item()
                / frames_per_batch,
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_per_batch) // test_interval < (
                collected_frames // test_interval
            ):
                model.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    model, test_env, num_episodes=num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_rewards,
                        "eval/eval_time": eval_time,
                    }
                )
                model.train()

        # Log all the information
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()

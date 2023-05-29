# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DDPG Example.

This is a simple self-contained example of a DDPG training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""

import hydra

import numpy as np
import torch
import torch.cuda
import tqdm
from torch import optim
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    make_collector,
    make_ddpg_agent,
    make_environment,
    make_loss_module,
    make_replay_buffer,
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = torch.device(cfg.device)

    exp_name = generate_exp_name("DDPG", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="ddpg_logging",
        experiment_name=exp_name,
        wandb_kwargs={"mode": cfg.mode},
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create Environments
    train_env, eval_env = make_environment(cfg)

    # Create Agent
    model, exploration_policy = make_ddpg_agent(train_env, eval_env, device)

    # Create Loss Module and Target Updater
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Make Off-Policy Collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.batch_size,
        prb=cfg.prb,
        buffer_size=cfg.buffer_size,
        device=device,
    )

    # Make Optimizers
    critic_params = list(loss_module.value_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(actor_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer_critic = optim.Adam(
        critic_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    rewards = []
    rewards_eval = []

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    r0 = None
    q_loss = None

    for i, tensordict in enumerate(collector):
        # update weights of the inference policy
        collector.update_policy_weights_()

        if r0 is None:
            r0 = tensordict["next", "reward"].sum(-1).mean().item()
        pbar.update(tensordict.numel())

        tensordict = tensordict.view(-1)
        current_frames = tensordict.numel()
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # optimization steps
        if collected_frames >= cfg.init_random_frames:
            (
                actor_losses,
                q_losses,
            ) = ([], [])
            for _ in range(
                int(cfg.env_per_collector * cfg.frames_per_batch * cfg.utd_ratio)
            ):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size).clone()

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_value"]

                optimizer_critic.zero_grad()
                q_loss.backward(retain_graph=True)
                optimizer_critic.step()
                q_losses.append(q_loss.item())

                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()
                actor_losses.append(actor_loss.item())

                # update qnet_target params
                target_net_updater.step()

                # update priority
                if cfg.prb:
                    replay_buffer.update_priority(sampled_tensordict)

        rewards.append(
            (i, tensordict["next", "reward"].sum().item() / cfg.env_per_collector)
        )
        train_log = {
            "train_reward": rewards[-1][1],
            "collected_frames": collected_frames,
        }
        if q_loss is not None:
            train_log.update(
                {
                    "actor_loss": np.mean(actor_losses),
                    "q_loss": np.mean(q_losses),
                }
            )
        for key, value in train_log.items():
            logger.log_scalar(key, value, step=collected_frames)
        if abs(collected_frames % 25000) < cfg.frames_per_batch * cfg.frame_skip:
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                eval_rollout = eval_env.rollout(
                    cfg.max_frames_per_traj // cfg.frame_skip,
                    exploration_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                logger.log_scalar(
                    "evaluation_reward", rewards_eval[-1][1], step=collected_frames
                )
        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f})," + eval_str
            )

    collector.shutdown()


if __name__ == "__main__":
    main()

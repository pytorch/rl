# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CQL Example.

This is a self-contained example of an online CQL training script.

The helper functions are coded in the utils.py associated with this script.

"""

import hydra
import numpy as np
import torch
import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    make_collector,
    make_cql_model,
    make_cql_optimizer,
    make_environment,
    make_loss,
    make_replay_buffer,
)


@hydra.main(config_path=".", config_name="online_config")
def main(cfg: "DictConfig"):  # noqa: F821
    exp_name = generate_exp_name("CQL-online", cfg.env.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="cql_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode, "config": cfg},
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    device = torch.device(cfg.optim.device)

    # Make Env
    train_env, eval_env = make_environment(cfg, cfg.collector.env_per_collector)

    # Make Buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        device="cpu",
    )

    # Make Model
    model = make_cql_model(cfg, train_env, eval_env, device)

    # Make Collector
    collector = make_collector(cfg, train_env, actor_model_explore=model[0])

    # Make Loss
    loss_module, target_net_updater = make_loss(cfg.loss, model)

    # Make Optimizer
    optimizer = make_cql_optimizer(cfg.optim, loss_module)

    rewards = []
    rewards_eval = []
    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    r0 = None
    q_loss = None

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    env_per_collector = cfg.collector.env_per_collector
    eval_iter = cfg.logger.eval_iter
    frames_per_batch, frame_skip = cfg.collector.frames_per_batch, cfg.env.frame_skip
    eval_rollout_steps = cfg.collector.max_frames_per_traj // frame_skip

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
        if collected_frames >= init_random_frames:
            (actor_losses, q_losses, alpha_losses, alpha_primes) = ([], [], [], [])
            for _ in range(num_updates):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]
                alpha_prime_loss = loss_td["loss_alpha_prime"]
                loss = actor_loss + q_loss + alpha_loss + alpha_prime_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                q_losses.append(q_loss.item())
                actor_losses.append(actor_loss.item())
                alpha_losses.append(alpha_loss.item())
                alpha_primes.append(alpha_prime_loss.item())

                # update qnet_target params
                target_net_updater.step()

                # update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        rewards.append(
            (i, tensordict["next", "reward"].sum().item() / env_per_collector)
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
                    "alpha_loss": np.mean(alpha_losses),
                    "alpha_prime_loss": np.mean(alpha_primes),
                    "entropy": loss_td["entropy"],
                }
            )
        if logger is not None:
            for key, value in train_log.items():
                logger.log_scalar(key, value, step=collected_frames)
        if abs(collected_frames % eval_iter) < frames_per_batch * frame_skip:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                if logger is not None:
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

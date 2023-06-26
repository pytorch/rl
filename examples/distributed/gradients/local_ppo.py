# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO Example.

This is a self-contained example of a PPO training script.

Both state and pixel-based environments are supported.

The helper functions are coded in the utils_supp.py associated with this script.
"""
import copy
import hydra
import random
import itertools
import numpy as np
import torch
from torchrl.local_gradient_collector import GradientCollector
from tensordict import TensorDict

# Set seeds for reproducibility
# Set a seed for the random module
random.seed(int(2023))

# Set a seed for the numpy module
np.random.seed(int(2023))

# Set a seed for the torch module
torch.manual_seed(int(2023))


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    import torch
    from utils import (
        make_collector,
        make_data_buffer,
        make_logger,
        make_loss,
        make_optim,
        make_ppo_models,
        make_test_env,
        make_advantage_module,
    )

    # Correct for frame_skip
    cfg.collector.total_frames = cfg.collector.total_frames // cfg.env.frame_skip
    cfg.collector.frames_per_batch = (
        cfg.collector.frames_per_batch // cfg.env.frame_skip
    )
    cfg.loss.mini_batch_size = cfg.loss.mini_batch_size // cfg.env.frame_skip
    num_mini_batches = (cfg.collector.frames_per_batch // cfg.loss.mini_batch_size) * cfg.loss.ppo_epochs

    # Create local modules
    local_model_device = cfg.optim.device
    local_actor, local_critic, local_critic_head = make_ppo_models(cfg)
    local_actor = local_actor.to(local_model_device)
    local_critic = local_critic.to(local_model_device)
    local_critic_head = local_critic_head.to(local_model_device)
    local_loss_module, local_advantage = make_loss(cfg.loss, actor_network=local_actor, value_network=local_critic, value_head=local_critic_head)
    local_optim = make_optim(cfg.optim, actor_network=local_actor, value_network=local_critic_head)

    collector, state_dict = make_collector(cfg, local_actor)
    objective, advantage = make_loss(cfg.loss, actor_network=local_actor, value_network=local_critic, value_head=local_critic_head)
    buffer = make_data_buffer(cfg)

    grad_worker = GradientCollector(
        actor=local_actor,
        critic=local_critic,
        collector=collector,
        objective=local_loss_module,
        advantage=local_advantage,
        buffer=buffer,
        updates_per_batch=320,
        device=cfg.optim.device,
    )

    logger = None
    if cfg.logger.backend:
        logger = make_logger(cfg.logger)
    test_env = make_test_env(cfg.env, state_dict)
    record_interval = cfg.logger.log_interval
    frames_in_batch = cfg.collector.frames_per_batch
    collected_frames = 0

    for remote_grads in grad_worker:

        grad_norm = torch.nn.utils.clip_grad_norm_(local_loss_module.parameters(), max_norm=0.5)

        # Update local policy
        local_optim.step()
        print(f"optimisation step!, grad norm {grad_norm}")
        local_optim.zero_grad()

        # Update counter
        collected_frames += frames_in_batch

        collector.update_policy_weights_()

        # Test current policy
        if (
            logger is not None
            and (collected_frames - frames_in_batch) // record_interval
            < collected_frames // record_interval
        ):

            with torch.no_grad():
                test_env.eval()
                local_actor.eval()
                td_test = test_env.rollout(
                    policy=local_actor,
                    max_steps=10_000_000,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                ).clone()
                logger.log_scalar(
                    "reward_testing",
                    td_test["next", "reward"].sum().item(),
                    collected_frames,
                )
                local_actor.train()
                del td_test


if __name__ == "__main__":
    main()

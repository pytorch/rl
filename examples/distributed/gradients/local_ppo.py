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
    import tqdm
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from utils import (
        make_collector,
        make_data_buffer,
        make_logger,
        make_loss,
        make_optim,
        make_ppo_models,
        make_test_env,
    )

    # Correct for frame_skip
    cfg.collector.total_frames = cfg.collector.total_frames // cfg.env.frame_skip
    cfg.collector.frames_per_batch = (
        cfg.collector.frames_per_batch // cfg.env.frame_skip
    )
    mini_batch_size = cfg.loss.mini_batch_size = (
        cfg.loss.mini_batch_size // cfg.env.frame_skip
    )

    model_device = cfg.optim.device
    actor, critic, critic_head = make_ppo_models(cfg)

    collector, state_dict = make_collector(cfg, policy=actor)
    data_buffer = make_data_buffer(cfg)
    loss_module, adv_module = make_loss(
        cfg.loss,
        actor_network=actor,
        value_network=critic,
        value_head=critic_head,
    )
    optim = make_optim(cfg.optim, actor_network=actor, value_network=critic_head)

    batch_size = cfg.collector.total_frames * cfg.env.num_envs
    num_mini_batches = batch_size // mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // batch_size)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    grad_worker = GradientCollector(
        actor=actor,
        critic=critic,
        collector=collector,
        objective=loss_module,
        advantage=adv_module,
        buffer=data_buffer,
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

        grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=0.5)

        # Update local policy
        optim.step()
        print(f"optimisation step!, grad norm {grad_norm}")
        optim.zero_grad()

        # Update counter
        collected_frames += frames_in_batch

        collector.update_policy_weights_()

        # Test current policy
        if (
            logger is not None
            and (collected_frames - frames_in_batch) // record_interval
            < collected_frames // record_interval
        ):

            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                test_env.eval()
                actor.eval()
                # Generate a complete episode
                td_test = test_env.rollout(
                    policy=actor,
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
                actor.train()
                del td_test


if __name__ == "__main__":
    main()

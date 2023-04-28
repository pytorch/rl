# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO Example.

This is a self-contained example of a PPO training script.

Both state and pixel-based environments are supported.

The helper functions are coded in the utils.py associated with this script.
"""
import hydra
from copy import deepcopy
from torchrl.gradient_collector import GradientCollector
from tensordict import TensorDict

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
    )

    # Correct for frame_skip
    cfg.collector.total_frames = cfg.collector.total_frames // cfg.env.frame_skip
    cfg.collector.frames_per_batch = (
        cfg.collector.frames_per_batch // cfg.env.frame_skip
    )
    cfg.loss.mini_batch_size = cfg.loss.mini_batch_size // cfg.env.frame_skip
    batch_size = cfg.collector.total_frames * cfg.env.num_envs
    num_mini_batches = batch_size // cfg.loss.mini_batch_size

    # Create one copy of all modules
    local_model_device = cfg.optim.device
    local_actor, local_critic = make_ppo_models(cfg)
    local_actor = local_actor.to(local_model_device)
    local_critic = local_critic.to(local_model_device)
    # TODO: I should not need a local loss module, can I get the dict of name params somehow else?
    local_loss_module = make_loss(cfg.loss, actor_network=local_actor, value_network=local_critic)
    local_optim = make_optim(cfg.optim, actor_network=local_actor, value_network=local_critic)

    # Create a second copy of all modules
    distributed_model_device = cfg.optim.device
    distributed_actor = deepcopy(local_actor).to(distributed_model_device)
    distributed_critic = deepcopy(local_critic).to(distributed_model_device)
    distributed_collector = make_collector(cfg, policy=local_actor)
    distributed_data_buffer = make_data_buffer(cfg)
    distributed_loss_module = deepcopy(local_loss_module)
    distributed_optim = make_optim(cfg.optim, actor_network=distributed_actor, value_network=distributed_critic)

    grad_worker = GradientCollector(
        policy=distributed_actor,
        critic=distributed_critic,
        collector=distributed_collector,
        objective=distributed_loss_module,
        replay_buffer=distributed_data_buffer,
        optimizer=distributed_optim,
        updates_per_batch=num_mini_batches * cfg.loss.ppo_epochs,
        device=distributed_model_device,
    )

    for grads in grad_worker:

        # TODO: is there a better way ?
        # Apply gradients
        for name, param in local_loss_module.named_parameters():
            param.grad = grads.get(name)

        # Process grads
        grad_norm = torch.nn.utils.clip_grad_norm_(local_loss_module.parameters(), max_norm=0.5)

        # Update local policy
        local_optim.step()
        print("optimisation step!")

        # Update grad_worker policy, not needed in this dummy local example


if __name__ == "__main__":
    main()

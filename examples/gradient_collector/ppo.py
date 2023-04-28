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

    model_device = cfg.optim.device
    actor, critic = make_ppo_models(cfg)
    actor = actor.to(model_device)
    critic = critic.to(model_device)

    collector = make_collector(cfg, policy=actor)
    data_buffer = make_data_buffer(cfg)
    loss_module, adv_module = make_loss(
        cfg.loss, actor_network=actor, value_network=critic
    )
    optim = make_optim(cfg.optim, actor_network=actor, value_network=critic)

    batch_size = cfg.collector.total_frames * cfg.env.num_envs
    num_mini_batches = batch_size // cfg.loss.mini_batch_size

    grad_worker = GradientCollector(
        policy=actor,
        critic=critic,
        collector=collector,
        objective=loss_module,
        replay_buffer=data_buffer,
        optimizer=optim,
        updates_per_batch=num_mini_batches * cfg.loss.ppo_epochs,
        device=model_device
    )

    for grads in grad_worker:

        # TODO: is there a better way ?
        # Apply gradients
        for name, param in loss_module.named_parameters():
            param.grad = grads.get(name)

        # Process grads
        grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=0.5)

        # Update local policy
        optim.step()
        print("optimisation step!")

        # Update grad_worker policy, not needed in this dummy local example


if __name__ == "__main__":
    main()

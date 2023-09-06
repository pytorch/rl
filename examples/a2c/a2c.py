# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A2C Example.

This is a self-contained example of a A2C training script.

Both state and pixel-based environments are supported.

The helper functions are coded in the utils.py associated with this script.
"""
import hydra


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import torch
    import tqdm
    from utils import (
        make_a2c_models,
        make_collector,
        make_logger,
        make_loss,
        make_optim,
        make_test_env,
    )

    # Correct for frame_skip
    cfg.collector.total_frames = cfg.collector.total_frames // cfg.env.frame_skip
    cfg.collector.frames_per_batch = (
        cfg.collector.frames_per_batch // cfg.env.frame_skip
    )

    model_device = cfg.optim.device
    actor, critic = make_a2c_models(cfg)
    actor = actor.to(model_device)
    critic = critic.to(model_device)

    collector = make_collector(cfg, policy=actor)
    loss_module, adv_module = make_loss(
        cfg.loss, actor_network=actor, value_network=critic
    )
    optim = make_optim(cfg.optim, actor_network=actor, value_network=critic)

    batch_size = cfg.collector.total_frames * cfg.env.num_envs
    total_network_updates = cfg.collector.total_frames // batch_size

    scheduler = None
    if cfg.optim.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optim, total_iters=total_network_updates, start_factor=1.0, end_factor=0.1
        )

    logger = None
    if cfg.logger.backend:
        logger = make_logger(cfg.logger)
    test_env = make_test_env(cfg.env)
    record_interval = cfg.logger.log_interval
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    collected_frames = 0

    # Main loop
    r0 = None
    l0 = None
    for data in collector:

        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * cfg.env.frame_skip
        pbar.update(data.numel())
        data_view = data.reshape(-1)

        # Compute GAE
        with torch.no_grad():
            batch = adv_module(data_view)

        # Normalize advantage
        adv = batch.get("advantage")
        loc = adv.mean().item()
        scale = adv.std().clamp_min(1e-6).item()
        adv = (adv - loc) / scale
        batch.set("advantage", adv)

        # Forward pass A2C loss
        batch = batch.to(model_device)
        loss = loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backward pass + learning step
        loss_sum.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), max_norm=0.5
        )
        optim.step()
        if scheduler is not None:
            scheduler.step()
        optim.zero_grad()

        # Logging
        if r0 is None:
            r0 = data["next", "reward"].mean().item()
        if l0 is None:
            l0 = loss_sum.item()
        pbar.set_description(
            f"loss: {loss_sum.item(): 4.4f} (init: {l0: 4.4f}), reward: {data['next', 'reward'].mean(): 4.4f} (init={r0: 4.4f})"
        )
        if logger is not None:
            for key, value in loss.items():
                logger.log_scalar(key, value.item(), collected_frames)
            logger.log_scalar("grad_norm", grad_norm.item(), collected_frames)
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            if len(episode_rewards) > 0:
                logger.log_scalar(
                    "reward_training", episode_rewards.mean().item(), collected_frames
                )
        collector.update_policy_weights_()

        # Test current policy
        if (
            logger is not None
            and (collected_frames - frames_in_batch) // record_interval
            < collected_frames // record_interval
        ):

            with torch.no_grad():
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


if __name__ == "__main__":
    main()

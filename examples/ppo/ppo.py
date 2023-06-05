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


@hydra.main(config_path=".", config_name="config", version_base="1.1")
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
    print("actor", actor)
    print("critic", critic)

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

    scheduler = None
    if cfg.optim.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optim, total_iters=total_network_updates, start_factor=1.0, end_factor=0.1
        )

    logger = None
    if cfg.logger.backend:
        logger = make_logger(cfg.logger)
    test_env = make_test_env(cfg.env, state_dict)
    record_interval = cfg.logger.log_interval
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    collected_frames = 0

    # Main loop
    r0 = None
    l0 = None
    frame_skip = cfg.env.frame_skip
    ppo_epochs = cfg.loss.ppo_epochs
    total_done = 0
    for data in collector:

        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Log end-of-episode accumulated rewards for training
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if logger is not None and len(episode_rewards) > 0:
            logger.log_scalar(
                "reward_training", episode_rewards.mean().item(), collected_frames
            )

        losses = TensorDict(
            {}, batch_size=[ppo_epochs, -(frames_in_batch // -mini_batch_size)]
        )
        for j in range(ppo_epochs):
            # Compute GAE
            with torch.no_grad():
                data = adv_module(data.to(model_device)).cpu()

            data_reshape = data.reshape(-1)
            # Update the data buffer
            data_buffer.extend(data_reshape)

            for i, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(model_device)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, i] = loss.detach()

                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )

                # Backward pass
                loss_sum.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), max_norm=0.5
                )
                losses[j, i]["grad_norm"] = grad_norm

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
            if i + 1 != -(frames_in_batch // -mini_batch_size):
                print(
                    f"Should have had {- (frames_in_batch // -mini_batch_size)} iters but had {i}."
                )
        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        if logger is not None:
            for key, value in losses.items():
                logger.log_scalar(key, value.item(), collected_frames)
            logger.log_scalar("total_done", total_done, collected_frames)

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

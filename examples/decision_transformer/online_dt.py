# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Decision Transformer Example.
This is a self-contained example of an offline Decision Transformer training script.
The helper functions are coded in the utils.py associated with this script.
"""

import hydra
import torch
import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type

from utils import (
    # get_loc_std,
    # make_collector,
    make_env,
    make_logger,
    make_odt_loss,
    make_odt_model,
    make_odt_optimizer,
    make_offline_replay_buffer,
    # make_online_replay_buffer,
)


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    model_device = cfg.optim.device

    # loc, std = get_loc_std(cfg.replay_buffer.dataset)
    test_env = make_env(cfg.env)  # , loc, std)
    logger = make_logger(cfg.logger)
    offline_buffer = make_offline_replay_buffer(
        cfg.replay_buffer, cfg.env.reward_scaling
    )  # , loc, std

    inference_actor, actor = make_odt_model(cfg)
    policy = actor.to(model_device)
    inference_policy = inference_actor.to(model_device)

    loss_module = make_odt_loss(cfg.loss, actor)
    transformer_optim, temperature_optim, scheduler = make_odt_optimizer(
        cfg.optim, policy, loss_module
    )

    pbar = tqdm.tqdm(total=cfg.optim.pretrain_gradient_steps)

    r0 = None
    l0 = None
    print(" ***Pretraining*** ")
    # Pretraining
    for i in range(cfg.optim.pretrain_gradient_steps):
        pbar.update(i)
        data = offline_buffer.sample()
        # loss
        loss_vals = loss_module(data)
        # backprop
        transformer_loss = loss_vals["loss"]
        temperature_loss = loss_vals["loss_alpha"]

        transformer_optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
        transformer_loss.backward()
        transformer_optim.step()

        temperature_optim.zero_grad()
        temperature_loss.backward()
        temperature_optim.step()

        scheduler.step()

        # evaluation
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            inference_policy.eval()
            if i % cfg.logger.pretrain_log_interval == 0:
                eval_td = test_env.rollout(
                    max_steps=cfg.logger.eval_steps,
                    policy=inference_policy,
                    auto_cast_to_device=True,
                )
                inference_policy.train()
        if r0 is None:
            r0 = eval_td["next", "reward"].sum(1).mean().item() / cfg.env.reward_scaling
        if l0 is None:
            l0 = transformer_loss.item()

        for key, value in loss_vals.items():
            logger.log_scalar(key, value.item(), i)
        eval_reward = (
            eval_td["next", "reward"].sum(1).mean().item() / cfg.env.reward_scaling
        )
        logger.log_scalar("evaluation reward", eval_reward, i)

        pbar.set_description(
            f"[Pre-Training] loss: {transformer_loss.item(): 4.4f} (init: {l0: 4.4f}), evaluation reward: {eval_reward: 4.4f} (init={r0: 4.4f})"
        )
    # print("\n ***Online Finetuning*** ")
    # collector = make_collector(cfg, inference_policy)
    # online_buffer = make_online_replay_buffer(
    #     offline_buffer, cfg.replay_buffer, cfg.env.reward_scaling
    # )
    # collected_frames = 0

    # pbar = tqdm.tqdm(total=cfg.env.total_online_frames)
    # r0 = None

    # for j, tensordict in enumerate(collector):
    #     # update weights of the inference policy
    #     collector.update_policy_weights_()

    #     episode_reward = (
    #         tensordict["next", "episode_reward"][tensordict["next", "done"]]
    #         .mean()
    #         .item()
    #         / cfg.env.reward_scaling
    #     )
    #     if r0 is None:
    #         r0 = episode_reward

    #     current_frames = tensordict.numel()
    #     pbar.update(current_frames)

    #     tensordict = tensordict.reshape(-1)
    #     # only used for logging
    #     tensordict.del_("episode_reward")

    #     online_buffer.extend(tensordict.cpu().clone().detach())
    #     collected_frames += current_frames

    #     # optimization steps
    #     for _ in range(int(cfg.optim.updates_per_episode)):
    #         sampled_tensordict = online_buffer.sample().clone()

    #         loss_vals = loss_module(sampled_tensordict)

    #         # backprop
    #         transformer_loss = loss_vals["loss"]
    #         temperature_loss = loss_vals["loss_alpha"]

    #         transformer_optim.zero_grad()
    #         torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.25)
    #         transformer_loss.backward()
    #         transformer_optim.step()

    #         temperature_optim.zero_grad()
    #         temperature_loss.backward()
    #         temperature_optim.step()

    #         scheduler.step()

    #     train_target_return = (
    #         tensordict["return_to_go"][:, 0].mean() / cfg.env.reward_scaling
    #     )
    #     train_log = {
    #         "collect reward": episode_reward,
    #         "collected_frames": collected_frames,
    #         "collect target_return": train_target_return.item()
    #         / cfg.env.reward_scaling,
    #     }

    #     for key, value in train_log.items():
    #         logger.log_scalar(key, value, step=j)

    #     with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
    #         if j % cfg.logger.fintune_log_interval == 0:
    #             eval_td = test_env.rollout(
    #                 max_steps=cfg.logger.eval_steps * cfg.env.num_eval_envs,
    #                 policy=inference_policy,
    #                 auto_cast_to_device=True,
    #             )
    #     eval_reward = (
    #         eval_td["next", "reward"].sum(1).mean().item() / cfg.env.reward_scaling
    #     )
    #     eval_target_return = (
    #         eval_td["return_to_go"][:, 0].mean() / cfg.env.reward_scaling
    #     )
    #     eval_log = {
    #         "fine-tune evaluation reward": eval_reward,
    #         "evaluation target_return": eval_target_return.item()
    #         / cfg.env.reward_scaling,
    #     }
    #     for key, value in eval_log.items():
    #         logger.log_scalar(key, value, step=j)
    #     pbar.set_description(
    #         f"[Fine-Tuning] loss: {transformer_loss.item(): 4.4f} (init: {l0: 4.4f}), evaluation reward: {eval_reward: 4.4f} (init={r0: 4.4f})"
    #     )

    # collector.shutdown()


if __name__ == "__main__":
    main()

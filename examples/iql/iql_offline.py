# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""IQL Example.

This is a self-contained example of an offline IQL training script.

The helper functions are coded in the utils.py associated with this script.

"""

import hydra
import torch
import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type

from utils import (
    get_stats,
    make_iql_model,
    make_iql_optimizer,
    make_logger,
    make_loss,
    make_offline_replay_buffer,
    make_parallel_env,
)


@hydra.main(config_path=".", config_name="offline_config")
def main(cfg: "DictConfig"):  # noqa: F821
    model_device = cfg.optim.device

    state_dict = get_stats(cfg.env)
    evaluation_env = make_parallel_env(cfg.env, state_dict=state_dict)
    logger = make_logger(cfg.logger)
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer, state_dict)

    actor_network, qvalue_network, value_network = make_iql_model(cfg)
    policy = actor_network.to(model_device)
    qvalue_network = qvalue_network.to(model_device)
    value_network = value_network.to(model_device)

    loss, target_net_updater = make_loss(
        cfg.loss, policy, qvalue_network, value_network
    )
    optim = make_iql_optimizer(cfg.optim, policy, qvalue_network, value_network)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    r0 = None
    l0 = None

    gradient_steps = cfg.optim.gradient_steps
    evaluation_interval = cfg.logger.evaluation_interval
    eval_steps = cfg.logger.eval_steps

    for i in range(gradient_steps):
        pbar.update(i)
        data = replay_buffer.sample()
        # loss
        loss_vals = loss(data)
        # backprop
        actor_loss = loss_vals["loss_actor"]
        q_loss = loss_vals["loss_qvalue"]
        value_loss = loss_vals["loss_value"]
        loss_val = actor_loss + q_loss + value_loss

        optim.zero_grad()
        loss_val.backward()
        optim.step()
        target_net_updater.step()

        # evaluation
        if i % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_td = evaluation_env.rollout(
                    max_steps=eval_steps, policy=policy, auto_cast_to_device=True
                )

        if r0 is None:
            r0 = eval_td["next", "reward"].sum(1).mean().item()
        if l0 is None:
            l0 = loss_val.item()

        for key, value in loss_vals.items():
            logger.log_scalar(key, value.item(), i)
        eval_reward = eval_td["next", "reward"].sum(1).mean().item()
        logger.log_scalar("evaluation_reward", eval_reward, i)

        pbar.set_description(
            f"loss: {loss_val.item(): 4.4f} (init: {l0: 4.4f}), evaluation_reward: {eval_reward: 4.4f} (init={r0: 4.4f})"
        )


if __name__ == "__main__":
    main()

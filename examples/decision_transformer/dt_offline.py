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
from torchrl.envs.utils import set_exploration_mode

from utils import (
    # get_stats,
    make_decision_transformer_model,
    make_dt_optimizer,
    # make_logger,
    make_loss,
    make_offline_replay_buffer,
    # make_parallel_env,
    make_test_env,
)


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    model_device = cfg.optim.device

    # state_dict = get_stats(cfg.env)
    evaluation_env = make_test_env(cfg.env)
    # logger = make_logger(cfg.logger)
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    actor = make_decision_transformer_model(cfg)
    policy = actor.to(model_device)

    loss, target_net_updater = make_loss(cfg.loss, policy)
    optim = make_dt_optimizer(cfg.optim, policy)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    r0 = None
    l0 = None

    for i in range(cfg.optim.gradient_steps):
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
        if i % cfg.env.evaluation_interval == 0:
            with set_exploration_mode("random"), torch.no_grad():
                eval_td = evaluation_env.rollout(
                    max_steps=1000, policy=policy, auto_cast_to_device=True
                )

        if r0 is None:
            r0 = eval_td["next", "reward"].sum(1).mean().item()
        if l0 is None:
            l0 = loss_val.item()

        # for key, value in loss_vals.items():
        #     logger.log_scalar(key, value.item(), i)
        # eval_reward = eval_td["next", "reward"].sum(1).mean().item()
        # logger.log_scalar("evaluation reward", eval_reward, i)

        # pbar.set_description(
        #     f"loss: {loss_val.item(): 4.4f} (init: {l0: 4.4f}), evaluation reward: {eval_reward: 4.4f} (init={r0: 4.4f})"
        # )


if __name__ == "__main__":
    main()

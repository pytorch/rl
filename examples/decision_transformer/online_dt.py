# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Online Decision Transformer Example.
This is a self-contained example of an Online Decision Transformer training script.
The helper functions are coded in the utils.py associated with this script.
"""

import time

import hydra
import numpy as np
import torch
import tqdm
from torchrl.envs.libs.gym import set_gym_backend

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module import DecisionTransformerInferenceWrapper

from utils import (
    log_metrics,
    make_env,
    make_logger,
    make_odt_loss,
    make_odt_model,
    make_odt_optimizer,
    make_offline_replay_buffer,
)


@set_gym_backend("gym")  # D4RL uses gym so we make sure gymnasium is hidden
@hydra.main(config_path=".", config_name="odt_config")
def main(cfg: "DictConfig"):  # noqa: F821
    model_device = cfg.optim.device

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create logger
    logger = make_logger(cfg)

    # Create offline replay buffer
    offline_buffer, obs_loc, obs_std = make_offline_replay_buffer(
        cfg.replay_buffer, cfg.env.reward_scaling
    )

    # Create test environment
    test_env = make_env(cfg.env, obs_loc, obs_std)

    # Create policy model
    actor = make_odt_model(cfg)
    policy = actor.to(model_device)

    # Create loss
    loss_module = make_odt_loss(cfg.loss, policy)

    # Create optimizer
    transformer_optim, temperature_optim, scheduler = make_odt_optimizer(
        cfg.optim, loss_module
    )

    # Create inference policy
    inference_policy = DecisionTransformerInferenceWrapper(
        policy=policy,
        inference_context=cfg.env.inference_context,
    ).to(model_device)

    pbar = tqdm.tqdm(total=cfg.optim.pretrain_gradient_steps)

    pretrain_gradient_steps = cfg.optim.pretrain_gradient_steps
    clip_grad = cfg.optim.clip_grad
    eval_steps = cfg.logger.eval_steps
    pretrain_log_interval = cfg.logger.pretrain_log_interval
    reward_scaling = cfg.env.reward_scaling

    print(" ***Pretraining*** ")
    # Pretraining
    start_time = time.time()
    for i in range(pretrain_gradient_steps):
        pbar.update(i)
        # sample data
        data = offline_buffer.sample()
        # compute loss
        loss_vals = loss_module(data.to(model_device))
        transformer_loss = loss_vals["loss_log_likelihood"] + loss_vals["loss_entropy"]
        temperature_loss = loss_vals["loss_alpha"]

        transformer_optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        transformer_loss.backward()
        transformer_optim.step()

        temperature_optim.zero_grad()
        temperature_loss.backward()
        temperature_optim.step()

        scheduler.step()

        # log metrics
        to_log = {
            "loss_log_likelihood": loss_vals["loss_log_likelihood"].item(),
            "loss_entropy": loss_vals["loss_entropy"].item(),
            "loss_alpha": loss_vals["loss_alpha"].item(),
        }

        # evaluation
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            inference_policy.eval()
            if i % pretrain_log_interval == 0:
                eval_td = test_env.rollout(
                    max_steps=eval_steps,
                    policy=inference_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
                inference_policy.train()
            to_log["evaluation_reward"] = (
                eval_td["next", "reward"].sum(1).mean().item() / reward_scaling
            )

        log_metrics(logger, to_log, i)

    pbar.close()
    print(f"Training time: {time.time() - start_time}")


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Decision Transformer Example.
This is a self-contained example of an offline Decision Transformer training script.
The helper functions are coded in the utils.py associated with this script.
"""
import time

import hydra
import numpy as np
import torch
import tqdm
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.libs.gym import set_gym_backend

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module import DecisionTransformerInferenceWrapper

from utils import (
    log_metrics,
    make_dt_loss,
    make_dt_model,
    make_dt_optimizer,
    make_env,
    make_logger,
    make_offline_replay_buffer,
)


@hydra.main(config_path="", config_name="dt_config", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    set_gym_backend(cfg.env.backend).set()

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
    actor = make_dt_model(cfg)
    policy = actor.to(model_device)

    # Create loss
    loss_module = make_dt_loss(cfg.loss, actor)

    # Create optimizer
    transformer_optim, scheduler = make_dt_optimizer(cfg.optim, loss_module)

    # Create inference policy
    inference_policy = DecisionTransformerInferenceWrapper(
        policy=policy,
        inference_context=cfg.env.inference_context,
    ).to(model_device)
    inference_policy.set_tensor_keys(
        observation="observation_cat",
        action="action_cat",
        return_to_go="return_to_go_cat",
    )

    pbar = tqdm.tqdm(total=cfg.optim.pretrain_gradient_steps)

    pretrain_gradient_steps = cfg.optim.pretrain_gradient_steps
    clip_grad = cfg.optim.clip_grad
    eval_steps = cfg.logger.eval_steps
    pretrain_log_interval = cfg.logger.pretrain_log_interval
    reward_scaling = cfg.env.reward_scaling

    torchrl_logger.info(" ***Pretraining*** ")
    # Pretraining
    start_time = time.time()
    for i in range(pretrain_gradient_steps):
        pbar.update(1)

        # Sample data
        data = offline_buffer.sample()
        # Compute loss
        loss_vals = loss_module(data.to(model_device))
        transformer_loss = loss_vals["loss"]

        transformer_optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        transformer_loss.backward()
        transformer_optim.step()

        scheduler.step()

        # Log metrics
        to_log = {"train/loss": loss_vals["loss"]}

        # Evaluation
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            if i % pretrain_log_interval == 0:
                eval_td = test_env.rollout(
                    max_steps=eval_steps,
                    policy=inference_policy,
                    auto_cast_to_device=True,
                )
            to_log["eval/reward"] = (
                eval_td["next", "reward"].sum(1).mean().item() / reward_scaling
            )
        if logger is not None:
            log_metrics(logger, to_log, i)

    pbar.close()
    torchrl_logger.info(f"Training time: {time.time() - start_time}")


if __name__ == "__main__":
    main()

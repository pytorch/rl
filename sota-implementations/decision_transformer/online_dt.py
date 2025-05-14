# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Online Decision Transformer Example.
This is a self-contained example of an Online Decision Transformer training script.
The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from tensordict.nn import CudaGraphModule
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module import DecisionTransformerInferenceWrapper
from torchrl.record import VideoRecorder
from utils import (
    dump_video,
    log_metrics,
    make_env,
    make_logger,
    make_odt_loss,
    make_odt_model,
    make_odt_optimizer,
    make_offline_replay_buffer,
)


@hydra.main(config_path="", config_name="odt_config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821
    set_gym_backend(cfg.env.backend).set()

    model_device = cfg.optim.device
    if model_device in ("", None):
        if torch.cuda.is_available():
            model_device = "cuda:0"
        else:
            model_device = "cpu"
    model_device = torch.device(model_device)

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
    test_env = make_env(cfg.env, obs_loc, obs_std, from_pixels=cfg.logger.video)
    if cfg.logger.video:
        test_env = test_env.append_transform(
            VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )

    # Create policy model
    policy = make_odt_model(cfg, device=model_device)

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
        device=model_device,
    )
    inference_policy.set_tensor_keys(
        observation="observation_cat",
        action="action_cat",
        return_to_go="return_to_go_cat",
    )

    def update(data):
        transformer_optim.zero_grad(set_to_none=True)
        temperature_optim.zero_grad(set_to_none=True)
        # Compute loss
        loss_vals = loss_module(data.to(model_device))
        transformer_loss = loss_vals["loss_log_likelihood"] + loss_vals["loss_entropy"]
        temperature_loss = loss_vals["loss_alpha"]

        (temperature_loss + transformer_loss).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)

        transformer_optim.step()
        temperature_optim.step()

        return loss_vals.detach()

    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            compile_mode = "default"
        update = torch.compile(update, mode=compile_mode, dynamic=False)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        if cfg.optim.optimizer == "lamb":
            raise ValueError(
                "cudagraphs isn't compatible with the Lamb optimizer. Use optim.optimizer=Adam instead."
            )
        update = CudaGraphModule(update, warmup=50)

    pbar = tqdm.tqdm(total=cfg.optim.pretrain_gradient_steps)

    pretrain_gradient_steps = cfg.optim.pretrain_gradient_steps
    clip_grad = cfg.optim.clip_grad
    eval_steps = cfg.logger.eval_steps
    pretrain_log_interval = cfg.logger.pretrain_log_interval
    reward_scaling = cfg.env.reward_scaling

    torchrl_logger.info(" ***Pretraining*** ")
    # Pretraining
    for i in range(pretrain_gradient_steps):
        timeit.printevery(1000, pretrain_gradient_steps, erase=True)
        pbar.update(1)
        with timeit("sample"):
            # Sample data
            data = offline_buffer.sample()

        with timeit("update"):
            torch.compiler.cudagraph_mark_step_begin()
            loss_vals = update(data.to(model_device))

        scheduler.step()

        # Log metrics
        metrics_to_log = {
            "train/loss_log_likelihood": loss_vals["loss_log_likelihood"],
            "train/loss_entropy": loss_vals["loss_entropy"],
            "train/loss_alpha": loss_vals["loss_alpha"],
            "train/alpha": loss_vals["alpha"],
            "train/entropy": loss_vals["entropy"],
        }

        # Evaluation
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            inference_policy.eval()
            if i % pretrain_log_interval == 0:
                eval_td = test_env.rollout(
                    max_steps=eval_steps,
                    policy=inference_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
                test_env.apply(dump_video)
                inference_policy.train()
            metrics_to_log["eval/reward"] = (
                eval_td["next", "reward"].sum(1).mean().item() / reward_scaling
            )

        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, i)

    pbar.close()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()

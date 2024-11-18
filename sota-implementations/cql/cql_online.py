# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CQL Example.

This is a self-contained example of an online CQL training script.

It works across Gym and MuJoCo over a variety of tasks.

The helper functions are coded in the utils.py associated with this script.

"""
import warnings

import hydra
import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_continuous_cql_optimizer,
    make_continuous_loss,
    make_cql_model,
    make_environment,
    make_replay_buffer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="online_config")
def main(cfg: "DictConfig"):  # noqa: F821
    # Create logger
    exp_name = generate_exp_name("CQL-online", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="cql_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Create env
    train_env, eval_env = make_environment(
        cfg,
        cfg.env.train_num_envs,
        cfg.env.eval_num_envs,
        logger=logger,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        device="cpu",
    )

    # create agent
    model = make_cql_model(cfg, train_env, eval_env, device)

    compile_mode = None
    if cfg.compile.compile:
        if cfg.compile.compile_mode not in (None, ""):
            compile_mode = cfg.compile.compile_mode
        elif cfg.compile.cudagraphs:
            compile_mode = "default"
        else:
            compile_mode = "reduce-overhead"

    # Create collector
    collector = make_collector(
        cfg,
        train_env,
        actor_model_explore=model[0],
        compile=cfg.compile.compile,
        compile_mode=compile_mode,
        cudagraph=cfg.compile.cudagraphs,
    )

    # Create loss
    loss_module, target_net_updater = make_continuous_loss(
        cfg.loss, model, device=device
    )

    # Create optimizer
    (
        policy_optim,
        critic_optim,
        alpha_optim,
        alpha_prime_optim,
    ) = make_continuous_cql_optimizer(cfg, loss_module)
    optimizer = group_optimizers(
        policy_optim, critic_optim, alpha_optim, alpha_prime_optim
    )

    def update(sampled_tensordict):

        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        cql_loss = loss_td["loss_cql"]
        q_loss = q_loss + cql_loss
        alpha_loss = loss_td["loss_alpha"]
        alpha_prime_loss = loss_td["loss_alpha_prime"]

        total_loss = alpha_loss + actor_loss + alpha_prime_loss + q_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # update qnet_target params
        target_net_updater.step()

        return loss_td.detach()

    if compile_mode:
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    frames_per_batch = cfg.collector.frames_per_batch
    evaluation_interval = cfg.logger.log_interval
    eval_rollout_steps = cfg.logger.eval_steps

    c_iter = iter(collector)
    for i in range(len(collector)):
        with timeit("collecting"):
            tensordict = next(c_iter)
        pbar.update(tensordict.numel())
        # update weights of the inference policy
        collector.update_policy_weights_()

        with timeit("rb - extend"):
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
            # add to replay buffer
            replay_buffer.extend(tensordict)
            collected_frames += current_frames

        if collected_frames >= init_random_frames:
            log_loss_td = TensorDict(batch_size=[num_updates], device=device)
            for j in range(num_updates):
                with timeit("rb - sample"):
                    # sample from replay buffer
                    sampled_tensordict = replay_buffer.sample().to(device)

                with timeit("update"):
                    torch.compiler.cudagraph_mark_step_begin()
                    loss_td = update(sampled_tensordict)
                log_loss_td[j] = loss_td.detach()
                # update priority
                if prb:
                    with timeit("rb - update priority"):
                        replay_buffer.update_priority(sampled_tensordict)

        episode_rewards = tensordict["next", "episode_reward"][
            tensordict["next", "done"]
        ]
        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][
                tensordict["next", "done"]
            ]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            metrics_to_log["train/loss_actor"] = log_loss_td.get("loss_actor").mean()
            metrics_to_log["train/loss_qvalue"] = log_loss_td.get("loss_qvalue").mean()
            metrics_to_log["train/loss_alpha"] = log_loss_td.get("loss_alpha").mean()
            metrics_to_log["train/loss_alpha_prime"] = log_loss_td.get(
                "loss_alpha_prime"
            ).mean()
            metrics_to_log["train/entropy"] = log_loss_td.get("entropy").mean()
            if i % 10 == 0:
                metrics_to_log.update(timeit.todict(prefix="time"))

        # Evaluation
        with timeit("eval"):
            prev_test_frame = ((i - 1) * frames_per_batch) // evaluation_interval
            cur_test_frame = (i * frames_per_batch) // evaluation_interval
            final = current_frames >= collector.total_frames
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ), torch.no_grad():
                    eval_rollout = eval_env.rollout(
                        eval_rollout_steps,
                        model[0],
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                    eval_env.apply(dump_video)
                    metrics_to_log["eval/reward"] = eval_reward

        log_metrics(logger, metrics_to_log, collected_frames)
        if i % 10 == 0:
            timeit.print()
            timeit.erase()

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()

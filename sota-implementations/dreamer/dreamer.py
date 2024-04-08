# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import hydra
import torch
import torch.cuda
import tqdm
from dreamer_utils import (
    log_metrics,
    make_collector,
    make_dreamer,
    make_environments,
    make_replay_buffer,
)

# mixed precision training
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
from torchrl.record.loggers import generate_exp_name, get_logger


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    # cfg = correct_for_frame_skip(cfg)

    if torch.cuda.is_available() and cfg.networks.device == "":
        device = torch.device("cuda:0")
    elif cfg.networks.device:
        device = torch.device(cfg.networks.device)
    else:
        device = torch.device("cpu")

    # Create logger
    exp_name = generate_exp_name("Dreamer", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="dreamer_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode},  # "config": cfg},
        )

    train_env, test_env = make_environments(cfg=cfg, parallel_envs=cfg.env.n_parallel_envs)

    # Make dreamer components
    action_key = "action"
    value_key = "state_value"
    world_model, model_based_env, actor_model, value_model, policy = make_dreamer(
        config=cfg,
        device=device,
        action_key=action_key,
        value_key=value_key,
        use_decoder_in_env=False,
    )

    # Losses
    world_model_loss = DreamerModelLoss(world_model)
    # Adapt loss keys to gym backend
    if cfg.env.backend == "gym":
        world_model_loss.set_keys(pixels="observation", reco_pixels="reco_observation")

    actor_loss = DreamerActorLoss(
        actor_model,
        value_model,
        model_based_env,
        imagination_horizon=cfg.optimization.imagination_horizon,
        discount_loss=True,
    )
    value_loss = DreamerValueLoss(value_model, discount_loss=True)

    # world_model_loss = torch.compile(world_model_loss)
    # actor_loss = torch.compile(actor_loss)
    # value_loss = torch.compile(value_loss)

    # Make collector
    collector = make_collector(cfg, train_env, policy)

    # Make replay buffer
    batch_length = cfg.optimization.batch_length
    replay_buffer = make_replay_buffer(
        batch_size=cfg.replay_buffer.batch_size,
        batch_seq_len=batch_length,
        buffer_size=cfg.replay_buffer.buffer_size,
        buffer_scratch_dir=cfg.replay_buffer.scratch_dir,
        device=cfg.networks.device,
        pixel_obs=cfg.env.from_pixels,
        grayscale=cfg.env.grayscale,
        image_size=cfg.env.image_size,
    )

    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    # Make optimizer
    world_model_opt = torch.optim.Adam(
        world_model.parameters(), lr=cfg.optimization.world_model_lr
    )
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=cfg.optimization.actor_lr)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.optimization.value_lr)

    # Grad scaler for mixed precision training https://pytorch.org/docs/stable/amp.html
    scaler1 = GradScaler()
    scaler2 = GradScaler()
    scaler3 = GradScaler()

    init_random_frames = cfg.collector.init_random_frames
    batch_size = cfg.optimization.batch_size
    optim_steps_per_batch = cfg.optimization.optim_steps_per_batch
    grad_clip = cfg.optimization.grad_clip
    frames_per_batch = cfg.collector.frames_per_batch
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.logger.eval_rollout_steps

    t_collect_init = time.time()
    for i, tensordict in enumerate(collector):
        t_collect = time.time() - t_collect_init

        t_preproc_init = time.time()
        pbar.update(tensordict.numel())
        current_frames = tensordict.numel()
        collected_frames += current_frames

        ep_reward = tensordict.get("episode_reward")[:, -1]
        replay_buffer.extend(tensordict.cpu())
        t_preproc = time.time() - t_preproc_init

        if collected_frames >= init_random_frames:
            t_loss_actor = 0.0
            t_loss_critic = 0.0
            t_loss_model = 0.0

            for _ in range(optim_steps_per_batch):
                # sample from replay buffer
                t_sample_init = time.time()
                sampled_tensordict = replay_buffer.sample(batch_size).reshape(-1, batch_length)
                t_sample = time.time() - t_sample_init

                t_loss_model_init = time.time()
                # update world model
                with autocast(dtype=torch.float16):
                    model_loss_td, sampled_tensordict = world_model_loss(
                        sampled_tensordict
                    )
                    loss_world_model = (
                        model_loss_td["loss_model_kl"]
                        + model_loss_td["loss_model_reco"]
                        + model_loss_td["loss_model_reward"]
                    )

                world_model_opt.zero_grad()
                scaler1.scale(loss_world_model).backward()
                scaler1.unscale_(world_model_opt)
                clip_grad_norm_(world_model.parameters(), grad_clip)
                scaler1.step(world_model_opt)
                scaler1.update()
                t_loss_model += (time.time()-t_loss_model_init)

                # update actor network
                t_loss_actor_init = time.time()
                with autocast(dtype=torch.float16):
                    actor_loss_td, sampled_tensordict = actor_loss(sampled_tensordict)

                actor_opt.zero_grad()
                scaler2.scale(actor_loss_td["loss_actor"]).backward()
                scaler2.unscale_(actor_opt)
                clip_grad_norm_(actor_model.parameters(), grad_clip)
                scaler2.step(actor_opt)
                scaler2.update()
                t_loss_actor += time.time() - t_loss_actor_init

                # update value network
                t_loss_critic_init = time.time()
                with autocast(dtype=torch.float16):
                    value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)

                value_opt.zero_grad()
                scaler3.scale(value_loss_td["loss_value"]).backward()
                scaler3.unscale_(value_opt)
                clip_grad_norm_(value_model.parameters(), grad_clip)
                scaler3.step(value_opt)
                scaler3.update()
                t_loss_critic += time.time() - t_loss_critic_init

        metrics_to_log = {"reward": ep_reward.mean().item()}
        if collected_frames >= init_random_frames:
            loss_metrics = {
                "loss_model_kl": model_loss_td["loss_model_kl"].item(),
                "loss_model_reco": model_loss_td["loss_model_reco"].item(),
                "loss_model_reward": model_loss_td["loss_model_reward"].item(),
                "loss_actor": actor_loss_td["loss_actor"].item(),
                "loss_value": value_loss_td["loss_value"].item(),
                "t_loss_actor": t_loss_actor,
                "t_loss_critic": t_loss_critic,
                "t_loss_model": t_loss_model,
                "t_sample": t_sample,
                "t_preproc": t_preproc,
                "t_collect": t_collect,
            }
            metrics_to_log.update(loss_metrics)

        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)

        policy.step(current_frames)
        collector.update_policy_weights_()
        # Evaluation
        if (i % eval_iter) == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_rollout = test_env.rollout(
                    eval_rollout_steps,
                    policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                eval_metrics = {"eval/reward": eval_reward}
                if logger is not None:
                    log_metrics(logger, eval_metrics, collected_frames)
        t_collect_init = time.time()

if __name__ == "__main__":
    main()

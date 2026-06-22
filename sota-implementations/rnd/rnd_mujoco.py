# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
PPO + Random Network Distillation (RND) on MuJoCo continuous-control tasks.

RND augments the extrinsic environment reward with a curiosity-driven
intrinsic bonus: the MSE between a trainable *predictor* network and a
randomly-initialised, permanently-frozen *target* network evaluated on the
next observation.  States the agent has visited frequently yield small
prediction errors (low bonus); novel states yield large errors (high bonus).

Reference:
    Burda et al., "Exploration by Random Network Distillation" (2018).
    https://arxiv.org/abs/1810.12894
"""

from __future__ import annotations

import hydra
from torchrl._utils import compile_with_warmup, get_available_device


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821
    import torch
    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.envs.transforms import RNDTransform
    from torchrl.objectives import ClipPPOLoss, group_optimizers
    from torchrl.objectives.rnd import RNDLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_mujoco import eval_model, make_env, make_ppo_models, make_rnd_networks

    torch.set_float32_matmul_precision("high")

    device = (
        torch.device(cfg.optim.device) if cfg.optim.device else get_available_device()
    )

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            compile_mode = "default" if cfg.compile.cudagraphs else "reduce-overhead"

    # ------------------------------------------------------------------
    # RND networks — must be created before the env so the transform can
    # reference them and share normalisation statistics with the loss.
    # ------------------------------------------------------------------

    # Build a temporary proof env to get the observation dimension.
    proof_env = make_env(cfg.env.env_name, device=device)
    obs_dim = proof_env.observation_spec["observation"].shape[-1]
    proof_env.close()

    target_net, predictor_net = make_rnd_networks(
        obs_dim=obs_dim,
        embed_dim=cfg.rnd.embed_dim,
        device=device,
    )

    # The transform computes the intrinsic reward at each env step and
    # maintains running statistics for obs / reward normalisation.
    rnd_transform = RNDTransform(
        target_network=target_net,
        predictor_network=predictor_net,
        obs_clip=cfg.rnd.obs_clip,
        reward_clip=cfg.rnd.reward_clip,
    )

    # ------------------------------------------------------------------
    # PPO actor / critic
    # ------------------------------------------------------------------
    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    # ------------------------------------------------------------------
    # Collector — env is wrapped with RNDTransform so every step
    # produces an "intrinsic_reward" entry in the next tensordict.
    # ------------------------------------------------------------------
    collector = Collector(
        create_env_fn=make_env(cfg.env.env_name, device, rnd_transform=rnd_transform),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        max_frames_per_traj=-1,
        compile_policy={"mode": compile_mode, "warmup": 1} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )

    # ------------------------------------------------------------------
    # Replay buffer (on-policy — PPO consumes each batch exactly once)
    # ------------------------------------------------------------------
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.collector.frames_per_batch,
            compilable=cfg.compile.compile,
            device=device,
        ),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
        compilable=cfg.compile.compile,
    )

    # ------------------------------------------------------------------
    # PPO loss + GAE advantage
    # ------------------------------------------------------------------
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
        vectorized=not cfg.compile.compile,
    )

    ppo_loss = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coeff=cfg.loss.entropy_coeff,
        critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )

    # ------------------------------------------------------------------
    # RND loss — shares obs_rms with the transform so normalisation is
    # consistent between collection time and training time.
    # ------------------------------------------------------------------
    rnd_loss = RNDLoss(
        predictor_network=predictor_net,
        target_network=target_net,
        obs_rms=rnd_transform.obs_rms,  # shared, populated after first step
        obs_clip=cfg.rnd.obs_clip,
        update_fraction=cfg.rnd.update_fraction,
    )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    ppo_optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    predictor_optim = torch.optim.Adam(
        predictor_net.parameters(),
        lr=cfg.rnd.predictor_lr,
        eps=1e-5,
    )

    # ------------------------------------------------------------------
    # Logger + test env
    # ------------------------------------------------------------------
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name(
            "PPO-RND", f"{cfg.logger.exp_name}_{cfg.env.env_name}"
        )
        logger = get_logger(
            cfg.logger.backend,
            logger_name="rnd",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )
        logger_video = cfg.logger.video
    else:
        logger_video = False

    test_env = make_env(cfg.env.env_name, device, from_pixels=logger_video)
    if logger_video:
        test_env = test_env.append_transform(
            VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    test_env.eval()

    # ------------------------------------------------------------------
    # Update function
    # ------------------------------------------------------------------
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = torch.tensor(cfg.optim.lr, device=device)
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_rnd_intrinsic_coeff = cfg.rnd.intrinsic_coeff

    def update(batch, num_network_updates):
        # Anneal LR and clip epsilon
        alpha = torch.ones((), device=device)
        if cfg_optim_anneal_lr:
            alpha = 1 - (num_network_updates / total_network_updates)
            for group in ppo_optim.param_groups:
                group["lr"] = cfg_optim_lr * alpha
        if cfg_loss_anneal_clip_eps:
            ppo_loss.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
        num_network_updates = num_network_updates + 1

        # PPO update
        ppo_optim.zero_grad(set_to_none=True)
        ppo_losses = ppo_loss(batch)
        (
            ppo_losses["loss_objective"]
            + ppo_losses["loss_critic"]
            + ppo_losses["loss_entropy"]
        ).backward()
        ppo_optim.step()

        # RND predictor update
        predictor_optim.zero_grad(set_to_none=True)
        rnd_losses = rnd_loss(batch)
        rnd_losses["loss_predictor"].backward()
        predictor_optim.step()

        return (
            ppo_losses.detach().update(rnd_losses.detach()).set("alpha", alpha),
            num_network_updates,
        )

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)
        adv_module = compile_with_warmup(adv_module, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
        adv_module = CudaGraphModule(adv_module)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    collected_frames = 0
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    collector_iter = iter(collector)
    total_iter = len(collector)

    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Log extrinsic episode rewards when episodes finish.
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )

        # Log mean intrinsic reward for this batch.
        metrics_to_log["train/intrinsic_reward"] = (
            data["next", "intrinsic_reward"].mean().item()
        )

        with timeit("training"):
            # Mix intrinsic and extrinsic reward before computing advantages.
            # The RNDTransform wrote "intrinsic_reward" into ("next", ...) at
            # collection time, so it is already in `data`.
            data["next", "reward"] = (
                data["next", "reward"]
                + cfg_rnd_intrinsic_coeff * data["next", "intrinsic_reward"]
            )

            for j in range(cfg_loss_ppo_epochs):
                with torch.no_grad(), timeit("adv"):
                    torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if compile_mode:
                        data = data.clone()

                with timeit("rb - extend"):
                    data_buffer.extend(data.reshape(-1))

                for k, batch in enumerate(data_buffer):
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss, num_network_updates = update(
                            batch, num_network_updates=num_network_updates
                        )
                        loss = loss.clone()
                    num_network_updates = num_network_updates.clone()
                    losses[j, k] = loss.select(
                        "loss_critic",
                        "loss_entropy",
                        "loss_objective",
                        "loss_predictor",
                    )

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log[f"train/{key}"] = value.item()
        metrics_to_log["train/lr"] = loss["alpha"] * cfg_optim_lr
        metrics_to_log["train/clip_epsilon"] = (
            loss["alpha"] * cfg_loss_clip_epsilon
            if cfg_loss_anneal_clip_eps
            else cfg_loss_clip_epsilon
        )

        with (
            torch.no_grad(),
            set_exploration_type(ExplorationType.DETERMINISTIC),
            timeit("eval"),
        ):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                actor.eval()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                metrics_to_log["eval/reward"] = test_rewards.mean()
                actor.train()

        if logger:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            logger.log_metrics(metrics_to_log, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()

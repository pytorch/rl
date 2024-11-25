# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import hydra
import torch

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="", config_name="config_atari", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    from copy import deepcopy

    import torch.optim
    import tqdm
    from tensordict import from_module
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import A2CLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_atari import eval_model, make_parallel_env, make_ppo_models

    device = cfg.loss.device
    if not device:
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    else:
        device = torch.device(device)

    # Correct for frame_skip
    frame_skip = 4
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    mini_batch_size = cfg.loss.mini_batch_size // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip

    # Create models (check utils_atari.py)
    actor, critic, critic_head = make_ppo_models(cfg.env.env_name, device=device)
    with from_module(actor).data.to("meta").to_module(actor):
        actor_eval = deepcopy(actor)
        actor_eval.eval()
    from_module(actor).data.to_module(actor_eval)

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=True,
        vectorized=not cfg.compile.compile,
        device=device,
    )
    loss_module = A2CLoss(
        actor_network=actor,
        critic_network=critic,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
    )

    # use end-of-life as done key
    adv_module.set_keys(done="end-of-life", terminated="end-of-life")
    loss_module.set_keys(done="end-of-life", terminated="end-of-life")

    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=torch.tensor(cfg.optim.lr, device=device),
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
        capturable=device.type == "cuda",
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("A2C", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="a2c",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create test environment
    test_env = make_parallel_env(cfg.env.env_name, 1, device, is_test=True)
    test_env.set_seed(0)
    if cfg.logger.video:
        test_env = test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["pixels"]
            ),
        )
    test_env.eval()

    # update function
    def update(batch, max_grad_norm=cfg.optim.max_grad_norm):
        # Forward pass A2C loss
        loss = loss_module(batch)

        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backward pass
        loss_sum.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=max_grad_norm
        )

        # Update the networks
        optim.step()
        optim.zero_grad(set_to_none=True)

        return (
            loss.select("loss_critic", "loss_entropy", "loss_objective")
            .detach()
            .set("grad_norm", gn)
        )

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"
        update = torch.compile(update, mode=compile_mode)
        adv_module = torch.compile(adv_module, mode=compile_mode)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
        adv_module = CudaGraphModule(adv_module)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_parallel_env(cfg.env.env_name, cfg.env.num_envs, device),
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        policy_device=device,
        compile_policy={"mode": compile_mode} if cfg.compile.compile else False,
        cudagraph_policy=cfg.compile.cudagraphs,
    )

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (total_frames // frames_per_batch) * num_mini_batches
    lr = cfg.optim.lr

    c_iter = iter(collector)
    for i in range(len(collector)):
        with timeit("collecting"):
            data = next(c_iter)

        log_info = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Get training rewards and lengths
        episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "terminated"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        losses = []

        # Compute GAE
        with torch.no_grad(), timeit("advantage"):
            torch.compiler.cudagraph_mark_step_begin()
            data = adv_module(data)
        data_reshape = data.reshape(-1)

        # Update the data buffer
        with timeit("rb - emptying"):
            data_buffer.empty()
        with timeit("rb - extending"):
            data_buffer.extend(data_reshape)

        with timeit("optim"):
            for batch in data_buffer:

                # Linearly decrease the learning rate and clip epsilon
                with timeit("optim - lr"):
                    alpha = 1.0
                    if cfg.optim.anneal_lr:
                        alpha = 1 - (num_network_updates / total_network_updates)
                        for group in optim.param_groups:
                            group["lr"].copy_(lr * alpha)

                num_network_updates += 1

                with timeit("update"):
                    torch.compiler.cudagraph_mark_step_begin()
                    loss = update(batch).clone()
                losses.append(loss)

        # Get training losses
        losses = torch.stack(losses).float().mean()

        for key, value in losses.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": lr * alpha,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch * frame_skip) // test_interval < (
                i * frames_in_batch * frame_skip
            ) // test_interval:
                test_rewards = eval_model(
                    actor_eval, test_env, num_episodes=cfg.logger.num_test_episodes
                )
                log_info.update(
                    {
                        "test/reward": test_rewards.mean(),
                    }
                )
        if i % 200 == 0:
            log_info.update(timeit.todict(prefix="time"))
            timeit.print()
            timeit.erase()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
from tensordict.nn import CudaGraphModule
from torchrl._utils import logger as torchrl_logger
from torchrl.record import VideoRecorder


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import time

    import torch.optim
    import tqdm

    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import A2CLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_mujoco import eval_model, make_env, make_ppo_models

    # Define paper hyperparameters

    device = cfg.loss.device
    if not device:
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    else:
        device = torch.device(device)

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        cfg.collector.total_frames // cfg.collector.frames_per_batch
    ) * num_mini_batches

    # Create models (check utils_mujoco.py)
    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.collector.frames_per_batch, device=device),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        vectorized=not cfg.loss.compile,
    )
    loss_module = A2CLoss(
        actor_network=actor,
        critic_network=critic,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=torch.tensor(cfg.optim.lr, device=device),
        capturable=device.type == "cuda",
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=torch.tensor(cfg.optim.lr, device=device),
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
    test_env = make_env(cfg.env.env_name, device, from_pixels=cfg.logger.video)
    test_env.set_seed(0)
    if cfg.logger.video:
        test_env = test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["pixels"]
            ),
        )

    def update(batch):
        # Forward pass A2C loss
        loss = loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss.get("loss_entropy", 0.0)

        # Backward pass
        (actor_loss + critic_loss).backward()

        # Update the networks
        actor_optim.step()
        critic_optim.step()

        actor_optim.zero_grad(set_to_none=True)
        critic_optim.zero_grad(set_to_none=True)
        return loss.select("loss_critic", "loss_objective").detach()  # , "loss_entropy"

    if cfg.loss.compile:
        compile_mode = cfg.loss.compile_mode
        if compile_mode in ("", None):
            if cfg.loss.cudagraphs:
                compile_mode = None
            else:
                compile_mode = "reduce-overhead"

        update = torch.compile(update, mode=compile_mode)
        actor = torch.compile(actor, mode=compile_mode)
        adv_module = torch.compile(adv_module, mode=compile_mode)

    if cfg.loss.cudagraphs:
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
        actor = CudaGraphModule(actor, warmup=10)
        adv_module = CudaGraphModule(adv_module)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        trust_policy=True,
    )

    test_env.eval()
    lr = cfg.optim.lr

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    sampling_start = time.time()
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Get training rewards and lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        losses = []
        training_start = time.time()

        # Compute GAE
        with torch.no_grad():
            data = adv_module(data)
        data_reshape = data.reshape(-1)

        # Update the data buffer
        data_buffer.extend(data_reshape)

        for batch in data_buffer:

            # Linearly decrease the learning rate and clip epsilon
            alpha = 1.0
            if cfg.optim.anneal_lr:
                alpha = 1 - (num_network_updates / total_network_updates)
                for group in actor_optim.param_groups:
                    group["lr"].copy_(lr * alpha)
                for group in critic_optim.param_groups:
                    group["lr"].copy_(lr * alpha)
            num_network_updates += 1
            torch.compiler.cudagraph_mark_step_begin()
            loss = update(batch)
            losses.append(loss)

        # Get training losses
        training_time = time.time() - training_start
        losses = torch.stack(losses).float().mean()
        for key, value in losses.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg.optim.lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            prev_test_frame = ((i - 1) * frames_in_batch) // cfg.logger.test_interval
            cur_test_frame = (i * frames_in_batch) // cfg.logger.test_interval
            final = collected_frames >= collector.total_frames
            if prev_test_frame < cur_test_frame or final:
                actor.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg.logger.num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "test/reward": test_rewards.mean(),
                        "test/eval_time": eval_time,
                    }
                )
                actor.train()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        sampling_start = time.time()
        torch.compiler.cudagraph_mark_step_begin()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()

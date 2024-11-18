# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import hydra
import torch.nn
import torch.optim
import tqdm

from tensordict.nn import CudaGraphModule, TensorDictSequential
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_cartpole import eval_model, make_dqn_model, make_env


@hydra.main(config_path="", config_name="config_cartpole", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Make the components
    model = make_dqn_model(cfg.env.env_name)

    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=model.spec,
    )
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    ).to(device)

    # Create the replay buffer
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=10,
        storage=LazyTensorStorage(
            max_size=cfg.buffer.buffer_size,
            device="cpu",
        ),
        batch_size=cfg.buffer.batch_size,
    )

    # Create the loss module
    loss_module = DQNLoss(
        value_network=model,
        loss_function="l2",
        delay_value=True,
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)
    loss_module = loss_module.to(device)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("DQN", f"CartPole_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="dqn",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create the test environment
    test_env = make_env(cfg.env.env_name, "cpu", from_pixels=cfg.logger.video)
    if cfg.logger.video:
        test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["pixels"]
            ),
        )

    def update(sampled_tensordict):
        loss_td = loss_module(sampled_tensordict)
        q_loss = loss_td["loss"]
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        target_net_updater.step()
        return q_loss.detach()

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    # Create the collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, "cpu"),
        policy=model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device="cpu",
        storing_device="cpu",
        max_frames_per_traj=-1,
        init_random_frames=cfg.collector.init_random_frames,
        compile_policy={"mode": compile_mode} if compile_mode is not None else False,
        cudagraph_policy=cfg.compile.cudagraphs,
    )

    # Main loop
    collected_frames = 0
    num_updates = cfg.loss.num_updates
    batch_size = cfg.buffer.batch_size
    test_interval = cfg.logger.test_interval
    num_test_episodes = cfg.logger.num_test_episodes
    frames_per_batch = cfg.collector.frames_per_batch
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    init_random_frames = cfg.collector.init_random_frames
    q_losses = torch.zeros(num_updates, device=device)

    c_iter = iter(collector)
    for i in range(len(collector)):
        with timeit("collecting"):
            data = next(c_iter)

        log_info = {}
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()

        with timeit("rb - extend"):
            replay_buffer.extend(data)
        collected_frames += current_frames
        greedy_module.step(current_frames)

        # Get and log training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            log_info.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    "train/episode_length": episode_length_mean,
                }
            )

        if collected_frames < init_random_frames:
            if collected_frames < init_random_frames:
                if logger:
                    for key, value in log_info.items():
                        logger.log_scalar(key, value, step=collected_frames)
                continue

        # optimization steps
        for j in range(num_updates):
            with timeit("rb - sample"):
                sampled_tensordict = replay_buffer.sample(batch_size)
                sampled_tensordict = sampled_tensordict.to(device)
            with timeit("update"):
                q_loss = update(sampled_tensordict)
            q_losses[j].copy_(q_loss)

        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_values": (data["action_value"] * data["action"]).sum().item()
                / frames_per_batch,
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
            }
        )

        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            prev_test_frame = ((i - 1) * frames_per_batch) // test_interval
            cur_test_frame = (i * frames_per_batch) // test_interval
            final = current_frames >= collector.total_frames
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                model.eval()
                test_rewards = eval_model(model, test_env, num_test_episodes)
                model.train()
                log_info.update(
                    {
                        "eval/reward": test_rewards,
                    }
                )

        if i % 200 == 0:
            timeit.print()
            log_info.update(timeit.todict(prefix="time"))
            timeit.erase()

        # Log all the information
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DQN: Reproducing experimental results from Mnih et al. 2015 for the
Deep Q-Learning Algorithm on Atari Environments.
"""
from __future__ import annotations

import functools
import warnings

import hydra
import torch.nn
import torch.optim
import tqdm
from tensordict.nn import CudaGraphModule, TensorDictSequential
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_atari import eval_model, make_dqn_model, make_env

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="", config_name="config_atari", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Correct for frame_skip
    frame_skip = 4
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    init_random_frames = cfg.collector.init_random_frames // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip

    # Make the components
    model = make_dqn_model(
        cfg.env.env_name,
        gym_backend=cfg.env.backend,
        frame_skip=frame_skip,
        device=device,
    )
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=model.spec,
        device=device,
    )
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    )

    # Create the replay buffer
    if cfg.buffer.scratch_dir in ("", None):
        storage_cls = LazyMemmapStorage
    else:
        storage_cls = functools.partial(
            LazyMemmapStorage, scratch_dir=cfg.buffer.scratch_dir
        )

    def transform(td):
        return td.to(device)

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        storage=storage_cls(
            max_size=cfg.buffer.buffer_size,
        ),
        batch_size=cfg.buffer.batch_size,
    )
    if transform is not None:
        replay_buffer.append_transform(transform)

    # Create the loss module
    loss_module = DQNLoss(
        value_network=model,
        loss_function="l2",
        delay_value=True,
    )
    loss_module.set_keys(done="end-of-life", terminated="end-of-life")
    loss_module.make_value_estimator(gamma=cfg.loss.gamma, device=device)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("DQN", f"Atari_mnih15_{cfg.env.env_name}")
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
    test_env = make_env(
        cfg.env.env_name,
        frame_skip,
        device,
        gym_backend=cfg.env.backend,
        is_test=True,
    )
    if cfg.logger.video:
        test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["pixels"]
            ),
        )
    test_env.eval()

    def update(sampled_tensordict):
        loss_td = loss_module(sampled_tensordict)
        q_loss = loss_td["loss"]
        optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(loss_module.parameters()), max_norm=max_grad
        )
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
        create_env_fn=make_env(
            cfg.env.env_name, frame_skip, device, gym_backend=cfg.env.backend
        ),
        policy=model_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        init_random_frames=init_random_frames,
        compile_policy={"mode": compile_mode, "fullgraph": True}
        if compile_mode is not None
        else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )

    # Main loop
    collected_frames = 0
    num_updates = cfg.loss.num_updates
    max_grad = cfg.optim.max_grad_norm
    num_test_episodes = cfg.logger.num_test_episodes
    q_losses = torch.zeros(num_updates, device=device)
    pbar = tqdm.tqdm(total=total_frames)

    c_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)
        with timeit("collecting"):
            data = next(c_iter)
        metrics_to_log = {}
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel() * frame_skip
        collected_frames += current_frames
        greedy_module.step(current_frames)
        with timeit("rb - extend"):
            replay_buffer.extend(data)

        # Get and log training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            metrics_to_log.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    "train/episode_length": episode_length_mean,
                }
            )

        if collected_frames < init_random_frames:
            if logger:
                for key, value in metrics_to_log.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue

        # optimization steps
        for j in range(num_updates):
            with timeit("rb - sample"):
                sampled_tensordict = replay_buffer.sample()
            with timeit("update"):
                q_loss = update(sampled_tensordict)
            q_losses[j].copy_(q_loss)

        # Get and log q-values, loss, epsilon, sampling time and training time
        metrics_to_log.update(
            {
                "train/q_values": data["chosen_action_value"].sum() / frames_per_batch,
                "train/q_loss": q_losses.mean(),
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
                test_rewards = eval_model(
                    model, test_env, num_episodes=num_test_episodes
                )
                metrics_to_log.update(
                    {
                        "eval/reward": test_rewards,
                    }
                )
                model.train()

        # Log all the information
        if logger:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            for key, value in metrics_to_log.items():
                logger.log_scalar(key, value, step=collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()

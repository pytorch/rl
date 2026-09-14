# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO with GTrXL on CartPole with hidden velocities and dense window replay."""
from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.collectors import Collector
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers import get_logger
from utils import make_env, make_policy, make_window_records


@torch.no_grad()
def evaluate(policy, num_envs, seed):
    """Measure first-episode returns with deterministic actions on fresh streams."""
    env = make_env(num_envs, seed)
    env.append_transform(policy[0].make_tensordict_primer())
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            data = env.rollout(500, policy=policy, break_when_any_done=False)
        done = data["next", "done"].squeeze(-1)
        first = done.long().argmax(-1)
        return (
            data["next", "episode_reward"].squeeze(-1).gather(-1, first[:, None]).mean()
        )
    finally:
        env.close()


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    # This small Gym example deliberately colocates environment, policy and replay.
    if device.type != "cpu":
        raise ValueError("This CartPole reference runs on CPU; set device=cpu")
    if cfg.replay.window_length < 1 or cfg.collector.steps_per_batch < 1:
        raise ValueError("Window and rollout lengths must be positive")
    policy, transformer, actor, critic = make_policy(cfg.network, device)
    env = make_env(cfg.env.num_envs, cfg.seed)
    env.append_transform(transformer.make_tensordict_primer())
    frames_per_batch = cfg.env.num_envs * cfg.collector.steps_per_batch
    record_count = cfg.env.num_envs * math.ceil(
        cfg.collector.steps_per_batch / cfg.replay.window_length
    )
    if cfg.replay.storage == "memmap":
        storage = LazyMemmapStorage(record_count, scratch_dir=cfg.replay.scratch_dir)
    elif cfg.replay.storage == "tensor":
        storage = LazyTensorStorage(record_count, device=device)
    else:
        raise ValueError("replay.storage must be tensor or memmap")
    replay = TensorDictReplayBuffer(
        storage=storage,
        batch_size=cfg.replay.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
    )
    # The encoder runs once per minibatch. The PPO heads consume its fresh features
    # and both losses backpropagate through the shared encoder.
    loss_module = ClipPPOLoss(
        actor,
        critic,
        functional=False,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_coeff,
        normalize_advantage=False,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optim.lr)
    advantage = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=None,
        average_gae=False,
    )
    value_policy = TensorDictSequential(transformer, critic)
    collector = Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        auto_register_policy_transforms=True,
    )
    logger = (
        get_logger(cfg.logger.backend, "gtrxl", logger_name="gtrxl_ppo")
        if cfg.logger.backend
        else None
    )
    metrics = []
    frames = 0
    next_eval = cfg.logger.eval_interval
    initial_return = evaluate(policy, cfg.logger.eval_envs, cfg.seed + 10000)
    torchrl_logger.info(
        "seed=%s initial evaluation return=%.1f", cfg.seed, initial_return
    )
    try:
        for data in collector:
            frames += data.numel()
            with torch.no_grad(), timeit("gtrxl/advantages"):
                # Interior bootstrap values were already computed at the next
                # actor step. Evaluate only episode and rollout boundaries.
                next_value = data["state_value"].roll(-1, dims=1)
                boundary = data["next", "done"].squeeze(-1).clone()
                boundary[:, -1] = True
                bootstrap = data["next"][boundary].clone()
                bootstrap["is_init"] = torch.zeros(
                    bootstrap.batch_size + (1,), dtype=torch.bool, device=device
                )
                next_value[boundary] = value_policy(bootstrap)["state_value"]
                data["next", "state_value"] = next_value
                advantage(data)
                adv = data["advantage"]
                data["advantage"] = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            # Storage's index is a window id. M belongs only to initial_state;
            # T belongs only to transitions. No SliceSampler or flattening time.
            records = make_window_records(data, cfg.replay.window_length)
            replay.empty()
            replay.extend(records)
            with timeit("gtrxl/training"):
                for _ in range(cfg.loss.epochs):
                    for sample in replay:
                        transitions = sample["transitions"]
                        inputs = TensorDict(
                            {
                                "observation": transitions["observation"],
                                "is_init": transitions["is_init"],
                                "state": sample["initial_state"],
                            },
                            sample.batch_size,
                            device=device,
                        )
                        with set_recurrent_mode(True):
                            transformer(inputs)
                        transitions["features"] = inputs["features"]
                        losses = loss_module(transitions)
                        loss = (
                            losses["loss_objective"]
                            + losses["loss_critic"]
                            + losses["loss_entropy"]
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(),
                            cfg.optim.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
            # This retains caller-owned carry across updates. Saved activations
            # can be stale, exactly as with stored GRU/LSTM carries.
            collector.update_policy_weights_()
            if frames >= next_eval or frames >= cfg.collector.total_frames:
                reward = evaluate(policy, cfg.logger.eval_envs, cfg.seed + 10000)
                row = TensorDict(
                    {
                        "frames": torch.tensor(frames),
                        "return": reward,
                        "loss": loss.detach(),
                    },
                    [],
                )
                metrics.append(row)
                torchrl_logger.info(
                    "seed=%s frames=%s evaluation return=%.1f loss=%.4f",
                    cfg.seed,
                    frames,
                    reward,
                    loss.detach(),
                )
                if logger:
                    logger.log_scalar("eval/return", reward.item(), frames)
                next_eval = frames + cfg.logger.eval_interval
    finally:
        collector.shutdown()
    history = torch.stack(metrics)
    result = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "initial_return": initial_return.item(),
        "metrics": history.to_dict(),
        "timings": timeit.todict(),
        "replay_state_bytes": records["initial_state", "memory"].numel()
        * records["initial_state", "memory"].element_size(),
        "per_step_state_bytes": data["state", "memory"].numel()
        * data["state", "memory"].element_size(),
    }
    result["metrics"] = {
        key: value.tolist() for key, value in result["metrics"].items()
    }
    output = Path(cfg.logger.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

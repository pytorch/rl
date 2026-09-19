# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torchrl.collectors import Collector
from utils import evaluate, make_agent, make_data_and_envs, REPLAY_KEYS


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    replay, env, eval_env = make_data_and_envs(cfg)
    policy, loss_module, updater = make_agent(cfg, env, device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    step = 0

    def update():
        losses = loss_module(replay.sample().to(device))
        total = sum(losses[key] for key in loss_module.out_keys)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        # Match the reference update's use of pre-optimizer critic parameters.
        updater.step()
        optimizer.step()
        return {f"training/{key}": losses[key].detach() for key in loss_module.out_keys}

    def record(metrics, step):
        if (
            step % cfg.evaluation.interval == 0
            or step == cfg.optim.offline_steps + cfg.optim.online_steps
        ):
            metrics.update(evaluate(policy, eval_env, cfg))
        if step % cfg.log_interval == 0 or "evaluation/return" in metrics:
            metrics = {key: float(value) for key, value in metrics.items()}
            metrics.update(step=step, replay_size=len(replay))
            with (output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")

    try:
        record({}, step)
        for step in range(1, cfg.optim.offline_steps + 1):
            record(update(), step)
        if cfg.optim.online_steps:
            collector = Collector(
                env,
                policy,
                frames_per_batch=1,
                total_frames=cfg.optim.online_steps,
                policy_device=device,
                env_device="cpu",
                storing_device="cpu",
                auto_register_policy_transforms=True,
            )
            try:
                for step, transition in enumerate(
                    collector, cfg.optim.offline_steps + 1
                ):
                    replay.extend(transition.reshape(-1).select(*REPLAY_KEYS))
                    record(update(), step)
                    collector.update_policy_weights_()
            finally:
                collector.shutdown()
        torch.save(
            {
                "loss": loss_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "step": step,
            },
            output_dir / "checkpoint.pt",
        )
    finally:
        if not env.is_closed:
            env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

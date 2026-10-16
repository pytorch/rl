# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Transitive RL (Park et al., 2026) on OGBench goal-conditioned datasets.

Reference: https://arxiv.org/abs/2510.22512
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensordict.nn import CudaGraphModule

from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.record.loggers import get_logger
from utils import evaluate, make_agent, make_replay_buffer


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    replay, eval_env, goal_dim = make_replay_buffer(cfg, device)
    eval_env.set_seed(cfg.seed)
    policy, loss_module, updater = make_agent(cfg, replay, goal_dim, device)
    optimizer = torch.optim.Adam(
        loss_module.parameters(), lr=cfg.optim.lr, capturable=cfg.compile.cudagraphs
    )

    def update(batch):
        losses = loss_module(batch)
        total = sum(v for k, v in losses.items() if k.startswith("loss_"))
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        updater.step()
        return losses.detach()

    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode or (
            "default" if cfg.compile.cudagraphs else "reduce-overhead"
        )
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)
    if cfg.compile.cudagraphs:
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=str(output_dir),
        experiment_name="transitive_rl",
        wandb_kwargs={
            "project": cfg.logger.project,
            "group": cfg.logger.group,
            "mode": cfg.logger.mode,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )

    for step in range(1, cfg.optim.steps + 1):
        with timeit("train/sample"):
            batch = replay.sample()
        with timeit("train/update"):
            losses = update(batch)
        if step % cfg.log_interval == 0:
            metrics = losses.apply(lambda x: x.mean()).to_dict()
            metrics = {f"training/{k}": float(v) for k, v in metrics.items()}
            timings = timeit.todict(percall=True)
            metrics.update(
                {
                    "timing/step_s": timings["train/sample"] + timings["train/update"],
                    "timing/sample_s": timings["train/sample"],
                    "timing/update_s": timings["train/update"],
                }
            )
            timeit.erase()
            logger.log_metrics(metrics, step=step)
            torchrl_logger.info(
                f"step {step} loss_qvalue {metrics['training/loss_qvalue']:.4f} "
                f"q_mean {metrics['training/q_mean']:.3f} "
                f"{1 / metrics['timing/step_s']:.1f} it/s"
            )
        if step % cfg.evaluation.interval == 0:
            with timeit("eval"):
                results = evaluate(policy, eval_env, cfg)
            logger.log_metrics({k: float(v) for k, v in results.items()}, step=step)
            torchrl_logger.info(
                f"step {step} eval overall {float(results['evaluation/overall']):.3f}"
            )
    eval_env.close()


if __name__ == "__main__":
    main()

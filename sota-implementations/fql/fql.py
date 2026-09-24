# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchrl.checkpoint import (
    Checkpoint,
    CheckpointRotation,
    resolve_checkpoint_path,
    resume_config,
)
from torchrl.collectors import Collector
from torchrl.record.loggers import get_logger
from torchrl.trainers.algorithms import OfflineToOnlineTrainer
from utils import evaluate, make_agent, make_data_and_envs


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    resume_path = None
    if cfg.resume:
        resume_path = resolve_checkpoint_path(to_absolute_path(cfg.resume))
        cfg = resume_config(cfg, resume_path)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    replay, env, eval_env = make_data_and_envs(cfg)
    policy, loss_module, updater = make_agent(cfg, env, device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=str(output_dir),
        experiment_name="fql",
        state_dict=(
            Checkpoint.read_component(resume_path, "logger", default=None)
            if resume_path
            else None
        ),
        wandb_kwargs={
            "project": cfg.logger.project,
            "mode": cfg.logger.mode,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )
    collector = (
        partial(
            Collector,
            env,
            policy,
            frames_per_batch=1,
            total_frames=cfg.optim.online_steps,
            policy_device=device,
            env_device="cpu",
            storing_device="cpu",
            auto_register_policy_transforms=True,
        )
        if cfg.optim.online_steps
        else None
    )
    trainer = OfflineToOnlineTrainer(
        loss_module=loss_module,
        optimizer=optimizer,
        replay_buffer=replay,
        target_net_updater=updater,
        offline_steps=cfg.optim.offline_steps,
        collector=collector,
        total_frames=cfg.optim.online_steps,
        device=device,
        compile_loss=cfg.optim.compile_loss,
        logger=logger,
        enable_logging=False,
        auto_log_optim_steps=False,
        progress_bar=False,
        checkpoint=Checkpoint(config=OmegaConf.to_container(cfg, resolve=False)),
        checkpoint_rotation=CheckpointRotation(output_dir / "checkpoints", keep_last=1),
    )

    def record(step: int, losses=None) -> None:
        if logger is None:
            return
        if losses is not None and step % cfg.log_interval == 0:
            logger.with_prefix("training").log_metrics(
                losses.select(*loss_module.out_keys), step=step
            )
        if (
            step % cfg.evaluation.interval == 0
            or step == cfg.optim.offline_steps + cfg.optim.online_steps
        ):
            logger.log_metrics(evaluate(policy, eval_env, cfg, stop=stop), step=step)

    trainer.register_op("post_optim_complete_log", record)
    try:
        if resume_path:
            trainer.load_from_file(resume_path, map_location=device)
        with trainer.stop_on_signal() as stop:
            record(trainer.completed_steps)
            trainer.train()
        if stop.requested:
            raise SystemExit(130)
    finally:
        if not env.is_closed:
            env.close()
        eval_env.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()

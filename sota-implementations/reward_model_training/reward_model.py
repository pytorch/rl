# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""RLHF reward-model training recipe.

Trains a scalar reward model from pairwise human-preference data using the
Bradley-Terry objective :class:`~torchrl.objectives.llm.RewardModelLoss`. This is the
reward-modelling stage that precedes policy optimization (e.g. GRPO/PPO) in RLHF
pipelines.

The backbone is any Hugging Face ``AutoModelForSequenceClassification`` with a
single-output head. The helper functions live in the ``utils.py`` next to this script.

Examples:
    Train on the default summarization-comparisons dataset with a small Qwen backbone::

        python reward_model.py model.name=Qwen/Qwen2.5-0.5B

    Run a quick hermetic smoke test (tiny from-scratch model, synthetic data)::

        python reward_model.py model.name= data.dataset_name= \\
            optim.max_iters=3 logger.backend=
"""
from __future__ import annotations

import hydra
import torch
import tqdm
from torchrl._utils import get_available_device, logger as torchrl_logger, timeit
from torchrl.objectives.llm import RewardModelLoss
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    get_vocab_size,
    log_metrics,
    make_dataset,
    make_optimizer,
    make_replay_buffer,
    make_reward_model,
    make_tokenizer,
)


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg):  # noqa: F821
    # Logger
    exp_name = generate_exp_name("RewardModel", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="reward_model_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(int(cfg.seed))
    device = (
        torch.device(cfg.model.device) if cfg.model.device else get_available_device()
    )

    # Model + data
    with timeit("setup/model"):
        score_network = make_reward_model(cfg, device)
    tokenizer = make_tokenizer(cfg)
    vocab_size = get_vocab_size(tokenizer, score_network)

    with timeit("setup/data"):
        train_data = make_dataset(cfg, tokenizer, cfg.data.split_train, vocab_size)
        val_data = make_dataset(cfg, tokenizer, cfg.data.split_val, vocab_size)
        train_rb = make_replay_buffer(train_data, cfg.data.batch_size, device)
        val_rb = make_replay_buffer(val_data, cfg.data.batch_size, device)

    # Loss + optimizer
    loss_module = RewardModelLoss(
        score_network,
        reduction=cfg.loss.reduction,
        center_coeff=cfg.loss.center_coeff,
        device=device,
    )
    optimizer = make_optimizer(cfg, score_network)

    n_trainable = sum(p.numel() for p in score_network.parameters() if p.requires_grad)
    torchrl_logger.info(f"Reward model with {n_trainable} trainable parameters.")

    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        score_network.eval()
        losses, accuracies = [], []
        for _ in range(int(cfg.logger.eval_iters)):
            batch = val_rb.sample()
            out = loss_module(batch)
            losses.append(out.loss_reward_model)
            accuracies.append(out.accuracy)
        score_network.train()
        return (
            torch.stack(losses).mean().item(),
            torch.stack(accuracies).mean().item(),
        )

    clip_grad = cfg.optim.clip_grad
    max_iters = int(cfg.optim.max_iters)
    pbar = tqdm.tqdm(range(1, max_iters + 1))
    for it in pbar:
        timeit.printevery(num_prints=1000, total_count=max_iters, erase=True)

        with timeit("train/sample"):
            batch = train_rb.sample()

        with timeit("train/forward"):
            loss_out = loss_module(batch)
            loss = loss_out.loss_reward_model
            if loss_out.loss_center is not None:
                loss = loss + loss_out.loss_center

        optimizer.zero_grad(set_to_none=True)
        with timeit("train/backward"):
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            score_network.parameters(), clip_grad
        )
        with timeit("train/optim_step"):
            optimizer.step()

        metrics_to_log = {
            "train/loss": loss.item(),
            "train/accuracy": loss_out.accuracy.item(),
            "train/grad_norm": grad_norm.item(),
        }
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", acc=f"{loss_out.accuracy.item():.3f}"
        )

        if it % int(cfg.logger.eval_iter) == 0:
            val_loss, val_acc = evaluate()
            metrics_to_log["eval/loss"] = val_loss
            metrics_to_log["eval/accuracy"] = val_acc
            torchrl_logger.info(
                f"iter {it}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if it % int(cfg.checkpoint.save_iter) == 0:
            score_network.module.model.save_pretrained(cfg.checkpoint.save_dir)

        if logger is not None and it % int(cfg.logger.log_interval) == 0:
            metrics_to_log.update(timeit.todict(prefix="time"))
            log_metrics(logger, metrics_to_log, it)

    score_network.module.model.save_pretrained(cfg.checkpoint.save_dir)
    pbar.close()


if __name__ == "__main__":
    main()

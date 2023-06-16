# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import hydra
import torch
from data import get_reward_dataloader
from models.reward import init_reward_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_file_logger, setup, resolve_name_or_path


def _accuracy(chosen_end_scores, rejected_end_scores):
    return (
        sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    ).item()


# TODO: eliminate redundant repeated definition
# helps estimate an arbitrarily accurate loss over either split using many batches
def create_loss_estimator(eval_iters, ctx):
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        model.eval()
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            chosen_batch, rejected_batch = next(dataloader)
            with ctx:
                model(chosen_batch)
                model(rejected_batch)
            losses[k] = model.compute_reward_loss(chosen_batch, rejected_batch).item()
            accs[k] = _accuracy(chosen_batch.end_scores, rejected_batch.end_scores)
        model.train()
        return losses.mean(), accs.mean()

    return estimate_loss


@hydra.main(version_base="1.1", config_path="config", config_name="train_reward")
def main(cfg):
    loss_logger = get_file_logger("loss_logger", "reward_loss_logger.log")

    data_cfg = cfg.data
    model_cfg = cfg.model
    reward_model_cfg = cfg.reward_model
    train_cfg = cfg.train

    eval_interval = cfg.io.eval_interval
    log_interval = cfg.io.log_interval
    eval_iters = cfg.io.eval_iters
    reward_out_dir = reward_model_cfg.out_dir

    max_iters = train_cfg.max_iters
    always_save_checkpoint = train_cfg.always_save_checkpoint

    device = cfg.sys.device
    dtype = cfg.sys.dtype
    compile_ = cfg.sys.compile

    ctx = setup(device=device, dtype=dtype)

    train_loader = get_reward_dataloader(data_cfg, device=device, split="train")
    val_loader = get_reward_dataloader(data_cfg, device=device, split="valid1")

    if reward_model_cfg.init_from == "resume":
        model = init_reward_model(
            reward_model_path=resolve_name_or_path(reward_model_cfg.out_dir),
            device=device,
            compile_=compile_,
        )
    else:
        model = init_reward_model(
            transformer_path=resolve_name_or_path(model_cfg.name_or_path),
            device=device,
            compile_=compile_,
        )
    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # ######## INIT TRAINING FUNCTIONS ########

    optimizer = torch.optim.AdamW(model.parameters(), **train_cfg.optimizer)
    scheduler = None
    if train_cfg.decay_lr:
        scheduler = CosineAnnealingLR(optimizer, **train_cfg.scheduler)

    estimate_loss = create_loss_estimator(eval_iters, ctx)

    best_val_loss = float("inf")

    t0 = time.time()
    for it in range(1, max_iters + 1):
        chosen_batch, rejected_batch = next(train_loader)

        with ctx:
            model(chosen_batch)
            model(rejected_batch)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_reward_loss(chosen_batch, rejected_batch)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if it % eval_interval == 0:
            val_loss, val_acc = estimate_loss(model, val_loader)
            train_loss, train_acc = estimate_loss(model, train_loader)

            msg = (f"VALID: {it=}: {train_loss=:.4f}, {val_loss=:.4f}, "
                   f"{train_acc=:.4f}, {val_acc=:.4f}")
            print(msg)
            loss_logger.info(msg)
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if it > 0:
                    msg = f"saving checkpoint to {reward_out_dir}"
                    print(msg)
                    loss_logger.info(msg)
                    model.module.save_pretrained(reward_out_dir)
        elif it % log_interval == 0:
            loss = loss.item()
            acc = _accuracy(chosen_batch.end_scores, rejected_batch.end_scores)
            msg = f"TRAIN: {it=}: {loss=:.4f}, {acc=:.4f} time={dt*1000:.2f}ms"
            print(msg)
            loss_logger.info(msg)


if __name__ == "__main__":
    main()

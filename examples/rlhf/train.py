# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train the transformer model. Configurable via config/train.yaml, but any argument can
also be overridden at the command line.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""
import time

import hydra
import torch
from models.transformer import init_transformer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchrl._utils import logger as torchrl_logger

from torchrl.data.rlhf.dataset import get_dataloader
from torchrl.data.rlhf.prompt import PromptData
from utils import get_file_logger, resolve_name_or_path, setup


def create_loss_estimator(eval_iters, ctx):
    # helps estimate an arbitrarily accurate loss over either split using many batches

    @torch.no_grad()
    def estimate_loss(model, dataloader):
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = next(dataloader)
            batch.batch_size = []
            with ctx:
                model(batch)
            losses[k] = batch.loss.item()
        model.train()
        return losses.mean()

    return estimate_loss


@hydra.main(version_base="1.1", config_path="config", config_name="train")
def main(cfg):
    loss_logger = get_file_logger("loss_logger", "transformer_loss_logger.log")

    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train

    eval_interval = cfg.io.eval_interval
    log_interval = cfg.io.log_interval
    eval_iters = cfg.io.eval_iters
    out_dir = model_cfg.out_dir

    grad_clip = train_cfg.grad_clip
    max_iters = train_cfg.max_iters
    always_save_checkpoint = train_cfg.always_save_checkpoint
    gradient_accumulation_steps = train_cfg.gradient_accumulation_steps

    device = cfg.sys.device
    dtype = cfg.sys.dtype
    compile_ = cfg.sys.compile

    ctx = setup(cfg.sys)

    train_loader = get_dataloader(
        data_cfg.batch_size,
        data_cfg.block_size,
        PromptData,
        device,
        dataset_name="CarperAI/openai_summarize_tldr",
        split="train",
    )
    val_loader = get_dataloader(
        data_cfg.batch_size,
        data_cfg.block_size,
        PromptData,
        device,
        dataset_name="CarperAI/openai_summarize_tldr",
        split="valid",
    )

    model = init_transformer(
        resolve_name_or_path(model_cfg.name_or_path),
        model_cfg.dropout,
        device,
        compile_model=compile_,
    )
    optimizer = torch.optim.AdamW(model.parameters(), **train_cfg.optimizer)
    scheduler = None
    if train_cfg.decay_lr:
        scheduler = CosineAnnealingLR(optimizer, **train_cfg.scheduler)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    estimate_loss = create_loss_estimator(eval_iters, ctx)

    best_val_loss = float("inf")

    t0 = time.time()
    next_batch = next(train_loader)  # fetch the very first batch
    for it in range(1, max_iters + 1):
        for _ in range(gradient_accumulation_steps):
            batch = next_batch
            # TODO: can we handle this better with a differently structured tensorclass?
            batch.batch_size = []
            with ctx:
                model(batch)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            next_batch = next(train_loader)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(batch.loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if it % eval_interval == 0:
            # evaluate the loss on train/val sets and write checkpoints
            train_loss = estimate_loss(model, train_loader)
            val_loss = estimate_loss(model, val_loader)
            msg = f"VALID: {it=}: {train_loss=:.4f}, {val_loss=:.4f}"
            torchrl_logger.info(msg)
            loss_logger.info(msg)
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if it > 0:
                    msg = f"saving checkpoint to {out_dir}"
                    torchrl_logger.info(msg)
                    loss_logger.info(msg)
                    model.module.save_pretrained(out_dir)
        elif it % log_interval == 0:
            # loss as float. note: this is a CPU-GPU sync point
            loss = batch.loss.item()
            msg = f"TRAIN: {it=}: {loss=:.4f}, time {dt*1000:.2f}ms"
            torchrl_logger.info(msg)
            loss_logger.info(msg)


if __name__ == "__main__":
    main()

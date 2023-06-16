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

import torch

from data.openai_summarize_tldr import get_prompt_dataloader
from models.transformer import init_transformer
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_file_logger, load_and_update_config, setup


def create_loss_estimator(eval_iters, ctx):
    # helps estimate an arbitrarily accurate loss over either split using many batches

    @torch.no_grad()
    def estimate_loss(model, dataloader):
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = next(dataloader)
            with ctx:
                model(batch)
            losses[k] = batch.loss.item()
        model.train()
        return losses.mean()

    return estimate_loss


def main():
    loss_logger = get_file_logger("loss_logger", "transformer_loss_logger.log")
    # load and extract configuration
    config = load_and_update_config("config/train.yaml")

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    eval_interval = config["io"]["eval_interval"]
    log_interval = config["io"]["log_interval"]
    eval_iters = config["io"]["eval_iters"]
    out_dir = model_config["out_dir"]

    grad_clip = train_config["grad_clip"]
    max_iters = train_config["max_iters"]
    always_save_checkpoint = train_config["always_save_checkpoint"]
    gradient_accumulation_steps = train_config["gradient_accumulation_steps"]

    device = config["sys"]["device"]
    dtype = config["sys"]["dtype"]
    compile_ = config["sys"]["compile"]

    ctx = setup(device=device, dtype=dtype)

    train_loader = get_prompt_dataloader(data_config, device=device, split="train")
    val_loader = get_prompt_dataloader(data_config, device=device, split="valid")

    model = init_transformer(
        model_config["name_or_path"],
        model_config["dropout"],
        device,
        compile_=compile_,
    )
    optimizer = torch.optim.AdamW(model.parameters(), **train_config["optimizer"])
    scheduler = None
    if train_config["decay_lr"]:
        scheduler = CosineAnnealingLR(optimizer, **train_config["scheduler"])

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    estimate_loss = create_loss_estimator(eval_iters, ctx)

    best_val_loss = float("inf")

    t0 = time.time()
    next_batch = next(train_loader)  # fetch the very first batch
    for it in range(1, max_iters + 1):
        for _ in range(gradient_accumulation_steps):
            batch = next_batch
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
            print(msg)
            loss_logger.info(msg)
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                if it > 0:
                    msg = f"saving checkpoint to {out_dir}"
                    print(msg)
                    loss_logger.info(msg)
                    model.module.save_pretrained(out_dir)
        elif it % log_interval == 0:
            # loss as float. note: this is a CPU-GPU sync point
            loss = batch.loss.item()
            msg = f"TRAIN: {it=}: {loss=:.4f}, time {dt*1000:.2f}ms"
            print(msg)
            loss_logger.info(msg)


if __name__ == "__main__":
    main()

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from shared import (
    create_infinite_dataloader,
    create_lr_scheduler,
    init_model,
    load_checkpoint,
    setup,
)
from tensordict.nn import TensorDictModule
from tensordict.prototype import tensorclass
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from utils import init_ddp, load_and_update_config

HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        batch = torch.stack(batch, dim=0).contiguous()
        batch.batch_size = []
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class Data:
    prompt: torch.Tensor
    target: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class PairedDataset(Dataset):
    def __init__(self, path, block_size):
        self._memmap = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __getitem__(self, idx):
        return Data(
            prompt=torch.from_numpy(
                self._memmap[idx : idx + self.block_size].astype(np.int64)
            ),
            target=torch.from_numpy(
                self._memmap[idx + 1 : idx + self.block_size + 1].astype(np.int64)
            ),
            batch_size=[self.block_size],
        )

    def __len__(self):
        # how many sequences of length block_size + 1 can we extract from the data?
        # the valid starting points for such a sequence are those tokens that aren't in
        # the final block_size positions. so it's just the length of the overall
        # sequence minus the block_size
        return len(self._memmap) - self.block_size


def create_datasets(config):
    data_dir = HERE / "nanoGPT" / "data" / config["dataset"]
    train_data = PairedDataset(data_dir / "train.bin", block_size=config["block_size"])
    val_data = PairedDataset(data_dir / "val.bin", block_size=config["block_size"])

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader


def init_scaler(config):
    # initialize a GradScaler. If enabled=False scaler is a no-op
    return torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))


def init_optimizer(model, config):
    # optimizer
    optimizer = model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        "cuda" if "cuda" in config["device"] else "cpu",
    )
    if config["init_from"] == "resume":
        checkpoint = load_checkpoint(config)
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def create_loss_estimator(config):
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            batch = next(dataloader)
            with ctx:
                model(batch)
            losses[k] = batch.loss.item()
        return losses.mean()

    return estimate_loss


def train(config):
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls
    model, model_kwargs = init_model(config)
    model.to(config["device"])
    scaler = init_scaler(config)
    optimizer = init_optimizer(model, config)

    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(
        model, in_keys=["prompt", "target"], out_keys=["logits", "loss"]
    )

    if config["is_ddp"]:
        model = DDP(model, device_ids=[config["ddp_local_rank"]])
    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    if config["decay_lr"]:
        lr_scheduler = create_lr_scheduler(config)
    else:

        def lr_scheduler(_):
            return config["learning_rate"]

    estimate_loss = create_loss_estimator(config)

    train_loader, val_loader = get_dataloaders(config)

    # training loop
    next_batch = next(train_loader)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = (
        model.module.module if config["is_ddp"] else model.module
    )  # unwrap DDP container if needed
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config["eval_interval"] == 0 and config["master_process"]:
            model.eval()
            losses = {
                "train": estimate_loss(model, train_loader),
                "val": estimate_loss(model, val_loader),
            }
            model.train()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config["out_dir"], "ckpt.pt"))
        if iter_num == 0 and config["eval_only"]:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(config["gradient_accumulation_steps"]):
            if config["is_ddp"]:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == config["gradient_accumulation_steps"] - 1
                )
            batch = next_batch
            with ctx:
                model(batch)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            next_batch = next(train_loader)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(batch.loss).backward()
        # clip the gradient
        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config["log_interval"] == 0 and config["master_process"]:
            # loss as float. note: this is a CPU-GPU sync point
            lossf = batch.loss.item()
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    config["batch_size"] * config["gradient_accumulation_steps"], dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                f"mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break


if __name__ == "__main__":
    config = load_and_update_config("config/train.yaml")
    config.update(init_ddp(config["backend"], config["device"]))

    ctx = setup(config)
    train(config)

    if config["is_ddp"]:
        destroy_process_group()

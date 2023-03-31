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

import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from nanoGPT.model import GPT, GPTConfig
from tensordict.nn import TensorDictModule
from tensordict.prototype import tensorclass
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
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


def create_infinite_dataloader(data, config):
    """
    Creates a dataloader and yields batches from it indefinitely, so that we can request
    batches whenever we like with next.
    """
    dl = DataLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=True,  # TODO: perhaps validation set shouldn't be shuffled?
        collate_fn=Collate(config["device"]),
        drop_last=True,
    )
    while True:
        yield from dl


def load_checkpoint(config):
    ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
    return torch.load(ckpt_path, map_location=config["device"])


def init_model_scratch(config, model_kwargs):
    # attempt to derive vocab_size from the dataset
    meta_path = HERE / "nanoGPT" / "data" / config["dataset"] / "meta.pkl"
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_kwargs["vocab_size"] = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    gptconf = GPTConfig(**model_kwargs)
    return GPT(gptconf)


def init_model_resume(config, model_kwargs):
    print(f"Resuming training from {config['out_dir']}")
    # resume training from a checkpoint.
    checkpoint = load_checkpoint(config)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_kwargs[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_kwargs)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k in state_dict:
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    config["iter_num"] = checkpoint["iter_num"]
    config["best_val_loss"] = checkpoint["best_val_loss"]
    return model


def init_model_gpt2(config, model_kwargs):
    print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
    # initialize from OpenAI GPT-2 weights
    override_args = {"dropout": config["dropout"]}
    model = GPT.from_pretrained(config["init_from"], override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_kwargs[k] = getattr(model.config, k)

    return model


def init_model(config):
    model_kwargs = {
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "bias": config["bias"],
        "vocab_size": None,
        "dropout": config["dropout"],
    }

    if config["init_from"] == "scratch":
        model = init_model_scratch(config, model_kwargs)
    elif config["init_from"] == "resume":
        model = init_model_resume(config, model_kwargs)
    elif config["init_from"].startswith("gpt2"):
        model = init_model_gpt2(config, model_kwargs)

    # crop down the model block size if desired, using model surgery
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        # so that the checkpoint will have the right value
        model_kwargs["block_size"] = config["block_size"]
    model.to(config["device"])

    # compile the model
    # TODO: is this fine here or do we need to init optimizer first?
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    return model, model_kwargs


def init_scaler(config):
    # initialize a GradScaler. If enabled=False scaler is a no-op
    return torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))


def init_optimizer(config):
    # optimizer
    optimizer = model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        device_type,
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


def create_lr_scheduler(config):
    # learning rate decay scheduler (cosine with warmup)
    def scheduler(it):
        # 1) linear warmup for warmup_iters steps
        if it < config["warmup_iters"]:
            return config["learning_rate"] * it / config["warmup_iters"]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config["lr_decay_iters"]:
            return config["min_lr"]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config["warmup_iters"]) / (
            config["lr_decay_iters"] - config["warmup_iters"]
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

    return scheduler


def train(
    model,
    model_kwargs,
    optimizer,
    train_data,
    val_data,
    config,
    master_process=True,
    ddp=False,
):
    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    if config["decay_lr"]:
        lr_scheduler = create_lr_scheduler(config)
    else:

        def lr_scheduler(_):
            return config["learning_rate"]

    estimate_loss = create_loss_estimator(config)

    train_loader = create_infinite_dataloader(train_data, config)
    val_loader = create_infinite_dataloader(val_data, config)

    # training loop
    next_batch = next(train_loader)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    # unwrap DDP container if needed, and unwrap TensorDictModule
    raw_model = model.module.module if ddp else model.module
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % config["eval_interval"] == 0 and master_process:
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
            if ddp:
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
        if iter_num % config["log_interval"] == 0 and master_process:
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

    ddp_config = init_ddp(config["backend"], config["device"])

    if ddp_config["master_process"]:
        os.makedirs(config["out_dir"], exist_ok=True)
    torch.manual_seed(1337 + ddp_config["seed_offset"])

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in config["device"] else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = getattr(torch, config["dtype"])
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model, model_kwargs = init_model(config)
    scaler = init_scaler(config)
    optimizer = init_optimizer(config)

    model = TensorDictModule(
        model, in_keys=["prompt", "target"], out_keys=["logits", "loss"]
    )
    # wrap model into DDP container
    if ddp_config["is_ddp"]:
        model = DDP(model, device_ids=[ddp_config["ddp_local_rank"]])

    train_data, val_data = create_datasets(config)

    train(
        model,
        model_kwargs,
        optimizer,
        train_data,
        val_data,
        config,
        master_process=ddp_config["master_process"],
        ddp=ddp_config["is_ddp"],
    )

    if ddp_config["is_ddp"]:
        destroy_process_group()

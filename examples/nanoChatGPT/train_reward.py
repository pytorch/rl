import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset
from model import GPT, RLHF, GPTConfig
from tensordict.nn import TensorDictModule
from tensordict.prototype import tensorclass
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import init_ddp, load_and_update_config

HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class PairwiseDataset:
    chosen: torch.Tensor
    rejected: torch.Tensor
    reward: Optional[torch.Tensor] = None
    

    @classmethod
    def from_dataset(cls, dataset, max_length):
        # TODO: check dtypes
        data = cls(
            chosen=torch.zeros(len(dataset), max_length, dtype=torch.int32),
            rejected=torch.zeros(len(dataset), max_length, dtype=torch.int32),
            batch_size=[len(dataset)],
        )
        enc = tiktoken.get_encoding("gpt2")
        i = 0

        for sample in tqdm(dataset, total=len(dataset)):
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            if len(chosen.split()) < 5 or len(rejected.split()) < 5:
                continue

            chosen = "\n".join([prompt, chosen])
            rejected = "\n".join([prompt, rejected])

            chosen = enc.encode(
                "<|startoftext|>" + chosen + "<|endoftext|>", allowed_special="all"
            )[-max_length:]
            rejected = enc.encode(
                "<|startoftext|>" + rejected + "<|endoftext|>", allowed_special="all"
            )[-max_length:]

            if chosen == rejected:
                continue

            data[i] = cls(chosen=torch.Tensor(chosen), rejected=torch.Tensor(rejected), batch_size=[])
            i += 1

        # index because we will have skipped some datapoints
        return data[:i]


def setup(config):
    if config["master_process"]:
        os.makedirs(config["out_dir"], exist_ok=True)

    torch.manual_seed(1337 + config["seed_offset"])
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # for later use in torch.autocast
    device_type = "cuda" if "cuda" in config["device"] else "cpu"
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def create_datasets(config):
    # Make pairwise datasets for training
    print("Creating pairwise datasets")
    data_path = "CarperAI/openai_summarize_comparisons"
    train_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="train"), max_length=config["block_size"]
    )
    val_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="test"), max_length=config["block_size"]
    )

    return train_data, val_data


def get_dataloaders(config):

    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(train_data, config)
    val_loader = create_infinite_dataloader(val_data, config)

    return train_loader, val_loader


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

    return model, model_kwargs


# helps estimate an arbitrarily accurate loss over either split using many batches
def create_loss_estimator(config):
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            batch = next(dataloader)
            with ctx:
                reward_chosen = model(batch.chosen)
                reward_rejected = model(batch.rejected)
                loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
            losses[k] = loss.item()
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


def train_reward_model(config):

    # GET DATA
    train_loader, val_loader = get_dataloaders(config)

    # FIXME: Don't like this. include it into model
    model, model_kwargs = init_model(config)
    model = RLHF(model, mode="reward", discrete_reward=False)

    print("Config of model: ", model.config)

    if config["init_multihead_from"] == "scratch":
        print("initializing multihead from scratch")
    else:
        if config["init_multihead_from"] == "resume":
            print(f"Resuming training from {config['out_dir_multihead']}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(config["out_dir_multihead"], "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=config["device"])
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)

    model.to(config["device"])

    model = TensorDictModule(
        model, in_keys=["input"], out_keys=["reward"]
    )

    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # FIXME: which one?
    # optimizer = torch.optim.AdamW(model.model.reward_head.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

    # wrap model into DDP container
    if config["is_ddp"]:
        model = DDP(model, device_ids=[config["ddp_local_rank"]])

    # training loop
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    config["running_mfu"] = -1.0
    raw_model = (
        model.module.module if config["is_ddp"] else model.module
    )  # unwrap DDP container if needed
    loss = None
    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    estimate_loss = create_loss_estimator(config)

    if config["decay_lr"]:
        lr_scheduler = create_lr_scheduler(config)
    else:

        def lr_scheduler(_):
            return config["learning_rate"]

    while True:
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(iter_num)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # # every once in a while evaluate the loss on train and val sets
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

        batch = next(train_loader)

        # TODO: check why is different from std model (missing micro gradients)
        
        # TODO: combine evaluate_loss function with this. it's almost the same thing
        # evaluate the loss
        reward_chosen = model(batch.chosen)
        reward_rejected = model(batch.rejected)
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # timing and logging
        t1 = time.time()
        # dt = t1 - t0
        t0 = t1
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break

    if config["is_ddp"]:
        destroy_process_group()


if __name__ == "__main__":
    config = load_and_update_config("config/train_reward.yaml")
    # set up distributed training
    config.update(init_ddp(config["backend"], config["device"]))

    ctx = setup(config)
    train_reward_model(config)
